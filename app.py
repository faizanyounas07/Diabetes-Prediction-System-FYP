"""
Diabetes Prediction System — Streamlit App
GCUF | Department of Computer Science
"""

import streamlit as st
import pandas as pd
import numpy as np
import os, io, tempfile
from sklearn.model_selection import cross_val_score


# ── local imports ──────────────────────────────────────────────────────────────
from utils.preprocessing import load_data, clean_data, get_summary, split_and_scale
from utils.ml_models import (
    train_all, evaluate_all, get_confusion,
    get_feature_importance, get_crossval, save_model, predict_single
)
from utils.visualizations import (
    fig_class_balance, fig_correlation_heatmap, fig_feature_distributions,
    fig_boxplots, fig_accuracy_comparison, fig_confusion_matrix,
    fig_roc_curves, fig_feature_importance
)
from utils.report import generate_pdf

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background: linear-gradient(180deg,#0D47A1,#1976D2); }
[data-testid="stSidebar"] * { color: white !important; }
.metric-card {
    background:#E3F2FD; border-radius:12px; padding:18px 14px;
    text-align:center; border-left:4px solid #1565C0;
}
.metric-card h2 { font-size:2rem; color:#0D47A1; margin:0; }
.metric-card p  { font-size:.85rem; color:#555; margin:0; }
.result-diabetic    { background:#FFEBEE; border:2px solid #F44336; border-radius:12px; padding:20px; }
.result-nondiabetic { background:#E8F5E9; border:2px solid #4CAF50; border-radius:12px; padding:20px; }
.disclaimer { background:#FFF3E0; border-left:4px solid #FF9800;
              padding:12px 16px; border-radius:6px; font-size:.85rem; }
</style>
""", unsafe_allow_html=True)

# ── session state ──────────────────────────────────────────────────────────────
for key in ['df', 'df_clean', 'trained', 'metrics', 'scaler',
            'X_test', 'y_test', 'feature_names', 'best_model_name',
            'last_prediction', 'X_train']:
    if key not in st.session_state:
        st.session_state[key] = None

# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🩺 Diabetes Prediction\n**System**")
    st.markdown("---")
    page = st.radio("Navigate", [
        "📂 Data Upload & EDA",
        "🤖 Train & Evaluate Models",
        "🔮 Predict Diabetes Risk",
        "📄 Export Report",
    ])
    st.markdown("---")
    st.markdown("**GCUF** | Computer Science\nFaizan · Talha · Hamza")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Data Upload & EDA
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📂 Data Upload & EDA":
    st.title("📂 Dataset Upload & Exploratory Analysis")

    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded = st.file_uploader("Upload CSV dataset", type=['csv'])
    with col2:
        use_sample = st.button("📦 Use Pima Indians Sample Dataset", use_container_width=True)

    # Load sample dataset programmatically if requested
    if use_sample:
        from sklearn.datasets import make_classification
        # Generate a realistic approximation of Pima dataset
        np.random.seed(42)
        n = 768
        data = {
            'Pregnancies':           np.random.poisson(3.8, n).clip(0, 17).astype(int),
            'Glucose':               np.random.normal(121, 32, n).clip(44, 199).astype(int),
            'BloodPressure':         np.random.normal(69, 19, n).clip(0, 122).astype(int),
            'SkinThickness':         np.random.normal(20, 16, n).clip(0, 99).astype(int),
            'Insulin':               np.random.exponential(80, n).clip(0, 846).astype(int),
            'BMI':                   np.random.normal(32, 7, n).clip(0, 67).round(1),
            'DiabetesPedigreeFunction': np.random.exponential(0.47, n).clip(0.078, 2.42).round(3),
            'Age':                   np.random.normal(33, 12, n).clip(21, 81).astype(int),
        }
        df_tmp = pd.DataFrame(data)
        # Make outcome correlated with glucose + BMI
        score = (df_tmp['Glucose'] / 200 + df_tmp['BMI'] / 80 +
                 df_tmp['Age'] / 100 + np.random.normal(0, 0.2, n))
        df_tmp['Outcome'] = (score > score.quantile(0.65)).astype(int)
        st.session_state['df'] = df_tmp
        st.success("✅ Sample dataset loaded — 768 records, 8 features.")

    if uploaded:
        st.session_state['df'] = load_data(uploaded)
        st.success(f"✅ Uploaded: {uploaded.name}")

    df = st.session_state['df']
    if df is None:
        st.info("👆 Upload a CSV file or use the sample dataset to begin.")
        st.stop()

    # Clean
    df_clean = clean_data(df)
    st.session_state['df_clean'] = df_clean
    summary = get_summary(df_clean)

    # Quick stats
    st.markdown("### Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><h2>{summary["shape"][0]}</h2><p>Records</p></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><h2>{summary["shape"][1]}</h2><p>Features</p></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><h2>{summary["class_balance"].get(1, 0)}</h2><p>Diabetic</p></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><h2>{summary["class_balance"].get(0, 0)}</h2><p>Non-Diabetic</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Preview (first 10 rows)")
    st.dataframe(df_clean.head(10), use_container_width=True)

    st.subheader("Descriptive Statistics")
    st.dataframe(summary['describe'].style.background_gradient(cmap='Blues'), use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Visualizations")

    tab1, tab2, tab3, tab4 = st.tabs(["Class Balance", "Correlations", "Distributions", "Box Plots"])

    with tab1:
        st.pyplot(fig_class_balance(df_clean))

    with tab2:
        st.pyplot(fig_correlation_heatmap(df_clean))

    with tab3:
        st.pyplot(fig_feature_distributions(df_clean))

    with tab4:
        st.pyplot(fig_boxplots(df_clean))


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Train & Evaluate
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Train & Evaluate Models":
    st.title("🤖 Model Training & Evaluation")

    if st.session_state['df_clean'] is None:
        st.warning("⚠️ Please upload a dataset first.")
        st.stop()

    df_clean = st.session_state['df_clean']

    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
    with col2:
        cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)

    if st.button("🚀 Train All Models", type="primary", use_container_width=True):
        with st.spinner("Training Logistic Regression, Decision Tree, Random Forest, SVM..."):
            X_train, X_test, y_train, y_test, scaler, feat_names = split_and_scale(
                df_clean, test_size=test_size
            )
            trained = train_all(X_train, y_train)
            metrics = evaluate_all(trained, X_test, y_test)

            best_name = metrics.iloc[0]['Model']
            save_model(trained[best_name], scaler,
                 path="models/best_model.pkl")

            st.session_state.update({
                'trained': trained, 'metrics': metrics, 'scaler': scaler,
                'X_test': X_test, 'y_test': y_test, 'X_train': X_train,
                'feature_names': feat_names, 'best_model_name': best_name,
            })
        st.success(f"✅ All 4 models trained! Best model: **{best_name}**")

    if st.session_state['trained'] is None:
        st.stop()

    trained = st.session_state['trained']
    metrics = st.session_state['metrics']
    X_test  = st.session_state['X_test']
    y_test  = st.session_state['y_test']
    X_train = st.session_state['X_train']
    feat_names = st.session_state['feature_names']
    best_name  = st.session_state['best_model_name']

    st.markdown("---")
    st.subheader("📈 Performance Metrics")
    st.dataframe(
        metrics.style.background_gradient(subset=['Accuracy','Precision','Recall','F1-Score','ROC-AUC'],
                                          cmap='Blues'),
        use_container_width=True
    )

    st.subheader("Cross-Validation (Best Model)")
    cv_scores = cross_val_score(trained[best_name], X_train, y_train,
                            cv=cv_folds, scoring='accuracy')
    cvc1, cvc2 = st.columns(2)
    cvc1.metric("CV Mean Accuracy", f"{cv_scores.mean():.4f}")
    cvc2.metric("CV Std Dev", f"{cv_scores.std():.4f}")

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["Model Comparison", "ROC Curves", "Confusion Matrix", "Feature Importance"])

    with tab1:
        st.pyplot(fig_accuracy_comparison(metrics))

    with tab2:
        st.pyplot(fig_roc_curves(trained, X_test, y_test))

    with tab3:
        model_choice = st.selectbox("Select Model", list(trained.keys()))
        cm = get_confusion(trained[model_choice], X_test, y_test)
        st.pyplot(fig_confusion_matrix(cm, model_choice))

    with tab4:
        fi_choice = st.selectbox("Select Model for Importance", ["Random Forest", "Logistic Regression"])
        fi_df = get_feature_importance(trained[fi_choice], feat_names)
        if fi_df is not None:
            st.pyplot(fig_feature_importance(fi_df, fi_choice))
        else:
            st.info("Feature importance not available for this model.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Predict
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict Diabetes Risk":
    st.title("🔮 Diabetes Risk Prediction")

    if st.session_state['trained'] is None:
        st.warning("⚠️ Please train models first.")
        st.stop()

    trained   = st.session_state['trained']
    scaler    = st.session_state['scaler']
    best_name = st.session_state['best_model_name']

    st.markdown(f"Using model: **{best_name}** *(highest accuracy)*")
    st.markdown("---")
    st.subheader("Enter Patient Health Parameters")

    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Pregnancies",          0, 20, 1)
        glucose     = st.number_input("Glucose Level (mg/dL)",44, 200, 110)
        bp          = st.number_input("Blood Pressure (mmHg)",24, 122, 70)
        skin        = st.number_input("Skin Thickness (mm)",  0, 100, 20)
    with col2:
        insulin     = st.number_input("Insulin (mu U/mL)",    0, 850, 80)
        bmi         = st.number_input("BMI",                  10.0, 70.0, 28.0, step=0.1)
        dpf         = st.number_input("Diabetes Pedigree Function", 0.05, 2.5, 0.47, step=0.01)
        age         = st.number_input("Age",                  21, 81, 33)

    if st.button("🔍 Predict", type="primary", use_container_width=True):
        values = [pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]
        result = predict_single(trained[best_name], scaler, values)
        st.session_state['last_prediction'] = result

        st.markdown("---")
        if result['prediction'] == 1:
            st.markdown(f"""
            <div class="result-diabetic">
            <h2 style="color:#C62828">⚠️ Diabetic</h2>
            <p style="font-size:1.1rem">Risk Probability: <b>{result['probability_diabetic']}%</b></p>
            <p>The model predicts a HIGH likelihood of diabetes based on the entered health parameters.</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-nondiabetic">
            <h2 style="color:#2E7D32">✅ Non-Diabetic</h2>
            <p style="font-size:1.1rem">Non-Diabetic Probability: <b>{result['probability_non_diabetic']}%</b></p>
            <p>The model predicts a LOW likelihood of diabetes based on the entered health parameters.</p>
            </div>""", unsafe_allow_html=True)

        # Probability gauge
        col1, col2 = st.columns(2)
        col1.metric("Diabetic Probability",     f"{result['probability_diabetic']}%")
        col2.metric("Non-Diabetic Probability", f"{result['probability_non_diabetic']}%")

        # Feature importance for this prediction
        fi_df = get_feature_importance(trained[best_name], st.session_state['feature_names'])
        if fi_df is not None:
            st.subheader("Key Factors Influencing This Prediction")
            st.pyplot(fig_feature_importance(fi_df, best_name))

    st.markdown("---")
    st.markdown("""
    <div class="disclaimer">
    ⚠️ <b>Medical Disclaimer:</b> This system is for <b>educational and research purposes only</b>.
    Predictions do not constitute professional medical advice and should not replace
    consultation with a qualified healthcare professional.
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Export Report
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📄 Export Report":
    st.title("📄 Export Analytical Report")

    if st.session_state['trained'] is None or st.session_state['df_clean'] is None:
        st.warning("⚠️ Complete training before exporting.")
        st.stop()

    df_clean   = st.session_state['df_clean']
    trained    = st.session_state['trained']
    metrics    = st.session_state['metrics']
    X_test     = st.session_state['X_test']
    y_test     = st.session_state['y_test']
    scaler     = st.session_state['scaler']
    feat_names = st.session_state['feature_names']
    best_name  = st.session_state['best_model_name']
    last_pred  = st.session_state['last_prediction']

    st.success("✅ All components are ready. Click below to generate the PDF report.")

    if st.button("📥 Generate & Download PDF Report", type="primary", use_container_width=True):
        with st.spinner("Generating report..."):
            summary = get_summary(df_clean)
            dataset_info = {
                "Total Records":    summary['shape'][0],
                "Total Features":   summary['shape'][1],
                "Diabetic Cases":   summary['class_balance'].get(1, 0),
                "Non-Diabetic":     summary['class_balance'].get(0, 0),
                "Best ML Model":    best_name,
                "Best Accuracy":    metrics.iloc[0]['Accuracy'],
                "Best ROC-AUC":     metrics.iloc[0]['ROC-AUC'],
            }
            figs = {
                "Class Distribution":        fig_class_balance(df_clean),
                "Correlation Heatmap":       fig_correlation_heatmap(df_clean),
                "Model Performance Chart":   fig_accuracy_comparison(metrics),
                "ROC Curves":                fig_roc_curves(trained, X_test, y_test),
            }
            cm = get_confusion(trained[best_name], X_test, y_test)
            figs[f"Confusion Matrix ({best_name})"] = fig_confusion_matrix(cm, best_name)

            fi_df = get_feature_importance(trained[best_name], feat_names)
            if fi_df is not None:
                figs[f"Feature Importance ({best_name})"] = fig_feature_importance(fi_df, best_name)

            tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            generate_pdf(dataset_info, metrics, best_name, last_pred, figs, tmp.name)

            with open(tmp.name, 'rb') as f:
                pdf_bytes = f.read()

        st.download_button(
            label="⬇️ Download PDF Report",
            data=pdf_bytes,
            file_name="Diabetes_Prediction_Report.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    st.markdown("---")
    st.subheader("Export Metrics as CSV")
    if metrics is not None:
        csv = metrics.to_csv(index=False)
        st.download_button("⬇️ Download Metrics CSV", csv,
                           file_name="model_metrics.csv", mime="text/csv",
                           use_container_width=True)
