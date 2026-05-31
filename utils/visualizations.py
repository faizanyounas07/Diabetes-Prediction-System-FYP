import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import pandas as pd
import numpy as np


PALETTE = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0"]
sns.set_theme(style="whitegrid", palette="muted")


def fig_class_balance(df: pd.DataFrame):
    counts = df['Outcome'].value_counts()
    labels = ['Non-Diabetic (0)', 'Diabetic (1)']
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, counts.values, color=["#4CAF50", "#F44336"], edgecolor='white', width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, str(val),
                ha='center', va='bottom', fontweight='bold')
    ax.set_title("Class Distribution", fontsize=13, fontweight='bold')
    ax.set_ylabel("Count")
    plt.tight_layout()
    return fig


def fig_correlation_heatmap(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(df.corr(), dtype=bool))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                mask=mask, linewidths=0.5, ax=ax, annot_kws={"size": 8})
    ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


def fig_feature_distributions(df: pd.DataFrame):
    features = [c for c in df.columns if c != 'Outcome']
    n = len(features)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3))
    axes = axes.flatten()
    for i, feat in enumerate(features):
        sns.histplot(data=df, x=feat, hue='Outcome', kde=True, ax=axes[i],
                     palette=["#4CAF50", "#F44336"], alpha=0.6, legend=(i == 0))
        axes[i].set_title(feat, fontweight='bold', fontsize=10)
        axes[i].set_xlabel("")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Feature Distributions by Outcome", fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    return fig


def fig_boxplots(df: pd.DataFrame):
    features = [c for c in df.columns if c != 'Outcome']
    n = len(features)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3))
    axes = axes.flatten()
    for i, feat in enumerate(features):
        sns.boxplot(data=df, x='Outcome', y=feat, ax=axes[i],
            hue='Outcome', palette={0: "#4CAF50", 1: "#F44336"}, legend=False)
        axes[i].set_xticks([0, 1])
        axes[i].set_xticklabels(['Non-Diabetic', 'Diabetic'])

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Box Plots by Outcome", fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def fig_accuracy_comparison(metrics_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(metrics_df))
    width = 0.18
    metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
    for i, (col, color) in enumerate(zip(metric_cols, colors)):
        bars = ax.bar(x + i * width, metrics_df[col], width, label=col, color=color, alpha=0.85)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(metrics_df["Model"], rotation=15, ha='right', fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_title("Model Performance Comparison", fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylabel("Score")
    plt.tight_layout()
    return fig


def fig_confusion_matrix(cm: np.ndarray, model_name: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Non-Diabetic', 'Diabetic'],
                yticklabels=['Non-Diabetic', 'Diabetic'],
                linewidths=1, linecolor='white', annot_kws={"size": 14})
    ax.set_title(f"Confusion Matrix — {model_name}", fontweight='bold')
    ax.set_xlabel("Predicted", fontweight='bold')
    ax.set_ylabel("Actual", fontweight='bold')
    plt.tight_layout()
    return fig


def fig_roc_curves(trained: dict, X_test, y_test):
    from sklearn.metrics import roc_curve, roc_auc_score
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = PALETTE + ["#FF9800"]
    for (name, model), color in zip(trained.items(), colors):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, lw=2)
    ax.plot([0, 1], [0, 1], 'k--', lw=1.2, label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontweight='bold')
    ax.set_ylabel("True Positive Rate", fontweight='bold')
    ax.set_title("ROC Curves — All Models", fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


def fig_feature_importance(fi_df: pd.DataFrame, model_name: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = sns.color_palette("Blues_r", len(fi_df))
    bars = ax.barh(fi_df['Feature'][::-1], fi_df['Importance'][::-1], color=colors)
    ax.set_title(f"Feature Importance — {model_name}", fontsize=13, fontweight='bold')
    ax.set_xlabel("Importance Score", fontweight='bold')
    plt.tight_layout()
    return fig
