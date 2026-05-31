import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, Image as RLImage, HRFlowable
)
import pandas as pd


def _fig_to_image(fig, width_cm=14):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img = RLImage(buf, width=width_cm * cm)
    img.hAlign = 'CENTER'
    return img


def generate_pdf(
    dataset_info: dict,
    metrics_df: pd.DataFrame,
    best_model_name: str,
    prediction_result: dict | None,
    figs: dict,
    output_path: str
):
    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title2', parent=styles['Title'],
                                 fontSize=18, textColor=colors.HexColor('#1565C0'), spaceAfter=6)
    h2 = ParagraphStyle('H2', parent=styles['Heading2'],
                        fontSize=13, textColor=colors.HexColor('#0D47A1'), spaceBefore=12)
    body = styles['BodyText']
    body.fontSize = 10

    story = []

    # Title
    story.append(Paragraph("Diabetes Prediction System — Analytical Report", title_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#1565C0')))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("Government College University Faisalabad | Department of Computer Science", body))
    story.append(Spacer(1, 0.5*cm))

    # Dataset info
    story.append(Paragraph("1. Dataset Summary", h2))
    rows = [["Property", "Value"]] + [[k, str(v)] for k, v in dataset_info.items()]
    t = Table(rows, colWidths=[6*cm, 10*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1565C0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#E3F2FD'), colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#BBDEFB')),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('PADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.5*cm))

    # Model metrics
    story.append(Paragraph("2. Model Performance Metrics", h2))
    metric_rows = [["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]]
    for _, row in metrics_df.iterrows():
        metric_rows.append([row['Model'], str(row['Accuracy']), str(row['Precision']),
                             str(row['Recall']), str(row['F1-Score']), str(row['ROC-AUC'])])
    mt = Table(metric_rows, colWidths=[4.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm])
    mt.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1565C0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#E3F2FD'), colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#BBDEFB')),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('PADDING', (0, 0), (-1, -1), 5),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
    ]))
    story.append(mt)
    story.append(Paragraph(f"<b>Best Model:</b> {best_model_name}", body))
    story.append(Spacer(1, 0.5*cm))

    # Charts
    story.append(Paragraph("3. Visualizations", h2))
    for label, fig in figs.items():
        story.append(Paragraph(label, ParagraphStyle('cap', parent=body, fontSize=9,
                                                      textColor=colors.grey, spaceAfter=4)))
        story.append(_fig_to_image(fig, width_cm=15))
        story.append(Spacer(1, 0.4*cm))

    # Prediction result
    if prediction_result:
        story.append(Paragraph("4. Prediction Result", h2))
        res_rows = [["Parameter", "Value"]] + [[k, str(v)] for k, v in prediction_result.items()]
        rt = Table(res_rows, colWidths=[8*cm, 8*cm])
        rt.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1565C0')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#E3F2FD'), colors.white]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#BBDEFB')),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('PADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(rt)
        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph(
            "⚠️ <b>Medical Disclaimer:</b> This prediction is for educational and research purposes only "
            "and does not constitute professional medical advice. Please consult a qualified healthcare professional.",
            ParagraphStyle('disc', parent=body, textColor=colors.HexColor('#B71C1C'), fontSize=9)
        ))

    doc.build(story)
    return output_path
