import streamlit as st
import base64
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import cv2
from datetime import datetime
from gradcam import model, make_gradcam_heatmap, create_gradcam_overlay
import io
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)

def set_background(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()

    st.markdown(
        f"""
        <style>

        /* Apply uploaded background image */
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}") !important;
            background-size: cover !important;
            background-position: center center !important;
            background-repeat: no-repeat !important;
            background-attachment: fixed !important;
        }}

        /* Make main content areas transparent */
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"] {{
            background: transparent !important;
        }}

        /* Transparent top header */
        [data-testid="stHeader"] {{
            background: transparent !important;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )




def generate_analysis_report(data):

    result = data["result"]
    predictions = np.array(data["predictions"]).flatten()

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        textColor=colors.HexColor("#7C3AED"),
        fontSize=22,
        spaceAfter=20
    )

    story = []

    story.append(Paragraph("RetinaAI", title_style))
    story.append(Paragraph(
        "Diabetic Retinopathy Analysis Report",
        styles["Heading2"]
    ))
    story.append(Spacer(1, 15))

    # Analysis details
    details = [
        ["Analysis Detail", "Result"],
        ["Image Filename", str(result["Filename"])],
        ["Analysis Date", str(result["Time"])],
        ["Predicted Grade", str(result["Grade"])],
        ["Prediction Score", f'{float(result["Probability"]) * 100:.2f}%']
    ]

    table = Table(details, colWidths=[180, 280])

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#7C3AED")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
    ]))

    story.append(table)
    story.append(Spacer(1, 25))

    # Five grade scores
    story.append(Paragraph(
        "Prediction Scores for All Grades",
        styles["Heading2"]
    ))

    score_data = [["DR Grade", "Prediction Score"]]

    for i, score in enumerate(predictions):
        score_data.append([
            str(CLASS_NAMES[i]),
            f"{float(score) * 100:.2f}%"
        ])

    score_table = Table(score_data, colWidths=[220, 240])

    score_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDD6FE")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 9)
    ]))

    story.append(score_table)
    story.append(Spacer(1, 25))

    # Understanding the Result
    story.append(Paragraph(
        "Understanding the Result",
        styles["Heading2"]
    ))

    grade_explanations = {
        "No DR": "No visible signs of diabetic retinopathy were detected. Regular eye examinations are still important.",
        "Mild DR": "Possible early retinal changes. Regular eye check-ups and good diabetes management are important.",
        "Moderate DR": "Possible moderate retinal changes. An ophthalmologist should assess the retina.",
        "Severe DR": "Possible advanced retinal changes. Prompt assessment by an ophthalmologist is recommended.",
        "Proliferative DR": "Possible sight-threatening retinal changes. Prompt specialist eye assessment is important."
    }

    predicted_grade = result["Grade"]
    explanation = grade_explanations.get(predicted_grade, "Unable to interpret this result.")

    story.append(Paragraph(
        f"<b>Predicted Grade:</b> {predicted_grade}",
        styles["BodyText"]
    ))

    story.append(Spacer(1, 8))

    story.append(Paragraph(
        explanation,
        styles["BodyText"]
    ))

    story.append(Spacer(1, 12))

    story.append(Paragraph(
        "<b>Understanding the Model Score:</b> The score represents "
        "the model's output for the predicted class. It is not a "
        "clinically confirmed probability or diagnosis.",
        styles["BodyText"]
    ))

    story.append(Spacer(1, 25))


    # Grad-CAM explanation
    story.append(Paragraph(
        "Explainable AI - Grad-CAM",
        styles["Heading2"]
    ))

    story.append(Paragraph(
        "Grad-CAM highlights image regions that influenced the model's "
        "prediction. Red and orange generally indicate stronger influence, "
        "while blue indicates lower influence. These highlighted regions "
        "do not confirm the presence of diabetic retinopathy lesions.",
        styles["BodyText"]
    ))

    story.append(Spacer(1, 20))

    # Disclaimer
    story.append(Paragraph(
        "Medical Disclaimer",
        styles["Heading2"]
    ))

    story.append(Paragraph(
        "RetinaAI is an educational and research prototype. It is not a "
        "clinically validated diagnostic system. Its predictions and scores "
        "should not be used as a substitute for professional eye examination "
        "or medical advice. Consult a qualified ophthalmologist for diagnosis "
        "and treatment.",
        styles["BodyText"]
    ))

    doc.build(story)

    return buffer.getvalue()

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

def show_dr_guidance(grade):

    guidance = {
        0: {
            "name": "No Diabetic Retinopathy",
            "severity": "No visible DR detected",
            "color": "success",
            "meaning": "No visible signs of diabetic retinopathy were detected in the uploaded image.",
            "symptoms": "Usually no DR-related symptoms.",
            "actions": [
                "Continue regular diabetic eye examinations.",
                "Maintain healthy blood glucose levels.",
                "Control blood pressure and cholesterol."
            ],
            "treatment": "No DR-specific treatment is generally needed. Continue preventive diabetes care.",
        },

        1: {
            "name": "Mild Non-Proliferative DR",
            "severity": "Early stage",
            "color": "success",
            "meaning": "Small changes in retinal blood vessels, such as microaneurysms, may be present.",
            "symptoms": "Usually no noticeable symptoms.",
            "actions": [
                "Schedule regular eye examinations.",
                "Maintain good blood sugar control.",
                "Monitor blood pressure and cholesterol."
            ],
            "treatment": "Usually monitoring and diabetes management are recommended. Treatment depends on eye examination findings."
        },

        2: {
            "name": "Moderate Non-Proliferative DR",
            "severity": "Moderate stage",
            "color": "warning",
            "meaning": "More retinal blood vessels may be damaged or blocked, and small retinal haemorrhages may occur.",
            "symptoms": "Some people experience blurred vision, although symptoms may be absent.",
            "actions": [
                "Arrange an ophthalmologist examination.",
                "Follow the recommended retinal monitoring schedule.",
                "Manage blood sugar, blood pressure and cholesterol."
            ],
            "treatment": "Monitoring may be sufficient in some cases. Treatment may be needed if complications such as diabetic macular oedema develop."
        },

        3: {
            "name": "Severe Non-Proliferative DR",
            "severity": "High risk",
            "color": "danger",
            "meaning": "Extensive blockage of retinal blood vessels may reduce oxygen supply to the retina.",
            "symptoms": "Vision may remain normal, or blurred vision and other visual changes may occur.",
            "actions": [
                "Arrange prompt evaluation by an ophthalmologist.",
                "Do not delay recommended follow-up visits.",
                "Carefully manage diabetes and blood pressure."
            ],
            "treatment": "Close specialist monitoring is important. Laser or anti-VEGF treatment may be considered depending on progression and clinical findings."
        },

        4: {
            "name": "Proliferative Diabetic Retinopathy",
            "severity": "Sight-threatening stage",
            "color": "danger",
            "meaning": "Abnormal new blood vessels may grow on the retina and can bleed or cause retinal detachment.",
            "symptoms": "Floaters, blurred vision, dark areas or sudden vision changes may occur. Some people initially have no symptoms.",
            "actions": [
                "Seek prompt ophthalmologist assessment.",
                "Do not postpone specialist treatment.",
                "Seek immediate eye care for sudden vision loss, flashes or a sudden increase in floaters."
            ],
            "treatment": "Treatment may include anti-VEGF eye injections, laser photocoagulation or vitrectomy surgery, depending on the condition."
        }
    }

    info = guidance.get(int(grade))

    if info is None:
        st.warning("Unable to display DR guidance for this grade.")
        return

    st.markdown("---")
    st.header("Understanding Your Result")

    st.subheader(info["name"])
    st.info(f"Severity: {info['severity']}")

    st.markdown("### What does this mean?")
    st.write(info["meaning"])

    st.markdown("### Possible symptoms")
    st.write(info["symptoms"])

    st.markdown("### Recommended actions")
    for action in info["actions"]:
        st.markdown(f"- {action}")

    st.markdown("### Treatment and recovery")
    st.write(info["treatment"])

    st.markdown("### Prevention and healthy habits")
    st.markdown("""
    - Maintain blood glucose levels as advised by your doctor.
    - Control blood pressure and cholesterol.
    - Follow a balanced diet.
    - Exercise regularly as medically appropriate.
    - Avoid smoking.
    - Attend regular diabetic eye examinations.
    """)

    st.warning(
        "Medical disclaimer: This is educational guidance based on an AI "
        "prediction, not a confirmed diagnosis or personalised treatment plan. "
        "Consult an ophthalmologist for clinical assessment."
    )

st.set_page_config(
    page_title="RetinaAI | DR Detection",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>


@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
[data-testid="stMainBlockContainer"] {
    padding-top: 1rem !important;
}

/* Keep section headings properly positioned */
[data-testid="stMainBlockContainer"] h1 {
    margin-top: 0.5rem !important;
}

.stApp {
    background: transparent;
    color: #1f2937;
}
/* Main content text */
[data-testid="stMain"] {
    color: #1f2937;
}

[data-testid="stMain"] h1,
[data-testid="stMain"] h2,
[data-testid="stMain"] h3 {
    color: #172554 !important;
}

[data-testid="stMain"] p,
[data-testid="stMain"] li,
[data-testid="stMain"] label, {
    color: #374151;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #17152f, #211b42);
    border-right: 1px solid #343052;
}

[data-testid="stSidebar"] * {
    color: #f8fafc !important;
}

[data-testid="stSidebar"] label {
    font-weight: 600 !important;
}

[data-testid="stSidebar"] [role="radiogroup"] {
    gap: 12px;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    background: #252044;
    padding: 12px;
    border-radius: 9px;
    border: 1px solid #38315f;
}

[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: #38305f;
}

.main-title {
    color: #172554 !important;
    font-size: 32px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 25px;
    text-shadow: none;

}



.section-title {
    color: #172554 !important;
    font-size: 20px;
    font-weight: 700;
    margin-top: 15px;
    margin-bottom: 15px;
}

.metric-card {
    background: linear-gradient(135deg, #191e35, #242044);
    border: 1px solid #38365c;
    padding: 22px;
    border-radius: 14px;
    min-height: 125px;
}

.metric-label {
    color: #a5abc0;
    font-size: 13px;
    font-weight: 600;
}

.metric-value {
    color: #ffffff;
    font-size: 30px;
    font-weight: 800;
    margin-top: 10px;
}

.metric-note {
    color: #a78bfa;
    font-size: 12px;
    margin-top: 5px;
}

.panel {
    background: #171d32;
    border: 1px solid #303752;
    border-radius: 14px;
    padding: 20px;
}

.result-card {
    background: linear-gradient(135deg, #211b42, #252d4a);
    border: 1px solid #6755a6;
    border-radius: 14px;
    padding: 24px;
}

.result-grade {
    font-size: 29px;
    font-weight: 800;
    color: #c4b5fd;
}

.stButton > button {
    background: linear-gradient(90deg, #7556d8, #9274ed);
    color: white;
    border: none;
    border-radius: 9px;
    padding: 11px 20px;
    font-weight: 700;
    width: 100%;
}

.stButton > button:hover {
    background: #a78bfa;
    color: white;
}

.stTabs [data-baseweb="tab"] {
    color: #cbd5e1;
}

div[data-testid="stFileUploader"] {
    background: #171d32;
    border: 1px dashed #6655a0;
    border-radius: 12px;
    padding: 15px;
}

hr {
    border-color: #343752;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)
set_background("retinaai_background.png")

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

if "current_result" not in st.session_state:
    st.session_state.current_result = None

# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------

CLASS_NAMES = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR"
]

CLASS_DESCRIPTIONS = [
    "No visible diabetic retinopathy detected.",
    "Mild non-proliferative diabetic retinopathy.",
    "Moderate non-proliferative diabetic retinopathy.",
    "Severe non-proliferative diabetic retinopathy.",
    "Proliferative diabetic retinopathy."
]

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown("""
    <div style="padding:15px 5px 25px 5px;">
        <h1 style="
            color:#c4b5fd;
            font-size:42px;
            font-weight:800;
            margin-bottom:5px;
            line-height:1.2;
        ">
            RetinaAI
        </h1>
        <p style="color:#a5abc0;font-size:14px;margin-top:5px;">
            Intelligent Retinal Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### MAIN MENU")

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "DR Analysis",
            "Prediction History",
            "About Project"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown("### MODEL INFORMATION")

    st.markdown("""
    **Architecture:** EfficientNetB0

    **Explainability:** Grad-CAM

    **Classification:** 5 Grades

    **Dataset:** APTOS 2019
    """)

    st.markdown("---")

    st.caption("RetinaAI | Academic Research Prototype")
    st.caption("Not intended for clinical diagnosis.")

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def metric_card(label, value, note):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)


def get_prediction(image):
    image_resized = image.resize((224, 224))
    img_array = np.array(image_resized).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)[0]

    predicted_class = int(np.argmax(predictions))
    probability = float(predictions[predicted_class])

    heatmap = make_gradcam_heatmap(
        img_array,
        predicted_class
    )

    overlay = create_gradcam_overlay(
        np.array(image_resized),
        heatmap
    )

    return predicted_class, probability, predictions, heatmap, overlay, image_resized


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if page == "Dashboard":

    

    st.markdown(
        '<div class="main-title">Diabetic Retinopathy Detection and Explainable AI Analytics</div>',
        unsafe_allow_html=True
    )

    history = st.session_state.history

    total = len(history)

    no_dr = sum(1 for x in history if x["Grade"] == "No DR")
    detected = total - no_dr

    metric1, metric2, metric3, metric4 = st.columns(4)

    with metric1:
        metric_card("TOTAL ANALYSES", total, "Images processed")

    with metric2:
        metric_card("NO DR", no_dr, "Grade 0 predictions")

    with metric3:
        metric_card("DR DETECTED", detected, "Grades 1 to 4")

    with metric4:
        metric_card("MODEL", "EfficientNetB0", "Transfer learning")

    st.markdown(
        '<div class="section-title">Analysis Overview</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns([1.3, 1])

    with left:

        st.markdown('<div class="panel">', unsafe_allow_html=True)

        st.markdown("#### Diabetic Retinopathy Grade Distribution")

        if total > 0:

            df = pd.DataFrame({
                "Grade": [x["Grade"] for x in history]
            })

            counts = (
                df["Grade"]
                .value_counts()
                .reindex(CLASS_NAMES, fill_value=0)
                .reset_index()
            )

            counts.columns = ["Grade", "Count"]

            fig = px.bar(
                counts,
                x="Grade",
                y="Count",
                color="Grade",
                color_discrete_sequence=[
                    "#60a5fa",
                    "#a78bfa",
                    "#818cf8",
                    "#c084fc",
                    "#f472b6"
                ]
            )

            fig.update_layout(
                paper_bgcolor="#171d32",
                plot_bgcolor="#171d32",
                font_color="#e2e8f0",
                showlegend=False,
                margin=dict(l=10, r=10, t=20, b=10)
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("No analysis data available. Upload an image in DR Analysis to begin.")

        st.markdown('</div>', unsafe_allow_html=True)

    with right:

        

        st.markdown("#### Model Information")

        st.markdown("""
        **Model:** EfficientNetB0

        **Task:** Multi-class classification

        **Output:** Five DR severity grades

        **Visualization:** Grad-CAM

        **Input Size:** 224 × 224 pixels

        **Framework:** TensorFlow / Keras
        """)

    
        


    

    st.markdown(
        '<div class="section-title">Recent Predictions</div>',
        unsafe_allow_html=True
    )

    if history:
        recent = pd.DataFrame(history[-5:][::-1])

        st.dataframe(
            recent,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Prediction history will appear here after image analysis.")

# --------------------------------------------------
# DR ANALYSIS
# --------------------------------------------------

elif page == "DR Analysis":

    st.markdown(
        '<div class="main-title">Retinal Image Analysis</div>',
        unsafe_allow_html=True
    )
    
    

    st.markdown(
        '<div class="subtitle">Upload a retinal fundus image for AI-based diabetic retinopathy grading.</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload Retinal Fundus Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")

        st.markdown("#### Uploaded Image")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(
                image,
                caption="Original Retinal Image",
                use_container_width=True
            )

        with col2:

            st.markdown('<div class="panel">', unsafe_allow_html=True)

            st.markdown("#### Image Details")

            st.write("Filename:", uploaded_file.name)
            st.write("Image dimensions:", f"{image.width} × {image.height}")
            st.write("Format:", image.format or "RGB")

            st.markdown('</div>', unsafe_allow_html=True)

        if st.button("Analyze Retinal Image", use_container_width=True):

            with st.spinner("Analyzing retinal image..."):

                try:

                    predicted_class, probability, predictions, heatmap, overlay, processed_image = get_prediction(image)

                    result = {
                        "Grade": CLASS_NAMES[predicted_class],
                        "Probability": probability,
                        "Time": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                        "Filename": uploaded_file.name
                    }
                    st.session_state.history.append(result)
                    st.session_state.current_result = {
                        "result": result,
                        "predictions": predictions,
                        "image": processed_image,
                        "heatmap": heatmap,
                        "overlay": overlay
                    }

                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")

    if st.session_state.current_result:

        data = st.session_state.current_result

        result = data["result"]
        predictions = data["predictions"]

        st.markdown("---")

        st.markdown(
            '<div class="section-title">Analysis Results</div>',
            unsafe_allow_html=True
        )

        st.markdown(f"""
        <div class="result-card">
            <div style="color:#cbd5e1;font-size:14px;">
                PREDICTED DIABETIC RETINOPATHY GRADE
            </div>
            <div class="result-grade">
                {result["Grade"]}
            </div>
            <div style="color:#cbd5e1;margin-top:8px;">
                {CLASS_DESCRIPTIONS[CLASS_NAMES.index(result["Grade"])]}
            </div>
            <div style="color:#a78bfa;margin-top:12px;">
                Model score: {result["Probability"] * 100:.2f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.caption(
            "The model score is not a clinically calibrated probability."
        )

        st.markdown(
            '<div class="section-title">Explainable AI - Grad-CAM</div>',
            unsafe_allow_html=True
        )

        tab1, tab2, tab3 = st.tabs([
            "Original Image",
            "Grad-CAM Heatmap",
            "Overlay Visualization"
        ])

        with tab1:
            st.image(data["image"], use_container_width=True)

        with tab2:
            st.image(
                data["heatmap"],
                caption="Regions contributing to the predicted class score",
                use_container_width=True,
                clamp=True
            )

        with tab3:
            st.image(
                data["overlay"],
                caption="Original image with Grad-CAM overlay",
                use_container_width=True
            )

        st.markdown(
            '<div class="section-title">Class-wise Prediction Scores</div>',
            unsafe_allow_html=True
        )

        score_df = pd.DataFrame({
            "DR Grade": CLASS_NAMES,
            "Score": predictions
        })

        fig = px.bar(
            score_df,
            x="DR Grade",
            y="Score",
            color="DR Grade",
            text=score_df["Score"].apply(lambda x: f"{x*100:.1f}%"),
            color_discrete_sequence=[
                "#60a5fa",
                "#a78bfa",
                "#818cf8",
                "#c084fc",
                "#f472b6"
            ]
        )

        fig.update_layout(
            paper_bgcolor="#0b1020",
            plot_bgcolor="#171d32",
            font_color="#e2e8f0",
            showlegend=False,
            yaxis=dict(range=[0, 1], tickformat=".0%"),
            margin=dict(l=10, r=10, t=20, b=10)
        )

        st.plotly_chart(fig, use_container_width=True)
        # NEW: DR Explanation and Treatment Guidance
        st.markdown("---")
        show_dr_guidance(CLASS_NAMES.index(result["Grade"]))

        st.warning(
            "This is an academic research prototype, not a medical diagnostic tool. "
            "Consult a qualified eye-care professional for clinical assessment."
        )

        st.markdown("---")
        st.subheader("Download Analysis Report")

        report_data = generate_analysis_report(
            st.session_state.current_result
        )

        st.download_button(
            label="Download PDF Report",
            data=report_data,
            file_name="RetinaAI_Analysis_Report.pdf",
            mime="application/pdf",
            use_container_width=True
        )


# --------------------------------------------------
# PREDICTION HISTORY
# --------------------------------------------------

elif page == "Prediction History":

    st.markdown(
        '<div class="main-title">Prediction History</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Review previously analyzed retinal images.</div>',
        unsafe_allow_html=True
    )

    if st.session_state.history:

        df = pd.DataFrame(st.session_state.history)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Prediction Report",
            data=csv,
            file_name="retinaai_prediction_history.csv",
            mime="text/csv"
        )

        if st.button("Clear Prediction History"):
            st.session_state.history = []
            st.session_state.current_result = None
            st.rerun()

    else:
        st.info("No prediction history available yet.")

# --------------------------------------------------
# ABOUT PROJECT
# --------------------------------------------------

elif page == "About Project":

    st.markdown(
        '<div class="main-title">About RetinaAI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Explainable Deep Learning for Diabetic Retinopathy Grading</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="panel">', unsafe_allow_html=True)

    st.markdown("### Project Objective")

    st.write("""
    RetinaAI is an academic deep learning project designed to classify
    diabetic retinopathy severity from retinal fundus images.

    The system uses transfer learning with EfficientNetB0 and Grad-CAM
    visualization to provide an interpretable view of image regions
    influencing model predictions.
    """)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Technologies Used")

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card("DEEP LEARNING", "TensorFlow", "Model framework")

    with c2:
        metric_card("ARCHITECTURE", "EfficientNetB0", "Transfer learning")

    with c3:
        metric_card("EXPLAINABILITY", "Grad-CAM", "Visual interpretation")

    st.markdown("### Classification Categories")

    grade_df = pd.DataFrame({
        "Grade": [0, 1, 2, 3, 4],
        "Classification": CLASS_NAMES,
        "Description": CLASS_DESCRIPTIONS
    })

    st.dataframe(
        grade_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Dataset")

    st.write("""
    The project uses the APTOS 2019 Blindness Detection dataset.
    It contains retinal fundus images labeled according to five
    diabetic retinopathy severity grades.
    """)

    st.markdown("### Important Disclaimer")

    st.warning("""
    RetinaAI is developed for academic and educational research.
    It has not been clinically validated and must not be used
    to diagnose or make treatment decisions.
    """)



                   