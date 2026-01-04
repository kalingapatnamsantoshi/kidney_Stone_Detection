import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ======================================================
# PAGE CONFIG (FIRST STREAMLIT COMMAND)
# ======================================================
st.set_page_config(
    page_title="Kidney Stone Detector",
    page_icon="🩺",
    layout="centered"
)

# ======================================================
# FORCE LIGHT MODE + CUSTOM UI
# ======================================================
st.markdown("""
<style>
    :root {
        color-scheme: light;
    }

    html, body {
        background-color: #f7f9fc !important;
        color: #000000 !important;
    }

    .stApp {
        background-color: #f7f9fc;
    }

    .title-text {
        font-size: 36px;
        font-weight: 700;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 10px;
    }

    .subtitle-text {
        font-size: 16px;
        color: #555;
        text-align: center;
        margin-bottom: 30px;
    }

    .card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 25px;
        color: #000000;
    }

    .footer {
        text-align: center;
        color: #666;
        font-size: 13px;
        margin-top: 30px;
    }
</style>
""", unsafe_allow_html=True)

# ======================================================
# LOAD MODEL
# ======================================================
MODEL_PATH = "kidney_stone_detector.h5"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# ======================================================
# HEADER
# ======================================================
st.markdown('<div class="title-text">🩺 Kidney Stone Detection System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle-text">AI-powered medical image analysis using Deep Learning</div>',
    unsafe_allow_html=True
)

# ======================================================
# INFO CARD
# ======================================================
st.markdown("""
<div class="card">
<b>How this system works:</b><br>
Upload a kidney scan image and the AI model will analyze it to determine
whether a kidney stone is present or not.
</div>
""", unsafe_allow_html=True)

# ======================================================
# UPLOAD & RESULT CARD
# ======================================================
st.markdown('<div class="card">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📤 Upload Kidney Scan Image (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # 👇 SMALL, CLEAN IMAGE DISPLAY
    st.image(
        image,
        caption="Uploaded Image",
        width=300
    )

    # Preprocess
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    with st.spinner("🔍 Analyzing image..."):
        prediction = float(model.predict(img_array)[0])

    st.markdown("### 🧪 Analysis Result")

    if prediction > 0.5:
        st.error("🚨 **Kidney Stone Detected**")
        st.write("⚠️ Please consult a qualified medical professional.")
    else:
        st.success("✅ **No Kidney Stone Detected**")

    st.info(f"**Confidence Score:** {prediction:.2f}")

st.markdown('</div>', unsafe_allow_html=True)

# ======================================================
# DISCLAIMER
# ======================================================
st.markdown("""
<div class="card">
<b>⚠️ Medical Disclaimer:</b><br>
This application is for educational purposes only.
It should not be used as a final medical diagnosis.
Always consult a healthcare professional.
</div>
""", unsafe_allow_html=True)

# ======================================================
# FOOTER
# ======================================================
st.markdown("""
<div class="footer">
Built with ❤️ using Deep Learning & Streamlit<br>
Project by Santoshi
</div>
""", unsafe_allow_html=True)
