import streamlit as st
from PIL import Image

from services.inference import load_model, predict
from utils.preprocessing import preprocess

# UI SETUP
st.set_page_config(page_title="WildScan AI", layout="centered")

st.title("🐾 WildScan AI")
st.write("Upload an animal image to classify it into 8 categories.")

# LOAD MODEL
@st.cache_resource
def get_model():
    try:
        return load_model()
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

model = get_model()

# CLASS LABELS
classes = [
    "bird", "monkey_prosimian", "leopard", "hog",
    "civet_genet", "antelope_duiker", "blank", "rodent"
]

# FILE UPLOAD
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if model is None:
        st.warning("Model not loaded.")
    else:
        tensor = preprocess(image)
        result, confidence = predict(model, tensor, classes)

        st.success(f"Prediction: **{result}**")
        st.write(f"Confidence: {confidence:.2%}")