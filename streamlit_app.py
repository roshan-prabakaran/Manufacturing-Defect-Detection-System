import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Page configuration
st.set_page_config(
    page_title="Manufacturing Defect Detection",
    page_icon="🔍",
    layout="centered"
)

# Title and Description
st.title("🛡️ Manufacturing Defect Detection")
st.markdown("""
Upload an image of a casting product to detect if it has any surface defects.
This system uses a **Convolutional Neural Network (CNN)** trained on industrial quality inspection data.
""")

# Load the model
MODEL_PATH = 'best_defect_model.keras'

@st.cache_resource
def load_defect_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    return None

model = load_defect_model()

if model is None:
    st.error(f"Model file '{MODEL_PATH}' not found. Please train the model first using 'train.py'.")
else:
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Preprocessing
        st.write("🔍 Analyzing...")
        
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Resize to match model input (128x128)
        img_resized = image.resize((128, 128))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Prediction
        prediction = model.predict(img_array)[0][0]
        
        # Display Results
        st.divider()
        if prediction > 0.5:
            st.success(f"### Result: ✅ OK (Non-Defective)")
            st.metric("Confidence Score", f"{prediction*100:.2f}%")
        else:
            st.error(f"### Result: ❌ DEFECTIVE")
            st.metric("Confidence Score", f"{(1 - prediction)*100:.2f}%")
            
        st.info("Note: The model identifies surface defects like pinholes, burrs, and scratches.")

# Footer
st.divider()
st.caption("Developed for DL Project - Defect Detection using CNN")
