import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import os

# ----------------------------
# Load the model
@st.cache_resource
def load_unet_model():
    model_path = "unet_model.h5"
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found.")
        return None
    model = load_model(model_path, compile=False)
    return model

model = load_unet_model()

# ----------------------------
# Title and description
st.title("U-Net Segmentation en ligne 🧠")
st.write("Ce modèle effectue une segmentation d'image médicale avec U-Net.")

# Upload image
uploaded_file = st.file_uploader("Chargez une image (format .png ou .jpg)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Lire et afficher l'image
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    st.image(image, caption="Image d'entrée", use_column_width=True)

    # Prétraitement
    img_resized = image.resize((128, 128))
    input_array = np.array(img_resized) / 255.0
    input_array = np.expand_dims(input_array, axis=(0, -1))  # (1, 128, 128, 1)

    # Prédiction
    if model is not None:
        pred_mask = model.predict(input_array)[0, :, :, 0]
        mask_binary = (pred_mask > 0.5).astype(np.uint8) * 255

        # Affichage du masque
        st.subheader("Masque prédit :")
        st.image(mask_binary, use_column_width=True, clamp=True)
    else:
        st.warning("Le modèle n'a pas été chargé.")
