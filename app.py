import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import os
import io
from datetime import datetime

# ----------------------------
st.set_page_config(page_title="Segmentation U-Net", layout="centered")

st.title("🧠 Segmentation Médicale avec U-Net")
st.markdown("Ce modèle U-Net effectue une **segmentation automatique** sur des images médicales en niveaux de gris.")

# ----------------------------
# Choix de la taille
target_size = st.selectbox("📐 Taille d'entrée du modèle", options=[128, 256], index=0)

# ----------------------------
@st.cache_resource
def load_unet_model():
    model_path = "unet_model.h5"
    if not os.path.exists(model_path):
        st.error(f"Fichier modèle '{model_path}' introuvable.")
        return None
    model = load_model(model_path, compile=False)
    return model

model = load_unet_model()

# ----------------------------
# Chargement de l'image
uploaded_file = st.file_uploader("📤 Charger une image (format .jpg / .png)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # grayscale
    st.image(image, caption="🖼️ Image d'origine", use_column_width=True)

    # Prétraitement
    image_resized = image.resize((target_size, target_size))
    input_array = np.array(image_resized) / 255.0
    input_array = np.expand_dims(input_array, axis=(0, -1))  # (1, H, W, 1)

    if model is not None:
        with st.spinner("🔍 Segmentation en cours..."):
            pred_mask = model.predict(input_array)[0, :, :, 0]
            binary_mask = (pred_mask > 0.5).astype(np.uint8) * 255

        # Superposition
        orig_img = np.array(image_resized)
        overlay = cv2.addWeighted(orig_img.astype(np.uint8), 0.7, binary_mask.astype(np.uint8), 0.3, 0)

        # Affichage résultats
        st.subheader("📊 Résultat de segmentation :")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(image_resized, caption="Image Redimensionnée", use_column_width=True)
        with col2:
            st.image(binary_mask, caption="Masque", use_column_width=True, clamp=True)
        with col3:
            st.image(overlay, caption="Superposition", use_column_width=True)

        # Télécharger le masque
        st.markdown("### 💾 Télécharger le masque")
        mask_pil = Image.fromarray(binary_mask)
        buf = io.BytesIO()
        mask_pil.save(buf, format="PNG")
        st.download_button(
            label="📥 Télécharger le masque",
            data=buf.getvalue(),
            file_name=f"mask_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            mime="image/png"
        )

    else:
        st.warning("❌ Le modèle n'est pas chargé correctement.")
else:
    st.info("👈 Veuillez charger une image pour commencer.")
