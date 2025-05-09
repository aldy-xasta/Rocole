import streamlit as st
from PIL import Image
import numpy as np
# import tensorflow as tf  # Aktifkan jika model siap
# from tensorflow.keras.models import load_model

# ============================
# Konfigurasi Halaman
# ============================
st.set_page_config(page_title="RoCole Lite - Diagnosa Daun Kopi", layout="centered")

# ============================
# Sisipkan Custom CSS
# ============================
with open("css/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ============================
# Header / Branding
# ============================
st.markdown("<h1 class='main-title'>🌿 RoCole Lite</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Deteksi penyakit pada daun kopi</p>", unsafe_allow_html=True)
st.markdown("---")

# ============================
# Kotak Diagnosis
# ============================
st.markdown("""
    <div class="diagnose-card">
        <h5 class="diagnose-title">Start Diagnosis</h5>
        <p class="diagnose-desc">Upload gambar daun atau ambil foto untuk analisis penyakit daun kopi.</p>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    take_photo_clicked = st.button("Take a Photo", use_container_width=True)
with col2:
    upload_image_clicked = st.button("Upload Image", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# ============================
# Input Kamera / Upload
# ============================
image = None
if take_photo_clicked:
    camera_photo = st.camera_input("Ambil Foto Daun")
    if camera_photo:
        image = Image.open(camera_photo)

if upload_image_clicked:
    uploaded_file = st.file_uploader("Pilih file gambar daun", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)

# ============================
# Diagnosa
# ============================
if image is not None:
    st.image(image, caption=" Gambar Daun", use_column_width=True)

    if st.button("Diagnosa Sekarang"):
        with st.spinner("Menganalisis gambar..."):

            # ================================
            # Jika model sudah siap, aktifkan ini:
            # ================================
            # model = load_model("model/leaf_model.h5")
            # img = image.resize((224, 224))  # Sesuaikan input model
            # img_array = np.array(img) / 255.0
            # img_array = np.expand_dims(img_array, axis=0)
            # prediction = model.predict(img_array)
            # pred_label = np.argmax(prediction)
            # labels = ["Sehat", "Leaf Blight", "Leaf Rust"]
            # hasil = labels[pred_label]

            # ================================
            # Sementara (dummy)
            hasil = "Leaf Blight"

        st.success(f"Hasil Diagnosa: Daun menunjukkan gejala *{hasil}* ")
else:
    st.markdown("<p style='text-align: center; color: gray;'>Silakan pilih metode input untuk memulai diagnosis.</p>", unsafe_allow_html=True)

# ============================
# Footer
# ============================
st.markdown("---")
st.markdown("<p class='footer'>© DSAI2024 - COMVIS - GROUP 5</p>", unsafe_allow_html=True)
