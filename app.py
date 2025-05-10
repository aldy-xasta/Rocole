import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

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
# Load Model EfficientNet
# ============================
@st.cache_resource
def load_model():
    model = EfficientNet.from_name('efficientnet-b0', num_classes=3)
    model.load_state_dict(torch.load("model/GCLD_EfficientNet.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

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
        <p class="diagnose-desc">Upload gambar daun kopi untuk analisis penyakit secara otomatis.</p>
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ============================
# Upload Gambar
# ============================
image = None
uploaded_file = st.file_uploader("Upload file gambar daun", type=["jpg", "jpeg", "png"], key="file_uploader")

if uploaded_file is not None:
    try:
        filename = getattr(uploaded_file, "name", "").lower()
        if filename.endswith((".jpg", ".jpeg", ".png")):
            image = Image.open(uploaded_file)
        else:
            st.error("❌ Format file tidak valid. Gunakan JPG, JPEG, atau PNG.")
            st.stop()
    except Exception as e:
        st.error("❌ Gagal membaca file. Coba upload ulang.")
        st.stop()

# ============================
# Diagnosa
# ============================
if image is not None:
    st.image(image, caption="Gambar Daun", use_container_width=True)

    if st.button("Diagnosa Sekarang"):
        with st.spinner("Menganalisis gambar..."):
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
            image_rgb = image.convert("RGB")
            img_tensor = transform(image_rgb).unsqueeze(0)

            with torch.no_grad():
                output = model(img_tensor)
                pred_label = torch.argmax(output, dim=1).item()

            # Update label sesuai klasifikasi baru
            labels = ["Sehat", "Red Spider Mite", "Rush"]
            hasil = labels[pred_label]

        # Warna kartu hasil berdasarkan label
        if hasil == "Sehat":
             result_class = "result-healthy"
        elif hasil == "Red Spider Mite":
            result_class = "result-warning"
        else:  # Rush
            result_class = "result-sick"

        st.markdown(f"""
        <div class="result-card">
            <div class="diagnosis-result {result_class}">
                <strong>Hasil Diagnosa:</strong> Daun menunjukkan gejala <em>{hasil}</em>
            </div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown("<p style='text-align: center; color: gray;'>Silakan upload gambar terlebih dahulu.</p>", unsafe_allow_html=True)

# ============================
# Footer
# ============================
st.markdown("---")
st.markdown("<p class='footer'>© DSAI2024 - COMVIS - GROUP 5</p>", unsafe_allow_html=True)
