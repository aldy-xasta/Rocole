import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from efficientnet_pytorch import EfficientNet

# ============================
# Konfigurasi Halaman
# ============================
st.set_page_config(page_title="RoCole Lite - Diagnosa Daun Kopi", layout="centered")

with open("css/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ============================
# Load Models
# ============================
@st.cache_resource
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # COCO Pre-trained (Manual Load)
    coco_model = fasterrcnn_resnet50_fpn(pretrained=False)
    coco_weights = torch.load("model/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth", map_location=device)
    coco_model.load_state_dict(coco_weights)
    coco_model.to(device).eval()

    # Custom Leaf Detection Model
    leaf_model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = leaf_model.roi_heads.box_predictor.cls_score.in_features
    leaf_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    leaf_model.load_state_dict(torch.load("model/faster_rcnn_leaf_epoch_10.pth", map_location=device))
    leaf_model.to(device).eval()

    # EfficientNet Classifier
    classifier = EfficientNet.from_name('efficientnet-b0', num_classes=3)
    classifier.load_state_dict(torch.load("model/GCLD_EfficientNet.pth", map_location=device))
    classifier.to(device).eval()

    return coco_model, leaf_model, classifier, device

coco_model, leaf_model, classifier, device = load_models()

# ============================
# Processing Functions
# ============================
def detect_objects(model, image_pil, device, threshold=0.3):
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image_pil).to(device)
    with torch.no_grad():
        prediction = model([image_tensor])
    boxes = prediction[0]['boxes'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    return boxes[scores >= threshold].astype(int), scores[scores >= threshold]

def apply_grabcut(image_pil, box):
    image = np.array(image_pil)
    mask = np.zeros(image.shape[:2], np.uint8)
    x1, y1, x2, y2 = box
    rect = (x1, y1, x2 - x1, y2 - y1)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    segmented = image * mask2[:, :, np.newaxis]
    return segmented

# --- Enhanced GrabCut Function ---
def enhanced_grabcut(img, bbox, save_steps=False):
    # Extract bounding box coordinates
    x_min, y_min, x_max, y_max = bbox
    img = np.array(img)

    # Initialize result dictionary to store all stages if needed
    steps = [] if save_steps else None

    # STEP 1: Initial GrabCut preparation
    # -----------------------------------
    # Create initial mask
    mask = np.zeros(img.shape[:2], np.uint8)

    # Define rectangle for GrabCut (use the provided bounding box)
    rect = (x_min, y_min, x_max - x_min, y_max - y_min)

    # GrabCut model parameters
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Mark the area inside bounding box as probable foreground
    cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), cv2.GC_PR_FGD, -1)
    marked_img = img.copy()
    cv2.rectangle(marked_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # STEP 2: Run initial GrabCut
    # --------------------------
    # Apply GrabCut with rectangle initialization
    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # Create binary mask for results (0=background, 1=foreground)
    initial_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype('uint8')

    # STEP 3: Image enhancement for better leaf segmentation
    # ----------------------------------------------------
    # Convert to HSV for better contrast of green leaves
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)

    # Enhance saturation and value to make leaves more distinct
    s = cv2.equalizeHist(s)
    v = cv2.equalizeHist(v)

    # Merge back to HSV and convert to BGR
    enhanced_hsv = cv2.merge([h, s, v])
    enhanced_img = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)


    # STEP 4: Contour functions to refine leaf boundary (previously step 6.3)
    # ---------------------------------------------------------------------
    # Find contours in the initial mask
    contours, _ = cv2.findContours(initial_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a new mask for refined contours
    contour_mask = np.zeros_like(initial_mask)

    if len(contours) > 0:
        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Find the center of the bounding box
        bbox_center_x = (x_min + x_max) // 2
        bbox_center_y = (y_min + y_max) // 2

        # Select the largest contour that's near the center
        main_contour = contours[0]  # Default to largest

        # Check if multiple large contours exist
        if len(contours) > 1:
            min_dist = float('inf')
            for i, contour in enumerate(contours[:3]):  # Check top 3 contours
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    contour_cx = int(M["m10"] / M["m00"])
                    contour_cy = int(M["m01"] / M["m00"])

                    # Calculate distance from bbox center
                    dist = np.sqrt((contour_cx - bbox_center_x)**2 + (contour_cy - bbox_center_y)**2)

                    # If this contour is closer to center than our current selection, use it instead
                    if i == 0 or dist < min_dist:
                        min_dist = dist
                        main_contour = contour

        # Draw the main contour (filled)
        cv2.drawContours(contour_mask, [main_contour], 0, 1, -1)
    else:
        # If no contours found, use initial mask
        contour_mask = initial_mask

    # STEP 5: Apply morphological operations to smooth the contour mask (previously part of step 6.3)
    # ------------------------------------------------------------------------------------------
    kernel = np.ones((5, 5), np.uint8)
    morph_mask = cv2.morphologyEx(contour_mask, cv2.MORPH_CLOSE, kernel)
    morph_mask = cv2.morphologyEx(morph_mask, cv2.MORPH_OPEN, kernel)

    # STEP 6: Bitwise operation to isolate leaf (previously step 6.4)
    # ------------------------------------------------------------
    # Apply bitwise AND between original image and final mask
    final_result = cv2.bitwise_and(img, img, mask=morph_mask)

    # TAMBAHAN STEP
    # ------------------------------------------------------------
    gray = cv2.cvtColor(final_result, cv2.COLOR_BGR2GRAY)

    # Thresholding dengan Otsu untuk memisahkan objek dan latar belakang
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2_imshow(thresh)

    # Menghilangkan noise dengan operasi morfologi
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)

    # Area latar belakang pasti (sure background) dengan dilasi lebih banyak
    sure_bg = cv2.dilate(opening, kernel, iterations=5)

    # Mencari area foreground pasti menggunakan distance transform
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Menentukan area yang tidak diketahui
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Label marker untuk connected components
    ret, markers = cv2.connectedComponents(sure_fg)

    # Menambahkan satu pada semua label agar latar belakang pasti bukan 0, tapi 1
    markers = markers + 1

    # Menandai area yang tidak diketahui dengan nol
    markers[unknown == 255] = 0

    # Menerapkan algoritma watershed untuk segmentasi
    markers = cv2.watershed(final_result, markers)

    # Menandai batas kontur dengan warna merah
    # img[markers == -1] = [255, 0, 0]  # Batas kontur berwarna merah

    # # Menampilkan hasil segmentasi
    # cv2_imshow(img)

    # Memberikan warna yang berbeda pada setiap objek yang terdeteksi
    output_img = np.zeros_like(final_result, dtype=np.uint8)

    # Menggunakan warna berbeda untuk setiap objek yang terdeteksi
    for i in range(2, np.max(markers) + 1):  # Mulai dari 2 karena 1 adalah latar belakang
        output_img[markers == i] = [np.random.randint(200, 256), np.random.randint(0, 100), np.random.randint(0, 100)]  # Warna acak untuk setiap objek


    # Convert the output image to grayscale before finding contours
    gray_output = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
    # cv2_imshow(gray_output)
    ret, thresh = cv2.threshold(gray_output, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2_imshow(thresh)

    # Now find contours on the grayscale image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Menyorot kontur pada gambar asli
    result_image = img.copy()
    cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)  # Menggambar kontur dengan warna hijau

    # Membuat mask dengan kontur yang terdeteksi
    mask5 = np.zeros_like(gray_output)

    # Menggambar kontur pada mask
    cv2.drawContours(mask5, contours, -1, (255), thickness=cv2.FILLED)

    # Melakukan bitwise AND antara gambar asli dan mask
    final_output = cv2.bitwise_and(img, img, mask=mask5)

    # Menyimpan hasil ke file jika perlu
    file_result_image = img.copy()

    return final_output

def classify_leaf(image_pil):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image_pil.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        output = classifier(img_tensor)
        pred_prob = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred_label = np.argmax(pred_prob)
    labels = ["Healthy", "Red Spider Mite", "Rush"]
    return labels[pred_label], pred_prob[pred_label]

# ============================
# Streamlit Interface
# ============================
st.markdown("<h1 class='main-title'>🌿 RoCole Lite</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Deteksi & Diagnosa Penyakit Daun Kopi</p>", unsafe_allow_html=True)
st.markdown("---")

uploaded_file = st.file_uploader("Upload gambar daun kopi", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image_pil = Image.open(uploaded_file).convert("RGB")
        st.image(image_pil, caption="Gambar Diupload", use_container_width=True)

        if st.button("Diagnosa Sekarang"):
            with st.spinner("Memproses gambar..."):
                # Deteksi daun
                leaf_boxes, leaf_scores = detect_objects(leaf_model, image_pil, device)
                if len(leaf_boxes) == 0:
                    st.warning("Tidak ditemukan daun pada gambar.")
                else:
                    st.subheader("Hasil Deteksi Daun:")
                    for idx, (box, score) in enumerate(zip(leaf_boxes, leaf_scores), start=1):
                        st.markdown(f"**Daun #{idx}** - Confidence: `{score:.2f}`")
                        
                        # ROI Preview
                        x1, y1, x2, y2 = box
                        
                        #Bounding Box
                        img_np = np.array(image_pil).copy()
                        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        marked_img = Image.fromarray(img_np)
                        st.image(marked_img, caption=f"Bounding Box #{idx}", use_container_width=True)
                        # ROI Preview
                        roi = np.array(image_pil)[y1:y2, x1:x2]
                        st.image(roi, caption=f"📌 ROI - Daun #{idx+1}", use_container_width=True)

                        # GrabCut
                        grabcut_result = enhanced_grabcut(image_pil, box)
                        st.image(grabcut_result, caption=f"GrabCut - Daun #{idx}", use_container_width=True)

                        # Klasifikasi
                        grabcut_pil = Image.fromarray(grabcut_result)
                        diagnosis, conf = classify_leaf(grabcut_pil)
                        text = f"Diagnosa Daun #{idx}: {diagnosis}\nConfidence Score: {conf*100:.1f}%"

                        if diagnosis == "Healthy":
                            st.success(text)
                        elif diagnosis == "Red Spider Mite":
                            st.warning(text)
                        else:  # Rush
                            st.error(text)

    except Exception as e:
        st.error(f"Gagal memproses gambar. Error: {e}")

else:
    st.markdown(
        "<p style='text-align: center; color: gray;'>"
        "Silakan upload gambar untuk memulai diagnosis."
        "</p>",
        unsafe_allow_html=True
    )

st.markdown("---")
st.markdown("<p class='footer'>© DSAI2024 - COMVIS - GROUP 5</p>", unsafe_allow_html=True)
