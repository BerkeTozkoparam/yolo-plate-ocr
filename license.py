import streamlit as st
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr  # OCR kullanacaksan, yoksa burayı kaldırabilirsin

st.set_page_config(
    page_title="Berke Plate & Object Detection",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Berke Plate & Object Detection")
st.write("Live Detection + Optional OCR with Grayscale")

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_models():
    yolo_model = YOLO("best-2.pt")  # YOLO plaka modeli
    ocr_reader = easyocr.Reader(['en'])  # OCR
    return yolo_model, ocr_reader

model, reader = load_models()

# -----------------------------
# Input method
# -----------------------------
input_method = st.radio("Select input method:", ["Use Camera", "Upload Image"])
FRAME_WINDOW = st.image([])
PLATE_INFO = st.empty()

# -----------------------------
# Camera input
# -----------------------------
if input_method == "Use Camera":
    run = st.checkbox("Start Camera")
    ocr_btn = st.button("Read Plate (OCR)")  # OCR için tuş
    cap = None

    if run:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Kamera açılamadı. Lütfen izinleri ve bağlantıyı kontrol edin.")
        else:
            while run:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Kare alınamadı, kamerayı kontrol edin.")
                    break

                # Canlı tespit (renkli)
                results = model.predict(frame, conf=0.5)
                annotated_frame = results[0].plot()

                FRAME_WINDOW.image(annotated_frame[:, :, ::-1])  # BGR->RGB

                # OCR sadece tuşa basılınca
                if ocr_btn:
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray_frame_3ch = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
                    results_ocr = model.predict(gray_frame_3ch, conf=0.5)

                    plate_texts = []
                    for box in results_ocr[0].boxes.xyxy.cpu().numpy():
                        x1, y1, x2, y2 = map(int, box)
                        crop = gray_frame_3ch[y1:y2, x1:x2]
                        ocr_result = reader.readtext(crop)
                        if ocr_result:
                            plate_texts.append(" ".join([res[1] for res in ocr_result]))

                    PLATE_INFO.markdown(f"*📖 Recognized plates:* {', '.join(plate_texts)}" 
                                        if plate_texts else "*No plate recognized.*")

            cap.release()

# -----------------------------
# Image upload
# -----------------------------
elif input_method == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)

        # Siyah-beyaz yap
        gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        gray_img_3ch = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)

        results = model.predict(gray_img_3ch, conf=0.5)
        annotated_frame = results[0].plot()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        FRAME_WINDOW.image(annotated_frame)
        plate_texts = []
        for box in results[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            crop = gray_img_3ch[y1:y2, x1:x2]
            ocr_result = reader.readtext(crop)
            if ocr_result:
                plate_texts.append(" ".join([res[1] for res in ocr_result]))

        PLATE_INFO.markdown(f"*📖 Recognized plates:* {', '.join(plate_texts)}" 
                            if plate_texts else "*No plate recognized.*")
