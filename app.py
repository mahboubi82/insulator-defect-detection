# ─────────────────────────────────────────────
#  app.py — Streamlit web app
#  Insulator Defect Detection — YOLOv8
# ─────────────────────────────────────────────
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tempfile
import os
from pathlib import Path
from ultralytics import YOLO

# ── Page config ───────────────────────────────
st.set_page_config(
    page_title="Insulator Defect Detection",
    page_icon="⚡️",
    layout="wide",
)

# ── Constants ─────────────────────────────────
CLASSES = ["broken", "insulator", "pollution-flashover"]
COLORS  = {
    "broken":              (220, 0,   0),    # red
    "insulator":           (0,   200, 80),   # green
    "pollution-flashover": (160, 0,   210),  # purple
}
ICONS = {
    "broken":              "🔴",
    "insulator":           "🟢",
    "pollution-flashover": "🟣",
}

# ── Load model (cached) ───────────────────────
@st.cache_resource
def load_model():
    model_path = Path("best.pt")
    if not model_path.exists():
        st.error("❌ Model file 'best.pt' not found. Upload it to the repo root.")
        st.stop()
    return YOLO(str(model_path))

# ── Draw boxes on image ───────────────────────
def draw_boxes(image: np.ndarray, results, model) -> np.ndarray:
    img = image.copy()
    for box in results[0].boxes:
        cls_name = model.names[int(box.cls.item())]
        conf     = box.conf.item()
        x1,y1,x2,y2 = [int(v) for v in box.xyxy[0].tolist()]
        color    = COLORS.get(cls_name, (200, 200, 200))
        color_bgr = (color[2], color[1], color[0])

        cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, 2)
        label = f"{cls_name}  {conf*100:.1f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 8, y1), color_bgr, -1)
        cv2.putText(img, label, (x1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return img


# ── UI ────────────────────────────────────────
st.title("🔌 Insulator Defect Detection")
st.caption("NBPower · YOLOv8 · Detects broken, healthy, and pollution-flashover insulators")
st.divider()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    conf_thresh = st.slider("Confidence threshold", 0.1, 0.9, 0.25, 0.05)
    iou_thresh  = st.slider("IoU threshold (NMS)", 0.1, 0.9, 0.45, 0.05)
    st.divider()
    st.markdown("**Classes**")
    for cls in CLASSES:
        st.markdown(f"{ICONS[cls]} `{cls}`")
    st.divider()
    st.caption("Upload an image to detect insulator defects.")

# Upload
uploaded = st.file_uploader(
    "Upload an insulator image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    label_visibility="collapsed",
)

if uploaded:
    # Load image
    pil_img  = Image.open(uploaded).convert("RGB")
    img_np   = np.array(pil_img)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("Original")
        st.image(pil_img, use_column_width=True)

    # Run model
    model = load_model()
    with st.spinner("Running detection..."):
        results = model.predict(
            source=img_np,
            conf=conf_thresh,
            iou=iou_thresh,
            verbose=False,
        )

    # Draw boxes
    annotated = draw_boxes(img_np, results, model)

    with col2:
        st.subheader("Detections")
        st.image(annotated, use_column_width=True)

    # Results table
    st.divider()
    boxes = results[0].boxes

    if len(boxes) == 0:
        st.warning("⚠️ No insulators detected. Try lowering the confidence threshold.")
    else:
        st.subheader(f"🔍 {len(boxes)} detection(s) found")

        # Metric cards
        counts = {cls: 0 for cls in CLASSES}
        for box in boxes:
            counts[model.names[int(box.cls.item())]] += 1

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("🔴 Broken",              counts["broken"])
        mc2.metric("🟢 Healthy (insulator)", counts["insulator"])
        mc3.metric("🟣 Pollution-flashover", counts["pollution-flashover"])

        # Detail table
        st.subheader("Detail")
        rows = []
        for i, box in enumerate(boxes):
            cls_name = model.names[int(box.cls.item())]
            conf     = box.conf.item()
            x1,y1,x2,y2 = [int(v) for v in box.xyxy[0].tolist()]
            rows.append({
                "#":           i + 1,
                "Class":       f"{ICONS[cls_name]} {cls_name}",
                "Confidence":  f"{conf*100:.1f}%",
                "Box (x1,y1,x2,y2)": f"[{x1}, {y1}, {x2}, {y2}]",
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

        # Download annotated image
        annotated_pil = Image.fromarray(annotated)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            annotated_pil.save(tmp.name, quality=95)
            with open(tmp.name, "rb") as f:
                st.download_button(
                    "⬇️ Download annotated image",
                    data=f.read(),
                    file_name=f"detection_{uploaded.name}",
                    mime="image/jpeg",
                )

else:
    st.info("👆 Upload an image to get started")
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Insulators_on_a_power_line.jpg/640px-Insulators_on_a_power_line.jpg",
        caption="Example: power line insulators",
        use_column_width=True,
    )
