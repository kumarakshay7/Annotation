import os
import json
import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import base64
from io import BytesIO

# --- Setup directories for saving data ---
BASE_DIR = os.getcwd()
ANNOTATED_IMAGES_DIR = os.path.join(BASE_DIR, "annotated_images")
ANNOTATIONS_DIR = os.path.join(BASE_DIR, "annotations")

for folder in [ANNOTATED_IMAGES_DIR, ANNOTATIONS_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# --- Application Title ---
st.title("Annotation Tool (LabelImg Replica)")

# --- Sidebar: Configuration and Uploads ---
st.sidebar.header("Configuration")

# 1. Select Annotation Format
annotation_format = st.sidebar.selectbox("Select Annotation Format", ["Pascal VOC", "YOLO"])

# 2. Upload Images
uploaded_files = st.sidebar.file_uploader("Upload one or more images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

# 3. Label Management
st.sidebar.subheader("Custom Labels / Classes")
labels_input = st.sidebar.text_area("Enter one label per line", "object")
custom_labels = [label.strip() for label in labels_input.splitlines() if label.strip()]

# Save labels to file
if st.sidebar.button("Save Labels"):
    with open(os.path.join(ANNOTATIONS_DIR, "labels.txt"), "w") as f:
        f.write("\n".join(custom_labels))
    st.sidebar.success("Labels saved successfully!")

# --- Main Panel: Image Annotation ---
if uploaded_files:
    st.header("Annotate Images")
    image_names = [file.name for file in uploaded_files]
    selected_image_name = st.selectbox("Select an image to annotate", options=image_names)
    selected_file = next((f for f in uploaded_files if f.name == selected_image_name), None)

    if selected_file:
        image = Image.open(selected_file).convert("RGB")
        width, height = image.size
        st.subheader(f"Annotate: {selected_image_name}")
        st.image(image, caption="Original Image", use_column_width=True)

        # --- Drawable Canvas (Bypass background_image issue) ---
        st.markdown("### Draw Bounding Boxes on the Image")
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0.3, 0.3)",
            stroke_width=2,
            stroke_color="black",
            height=height,
            width=width,
            drawing_mode="rect",
            key="canvas",
        )

        # --- Process Annotations ---
        if canvas_result.json_data and "objects" in canvas_result.json_data:
            bounding_boxes = [obj for obj in canvas_result.json_data["objects"] if obj.get("type") == "rect"]

            if bounding_boxes:
                st.markdown("#### Assign Labels to Each Bounding Box")
                assigned_annotations = []

                for i, box in enumerate(bounding_boxes):
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.write(f"**Bounding Box {i+1}:** Coordinates: *(x: {int(box['left'])}, y: {int(box['top'])}, width: {int(box['width'])}, height: {int(box['height'])})*")
                    with col2:
                        label_choice = st.selectbox(f"Label for Box {i+1}", custom_labels or ["object"], key=f"label_{i}")
                    assigned_annotations.append({"label": label_choice, "x": box["left"], "y": box["top"], "width": box["width"], "height": box["height"]})

                # --- Save Annotations ---
                if st.button("Save Annotation"):
                    annotation_data = {"image_name": selected_image_name, "annotations": assigned_annotations}
                    json_path = os.path.join(ANNOTATIONS_DIR, f"{os.path.splitext(selected_image_name)[0]}.json")
                    with open(json_path, "w") as f:
                        json.dump(annotation_data, f, indent=4)
                    
                    image.save(os.path.join(ANNOTATED_IMAGES_DIR, selected_image_name))

                    # Save YOLO format annotations
                    txt_path = os.path.join(ANNOTATIONS_DIR, f"{os.path.splitext(selected_image_name)[0]}.txt")
                    with open(txt_path, "w") as f:
                        for ann in assigned_annotations:
                            x, y, w, h = ann["x"], ann["y"], ann["width"], ann["height"]
                            cx, cy, norm_w, norm_h = (x + w / 2) / width, (y + h / 2) / height, w / width, h / height
                            class_index = custom_labels.index(ann["label"]) if ann["label"] in custom_labels else 0
                            f.write(f"{class_index} {cx:.6f} {cy:.6f} {norm_w:.6f} {norm_h:.6f}\n")

                    st.success("Annotation saved successfully! JSON and TXT files created.")
            else:
                st.info("Draw at least one bounding box to save annotations.")
        else:
            st.info("Use the drawing tool above to add bounding boxes.")
