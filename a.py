import os
import json
import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import base64
from io import BytesIO

# Function to convert PIL image to base64 URL
def image_to_base64(image):
    """Convert a PIL image to a base64 URL for use in Streamlit."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

# Setup directories for saving data
BASE_DIR = os.getcwd()
ANNOTATED_IMAGES_DIR = os.path.join(BASE_DIR, "annotated_images")
ANNOTATIONS_DIR = os.path.join(BASE_DIR, "annotations")

for folder in [ANNOTATED_IMAGES_DIR, ANNOTATIONS_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Application Title
st.title("Annotation Tool (LabelImg Replica)")

# Sidebar: Configuration and Uploads
st.sidebar.header("Configuration")

# Select Annotation Format
annotation_format = st.sidebar.selectbox(
    "Select Annotation Format",
    options=["Pascal VOC", "YOLO"],
    index=0,
    help="Choose the annotation format for output."
)

# Upload Images
uploaded_files = st.sidebar.file_uploader(
    "Upload one or more images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

# Label Management: Create/Edit Custom Labels
st.sidebar.subheader("Custom Labels / Classes")
labels_input = st.sidebar.text_area(
    "Enter one label per line",
    value="object",
    help="Provide labels to assign to bounding boxes."
)
custom_labels = [label.strip() for label in labels_input.splitlines() if label.strip() != ""]

if st.sidebar.button("Save Labels"):
    labels_file_path = os.path.join(ANNOTATIONS_DIR, "labels.txt")
    with open(labels_file_path, "w") as f:
        for label in custom_labels:
            f.write(label + "\n")
    st.sidebar.success("Labels saved successfully!")

# Main Panel: Image Annotation
if uploaded_files:
    st.header("Annotate Images")
    image_names = [uploaded_file.name for uploaded_file in uploaded_files]
    selected_image_name = st.selectbox("Select an image to annotate", options=image_names)
    selected_file = next((f for f in uploaded_files if f.name == selected_image_name), None)
    
    if selected_file is not None:
        image = Image.open(selected_file).convert("RGB")
        width, height = image.size
        st.subheader(f"Annotate: {selected_image_name}")
        st.image(image, caption="Original Image", use_column_width=True)
        
        # Convert image to base64 URL for Streamlit compatibility
        background_image_url = image_to_base64(image)
        
        # Drawable Canvas for Annotation
        st.markdown("### Draw Bounding Boxes on the Image")
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=2,
            stroke_color="black",
            background_image=None,
            background_image_url=background_image_url,
            update_streamlit=True,
            height=height,
            width=width,
            drawing_mode="rect",
            key="canvas",
        )

        if canvas_result.json_data is not None:
            objects = canvas_result.json_data.get("objects", [])
            bounding_boxes = [obj for obj in objects if obj.get("type") == "rect"]
            
            if bounding_boxes:
                st.markdown("#### Assign Labels to Each Bounding Box")
                assigned_annotations = []
                for i, box in enumerate(bounding_boxes):
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.write(f"**Bounding Box {i+1}:**")
                        st.write(
                            f"Coordinates: *(x: {int(box.get('left', 0))}, y: {int(box.get('top', 0))}, "
                            f"width: {int(box.get('width', 0))}, height: {int(box.get('height', 0))})*"
                        )
                    with col2:
                        default_label = custom_labels[0] if custom_labels else "object"
                        label_choice = st.selectbox(
                            f"Select label for box {i+1}",
                            options=custom_labels if custom_labels else [default_label],
                            key=f"label_{i}"
                        )
                    assigned_annotations.append({
                        "label": label_choice,
                        "x": box.get("left", 0),
                        "y": box.get("top", 0),
                        "width": box.get("width", 0),
                        "height": box.get("height", 0)
                    })

                if st.button("Save Annotation"):
                    annotation_data = {
                        "image_name": selected_image_name,
                        "annotation_format": annotation_format,
                        "custom_labels": custom_labels,
                        "annotations": assigned_annotations
                    }
                    base_filename = os.path.splitext(selected_image_name)[0]
                    json_annotation_file = os.path.join(ANNOTATIONS_DIR, f"{base_filename}.json")
                    with open(json_annotation_file, "w") as f:
                        json.dump(annotation_data, f, indent=4)
                    image.save(os.path.join(ANNOTATED_IMAGES_DIR, selected_image_name))
                    
                    st.success("Annotation saved successfully!")
            else:
                st.info("Draw one or more bounding boxes on the image to begin annotation.")
        else:
            st.info("Use the drawing tool above to add bounding boxes.")
else:
    st.info("Upload images from the sidebar to start annotating.")
