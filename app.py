import streamlit as st
import cv2
import numpy as np
from PIL import Image

# -----------------------------------
# Page Configuration
# -----------------------------------

st.set_page_config(
    page_title="AI Face Detection",
    page_icon="😊",
    layout="wide"
)

# -----------------------------------
# Sidebar
# -----------------------------------

st.sidebar.title("😊 About Project")

st.sidebar.info("""
This project detects human faces
using OpenCV Haar Cascade.

Technology Used:
- Python
- OpenCV
- Streamlit
- NumPy
- Pillow
""")

st.sidebar.markdown("---")
st.sidebar.write("Developed by Manish Kumar")

# -----------------------------------
# Title
# -----------------------------------

st.title("😊 AI Face Detection System")

st.write(
    "Upload an image and the system will detect all human faces."
)

st.divider()

# -----------------------------------
# Load Face Detector
# -----------------------------------

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# -----------------------------------
# Upload Image
# -----------------------------------

uploaded_file = st.file_uploader(
    "📤 Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    image = image.convert("RGB")

    image_np = np.array(image)

    original_image = image_np.copy()

    # Convert RGB to BGR
    image_cv = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2BGR
    )

    gray = cv2.cvtColor(
        image_cv,
        cv2.COLOR_BGR2GRAY
    )

    # Loading Spinner
    with st.spinner("Detecting Faces..."):

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30,30)
        )

    # Draw Rectangle
    for (x,y,w,h) in faces:

        cv2.rectangle(
            image_cv,
            (x,y),
            (x+w,y+h),
            (0,255,0),
            2
        )

    detected_image = cv2.cvtColor(
        image_cv,
        cv2.COLOR_BGR2RGB
    )

    st.success("Face Detection Completed Successfully!")

    st.divider()

    # -----------------------------------
    # Display Images
    # -----------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📷 Original Image")

        st.image(
            original_image,
            use_container_width=True
        )

    with col2:

        st.subheader("😊 Detected Faces")

        st.image(
            detected_image,
            use_container_width=True
        )

    st.divider()

    # -----------------------------------
    # Result Section
    # -----------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "😊 Faces Detected",
            len(faces)
        )

    with col2:

        st.metric(
            "📏 Width",
            f"{original_image.shape[1]} px"
        )

    with col3:

        st.metric(
            "📐 Height",
            f"{original_image.shape[0]} px"
        )

    st.divider()

    # -----------------------------------
    # Download Image
    # -----------------------------------

    result = cv2.cvtColor(
        detected_image,
        cv2.COLOR_RGB2BGR
    )

    cv2.imwrite(
        "detected_faces.jpg",
        result
    )

    with open(
        "detected_faces.jpg",
        "rb"
    ) as file:

        st.download_button(
            label="⬇ Download Result Image",
            data=file,
            file_name="detected_faces.jpg",
            mime="image/jpeg"
        )

else:

    st.info("Please upload an image to start face detection.")

st.divider()

st.caption(
    "AI Face Detection using OpenCV Haar Cascade & Streamlit"
)