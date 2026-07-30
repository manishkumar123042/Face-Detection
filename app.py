import streamlit as st
import cv2
import numpy as np
import time
from PIL import Image

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------

st.set_page_config(
    page_title="AI Face Detection",
    page_icon="😀",
    layout="wide"
)

# -----------------------------------------------------
# CUSTOM CSS
# -----------------------------------------------------

st.markdown("""
<style>

.main{
    padding-top:20px;
}

.stButton>button{
    width:100%;
    background:#0066ff;
    color:white;
    border:none;
    border-radius:10px;
    font-size:18px;
    font-weight:bold;
    padding:12px;
}

.stButton>button:hover{
    background:#0047cc;
}

[data-testid="stMetricValue"]{
    font-size:30px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# LOAD HAAR CASCADE
# -----------------------------------------------------

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# -----------------------------------------------------
# SIDEBAR
# -----------------------------------------------------

st.sidebar.title("⚙ Face Detection Settings")

scale_factor = st.sidebar.slider(
    "Scale Factor",
    1.05,
    2.0,
    1.20,
    0.05
)

min_neighbors = st.sidebar.slider(
    "Min Neighbors",
    1,
    10,
    5
)

min_size = st.sidebar.slider(
    "Minimum Face Size",
    20,
    200,
    40
)

thickness = st.sidebar.slider(
    "Rectangle Thickness",
    1,
    8,
    2
)

color_name = st.sidebar.selectbox(
    "Rectangle Color",
    [
        "Green",
        "Blue",
        "Red",
        "Yellow"
    ]
)

blur_faces = st.sidebar.toggle(
    "Blur Faces"
)

st.sidebar.markdown("---")

st.sidebar.info("""
Model

OpenCV Haar Cascade

Input

Image

Language

Python

Framework

Streamlit
""")

# -----------------------------------------------------
# RECTANGLE COLORS
# -----------------------------------------------------

colors = {

    "Green": (0,255,0),

    "Blue": (255,0,0),

    "Red": (0,0,255),

    "Yellow": (0,255,255)

}

rectangle_color = colors[color_name]

# -----------------------------------------------------
# TITLE
# -----------------------------------------------------

st.title("😀 AI Face Detection System")

st.write(
    "Detect human faces from images using OpenCV Haar Cascade."
)

st.divider()

# -----------------------------------------------------
# IMAGE UPLOAD
# -----------------------------------------------------

uploaded_file = st.file_uploader(

    "📤 Upload Image",

    type=[
        "jpg",
        "jpeg",
        "png"
    ]

)

# -----------------------------------------------------
# SESSION HISTORY
# -----------------------------------------------------

if "history" not in st.session_state:

    st.session_state.history = []

# -----------------------------------------------------
# IF IMAGE UPLOADED
# -----------------------------------------------------

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    image = np.array(image)

    original = image.copy()

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    # ------------------------------------------

    col1,col2 = st.columns(2)

    with col1:

        st.subheader("🖼 Original Image")

        st.image(
            original,
            use_container_width=True
        )

    with col2:

        st.subheader("📋 Image Information")

        h,w,c = image.shape

        st.metric(
            "Width",
            f"{w}px"
        )

        st.metric(
            "Height",
            f"{h}px"
        )

        st.metric(
            "Channels",
            c
        )

        st.metric(
            "Resolution",
            f"{w*h:,}"
        )

    st.divider()

    # ------------------------------------------
    # DETECT BUTTON
    # ------------------------------------------

    detect = st.button(

        "🔍 Detect Faces",

        use_container_width=True,

        type="primary"

    )

    if detect:

        with st.spinner(
            "Detecting Faces..."
        ):

            start = time.time()

            faces = face_detector.detectMultiScale(

                gray,

                scaleFactor=scale_factor,

                minNeighbors=min_neighbors,

                minSize=(
                    min_size,
                    min_size
                )

            )

            end = time.time()

            processing_time = round(
                end-start,
                3
            )

            # ------------------------------------------
            # DRAW RECTANGLES
            # ------------------------------------------

            for (x, y, w, h) in faces:

                if blur_faces:

                    face = image[y:y+h, x:x+w]

                    blurred = cv2.GaussianBlur(
                        face,
                        (51, 51),
                        30
                    )

                    image[y:y+h, x:x+w] = blurred

                cv2.rectangle(
                    image,
                    (x, y),
                    (x+w, y+h),
                    rectangle_color,
                    thickness
                )

            # ------------------------------------------
            # RESULTS
            # ------------------------------------------

            st.success("✅ Face Detection Completed")

            st.divider()

            col1, col2 = st.columns(2)

            with col1:

                st.subheader("😀 Detected Faces")

                st.image(
                    image,
                    use_container_width=True
                )

            with col2:

                st.subheader("📊 Detection Summary")

                st.metric(
                    "Faces Detected",
                    len(faces)
                )

                st.metric(
                    "Processing Time",
                    f"{processing_time} sec"
                )

                if len(faces) == 0:

                    st.warning(
                        "No face detected."
                    )

                elif len(faces) == 1:

                    st.success(
                        "One face detected."
                    )

                else:

                    st.success(
                        f"{len(faces)} faces detected."
                    )

            st.divider()

            # ------------------------------------------
            # FACE COORDINATES
            # ------------------------------------------

            if len(faces) > 0:

                st.subheader("📍 Face Coordinates")

                coordinate_data = []

                for i, (x, y, w, h) in enumerate(faces, start=1):

                    coordinate_data.append({

                        "Face": i,

                        "X": x,

                        "Y": y,

                        "Width": w,

                        "Height": h

                    })

                st.dataframe(
                    coordinate_data,
                    use_container_width=True,
                    hide_index=True
                )

            st.divider()

            # ------------------------------------------
            # DETECTED FACE CROPS
            # ------------------------------------------

            if len(faces) > 0:

                st.subheader("🖼️ Cropped Faces")

                cols = st.columns(3)

                for i, (x, y, w, h) in enumerate(faces):

                    crop = original[
                        y:y+h,
                        x:x+w
                    ]

                    cols[i % 3].image(
                        crop,
                        caption=f"Face {i+1}",
                        use_container_width=True
                    )

            st.divider()

            # ------------------------------------------
            # HISTORY
            # ------------------------------------------

            st.session_state.history.append({

                "Faces": len(faces),

                "Time (sec)": processing_time

            })

            st.subheader("📜 Detection History")

            st.dataframe(

                st.session_state.history,

                use_container_width=True,

                hide_index=True

            )

            st.divider()

            # ------------------------------------------
            # DOWNLOAD IMAGE
            # ------------------------------------------

            result = cv2.cvtColor(
                image,
                cv2.COLOR_RGB2BGR
            )

            success, buffer = cv2.imencode(
                ".jpg",
                result
            )

            if success:

                st.download_button(

                    "📥 Download Result",

                    buffer.tobytes(),

                    "detected_faces.jpg",

                    "image/jpeg",

                    use_container_width=True

                )

            st.divider()

            # ------------------------------------------
            # ABOUT
            # ------------------------------------------

            with st.expander("ℹ️ About this Project"):

                st.write("""
This application detects human faces using the **OpenCV Haar Cascade Classifier**.

### Features

- Upload image
- Adjustable detection settings
- Blur detected faces
- Face counting
- Face coordinates
- Cropped face previews
- Download processed image
- Detection history

### Technologies

- Python
- Streamlit
- OpenCV
- NumPy
                """)

# -----------------------------------------------------
# FOOTER
# -----------------------------------------------------

st.markdown("---")

st.caption(
    "Developed by Manish Kumar | AI Face Detection using OpenCV & Streamlit"
)
