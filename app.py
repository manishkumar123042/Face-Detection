# AI Face Detection - Streamlit
import streamlit as st
import cv2
import numpy as np
import time
from PIL import Image

st.set_page_config(page_title="AI Face Detection", page_icon="😀", layout="wide")

st.markdown("""
<style>
.stButton>button{
width:100%;
background:#0E6FFF;
color:white;
border-radius:8px;
font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

st.title("😀 AI Face Detection")
st.write("Detect human faces using OpenCV Haar Cascade.")

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

st.sidebar.header("Detection Settings")

scale_factor = st.sidebar.slider("Scale Factor",1.05,2.0,1.20,0.05)
min_neighbors = st.sidebar.slider("Min Neighbors",1,10,5)

color_name = st.sidebar.selectbox(
    "Rectangle Color",
    ["Green","Blue","Red","Yellow"]
)

colors={
    "Green":(0,255,0),
    "Blue":(255,0,0),
    "Red":(0,0,255),
    "Yellow":(0,255,255)
}

rectangle_color=colors[color_name]

uploaded_file=st.file_uploader(
    "Upload an Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file:

    image=Image.open(uploaded_file).convert("RGB")
    image=np.array(image)

    original=image.copy()

    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

    if st.button("Detect Faces",type="primary"):

        start=time.time()

        faces=face_detector.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors
        )

        end=time.time()

        processing_time=end-start

        for (x,y,w,h) in faces:

            cv2.rectangle(
                image,
                (x,y),
                (x+w,y+h),
                rectangle_color,
                2
            )

        col1,col2=st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(original,use_container_width=True)

        with col2:
            st.subheader("Detected Image")
            st.image(image,use_container_width=True)

        st.divider()

        c1,c2=st.columns(2)

        with c1:
            st.metric("Faces Detected",len(faces))

        with c2:
            st.metric(
                "Processing Time",
                f"{processing_time:.3f} sec"
            )

        result=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        success,buffer=cv2.imencode(".jpg",result)

        if success:
            st.download_button(
                "Download Result",
                buffer.tobytes(),
                "detected_faces.jpg",
                "image/jpeg",
                use_container_width=True
            )

else:
    st.info("Upload an image to start face detection.")

st.markdown("---")
st.caption("Built with Streamlit + OpenCV Haar Cascade")
