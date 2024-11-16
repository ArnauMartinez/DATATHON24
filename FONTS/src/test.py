import streamlit as st
from PIL import Image
import base64
from training1 import *

# Add a black header that spans the entire width of the page and centers the text
st.markdown(
    """
    <style>
    .header {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        background-color: black;
        height: 150px; /* Increased height for visibility */
        display: flex;
        align-items: center; /* Vertically center the content */
        justify-content: flex-start; /* Align items to the left */
        z-index: 1000; /* Ensures it stays above other elements */
        padding-left: 20px; /* Add padding to the left of the header */
    }
    .header img {
        height: 100px; /* Set the height of the image */
        margin-right: 20px; /* Add space between the image and the text */
    }
    .header h1 {
        color: white;
        font-size: 25px;
        margin: 0; /* Remove default margin for proper centering */
    }
    .main-content {
        margin-top: 180px; /* Increased margin-top to push content down and avoid overlap with the header */
    }
    </style>
    <div class="header">
        <img src="data:image/jpeg;base64,{base64.b64encode(open('mango.png', 'rb').read()).decode()}" alt="Mango Image">
        <h1>Product Design Attributes</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Add a wrapper div for the rest of the content to prevent overlap with the header
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# File uploader and image display
uploaded_file = st.file_uploader("Upload an image", type=["jpg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    #st.image(image, caption="Uploaded Image", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)
