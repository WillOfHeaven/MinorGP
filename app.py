import numpy as np
import cv2
import matplotlib.pyplot as plt
import keras
from keras.models import load_model
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas

def main():
    if "button_id" not in st.session_state:
        st.session_state["button_id"] = ""
    if "color_to_label" not in st.session_state:
        st.session_state["color_to_label"] = {}
    PAGES = {
        "About": about_page,
        "Basic example": full_app,
    }
    page = st.sidebar.selectbox("Page:", options=list(PAGES.keys()))
    PAGES[page]()

    with st.sidebar:
        st.markdown("---")
        st.markdown(
            '<h6>Made in &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="16">&nbsp </h6>',
            unsafe_allow_html=True,
        )


def about_page():
    st.title("About This App")

    st.markdown("This application is a number recognition tool designed to showcase the power of combining Streamlit with deep learning models.")

    st.subheader("What it does:")

    st.write("""
    - Allows users to draw a digit (0-9) on a canvas.
    - Utilizes a Convolutional Neural Network (CNN) model to analyze the drawn digit.
    - Predicts the most likely number represented by the drawing.
    """)

    st.subheader("Technical Details:")

    st.write("""
    - Frontend framework: Streamlit (https://docs.streamlit.io/)
    - Drawing canvas: streamlit-drawable-canvas library (https://github.com/andfanilo/streamlit-drawable-canvas)
    - Machine learning model: User-defined CNN model trained on the MNIST dataset (replace with relevant details)
    """)

    st.subheader("Developed by:")

    st.write("""
    - Gaurav Rawat
    - Karthik Sharma Dhulipati
    - Mohak Kumar Srivastava
    - Naman Jain
    - Sambuddha Chatterjee
             """)

    st.subheader("Feedback :")

    st.write("""
    We welcome your feedback and contributions to this application! 
    - Feel free to report any issues or suggest improvements on [github](https://github.com/WillOfHeaven/MinorGP) (if applicable). 
    """)
def full_app():
    st.sidebar.header("Configuration")
    st.markdown(
        """
    Draw on the canvas, get the drawings back to Streamlit!
    * Configure canvas in the sidebar
    """
    )

    
    # Specify canvas parameters in application
    drawing_mode = "freedraw"
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 25)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ", "#FFFFFF")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#000000")
    bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
    realtime_update = st.sidebar.checkbox("Update in realtime", True)

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width ,
        stroke_color=stroke_color ,
        background_color=bg_color ,
        background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit=realtime_update,
        height=200,
        drawing_mode=drawing_mode,
        display_toolbar=st.sidebar.checkbox("Display toolbar", True),
        key="full_app",
    )

    #st.button("Clear canvas", key="clear_canvas")
    model = load_model("new_mnist(4x4)_epoch70.h5")
    model1 = load_model("mnist.h5")
    if canvas_result.image_data is not None:
        img = cv2.cvtColor(canvas_result.image_data, cv2.COLOR_RGBA2GRAY)  # Convert the image to grayscale
        img_resized = cv2.resize(img, (28, 28))  # Resize the image to 28x28
        img_resized1 = cv2.resize(img, (56, 56))  # Resize the image to 28x28
        st.image(img_resized1,caption="Resized Image with changed dimensions 56 x 56")
        img_reshaped = img_resized.reshape(1, 28, 28, 1)  # Reshape the image to match the model's expected input shape
        img_reshaped = img_reshaped.astype("float32")
        st.image(img_resized,caption="Resized Image")
        img_reshaped /= 255.0  # Normalize the image if your model was trained on normalized images
        response = model.predict(img_reshaped)  # Use the model to predict the digit
        response1 = model1.predict(img_reshaped)  # Use the model to predict the digit 
        st.header("Predicted Number new model new_mnist(4x4)_epoch70")
        st.header(np.argmax(response))  # The predicted digit is the one with the highest probability
        st.header("Predicted Number old modelv mnist.h5 the first one)
        st.header(np.argmax(response1))        
        # st.header("Image type as returned by the canvas component")
        #st.write(type(canvas_result.image_data))
        st.header("Matrix contiaining response data")
        st.write(response)
    


if __name__ == "__main__":
    st.set_page_config(
        page_title="Number Recognitions using ANN", page_icon=":pencil2:"
    )
    st.sidebar.subheader("Contents : ")
    main()
