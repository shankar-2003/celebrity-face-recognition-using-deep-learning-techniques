import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import base64

# Load the saved model
model_path = r"C:\Users\Admin\anaconda3\Lib\site-packages\pandas\io\parsers\__pycache__\colab_model.h5"
model = tf.keras.models.load_model(model_path)

# Load celebrity names from the text file with error handling
def load_celebrity_dict(file_path):
    celebrity_dict = {}
    with open(file_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        try:
            index, name = line.strip().split(',')
            celebrity_dict[int(index)] = name
        except ValueError:
            st.error(f"Error processing line: {line.strip()}")
            continue
    return celebrity_dict

celebrity_dict = load_celebrity_dict(r"C:\Users\Admin\anaconda3\Lib\site-packages\pandas\io\parsers\__pycache__\Indexing_names.txt")

# Function to load and preprocess the image
def load_and_prep_image(image, img_shape=160):
    image = Image.open(image)
    image = image.resize((img_shape, img_shape))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)  # add batch dimension
    return image

# Function to make predictions
def predict_image(image):
    image = load_and_prep_image(image)
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return celebrity_dict.get(predicted_class, "Unknown")

# Set the background image
def set_background_image(image_path):
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode()
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{image_data}");
        background-size: cover;
        background-attachment: fixed;
    }}
    .stApp .stTextInput label, .stApp .stFileUploader label {{
        color: white;
    }}
    .stApp .stTextInput input, .stApp .stFileUploader input {{
        background-color: #444;
        color: white;
    }}
    .stApp .stButton button {{
        background-color: #FF4B4B;
        color: white;
        border-radius: 10px;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Set background image
set_background_image(r"C:\Users\Admin\anaconda3\Lib\site-packages\pandas\io\parsers\__pycache__\__pycache__\background image.jpeg")  # Replace with your background image path

# Streamlit layout
st.markdown("<h1 style='text-align: center; color: white;'>Celebrity Image Classification</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white;'>Upload an image to classify the celebrity.</p>", unsafe_allow_html=True)

# File uploader for images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image at a reduced size with high quality
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', width=300)  # Decreased size with `width`

    st.write("")
    st.write("<p style='text-align: center; color: white;'>Classifying...</p>", unsafe_allow_html=True)

    # Make a prediction
    predicted_celebrity = predict_image(uploaded_file)

    # Display the prediction
    st.markdown(f"<h2 style='text-align: center; color: #FF4B4B;'>Predicted Celebrity: {predicted_celebrity}</h2>", unsafe_allow_html=True)

# Footer
st.markdown("<p style='text-align: center; color: white;'>Developed with using Streamlit and TensorFlow</p>", unsafe_allow_html=True)
