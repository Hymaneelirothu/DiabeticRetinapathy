import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model(r'best_model.keras')

# Function to preprocess the image
def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.array(image)
    image = image / 255.0  # Normalize to [0, 1] range
    image = np.expand_dims(image, axis=0)
    return image

# Function to preprocess additional data (if needed)
def preprocess_additional_data(data):
    # Ensure data is in the correct shape and format
    return np.expand_dims(data, axis=0)  # Example to match input shape

# Streamlit UI
st.title("Diabetic Retinopathy Detection App")
st.write("This app predicts the level of diabetic retinopathy from retinal images.")

# Upload an image
uploaded_file = st.file_uploader("Choose a retinal image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")

    # Preprocess the image for both models
    processed_image = preprocess_image(image, target_size=(224, 224))
    
    # Duplicate the input for both EfficientNet and ConvNeXt (since both expect the same input)
    model_inputs = [processed_image, processed_image]

    # Make prediction
    prediction = model.predict(model_inputs)
    prediction = np.squeeze(prediction)  # Remove batch dimension

    # Interpret the results
    labels = ['Healthy', 'Mild DR', 'Moderate DR', 'Proliferate DR', 'Severe DR']
    st.write("Prediction results:")
    for label, prob in zip(labels, prediction):
        st.write(f"{label}: {prob:.2%}")

    # Display the highest probability
    predicted_label = labels[np.argmax(prediction)]
    st.subheader(f"Predicted: {predicted_label}")
