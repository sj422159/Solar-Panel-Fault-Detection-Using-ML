import numpy as np
import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from io import BytesIO

# Load the model (assuming it's already trained and saved as 'solar_fault_classification_model.h5')
model = load_model("solar_fault_classification_model.h5")

# Example: Class labels (adjust based on your actual class labels)
class_labels = ["Bird Drop", "Clean", "Dusty", "Electrical Damage", "Cracks", "Snow Coverage"]

# Load and preprocess the image
def preprocess_image(img):
    img = img.resize((244, 244))  # Ensure target_size matches the model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize
    return img_array

# Streamlit app
st.title("Solar Panel Fault Detection")
st.write("Upload a solar panel image to detect potential faults.")

# Image uploader
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Open image file and display in Streamlit
    img = image.load_img(uploaded_image)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_array = preprocess_image(img)

    # Predict button
    if st.button("Predict Fault"):
        # Make predictions
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = class_labels[predicted_class]
        confidence = np.max(predictions[0])  # Get confidence level

        # Display results in Streamlit
        st.write(f"Predicted Fault: **{predicted_label}**")
        st.write(f"Confidence: **{confidence:.2f}**")
