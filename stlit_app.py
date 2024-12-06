import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import cv2

# Define a custom object for `mse` if required by the model
custom_objects = {"mse": tf.keras.losses.MeanSquaredError()}

# Load the model
@st.cache_resource
def load_local_model(model_path):
    return load_model(model_path, custom_objects=custom_objects)

model = load_local_model('C:/Users/___user___/Documents/Projects/stock_manage/bbox_model/model_object_detection.h5')

# Function to draw bounding boxes and object name
def draw_bounding_box_with_label(image, bbox, label, color=(255, 0, 0), thickness=2, font_scale=0.8, font=cv2.FONT_HERSHEY_SIMPLEX):
    """
    Draws a bounding box and label (name of object) on an image.
    
    Args:
        image: The input image as a NumPy array.
        bbox: The bounding box in [x_min, y_min, x_max, y_max] format.
        label: The label (name of object) to display on top of the bounding box.
        color: The color of the bounding box and text.
        thickness: The thickness of the bounding box lines.
        font_scale: The scale of the font.
        font: The font used for text.
    """
    x_min, y_min, x_max, y_max = bbox
    # Draw the bounding box
    image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
    
    # Add label text above the bounding box
    text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    text_x = x_min
    text_y = y_min - 10 if y_min - 10 > 10 else y_min + 10  # Position the text above or below the box

    image = cv2.putText(image, label, (text_x, text_y), font, font_scale, color, thickness)
    
    return image

# Streamlit UI
st.title("Bounding Box Detection with Object Name")

st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image file", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # Load and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image for the model
    st.write("Processing the image...")
    target_size = (128, 128)  # Replace with your model's input size
    img_array = np.array(image)
    original_size = img_array.shape[:2]  # Save original size for scaling
    img_resized = cv2.resize(img_array, target_size)  # Resize to model input size
    img_normalized = img_resized / 255.0  # Normalize
    img_batch = np.expand_dims(img_normalized, axis=0)  # Add batch dimension

    # Predict bounding box and object class
    st.write("Running prediction...")
    bbox_pred, class_pred = model.predict(img_batch)

    # Scale the bounding box back to the original image size
    h_scale = original_size[0] / target_size[0]
    w_scale = original_size[1] / target_size[1]
    bbox_scaled = bbox_pred[0] * [w_scale, h_scale, w_scale, h_scale]
    bbox_scaled = bbox_scaled.astype(int)

    # Assuming class_pred contains a string (e.g., 'cat', 'dog', etc.)
    label = str(class_pred[0])  # Convert class to string for the label

    st.write("### Predicted Bounding Box Coordinates and Class:")
    st.write(f"x_min: {bbox_scaled[0]}, y_min: {bbox_scaled[1]}, x_max: {bbox_scaled[2]}, y_max: {bbox_scaled[3]}")
    st.write(f"Predicted Class: {label}")

    # Draw the bounding box and label on the original image
    img_with_bbox = draw_bounding_box_with_label(img_array.copy(), bbox_scaled, label)

    # Display the image with bounding box and label
    st.image(img_with_bbox, caption="Image with Predicted Bounding Box and Label", use_column_width=True)
else:
    st.write("Upload an image file to start.")

# Footer
st.sidebar.info("Built with Streamlit and TensorFlow")