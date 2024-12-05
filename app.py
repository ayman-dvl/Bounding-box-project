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
    """
    Load a TensorFlow model from a local path.
    Handles custom loss functions or metrics.
    """
    return load_model(model_path, custom_objects=custom_objects)

model = load_local_model('C:/Users/___user___/Documents/Projects/stock_manage/model/model_object_detection.h5')

# Function to draw bounding boxes on the image
def draw_bounding_box(image, bbox, color=(255, 0, 0), thickness=2):
    """
    Draws a single bounding box on an image.

    Args:
        image: The input image as a NumPy array.
        bbox: The bounding box in [x_min, y_min, x_max, y_max] format.
        color: The color of the bounding box.
        thickness: The thickness of the bounding box lines.
    """
    x_min, y_min, x_max, y_max = bbox
    image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
    return image

# Streamlit UI
st.title("Bounding Box Detection with TensorFlow Model")

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

    # Make a prediction and convert the result to a numpy array if it's a list
    bbox_pred = model.predict(img_batch)

    # If bbox_pred is a list, convert it to a NumPy array
    bbox_pred = np.array(bbox_pred[0])

    # Print the shape of the prediction to check the output structure
    st.write("Bounding box prediction shape:", bbox_pred.shape)
    st.write("Bounding box prediction:", bbox_pred)

    # Assuming bbox_pred is now a 2D array with shape (1, 4)
    if bbox_pred.shape == (1, 4):
        # Scale the bounding box back to the original image size
        h_scale = original_size[0] / target_size[0]
        w_scale = original_size[1] / target_size[1]
        bbox_scaled = bbox_pred[0] * [w_scale, h_scale, w_scale, h_scale]

        # Convert to integer values
        bbox_scaled = bbox_scaled.astype(int)

        st.write(f"Scaled Bounding Box: {bbox_scaled}")

        # Display the bounding box coordinates
        x_min, y_min, x_max, y_max = bbox_scaled
        st.write(f"x_min: {x_min}, y_min: {y_min}, x_max: {x_max}, y_max: {y_max}")

        # Draw the bounding box on the original image
        img_with_bbox = draw_bounding_box(img_array.copy(), bbox_scaled)

        # Display the image with bounding box
        st.image(img_with_bbox, caption="Image with Predicted Bounding Box", use_column_width=True)
    else:
        st.write("Unexpected output format from the model. Expected shape: (1, 4).")

# Footer
st.sidebar.info("Built with Streamlit and TensorFlow")
