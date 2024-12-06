import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import cv2
from pyzbar.pyzbar import decode

# Define a custom object for `mse` if required by the model
custom_objects = {"mse": tf.keras.losses.MeanSquaredError()}

# Load the object detection model
@st.cache_resource
def load_local_model(model_path):
    return load_model(model_path, custom_objects=custom_objects)

model = load_local_model('C:/Users/___user___/Documents/Projects/stock_manage/bbox_model/model_object_detection.h5')

# Function to draw bounding boxes and object name
def draw_bounding_box_with_label(image, bbox, label, color=(255, 0, 0), thickness=2, font_scale=0.8, font=cv2.FONT_HERSHEY_SIMPLEX):
    x_min, y_min, x_max, y_max = bbox
    image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
    text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    text_x = x_min
    text_y = y_min - 10 if y_min - 10 > 10 else y_min + 10
    image = cv2.putText(image, label, (text_x, text_y), font, font_scale, color, thickness)
    return image

# Function to decode the barcode
def barcode_reader(image):
    image = np.array(image.convert('RGB'))
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    detected_barcodes = decode(img)
    output_img = img.copy()
    if not detected_barcodes:
        st.warning("Barcode Not Detected or the image is blank/corrupted!")
    else:
        for barcode in detected_barcodes:
            (x, y, w, h) = barcode.rect
            cv2.rectangle(output_img, (x-10, y-10), 
                          (x + w + 10, y + h + 10), 
                          (255, 0, 0), 2)
            if barcode.data:
                st.success(f"Data: {barcode.data.decode('utf-8')}")
                st.info(f"Type: {barcode.type}")
    output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
    return output_img

# Streamlit app interface
def main():
    st.title("Stock Management App")
    st.sidebar.header("Choose a Functionality")
    option = st.sidebar.selectbox("Select an operation:", ["Bounding Box Detection", "Barcode Reader"])

    st.sidebar.header("Upload Image")
    uploaded_file = st.sidebar.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if option == "Bounding Box Detection":
            st.write("Processing the image for bounding box detection...")
            target_size = (128, 128)  # Replace with your model's input size
            img_array = np.array(image)
            original_size = img_array.shape[:2]
            img_resized = cv2.resize(img_array, target_size)
            img_normalized = img_resized / 255.0
            img_batch = np.expand_dims(img_normalized, axis=0)

            st.write("Running prediction...")
            bbox_pred, class_pred = model.predict(img_batch)
            h_scale = original_size[0] / target_size[0]
            w_scale = original_size[1] / target_size[1]
            bbox_scaled = bbox_pred[0] * [w_scale, h_scale, w_scale, h_scale]
            bbox_scaled = bbox_scaled.astype(int)
            label = str(class_pred[0])

            st.write("### Predicted Bounding Box Coordinates and Class:")
            st.write(f"x_min: {bbox_scaled[0]}, y_min: {bbox_scaled[1]}, x_max: {bbox_scaled[2]}, y_max: {bbox_scaled[3]}")
            st.write(f"Predicted Class: {label}")

            img_with_bbox = draw_bounding_box_with_label(img_array.copy(), bbox_scaled, label)
            st.image(img_with_bbox, caption="Image with Predicted Bounding Box and Label", use_column_width=True)

        elif option == "Barcode Reader":
            st.write("Processing the image for barcode reading...")
            output_img = barcode_reader(image)
            st.image(output_img, caption="Processed Image with Barcode Highlighted", use_column_width=True)

    else:
        st.write("Upload an image file to start.")

# Run the app
if __name__ == "__main__":
    main()
