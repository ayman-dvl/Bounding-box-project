import cv2
import numpy as np
from pyzbar.pyzbar import decode
import streamlit as st
from PIL import Image

# Function to decode the barcode
def BarcodeReader(image):
    # Convert the uploaded image to a format usable by OpenCV
    image = np.array(image.convert('RGB'))
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Decode the barcode image
    detectedBarcodes = decode(img)

    # Create an output image with a highlighted rectangle
    output_img = img.copy()

    if not detectedBarcodes:
        st.warning("Barcode Not Detected or the image is blank/corrupted!")
    else:
        # Traverse through all detected barcodes in the image
        for barcode in detectedBarcodes:
            # Locate the barcode position in the image
            (x, y, w, h) = barcode.rect

            # Draw a rectangle around the barcode
            cv2.rectangle(output_img, (x-10, y-10), 
                          (x + w + 10, y + h + 10), 
                          (255, 0, 0), 2)

            # Display the barcode data and type
            if barcode.data:
                st.success(f"Data: {barcode.data.decode('utf-8')}")
                st.info(f"Type: {barcode.type}")

    # Convert BGR image back to RGB for display
    output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
    return output_img

# Streamlit app interface
def main():
    st.title("Barcode Reader App")
    st.write("Upload an image containing a barcode to extract its data.")

    # File uploader for the image
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Open the image
        image = Image.open(uploaded_file)

        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Perform barcode reading
        st.write("Processing...")
        output_img = BarcodeReader(image)

        # Display the processed image
        st.image(output_img, caption="Processed Image", use_column_width=True)

if __name__ == "__main__":
    main()