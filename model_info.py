import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Function to load the trained model with custom objects (e.g., mse)
def load_model_from_local(model_path):
    try:
        # Custom objects for metrics or loss functions
        custom_objects = {'mse': tf.keras.losses.MeanSquaredError}
        
        # Load the model with custom objects
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print(f"Model loaded successfully from {model_path}")
        
        # Print model summary
        model.summary()  # Prints the architecture of the model
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Function to preprocess the image before feeding it to the model
def preprocess_image(image_path, target_size=(128, 128)):
    # Load image
    img = Image.open(image_path)
    img = img.convert("RGB")  # Ensure it's in RGB format
    img_array = np.array(img)

    # Resize image
    img_resized = cv2.resize(img_array, target_size)

    # Normalize pixel values to between 0 and 1
    img_normalized = img_resized / 255.0

    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)

    return img_array, img_batch, img

# Function to draw the bounding box on the image
def draw_bounding_box(image, bbox, color=(0, 255, 0), thickness=2):
    x_min, y_min, x_max, y_max = bbox
    image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
    return image

# Test the model with an image
def test_model(model, image_path):
    # Preprocess the image
    img_array, img_batch, img_pil = preprocess_image(image_path)

    # Print input image shape and batch size
    print("Input Image Shape:", img_array.shape)
    print("Batch Image Shape:", img_batch.shape)

    # Predict bounding boxes
    bbox_pred = model.predict(img_batch)

    # Print the prediction output (raw)
    print("Bounding Box Prediction (Raw Output):")
    print(bbox_pred)

    # Check if the output is a list and convert it to a numpy array
    if isinstance(bbox_pred, list):
        bbox_pred = np.array(bbox_pred[0])
        print("Converted prediction to NumPy array.")

    # Inspect the shape of the output and print it
    print("Shape of raw prediction output:", bbox_pred.shape)

    # Check if we have a 2D array (batch size x prediction)
    if len(bbox_pred.shape) == 2:
        print("Prediction output has shape (batch_size, number_of_predictions)")

    # Extract and print each element in the prediction (for debugging)
    print("Extracting individual elements from the raw prediction:")
    for i, pred in enumerate(bbox_pred):
        print(f"Prediction {i+1}:")
        print(pred)
        if len(pred) == 4:
            print(f"Bounding Box Coordinates: x_min={pred[0]}, y_min={pred[1]}, x_max={pred[2]}, y_max={pred[3]}")
        else:
            print("Unexpected output format for bounding box.")

    # Extract bounding box coordinates (assuming first prediction is the bounding box)
    bbox = bbox_pred[0]  # Assuming first prediction in batch is the bounding box
    print("Extracted Bounding Box Coordinates:", bbox)

    # Check if the bounding box output has the expected shape
    if len(bbox) == 4:
        # Scale back the bounding box to the original image size
        original_size = img_array.shape[:2]  # (height, width)
        target_size = (128, 128)  # Make sure this matches the model's input size

        # Scaling factors
        h_scale = original_size[0] / target_size[0]
        w_scale = original_size[1] / target_size[1]

        # Apply scaling to bounding box
        bbox_scaled = bbox * [w_scale, h_scale, w_scale, h_scale]
        bbox_scaled = bbox_scaled.astype(int)

        # Draw bounding box on the image
        img_with_bbox = draw_bounding_box(img_array.copy(), bbox_scaled)

        # Display the results
        cv2.imshow("Predicted Bounding Box", img_with_bbox)
        cv2.waitKey(0)  # Wait for a key press to close
        cv2.destroyAllWindows()
    else:
        print("Unexpected output format for bounding box!")

# Main function to test the model
if __name__ == "__main__":
    # Path to the trained model and image
    model_path = 'C:/Users/___user___/Documents/Projects/stock_manage/model/model_object_detection.h5'
    image_path = 'C:/Users/___user___/Documents/Projects/box_detect/test/images/2-jpg__jpg-jpganti-clockwise_jpg.rf.c1573a3fb5b257bba7939c75d387086e.jpg'

    # Load the model
    model = load_model_from_local(model_path)

    if model:
        # Test the model with the provided image
        test_model(model, image_path)
    else:
        print("Model loading failed, cannot proceed with testing.")
