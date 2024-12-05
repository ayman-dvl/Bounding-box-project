import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Step 1: Prepare the Dataset
def prepare_data(image_paths, annotations, img_size=(128, 128)):
    """
    Prepares the dataset.
    - image_paths: List of image file paths.
    - annotations: List of bounding box annotations [x_min, y_min, x_max, y_max].
    - img_size: Tuple of desired image size (height, width).
    """
    images = []
    labels = []
    
    for img_path, bbox in zip(image_paths, annotations):
        # Read and resize the image
        img = cv2.imread(img_path)
        img = cv2.resize(img, img_size)
        images.append(img)
        
        # Normalize bounding box values to [0, 1] relative to the image dimensions
        h, w, _ = img.shape
        x_min, y_min, x_max, y_max = bbox
        labels.append([x_min / w, y_min / h, x_max / w, y_max / h])
    
    return np.array(images), np.array(labels)

# Step 2: Define the CNN Model
def create_model(input_shape):
    """
    Creates a CNN model for bounding box prediction.
    - input_shape: Shape of the input image (height, width, channels).
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='sigmoid')  # 4 outputs for [x_min, y_min, x_max, y_max]
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

# Step 3: Train the Model
def train_model(model, train_data, val_data, epochs=20, batch_size=32):
    """
    Trains the CNN model.
    - model: The CNN model.
    - train_data: Tuple (X_train, y_train).
    - val_data: Tuple (X_val, y_val).
    - epochs: Number of training epochs.
    - batch_size: Batch size.
    """
    history = model.fit(
        train_data[0], train_data[1],
        validation_data=val_data,
        epochs=epochs,
        batch_size=batch_size
    )
    return history

# Step 4: Inference and Visualization
def draw_bounding_box(image, bbox, color=(0, 255, 0), thickness=2):
    """
    Draws a bounding box on the image.
    - image: Input image (numpy array).
    - bbox: Bounding box coordinates [x_min, y_min, x_max, y_max] in normalized form.
    - color: Color of the bounding box (BGR format).
    - thickness: Thickness of the bounding box lines.
    """
    h, w, _ = image.shape
    x_min, y_min, x_max, y_max = bbox
    x_min, y_min, x_max, y_max = int(x_min * w), int(y_min * h), int(x_max * w), int(y_max * h)
    return cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)

def predict_and_visualize(model, image_path, img_size=(128, 128)):
    """
    Makes a prediction and visualizes the bounding box.
    - model: Trained CNN model.
    - image_path: Path to the input image.
    - img_size: Input size of the model.
    """
    # Read and preprocess the image
    img = cv2.imread(image_path)
    orig_img = img.copy()
    img = cv2.resize(img, img_size) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)

    # Predict bounding box
    pred_bbox = model.predict(img)[0]

    # Draw bounding box on the original image
    output_img = draw_bounding_box(orig_img, pred_bbox)
    plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

# Example Usage
if __name__ == "__main__":
    # Dummy dataset
    image_paths = ["image1.jpg", "image2.jpg"]  # Replace with your image file paths
    annotations = [[50, 50, 100, 100], [30, 30, 120, 120]]  # Replace with your bounding box annotations

    # Prepare data
    X, y = prepare_data(image_paths, annotations)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize images
    X_train, X_val = X_train / 255.0, X_val / 255.0

    # Create and train model
    model = create_model(X_train.shape[1:])
    train_model(model, (X_train, y_train), (X_val, y_val), epochs=10)

    # Test on a new image
    test_image_path = "test_image.jpg"  # Replace with a new image path
    predict_and_visualize(model, test_image_path)
