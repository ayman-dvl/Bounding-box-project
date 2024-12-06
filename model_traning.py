import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.models import Model
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2

train_image = 'C:/Users/___user___/Documents/Projects/box_detect/train/images'
train_label = 'C:/Users/___user___/Documents/Projects/box_detect/train/labels'
val_image = 'C:/Users/___user___/Documents/Projects/box_detect/valid/images'
val_label = 'C:/Users/___user___/Documents/Projects/box_detect/valid/labels'

# Custom YOLO Data Generator with error handling and padding for incorrect labels
def yolo_data_generator(image_folder, label_folder, batch_size=32, target_size=(128, 128)):
    image_files = sorted(f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg')))
    label_files = sorted(f for f in os.listdir(label_folder) if f.endswith('.txt'))
    
    num_samples = len(image_files)

    while True:  # Infinite loop for Keras generator
        for i in range(0, num_samples, batch_size):
            batch_images = []
            batch_bboxes = []
            batch_classes = []

            for j in range(i, min(i + batch_size, num_samples)):
                # Load and preprocess image
                image_path = os.path.join(image_folder, image_files[j])
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, target_size)  # Resize to target size
                batch_images.append(img / 255.0)  # Normalize image

                # Load and preprocess label file
                label_path = os.path.join(label_folder, label_files[j])
                with open(label_path, "r") as file:
                    lines = file.readlines()
                    bboxes = []
                    class_labels = []
                    
                    for line in lines:
                        # Split each line into parts (class label + bounding box values)
                        parts = line.strip().split()
                        
                        # Check if the line has the correct number of values
                        if len(parts) >= 5:  # Minimum 1 class label + 4 bounding box values
                            class_label = int(parts[0])  # Class label
                            
                            # Process bounding boxes (every 4 values after the class label)
                            for k in range(1, len(parts) - 3, 4):  # Iterate with step of 4 for bounding boxes
                                try:
                                    x_center, y_center, width, height = map(float, parts[k:k + 4])
                                    
                                    # Convert normalized coordinates to pixel coordinates
                                    x_min = int((x_center - width / 2) * target_size[0])
                                    y_min = int((y_center - height / 2) * target_size[1])
                                    x_max = int((x_center + width / 2) * target_size[0])
                                    y_max = int((y_center + height / 2) * target_size[1])

                                    # Store bounding box and class label
                                    bboxes.append([x_min, y_min, x_max, y_max])
                                    class_labels.append(class_label)

                                except ValueError:
                                    # If there's a problem converting the values, skip this bounding box
                                    print(f"Skipping invalid bounding box in file: {label_path}")
                                    continue

                    # If multiple boxes exist, take only the first box (or default box if none)
                    if bboxes:
                        bboxes = [bboxes[0]]  # Take the first bounding box
                        class_labels = [class_labels[0]]  # Take the first class label
                    else:
                        # If no boxes exist, use a default empty box and class 0
                        bboxes = [[0, 0, 0, 0]]
                        class_labels = [0]

                    batch_bboxes.append(bboxes)
                    batch_classes.append(class_labels)

            # Ensure bounding boxes have a consistent shape (batch_size, 4)
            batch_bboxes = np.array(batch_bboxes)
            batch_bboxes = np.squeeze(batch_bboxes, axis=1)  # Remove extra dimension for single box

            # Ensure class labels have a consistent shape (batch_size,)
            batch_classes = np.array(batch_classes)
            batch_classes = np.squeeze(batch_classes, axis=1)  # Remove extra dimension for single class

            # Yield the batch of images and corresponding labels
            yield (np.array(batch_images),
                   {'bbox': batch_bboxes, 'class': batch_classes})



num_images_train = len(os.listdir(train_image))
num_label_train = len(os.listdir(train_label))
num_images_val = len(os.listdir(val_image))
num_label_val = len(os.listdir(val_label))
print("num of images: ",num_images_train,'\n',
      "num of labels: ", num_label_train, '\n',
      "num of images: ", num_images_val, '\n',
      "num of labels: ", num_label_val, '\n')

input_layer = Input(shape=(128, 128, 3))

#layers
x = Conv2D(32, (3, 3), activation='relu')(input_layer)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)

# Bounding box output
bbox_output = Dense(4, activation='linear', name='bbox')(x)

# Classification output
class_output = Dense(1, activation='sigmoid', name='class')(x)

# Model
model = Model(inputs=input_layer, outputs=[bbox_output, class_output])

model.compile(
    optimizer='adam',
    loss={'bbox': 'mse', 'class': 'binary_crossentropy'},
    loss_weights={'bbox': 1.0, 'class': 0.5},
    metrics={'bbox': 'mae', 'class': 'accuracy'}
)

# Generators
train_generator = yolo_data_generator(image_folder=train_image, 
                                       label_folder=train_label, 
                                       batch_size=32, 
                                       target_size=(128, 128))
val_generator = yolo_data_generator(image_folder=val_image, 
                                     label_folder=val_label, 
                                     batch_size=32, 
                                     target_size=(128, 128))


# Training
history = model.fit(
    train_generator,
    steps_per_epoch=100,  # Number of batches per epoch
    epochs=20,
    validation_data=val_generator,
    validation_steps=20  # Number of batches per validation epoch
)

# Evaluation
loss, bbox_loss, class_loss, bbox_mae, class_accuracy = model.evaluate(val_generator, steps=20)
print(f"Bounding Box MAE: {bbox_mae}, Classification Accuracy: {class_accuracy}")

model.save("model_object_detection.h5")

