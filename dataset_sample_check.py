import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random



def yolo_data_generator(image_folder, label_folder,batch_size=32, target_size=(128, 128)):
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
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, target_size)  # Resize to target size
                batch_images.append(img_resized / 255.0)  # Normalize image

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

            # Visualize the first image in the batch along with its bounding boxes
            image_to_show = batch_images[random.randint(0, 31)] * 255.0  # Rescale to 0-255 for visualization
            image_to_show = image_to_show.astype(np.uint8)

            # Visualize using Matplotlib
            plt.figure(figsize=(6, 6))
            plt.imshow(image_to_show)
            ax = plt.gca()
            # Ensure that batch_classes[idx] is iterable (list of class labels)
            for bbox, class_id in zip(batch_bboxes, batch_classes):
                # Convert YOLO format to pixel coordinates
                x_center, y_center, width, height = bbox
                x_min = x_center - width / 2
                y_min = y_center - height / 2

                # Plot bounding box
                rect = plt.Rectangle((x_min, y_min), width, height,
                                    linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

                # Add class label text near the box
                plt.text(x_min, y_min - 5, f'Class {class_id}', color='red', fontsize=12)

            plt.axis('off')
            plt.show()


            # Yield the batch of images and corresponding labels
            yield (np.array(batch_images),
                   {'bbox': batch_bboxes, 'class': batch_classes})



def main():
    # Set the paths to the image and label folders
    image_folder = 'C:/Users/___user___/Documents/Projects/datasets/boxes/train/images'  # Replace with the actual path to your images
    label_folder = 'C:/Users/___user___/Documents/Projects/datasets/boxes/train/labels'  # Replace with the actual path to your labels

    # Create an instance of the data generator
    batch_size = 32
    target_size = (128, 128)
    generator = yolo_data_generator(image_folder, label_folder,batch_size, target_size)

    # Get a single batch of data
    images, labels = next(generator)

    # Print the images and their bounding boxes
    # This will visualize the first 10 images in the batch
    print("Displaying images and their bounding boxes:")
    # Visualizing the batch of images (already done in the generator itself)

if __name__ == "__main__":
    main()