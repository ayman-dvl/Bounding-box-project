Stock Management: Box Identification and Barcode Reader
========================================================

This project automates the management of stock by identifying, scanning, and storing boxes in an appropriate shelf. The key components include box detection, batch code scanning, damaged box detection, and shelf space detection.

Overview
--------

The project is divided into four main steps:

1. **Box Detection**:
   Ensures the presence of a box at the entry before proceeding to check if it is damaged.
   
2. **Batch Code Scanner**:
   Scans the barcode of each entering box for tracking and inventory management.
   
3. **Damaged Box Detection**:
   A dataset of damaged boxes is used. If a box is not detected as damaged, it is assumed to be non-damaged.

4. **Shelf Space Detection**:
   Predicts if a shelf has enough space to store the non-damaged box.

Datasets
--------

The labeled datasets used for training and testing were sourced from Roboflow in YOLOv11 format:

- `dataset_1`: Box Detection  
  `https://universe.roboflow.com/yolov7-scbtt/box-detection-xuvru/dataset/1`
  
- `dataset_2`: Cardboard Box Detection  
  `https://universe.roboflow.com/ece4191-xcxot/cardboard-box-detection-mxqjh/dataset/1`

- `dataset_3`: Box Detection  
  `https://universe.roboflow.com/yolov3tiny/box-detection-f04xv/dataset/3`

Installation
------------

Follow these steps to set up the environment for the project:

1. Create a virtual environment and activate it:

   .. code-block:: sh
   
      python -m venv (name of virtual environment)
      .\(name of virtual environment)\Scripts\activate

2. Install required dependencies:

   .. code-block:: sh
   
      pip install pipreqs
      pip install -r requirements.txt

Deployment
----------

Use Streamlit to deploy the models. Run the following command:

.. code-block:: sh
   
   streamlit run app.py

Thank You!
----------
We appreciate your attention and hope this project meets your expectations!
