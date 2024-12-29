Overview
--------

This project was created to take non-damaged boxes and store them in a shelf that contains enough space for the box to be packed.

The approach is divided into the creation of 3 CNN detection models (with only one currently included) and a batch number scanner:

- **Box Detection**: Ensures there is a box at the entry before processing to check for damages.
- **Batch Code Scanner**: Scans the barcode of each entering box.
- **Damaged Box Detection**: Uploads a dataset of damaged boxes and if the model doesn't detect it, the box is not damaged.
- **Shelf Space Detection**: Predicts if a shelf has space to pack the non-damaged box.

Features
--------

- **Box Identification**
- **Barcode Scanning**

Dataset Used for Training
-------------------------

The datasets used were sourced from Roboflow and are labeled in YOLOv11 format.

Links to datasets:

- `Dataset 1 <https://universe.roboflow.com/yolov7-scbtt/box-detection-xuvru/dataset/1>`_
- `Dataset 2 <https://universe.roboflow.com/ece4191-xcxot/cardboard-box-detection-mxqjh/dataset/1>`_
- `Dataset 3 <https://universe.roboflow.com/yolov3tiny/box-detection-f04xv/dataset/3>`_

Batch Number Scanner
--------------------
The batch number scanner is a simple barcode scanner that reads the barcode of each box entering the system.
It is imported from the ``pyzbar`` library.
Check the full project by *Lawrence Hudson* in the following link: `Pyzbar project <https://pypi.org/project/pyzbar/>`_