.. Project Name documentation master file, created by
   sphinx-quickstart on YYYY-MM-DD.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

===================================================
Stock management: Box identification and codebar reader
===================================================

**Description:**  
The base of this project is to take non-damaged boxes and store them into a shelf that contains enough space for that box to be packed.
The approach would be divided into the creation of three CNN detection models (only one is included so far) and a Batch number scanner :

- Box Detection (ensure there's a box at the entry before processing to check whether its damaged or not)
- Batch Code scanner (to scan the barcode of each entering box)
- Damaged box Detection (here we're not gonna go for a classification methodology , instead we're gonna upload a dataset of damaged boxes and if he doesnt detect it , it means that the box is in fact not damaged , thanks to the first layer of our process which is the first model of box detection)
- Shelf space detection (it's gonna predict if a shelf has space to pack the non damaged box there)

**Author(s):**  
AIT ABDOU AYMAN & OUHSSAIN ANOUAR.

**Version:**  
v1.0.0

**License:**  
Not licensed.

Table of Contents
=================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   installation
   usage
   development

IStock Management: Box Identification and Barcode Scanner
=========================================================

Overview
--------

This project automates warehouse management by identifying, scanning, and sorting boxes into appropriate shelves. It focuses on non-damaged boxes and ensures efficient storage by detecting available shelf space.

The approach is divided into the creation of three CNN detection models (with only one currently included) and a batch number scanner:

- **Box Detection**: Ensures there is a box at the entry before processing to check for damages.
- **Batch Code Scanner**: Scans the barcode of each entering box.

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

Installation
------------

For production environments, follow these steps:

1. Create a virtual environment and activate it.
2. Install the required dependencies.

Run the following commands:

.. code-block:: bash

    pip install pipreqs
    python -m venv (name_of_virtual_environment)
    .\(name_of_virtual_environment)\Scripts\activate
    pip install -r requirements.txt

Streamlit Deployment for the Models
------------------------------------

To deploy the models using Streamlit, use the following command:

.. code-block:: bash

    streamlit run app.py

Thank you for your attention!
-----------------------------
.. toctree::
   :maxdepth: 1

   glossary

