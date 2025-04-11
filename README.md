

## Stock management: Box identification and codebar reader



This project is still under development, but you can use some features and python functions to handle YOLO format datasets and to train models for bounding box detection using tensorflow framework.

The base of this project is to take non-damaged boxes and store them into a shelf that contains enough space for that box to be packed.
The approach would be divided into the creation of three CNN detection models (only one is included so far) and a Batch number scanner :

- Box Detection (ensure there's a box at the entry before processing to check whether its damaged or not)
- Batch Code scanner (to scan the barcode of each entering box)
- Damaged box Detection (here we're not gonna go for a classification methodology , instead we're gonna upload a dataset of damaged boxes and if he doesnt detect it , it means that the box is in fact not damaged , thanks to the first layer of our process which is the first model of box detection)
- Shelf space detection (it's gonna predict if a shelf has space to pack the non damaged box there)
- Here is the pipeline followed for our project as mentionned earlier 
![Capture](https://github.com/user-attachments/assets/b62e0636-3520-40b8-91d6-d740a3aa4231)

## Read the docs
[Documentation](https://bounding-box-project.readthedocs.io/en/master/index.html)



*Thank you for your attention!*
