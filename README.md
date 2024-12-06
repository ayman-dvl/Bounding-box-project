#Read_The_Doc

## Supply chain Optimization, Boxing and shelfing





The base of this project is to take non-damaged boxes and store them into a shelf that contains enough space for that box to be packed.
The approach would be divided into the creation of three CNN detection models and a Batch number scanner :

- Box Detection (ensure there's a box at the entry before processing to check whether its damaged or not)
- Batch Code scanner (to scan the barcode of each entering box)
- Damaged box Detection (here we're not gonna go for a classification methodology , instead we're gonna upload a dataset of damaged boxes and if he doesnt detect it , it means that the box is in fact not damaged , thanks to the first layer of our process which is the first model of box detection)
- Shelf space detection (it's gonna predict if a shelf has space to pack the non damaged box there)

## Features


- Import a HTML file and watch it magically convert to Markdown
- Drag and drop images (requires your Dropbox account be linked)
- Import and save files from GitHub, Dropbox, Google Drive and One Drive
- Drag and drop markdown and HTML files into Dillinger
- Export documents as Markdown, HTML and PDF

The dataset we used was taken from roboflow . It is a labelised data with a YOLOv11 format

> Links of datasets
> formatting syntax is to make it as readable
> as possible. The idea is that a
> Markdown-formatted document should be
> publishable as-is, as plain text, without
> looking like it's been marked up with tags
> or formatting instructions.







## Installation
Install the dependencies .

sh
pip install tensorflow
pip install pytorch
and other dependencies


For production environments, creation of a virtual environment and Scripts activation...

sh
python -m venv (name of virtual environement)
.\(name of virtual environment)\Scripts\activate



Streamlit Deployment for the Models.

sh
streamlit run app.py
gotta add the image of the interface
127.0.0.1:8000




*Thank you for your attention!*
