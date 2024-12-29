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

   pages/overview
   pages/installation
   pages/usage
   pages/contributing


