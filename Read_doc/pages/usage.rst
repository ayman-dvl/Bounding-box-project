Streamlit Deployment for the bounding box Models
================================================

To deploy the models using Streamlit, use the following command in terminal:

.. code-block:: bash

    streamlit run app.py

Train your own model for the bounding box detection
===================================================
If you want to train your own model for the bounding box detection, follow the steps below:

Check your dataset
------------------
1. change paths in the code to the path of the dataset you want to check.
.. image:: pages/images/folders_paths.jpg
    :alt: paths to folders
    :width: 600px
    :align: center

2. run the code in terminal
.. code-block:: bash

    streamlit run dataset_sample_check.py

Train the model
---------------
1. change paths in the code to the path of the dataset you want to train the model on.
The dataset should be divided into training and validation folders.
And each folder should contain the images and the labels.
The folder structure should look like this:

.. code-block:: text

    dataset/
    ├── train/
    │   ├── images/
    │   └── labels/
    └── val/
        ├── images/
        └── labels/

.. image:: pages/images/train_val_folders.jpg
    :alt: paths to training and validation folders
    :width: 600px
    :align: center

2. run the code in terminal
.. code-block:: bash

    streamlit run model_train.py

Evaluate the model
------------------
1. change paths in the code to the path of the dataset you want to evaluate the model on.
.. image:: pages/images/model_info_paths.jpg
    :alt: paths to model and test image
    :width: 600px
    :align: center
2. run the code in terminal
.. code-block:: bash

    streamlit run model_test.py

