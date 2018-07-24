## Table of Contents
- [About the Project](##About-the-Project)
- [Dependecies](##Dependencies)
- [Development](##Development)
- [About the Developers](##About-the-Developers)

## About the Project

This is a program made as part of the Drone Image Analysis project for the [International Rice Research Institute](http://irri.org/). It processes drone images for input to a deep learning neural network regression model that predicts rice phenotype.

## Dependecies

The programs require the following libraries and modules to be installed:

#### Image Processing

- `Python 2.7`
- `PIL` or `pillow`
- `scikit-learn`
- `Numpy`
- `Pandas` or `Geopandas`
- `gdal` or `ogr`
- `OpenCV 2.x`

#### Building Neural Network

- `Keras` + `Tensorflow 1.3.0`
- `sklearn`

These dependencies can be installed along with the environment of OpenCV using this command `conda install -n opencv -c conda-forge <module>` (assuming that you already have anaconda2).

## Development

To run the programs, open a terminal and follow these steps:

1.  Make sure that the drone image is in the same directory as detect.py. Go to the directory by typing `cd image-processing`
2.  Run this command <python detect.py>
    - This should generate a tif file which is the detected rice field inside the same directory.
3.  Run this command on your terminal `python extract.py`
    - This should generate a csv file that contains the extracted data from the drone image in a directory named model.
4.  Go to that directory by typing `cd model`
5.  Run this command `python dnn.py` to build and train the deep learning regression model.
    - This should generate logs containing the loss and mean absolute error of each epoch during training. The values are displayed automatically in the terminal.
6.  For a better visualization, run this command: `tensorboard --logdif=logs/` and enter this URL `http:localhost:6006` using any browser.

Note: Each script has its own documentation. 
To improve the development of the programs, you can always refer to online documentations.

## About the Developers

The program is written by [Loria Roie Grace Malingan](https://github.com/malinganloria), a BS Computer Science student at the University of the Philippines Los Baños. The other part of the project which is a web app is made by Jasper Arquilita, a co-intern.
