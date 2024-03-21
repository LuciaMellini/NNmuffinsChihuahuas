# Distinguishing muffins and Chihuahuas with NNs
The aim of this project is to train a neural network for the binary classification of images of muffins and Chihuahuas based in the relative [Kaggle dataset](https://www.kaggle.com/datasets/samuelcortinhas/muffin-vs-chihuahua-image-classification).

## Contents
This repository contains two main files:
* `project.ipynb` the Jupyter Notebook with all the adopted code described as necessary,
* `report.tex` the $\LaTeX$ source code for the report of the project; this document goes into more detail about the formal aspects of the used approaches and the observed results.

Also, it provides a `tuner` directory, containing precomputed hyperparameters for the proposed hypermodels. During the execution of the Jupyter Notebook, with `tuners` in the same directory, the `search` function offered by `keras_tuner` will automatically retrieve the precomputed information, reducing the computational time.

## Prerequisites
To be able to download the dataset it is necessary to authenticate with a Kaggle username and token<sup>[1](#fn1)</sup>. At this scope the second code block in the notebook `project.ipynb` is intended to be filled out with your personal username and key in the respective fields, like suggested below.
```python
    os.environ['KAGGLE_USERNAME'] = "<USERNAME>"
    os.environ['KAGGLE_KEY'] = "<KEY>"
```
<a name="fn1">1</a> To create a new token go to the [settings of your Kaggle account](https://www.kaggle.com/settings) under the *API* section, and push on the dedicated button.