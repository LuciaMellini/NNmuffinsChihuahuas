
import subprocess
import importlib
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import kerastuner as kt
from sklearn.metrics import zero_one_loss
from sklearn.model_selection import KFold
import keras
from keras.callbacks import EarlyStopping
import json
from os.path import exists

def install(package_name: str):
    """
    Imports a package if it is already installed, otherwise installs it using pip and then imports it.

    Args:
        package_name (str): The name of the package to import or install.
        alias (str): An optional alias to assign to the imported package.
    """
    try:
        importlib.import_module(package_name)
        print(f"{package_name} is already installed.")
    except ImportError:
        print(f"{package_name} is not installed. Installing...")
        subprocess.check_call(['pip', 'install', package_name])
        print(f"{package_name} has been installed.")
        
        
def remove_badly_encoded_images(directory: str) -> None:
    """
    Removes badly encoded images from the specified directory.
    
    Args:
        directory (str): The path to the directory containing the images.
        
    Returns:
        None
        
    """
    for filename in os.listdir(directory):
        if filename.endswith('.JPG'):
            try:
                img = Image.open(directory+filename)  
                img.verify()
                img.close()
            except (IOError, SyntaxError):
                os.remove(directory+filename)
                
                
def dataset_to_numpy_arrays(dataset:tf.data.Dataset) -> tuple[np.array, np.array]:   
    """
    Converts a TensorFlow dataset to NumPy arrays.

    Args:
        dataset (tf.data.Dataset): The TensorFlow dataset to convert.

    Returns:
        tuple: A tuple containing the NumPy arrays (X, y). X represents the features and y represents the labels.
    """
    dataset = dataset.unbatch()
    X = []
    y = []
    for features, labels in dataset:
        X.append(features.numpy())
        y.append(labels.numpy())
    X = np.array(X)
    y = np.array(y)
    return X, y


def rgb_to_grayscale(image: tf.Tensor) -> tf.Tensor:
    """
        Converts an RGB image to grayscale.
        
        Args:
            image (Tensor): The input RGB image Tensor.
            
        Returns:
            Tensor: The grayscale image Tensor corresponding to image.
    """
    grayscale_image = tf.image.rgb_to_grayscale(image)
    return grayscale_image


def resize_image(image:tf.Tensor, target_size: tuple) -> tf.Tensor:
    """
    Resizes the given image to the specified target size.

    Parameters:
        image (PIL.Image.Image): The image to be resized.
        target_size (tuple): The target size of the image in the format (width, height).

    Returns:
        PIL.Image.Image: The resized image.
    """
    resized_image = tf.image.resize(image, target_size)
    return resized_image


def display_hypermodel(hypermodel: kt.HyperModel) -> None:
    """
    Displays the architecture of a hypermodel.

    Args:
        hypermodel_class (kt.Hypermodel): The hypermodel class to display the architecture of.

    Returns:
        None
    """
    built_hypermodel = hypermodel.build(kt.HyperParameters())
    return keras.utils.plot_model(built_hypermodel, show_layer_names=True)


def tune_hypermodel(model_name: str, hypermodel: kt.HyperModel, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, directory: str, max_trials: int) -> kt.Tuner:
    """
    Tune hyperparameters of a given model using random search.

    Args:
        model_name (str): Name of the model.
        model_class (class): Class of the model.
        train_ds (tf.data.Dataset): Training dataset.
        val_ds (tf.data.Dataset): Validation dataset.
        directory (str): Directory to store the search results.
        max_trials (int): Maximum number of trials to run.

    Returns:
        tuner (kerastuner.tuners.RandomSearch): Tuner object after performing the search.
    """
    tuner = kt.RandomSearch(
        hypermodel,
        objective='val_accuracy',
        directory=directory,
        max_trials=max_trials,
        project_name=model_name)

    tuner.search(train_ds, epochs=10, validation_data=val_ds)

    return tuner


def tuners_get_best_hyperparameters(tuners: dict[str: kt.Tuner], hypermodels: dict[str: kt.HyperModel]) -> dict[str: kt.HyperParameters]:
    """
    Get the best hyperparameters for each hypermodel using the provided tuners.

    Args:
        tuners (dict[str: kt.Tuner]): A dictionary of tuners, where the keys are hypermodel names and the values are tuners.
        hypermodels (dict[str: kt.HyperModel]): A dictionary of hypermodels, where the keys are hypermodel names and the values are hypermodels.

    Returns:
        dict[str: kt.HyperParameters]: A dictionary of best hyperparameters, where the keys are hypermodel names and the values are the best hyperparameters.

    """
    best_hps = {}
    for hypermodel_name in hypermodels:
        best_hps[hypermodel_name] = tuners[hypermodel_name].get_best_hyperparameters()[0]
    return best_hps
        
def tuners_display_best_hyperparameters(tuners: dict[str: kt.Tuner], hypermodels: dict[str: kt.HyperModel]):
    """
    Display the best hyperparameters for each tuner.

    Args:
        tuners (dict[str: kt.Tuner]): A dictionary of tuners, where the keys are the tuner names and the values are the tuner objects.
        hypermodels (dict[str: kt.HyperModel]): A dictionary of hypermodels, where the keys are the hypermodel names and the values are the hypermodel objects.

    Returns:
        None
    """
    best_hps = tuners_get_best_hyperparameters(tuners, hypermodels)
    best_hps_values = {k: v.values for k, v in best_hps.items()}
    print(pd.DataFrame.from_dict(best_hps_values).fillna(0).astype(int).replace(0,"-"))
    
def tuners_display_metrics(tuners: dict[str: kt.Tuner], hypermodels: dict[str: kt.HyperModel]):
    """
    Display the best metrics for each hypermodel.

    Args:
        tuners (dict[str: kt.Tuner]): A dictionary of tuners, where the keys are hypermodel names and the values are tuners.
        hypermodels (dict[str: kt.HyperModel]): A dictionary of hypermodels, where the keys are hypermodel names and the values are hypermodels.

    Returns:
        None
    """
    best_metrics = {}
    for hypermodel_name in hypermodels:
        best_metrics[hypermodel_name] = tuners[hypermodel_name].oracle.get_best_trials(1)[0].score
    print(pd.DataFrame({ k:[v] for (k,v) in best_metrics.items()}, index = ["loss"]))
    
    
def get_best_models(tuners: dict[str: kt.Tuner], hypermodels: dict[str: kt.HyperModel]) -> dict[str: keras.Model]:
    """
    Returns the best models built using the given tuners and hypermodels.

    Args:
        tuners (dict[str: kt.Tuner]): A dictionary of tuners, where the keys are hypermodel names and the values are tuners.
        hypermodels (dict[str: kt.HyperModel]): A dictionary of hypermodels, where the keys are hypermodel names and the values are hypermodels.

    Returns:
        dict: A dictionary mapping hypermodel names to the best models obtained by setting the best hyperparameters.

    """
    best_model = {}
    best_hps = tuners_get_best_hyperparameters(tuners, hypermodels)
    for hypermodel_name, hypermodel in hypermodels.items():
        hps = best_hps[hypermodel_name]
        best_model[hypermodel_name[5:]] = hypermodel.build(hps)
    return best_model


def fold_validation(model: keras.Model, X: np.array, y: np.array, train_index: np.array, test_index: np.array) -> float:
    """
    Perform fold validation on a given model using the specified data and indices.

    Args:
        model (keras.Model): The model to be trained and validated.
        X (np.array): The input data.
        y (np.array): The target labels.
        train_index (np.array): The indices of the training portion of the data.
        test_index (np.array): The indices of the test portion of the data.

    Returns:
        float: The zero-one loss of the model on the test portion of the data.
    """
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    print("\nFitting model to train portion")
    callback = EarlyStopping(monitor='val_loss', patience=3)
    model.fit(X_train, y_train, epochs = 100, batch_size = 16, callbacks = [callback])
    
    print("\nPredicting labels for test portion")
    predicted_labels_continuous = model.predict(X_test)
    predicted_labels = [round(l[0]) for l in predicted_labels_continuous]
    loss = zero_one_loss(y_test, predicted_labels)
    return loss

def cross_validation(model: keras.Model, model_name: str, X: np.array, y: np.array, num_folds: int) -> list[float]:
    """
    Perform cross-validation on a given model.

    Args:
        model (keras.Model): The model to be evaluated.
        model_name (str): The name of the model.
        X (np.array): The input data.
        y (np.array): The target data.
        num_folds (int): The number of folds for cross-validation.

    Returns:
        list[float]: A list of loss values for each fold.
    """
    kf = KFold(n_splits=num_folds)
    
    losses = [0]*num_folds
    for i,(train_index, test_index) in enumerate(kf.split(X)):
        print(f"Fold: {i+1}")
        model.load_weights(f"{DATA_PATH}/{model_name}.h5")
        losses[i] = fold_validation(model, X, y, train_index, test_index)
        
    return losses


class LossesJson:
    """
    A class for managing loss data and saving it to a JSON file.

    Args:
        filename (str): The path to the JSON file.

    Methods:
        __init__(self, filename): Initializes the LossesJson object.
        __str__(self): Returns a string representation of the loss data.
        update(self, model_name, losses): Updates the loss data for a specific model.
        save(self): Saves the loss data to the JSON file.
        load(self): Loads the loss data from the JSON file.
        get_losses(self): Returns the current loss data.

    """

    def __init__(self, filename: str):
        """
        Initializes the LossesJson object.

        If the JSON file does not exist, it creates an empty file.

        Args:
            filename (str): The path to the JSON file.

        """
        self.filename = filename
        if not exists(filename):
            with open(filename, 'w') as file:
                json.dump({}, file, indent=4)
        self.load()

    def __str__(self):
        """
        Returns a string representation of the loss data, in the form of a pandas DataFrame that displays the loss data in a tabular format.

        Returns:
            str: A string representation of the loss data.

        """
        aesthetic_dict = {
            model: {
                **model_data["loss_folds"],
                "loss_cv": model_data["loss_cv"]
            }
            for model, model_data in self.losses.items()
        }
        return str(pd.DataFrame.from_dict(aesthetic_dict))

    def update(self, model_name: str, losses: list[float]):
        """
        Updates the loss data for a specific model.

        Args:
            model_name (str): The name of the model.
            losses (list[float]): A list of loss values.

        """
        def losses_to_dict(loss_cv: float, loss_folds: list[float]):
            losses_dict = {}
            num_folds = len(loss_folds)
            losses_dict["loss_folds"] = {}
            for fold in range(num_folds):
                fold_name = f"fold_{fold+1}"
                losses_dict["loss_folds"][fold_name] = loss_folds[fold]
            losses_dict["loss_cv"] = loss_cv
            return losses_dict

        self.losses[model_name] = losses_to_dict(np.mean(losses), losses)
        self.save()

    def save(self):
        """
        Saves the loss data to the JSON file.

        """
        with open(self.filename, 'w') as file:
            json.dump(self.losses, file, indent=4)

    def load(self):
        """
        Loads the loss data from the JSON file.

        """
        with open(self.filename, 'r') as file:
            self.losses = json.load(file)

    def get_losses(self):
        """
        Returns the current loss data.

        Returns:
            dict: The current loss data.

        """
        self.load()
        return self.losses


def save_execute_cross_validation_on_models(filename: str, models: dict[str:keras.Model], X: np.array, y: np.array, num_folds: int) -> LossesJson:
    
    # Initialize LossesJson object
    losses_json = LossesJson(filename)
    
    # Load losses from filename.json
    if exists(filename):
        print(f"Loading losses from {filename} for models: ")
    
    # Get models for which there are no precomputed losses
    models_without_losses = dict(models)
    for key in losses_json.get_losses():
        print(f"{key} ")
        models_without_losses.pop(key)
    
    # Execute cross validation on models without precomputed losses
    if len(models_without_losses):        
        for model_name, model_class in models_without_losses.items():
            print(f"### Cross_validation for model: {model_name} ###")
            model_class.save_weights(f"{DATA_PATH}/{model_name}.h5")
            losses = cross_validation(model_class, model_name, X,y, num_folds)
            print(f"### Cross_validation results for model: {model_name} ###")
            print(f"Losses: {losses}")
            print(f"Cross validation loss: {np.mean(losses)}")
            losses_json.update(model_name, losses)
    
    return losses_json