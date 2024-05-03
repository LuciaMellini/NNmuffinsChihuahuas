import keras_tuner as kt
import keras
from keras import layers
import numpy as np

class HyperModel_1(kt.HyperModel):
    """
    A hypermodel class for building a convolutional neural network.

    Args:
        input_shape (tuple[int, int]): The shape of the input data.

    Methods:
        __init__(self, input_shape: tuple[int, int]): Initializes the hypermodel.
        build(self, hp: kt.HyperParameters) -> keras.Model: Builds the convolutional neural network model.

    """

    def __init__(self, input_shape: tuple[int, int, int]):
        self.input_shape = input_shape

    def build(self, hp: kt.HyperParameters) -> keras.Model:
        """
        Build the convolutional neural network model.

        Args:
            hp (kt.HyperParameters): The hyperparameters for model configuration.

        Returns:
            keras.Model: The compiled model.

        """
        model = keras.Sequential([
            layers.Input(shape=self.input_shape, name='input'),
            layers.Conv2D(hp.Choice('filters_1', [16, 32, 64]), np.array(hp.Choice('kernel_size_1', range(3, 6))).repeat(2), activation='relu', padding = 'valid', name='convolutional_1'),
            layers.MaxPooling2D(np.array(hp.Choice('pool_size_1', range(2, 5))).repeat(2), name='maxPooling_1'),
            layers.Conv2D(hp.Choice('filters_2', [16, 32, 64]), np.array(hp.Choice('kernel_size_2', range(3, 6))).repeat(2), activation='relu', padding = 'valid', name='convolutional_2'),
            layers.MaxPooling2D(np.array(hp.Choice('pool_size_2', range(2, 5))).repeat(2), name='maxPooling_2'),
            layers.Flatten(name='flatten'),
            layers.Dense(1, activation='sigmoid', name='output')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return model
    
    
class HyperModel_2(kt.HyperModel): 
    """
    A hypermodel class for building a convolutional neural network.

    Args:
        input_shape (tuple[int, int]): The shape of the input data.

    Methods:
        __init__(self, input_shape: tuple[int, int]): Initializes the hypermodel.
        build(self, hp: kt.HyperParameters) -> keras.Model: Builds the convolutional neural network model.

    """
    def __init__(self, input_shape: tuple[int, int]):
        self.input_shape = input_shape

    def build(self, hp:kt.HyperParameters) -> keras.Model:
        """
        Build the convolutional neural network model.

        Args:
            hp (kt.HyperParameters): The hyperparameters for model configuration.

        Returns:
            keras.Model: The compiled model.
            
        """
        model = keras.Sequential([
            layers.Input(shape=self.input_shape, name='input'),
            layers.Conv2D(hp.Choice('filters_1', [16, 32, 64]), np.array(hp.Choice('kernel_size_1', range(3, 6))).repeat(2), activation='relu', padding = 'valid', name='convolutional_1'),
            layers.MaxPooling2D(np.array(hp.Choice('pool_size_1', range(2, 5))).repeat(2), name='maxPooling_1'),
            layers.Conv2D(hp.Choice('filters_2', [16, 32, 64]), np.array(hp.Choice('kernel_size_2', range(3, 6))).repeat(2), activation='relu', padding = 'valid', name='convolutional_2'),
            layers.MaxPooling2D(np.array(hp.Choice('pool_size_2', range(2, 5))).repeat(2), name='maxPooling_2'),
            layers.Flatten(name='flatten'),
            layers.Dense(hp.Choice('units',[32,64,96]), activation='relu', name = 'dense_1'),
            layers.Dense(1, activation='sigmoid', name='output')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return model


class HyperModel_3(kt.HyperModel):
    """
    A hypermodel class for building a convolutional neural network.

    Args:
        input_shape (tuple[int, int]): The shape of the input data.

    Methods:
        __init__(self, input_shape: tuple[int, int]): Initializes the hypermodel.
        build(self, hp: kt.HyperParameters) -> keras.Model: Builds the convolutional neural network model.

    """
    def __init__(self, input_shape: tuple[int, int]):
        self.input_shape = input_shape

    def build(self, hp: kt.HyperParameters) -> keras.Model:
        """
        Build the convolutional neural network model.

        Args:
            hp (kt.HyperParameters): The hyperparameters for model configuration.

        Returns:
            keras.Model: The compiled model.

        """
        model = keras.Sequential([
            layers.Input(shape=self.input_shape, name='input'),
            layers.Conv2D(hp.Choice('filters_1', [16, 32, 64]), np.array(hp.Choice('kernel_size_1', range(3, 6))).repeat(2), activation='relu', padding = 'same', name='convolutional_1'),
            layers.MaxPooling2D(np.array(hp.Choice('pool_size_1', range(2, 5))).repeat(2), name='maxPooling_1'),
            layers.Conv2D(hp.Choice('filters_2', [16, 32, 64]), np.array(hp.Choice('kernel_size_2', range(3, 6))).repeat(2), activation='relu', padding = 'same', name='convolutional_2'),
            layers.MaxPooling2D(np.array(hp.Choice('pool_size_2', range(2, 5))).repeat(2), name='maxPooling_2'),
            layers.Conv2D(hp.Choice('filters_3', [16, 32, 64]), np.array(hp.Choice('kernel_size_3', range(3, 6))).repeat(2), activation='relu', padding = 'same', name='convolutional_3'),
            layers.MaxPooling2D(np.array(hp.Choice('pool_size_3', range(2, 5))).repeat(2), name='maxPooling_3'),
            layers.Flatten(name='flatten'),
            layers.Dense(1, activation='sigmoid', name='output')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return model


class HyperModel_4(kt.HyperModel):
    """
    A hypermodel class for building a convolutional neural network.

    Args:
        input_shape (tuple[int, int]): The shape of the input data.

    Methods:
        __init__(self, input_shape: tuple[int, int]): Initializes the hypermodel.
        build(self, hp: kt.HyperParameters) -> keras.Model: Builds the convolutional neural network model.

    """
   
    def __init__(self, input_shape: tuple[int, int]):
        self.input_shape = input_shape

    def build(self, hp: kt.HyperParameters) -> keras.Model:
        """
        Build the convolutional neural network model.

        Args:
            hp (kt.HyperParameters): The hyperparameters for model configuration.

        Returns:
            keras.Model: The compiled model.

        """
        model = keras.Sequential([
            layers.Input(shape=self.input_shape, name='input'),
            layers.Conv2D(hp.Choice('filters_1', [16, 32, 64]), np.array(hp.Choice('kernel_size_1', range(3, 6))).repeat(2), activation='relu', padding = 'same', name='convolutional_1'),
            layers.MaxPooling2D(np.array(hp.Choice('pool_size_1', range(2, 5))).repeat(2), name='maxPooling_1'),
            layers.Conv2D(hp.Choice('filters_2', [16, 32, 64]), np.array(hp.Choice('kernel_size_2', range(3, 6))).repeat(2), activation='relu', padding = 'same', name='convolutional_2'),
            layers.MaxPooling2D(np.array(hp.Choice('pool_size_2', range(2, 5))).repeat(2), name='maxPooling_2'),
            layers.Conv2D(hp.Choice('filters_3', [16, 32, 64]), np.array(hp.Choice('kernel_size_3', range(3, 6))).repeat(2), activation='relu', padding = 'same', name='convolutional_3'),
            layers.MaxPooling2D(np.array(hp.Choice('pool_size_3', range(2, 5))).repeat(2), name='maxPooling_3'),
            layers.Flatten(name='flatten'),
            layers.Dense(hp.Choice('units',[32,64,96]), activation='relu', name = 'dense_1'),
            layers.Dense(1, activation='sigmoid', name='output')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return model