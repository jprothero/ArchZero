from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
from keras import utils as np_utils

class Dataset():
    def __init__(self, augment=False, batch_size=64):
        (self.X_train, self.y_train), (self.X_test, self.y_test) = cifar10.load_data()
        self.batch_size = batch_size

        self.y_train = np_utils.to_categorical(self.y_train, 10)
        self.y_test = np_utils.to_categorical(self.y_test, 10)

        if augment:
            self.train_datagen = ImageDataGenerator(
                  rotation_range=40,
                  width_shift_range=0.2,
                  height_shift_range=0.2,
                  shear_range=0.2,
                  zoom_range=0.2,
                  horizontal_flip=True,
                  fill_mode='nearest')
        else:
            self.train_datagen = ImageDataGenerator()

        self.test_datagen = ImageDataGenerator()

        self.X_train = self.X_train.astype('float32') / 255
        self.X_test = self.X_test.astype('float32') / 255

        X_train_mean = np.mean(self.X_train, axis = 0)
        self.X_train -= X_train_mean
        self.X_test -= X_train_mean

        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(
            self.X_test, self.y_test, test_size = 0.5)
        
        self.train_steps = len(self.X_train) // self.batch_size // 10 
        self.val_steps = len(self.X_val) // self.batch_size // 10 
        self.test_steps = len(self.X_test) // self.batch_size // 10 
        
        percentage = 1
        
        self.train_gen = self.train_datagen.flow(
                self.X_train[:int(len(self.X_train) * percentage)], self.y_train[:int(len(self.y_train) * percentage)],
                batch_size = self.batch_size)

        self.val_gen = self.test_datagen.flow(
                self.X_val[:int(len(self.X_val) * percentage)], self.y_val[:int(len(self.y_val) * percentage)],
                batch_size = self.batch_size)

        self.test_gen = self.test_datagen.flow(
                self.X_test[:int(len(self.X_test) * percentage)], self.y_test[:int(len(self.y_test) * percentage)],
                batch_size = self.batch_size)
        