import time
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Input,Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras import Input, Model
from keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint


# Directory where the checkpoints will be saved.
dir = "models/"

#reads a file. Each line has the format: label text
#Returns a list with the text and a list with the labels
def readData(fname):

    with open(fname, 'r', encoding="utf-8") as f:
        fileData = f.read()
  
    lines = fileData.split("\n")
    textData = list()
    textLabel = list()
    lineLength = np.zeros(len(lines))
    
    for i, aLine in enumerate(lines):     
        if not aLine:
            break  
        label = aLine.split(" ")[0]
        lineLength[i] = len(aLine.split(" "))
        if(label == "__label__1"):
            textLabel.append(0)
            textData.append(remove_prefix(aLine, "__label__1 "))

        elif(label == "__label__2"):
            textLabel.append(1)
            textData.append(remove_prefix(aLine, "__label__2 "))

        else:
            print("\nError in readData: ", i, aLine)
            exit()
    
    f.close()
    return textData, textLabel, int(np.average(lineLength)+2*np.std(lineLength))

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text  

def transformData(x_train, y_train, x_test, y_test, maxFeatures, seqLength):
    #transforms text input to int input based on the vocabulary
    #max_tokens = maxFeatures is the size of the vocabulary
    #output_sequence_length =  seqLength is the maximum length of the transformed text. Adds 0 is text length is shorter
    precLayer = layers.experimental.preprocessing.TextVectorization(max_tokens = maxFeatures, 
    standardize =  'lower_and_strip_punctuation', split = 'whitespace', output_mode = 'int', 
    output_sequence_length =  seqLength)
    precLayer.adapt(x_train)
    #print(precLayer.get_vocabulary())
    x_train_int = precLayer(x_train)
    y_train = tf.convert_to_tensor(y_train)
    #print(x_train_int)
    #print(y_train)
    x_test_int= precLayer(x_test)
    y_test = tf.convert_to_tensor(y_test)
    #print(x_test_int)
    #print(y_test)

    return x_train_int, y_train, x_test_int, y_test

def visualize_fit(history):
    """Visualize the fit of a model. 

    Args:
        history (list): list of metrics along the epochs.  
    """    
    history_dict = history.history
    print(history_dict.keys())
    history_dict.keys()

    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']

    epochs = range(1, len(loss_values) + 1)

    plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_values, 'b-o', label='Training loss')
    plt.plot(epochs, val_loss_values, 'r-o', label='Validation loss')

    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']

    plt.plot(epochs, acc, 'b-o', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r-o', label='Validation Accuracy')
    plt.title('Training and validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.ylim([0, 1])
    plt.legend()

    plt.tight_layout()
    plt.show()

def fitModel(model, x_train, y_train, ds_val, num_epochs=20, monitor='val_accuracy', model_name='best_model.h5', callbacks=[], batch_size=32):
    """Function to train a model. It saves the best model in a file. It also prints the evolution of the training process.


    Args:
        model (Model): The model to be trained.
        ds_train (_type_): The training dataset. 
        ds_val (_type_): The validation dataset.
        num_epochs (int, optional): Defaults to 20.
        monitor (str, optional): Metric to monitor and save the best model. Defaults to 'val_mean_absolute_error'. 
        model_name (str, optional): Name of the file where the best model will be saved. Defaults to 'best_model.h5'.
        callbacks (list, optional): List of callbacks to be used during training. Defaults to [].

    Returns:
        final_metrics (list): List with the final metrics of the model
    """    
    checkpoint = ModelCheckpoint(dir + model_name, save_best_only=True, save_weights_only=False, monitor=monitor, mode='auto', verbose=1)
    history = model.fit(x_train, y_train, verbose = 1, epochs=num_epochs, callbacks=callbacks+[checkpoint], validation_data=ds_val, batch_size=batch_size)
    visualize_fit(history)
    return history