import os
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

from src.phaseA import phaseA1
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def test_model_(path_to_imgs, scanned_path, TARGETS, MODEL = "DENV"):
    """ Test model using scanned images (not generated)"""

    test_datagen = ImageDataGenerator(rescale=1/255.)

    test_generator = test_datagen.flow_from_directory(
                                f"{os.getcwd()}/AmpliVision/data/scanned_{MODEL.removesuffix('.h5')}",
                                classes = TARGETS,
                                class_mode = "categorical",
                                shuffle = False,
                                target_size = (1024, 1024),
                                )

    model_path = f"{os.getcwd()}/AmpliVision/data/ML_models/61{MODEL}"
    test_model = tf.keras.models.load_model(model_path)
   
    # Predict using the model
    predictions = test_model.predict(test_generator)  
    print(predictions[0])
    print(np.argmax(predictions[0]))
    
    y_pred = np.argmax(predictions, axis=1)
    print(y_pred)

    y_true = test_generator.classes
    print(y_true)
    
    # Plot the evaluation results   
    plot_evaluation_results(y_pred, y_true, TARGETS)    

def test_model_generated(dataset, clf, TARGETS, TAG): 
    true_labels = []
    predictions = []
    losses = []

    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    N = 21
    for i, (x_batch, y_batch) in enumerate(dataset.take(N)):   
        # Make predictions using the trained classifier
        y_pred = clf.predict(x_batch)
        print(x_batch.shape)

        # Store true labels and predicted labels
        true_labels.extend(tf.argmax(y_batch, axis=1).numpy())
        predictions.extend(tf.argmax(y_pred, axis=1))
        losses.append(loss_fn(y_batch, y_pred).numpy())

        print(losses)

        if i >= N:
            break

        # Step 3: Generate confusion matrix
        plot_confusion_matrix(predictions, true_labels, TARGETS, TAG)

        # Step 4: Generate F1 score
        plot_f1_score(predictions, true_labels, TARGETS, TAG)
        
        plt.close('all')

    


" -------------------------------------------------------- "

def plot_evaluation_results(predictions: list, y_true: list, TARGETS):
    # Plot the results

    # Plot the F1 score
    plot_f1_score(predictions, y_true, TARGETS)

    # Plot the confusion matrix
    plot_confusion_matrix(predictions, y_true, TARGETS)

def plot_f1_score(predictions: list, y_true: list, TARGETS, TAG):
    # Calculate the F1 score
    f1_score = metrics.f1_score(y_true, predictions, average=None)

    # Plot the F1 score
    plt.clf()
    disp = sns.barplot(x=TARGETS, y=f1_score)
    disp.set_title("F1 Score")
    disp.set_xlabel("Class")
    disp.set_ylabel("F1 Score")

    path = f"{os.getcwd()}/{TAG}_F1_Score.png"
    print(f"saved F1 score at {path}.\n Score: {f1_score}")
    plt.savefig(path)

def plot_confusion_matrix(predictions: list, y_true: list, TARGETS, TAG):
    # Calculate the confusion matrix
    confusion_matrix = metrics.confusion_matrix(y_true, predictions)
 
    # Plot the confusion matrix
    plt.clf()
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels=TARGETS)
    disp.plot()
 
    path = f"{os.getcwd()}/{TAG}_confusion_matrix.png"
    print(f"saved confusion matrix at {path}")
    plt.savefig(path)


" -------------------------------------------------------- "

# UTILS
def preprocess_image(image):
    # Resize the image to match the input shape of the model
    image = tf.image.resize(image, (1024, 1024))
    # Normalize the image
    image = image / 255.0

    # add a batch dimension
    image = tf.expand_dims(image, axis=0)

    return image

def resize_image(image):
    
    # If the image is a NumPy array, convert to tensor
    if isinstance(image, np.ndarray):
        image = tf.convert_to_tensor(image)
    # Resize image to the target size
    image = tf.image.resize(image, [1024,1024])
    return image
