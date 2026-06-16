import os
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

from src.phaseA import phaseA1
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from .utils import ML_Utils
from src.config import CONFIG

def test_model_(test_model, DATASET = "scanned_MARKER"):
    """ Test model using scanned images (not generated)"""

    test_ds = ML_Utils.load_dataset(
        train_split= 1, 
        Keras_Preprocess = isinstance(test_model, tf.keras.Model), 
        data_path = f"{os.getcwd()}/AmpliVision/data/{DATASET}/test/",
        use_case = "Test"
    )
   
    # 3. Predict (Will safely stop because there is no .repeat())
    print(f"\n--- Testing model with {DATASET} images ---\n")
    predictions = test_model.predict(test_ds)
    y_pred = np.argmax(predictions, axis=1)
    print(np.bincount(y_pred))          # if ~all one number -> confirmed collapse
    print(predictions[:5])
    
    # Extract true labels from the dataset (Safe because dataset is NOT shuffled)
    y_true = np.concatenate([y for x, y in test_ds], axis=0)
    y_true_indices = np.argmax(y_true, axis=1)

    # 4. Evaluation 
    plot_evaluation_results(y_pred, y_true_indices)

    # save dataset named as "{PredictedLabel}_{TrueLabel}_index.png"
    save_test_images_with_predictions(test_ds, y_true_indices, y_pred, "LOAD")


def save_test_images_with_predictions(test_ds, y_true_indices, y_pred, dir):
    """
    Save ALL test images with their true and predicted labels.
    (No hardcoded 10-image-per-batch limit.)
    """
    output_dir = f"/home/matheus.berbet001/code/AmpliVision/z_TEST/{dir}"
    
    # Clear the output directory
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, file))
    else:
        os.makedirs(output_dir)
    
    index = 0
    
    # Iterate over batches
    for x_batch, y_batch in test_ds:
        batch_size = x_batch.shape[0]
        
        for i in range(batch_size):
            # Safety check: stop if we've exhausted the arrays
            if index >= len(y_true_indices) or index >= len(y_pred):
                break
            
            true_label = CONFIG.TARGETS[y_true_indices[index]]
            pred_label = CONFIG.TARGETS[y_pred[index]]
            filename = f"{index}_P:{pred_label}_T:{true_label}.png"
            filepath = os.path.join(output_dir, filename)
            
            # Resize and save the image
            img = tf.image.resize(x_batch[i].numpy(), [1242, 1242])
            img, _ = preprocess_test(img, "", True)
            plt.imsave(filepath, img.numpy())
            
            index += 1
        
        # Early exit if we've exhausted the test set
        if index >= len(y_true_indices):
            break
    
    print(f"Saved {index} test images with predictions to {output_dir}")
    return index
 

def test_model_generated(dataset, clf): 
    true_labels = []
    predictions = []
    losses = []

    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    test_ds = dataset.take(1)
    for i, (x_batch, y_batch) in enumerate(test_ds):   
        # Make predictions using the trained classifier
        y_pred = clf.predict(x_batch)
        print(x_batch.shape) # (Batch , Height, Width, Channels)

        # Store true labels and predicted labels
        true_labels.extend(tf.argmax(y_batch, axis=1).numpy())
        predictions.extend(tf.argmax(y_pred, axis=1))
        losses.append(loss_fn(y_batch, y_pred).numpy())

        for p, t in zip(y_pred, y_batch):
            print("Predicted:", p, " True:", t)


        # save image as "{PredictedLabel}_{TrueLabel}_index.png"
        for j, img in enumerate(x_batch):
            if j == 0:
                print(img.shape) # (Height, Width, Channels)
                print(img[:1, :1, 0])  # Print the head of the image array

            true_label = CONFIG.TARGETS[true_labels[j]]
            pred_label = CONFIG.TARGETS[predictions[j]]
            filename = f"{j}_P:{pred_label}_T:{true_label}.png"
            filepath = f"/home/matheus.berbet001/code/AmpliVision/z_TEST/GEN/{filename}"
            #resize and save the image
            img_clipped = tf.clip_by_value(img, 0.0, 255.0)
            img_uint8 = tf.cast(img_clipped, tf.uint8).numpy()
            plt.imsave(filepath, img_uint8)
        
    
        # Step 3: Generate confusion matrix
        plot_evaluation_results(predictions, true_labels)
          
        plt.close('all')
    


" -------------------------------------------------------- "

def plot_evaluation_results(predictions: list, y_true: list):
    # Plot the results

    # Plot the F1 score
    plot_f1_score(predictions, y_true)

    # Plot the confusion matrix
    plot_confusion_matrix(predictions, y_true)

def plot_f1_score(predictions: list, y_true: list):
    # Calculate the F1 score
    f1_score = metrics.f1_score(y_true, predictions, average=None, labels=range(len(CONFIG.TARGETS)))

    # Plot the F1 score
    plt.clf()
    disp = sns.barplot(x=CONFIG.TARGETS, y=f1_score)
    disp.set_title("F1 Score")
    disp.set_xlabel("Class")
    disp.set_ylabel("F1 Score")

    path = f"{os.getcwd()}/AmpliVision/data/ML_perform/f1_scores/_{CONFIG.TEST_DATASET}_{CONFIG.TAG}.png"
    print(f"saved F1 score at {path}.\n Score: {f1_score}")
    plt.savefig(path)

def plot_confusion_matrix(predictions: list, y_true: list):
    # Calculate the confusion matrix

    #print(y_true, predictions, CONFIG.TARGETS)

    #confusion_matrix = metrics.confusion_matrix(y_true, predictions)
 
    # Plot the confusion matrix
    plt.clf()
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_true, predictions, display_labels=sorted(CONFIG.TARGETS))
    disp.plot()
 
    path = f"{os.getcwd()}/AmpliVision/data/ML_perform/confusion_matrix/_{CONFIG.TEST_DATASET}_{CONFIG.TAG}.png"
    print(f"saved confusion matrix at {path}")
    plt.savefig(path)


def bgr_to_rgb(image):
    """Converts image from BGR (OpenCV/default) to RGB (Keras/TensorFlow)."""
    # The image is a NumPy array (H, W, C). Slicing [..., ::-1] reverses the color channels.
    return image[..., ::-1]

def preprocess_test(image, label, keras=True):
    if not keras:
        return image, label 
    
    # for saving the image
    image = tf.cast(image, tf.float32)
    image /= 255.0 
    return image, label