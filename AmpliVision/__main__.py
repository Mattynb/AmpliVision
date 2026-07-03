""" Entry point for multiple AmpliVision workflows """

import os
import sys
import matplotlib.pyplot as plt
import tensorflow as tf

from src.ML import models, ML_Utils
from src.pyod_workflow import run_pyod_workflow
from src.config import CONFIG


def main():
    print(f" --- Running {use_case} --- ")

    match use_case.upper():

        # Scans AMPLI test images and extracts results
        case 'SCAN':
            ML_Utils()
        
        # Trains LENET model using generated images
        case 'FULL':
            """ Full workflow: Train, Test, and PyOD """
            model_class = getattr(models, CONFIG.model_name, None)
            if model_class is None:
                print(f"ERROR: Model {model_name} not found in models.py, exiting...")
                exit(1)
    
            trained = model_class().run()
            
            from src.ML.models import KerasModelBase
            run_pyod_workflow(
                trained, 
                keras_preprocess=isinstance(model_class(), KerasModelBase)
            )

        case 'TRAIN':
            """ Train a CNN model using the model_name architecture to predict Ampli test diagnostics """

            model_class = getattr(models, CONFIG.model_name, None)
            if model_class is None:
                print(f"ERROR: Model {model_name} not found in models.py, exiting...")
                exit(1)

            model_class().run()

        case 'GAN':
            """ Train a CycleGAN to adapt synthetic images to look like real scanned blocks """            
            # Import here to avoid overhead when running standard classifiers
            from src.ML.gan import Sim2RealWorkflow
            
            CONFIG.TAG = f"CycleGAN_Sim2Real_{CONFIG.TAG}"
            
            gan_workflow = Sim2RealWorkflow()
            gan_workflow.build_model()
            gan_workflow.train_tf_model()
            
            print("\n✅ GAN Training Complete. Generator saved for production.")

        case 'TEST':
            """ Test a trained CNN model in predicting never seen Ampli tests """
            
            model_class = getattr(models, CONFIG.model_name, None)
            if model_class is None:
                print(f"ERROR: Model {model_name} not found in models.py, exiting...")
                exit(1)
                
            model_class().test_model()
        
        case 'PYOD':
            "Outlier Detection"
            run_pyod_workflow()

        case 'GENERATE_DATA':
            "Generate synthetic images using the configured parameters in CONFIG."
            
            # Utils.py is flipping R and B channels
            # The current expectation is that the generator yields bgr images
            # and it is up to load() functions to flip 
        
            # Change these: 
            # CONFIG.SAVE = True
            # CONFIG.TRAIN_DATASET = "GEN"
            # CONFIG.SIZE = [1242, 1242]
            # CONFIG.CROP_TO_TEST_AREA = False
            # CONFIG.NOISE = 0
            
            n_images = len(CONFIG.TARGETS) * 100         #CONFIG.EPOCHS * CONFIG.STEPS_PER_EPOCH * CONFIG.BATCH_N
            print(f"""
            Generating {n_images} synthetic images.
            {n_images / len(CONFIG.TARGETS)} per class approximately.
            Saving to: {CONFIG.path_to_store}
            """)

            RBG_GEN = ML_Utils().build_dataset(BATCH_N=1, Keras_Preprocess=False)
   
            for i, (img, label) in enumerate(RBG_GEN.take(n_images)):
                
                # FIX 1: Handle the batch dimension for the label
                true_label = CONFIG.TARGETS[int(tf.argmax(label[0]).numpy())]
                
                filename = f"{true_label}_{i//len(CONFIG.TARGETS)}.png"
                filepath = os.path.join(CONFIG.path_to_store, filename)
                
                # FIX 2: Handle the batch dimension for the image
                plt.imsave(filepath, img[0].numpy())
                
                # Optional print statement so you know it's not frozen
                if i % 10 == 0:
                    print(f"Generated {i}/{n_images} images...")
            
            print(f"\n✅ Data Generation Complete. {i+1} images saved to {CONFIG.path_to_store}\n")

        case 'CHECK_DATA':
            "checking if data is correct by displaying one image of each class."

            model_class = getattr(models, CONFIG.model_name, None)
            if model_class is None:
                print(f"ERROR: Model {CONFIG.model_name} not found in models.py, exiting...")
                exit(1)

            from src.ML.models import KerasModelBase
            import numpy as np # Make sure numpy is imported

            # Helper function to normalize images for saving
            def prep_for_save(img_tensor):
                img_arr = img_tensor.numpy()
                # Min-Max scaling to force the array into [0.0, 1.0]
                img_arr = (img_arr - np.min(img_arr)) / (np.max(img_arr) - np.min(img_arr) + 1e-8)
                return img_arr

            ds = model_class().MLU.build_dataset(
                BATCH_N = 1,
                Keras_Preprocess=isinstance(model_class, KerasModelBase)
            )
            
            print("\n --- Checking Data Generation --- \n")
            i = 0
            for img, label in ds.take(7):
                print(img.shape, label.shape) 
                print("image array head: ", img[0][:5, :5, 0])
                print("label: ", label[5:], "...")
                try: 
                    # Applied normalization here
                    plt.imsave(f"gen_sanity_test_img_{i}.png", prep_for_save(img[0]))
                    i += 1
                except Exception as e:
                    print("ERROR: You may be attempting to plot a graph in a headless process. Error: ", e)

            print("\n --- Checking Data Loading from TRAIN directory --- \n")
            CONFIG.BATCH_N = 1
            train_ds, _ = ML_Utils.load_dataset(
                Keras_Preprocess=isinstance(model_class, KerasModelBase),
                #skip_sentinel = True,
            )
            i=0
            for img, label in train_ds.take(7):
                print(img.shape, label.shape) 
                print("image array head: ", img[0][:5, :5, 0])
                print("label: ", label[5:], "...")
                try: 
                    # Applied normalization here
                    plt.imsave(f"load_sanity_test_img_{i}.png", prep_for_save(img[0]))
                    i += 1
                except Exception as e:
                    print("ERROR: You may be attempting to plot a graph in a headless process. Error: ", e)

            print("\n --- Checking Data Loading from TEST directory --- \n")
            CONFIG.BATCH_N = 1
            test_ds = ML_Utils.load_dataset(
                train_split= 1, 
                Keras_Preprocess = isinstance(model_class, tf.keras.Model), 
                data_path = f"{os.getcwd()}/AmpliVision/data/scanned_MARKER/test/",
                use_case = "Test",
                #skip_sentinel = True,
            )
            i=0
            # Fixed loop to iterate over test_ds instead of train_ds
            for img, label in test_ds.take(7): 
                print(img.shape, label.shape) 
                print("image array head: ", img[0][:5, :5, 0])
                print("label: ", label[5:], "...")
                try: 
                    # Note: You might want to rename the output file to distinguish from the train files
                    plt.imsave(f"test_load_sanity_img_{i}.png", prep_for_save(img[0]))
                    i += 1
                except Exception as e:
                    print("ERROR: You may be attempting to plot a graph in a headless process. Error: ", e)

            
        
        case 'VIEW':
            """ Visualize feature maps of convolutional layers for a given image using a trained model """
            from src.ML.visuals import visualize_feature_maps
            
            model_class = getattr(models, CONFIG.model_name, None)
            if model_class is None:
                print(f"ERROR: Model {model_name} not found in models.py, exiting...")
                exit(1)

            #trained_model = model_class().build_model()

            #H, W = tuple(CONFIG.SIZE)
            #input_shape_with_batch = (None, H, W, 3) 
    
            # Explicitly build the Sequential model so 'model.input' is defined.
            #trained_model.build(input_shape=input_shape_with_batch)

            # load trained model from disk. the 
            from tensorflow.keras.models import load_model

            path = "/home/matheus.berbet001/code/AmpliVision/AmpliVision/data/ML_models/ALEXNET_2025_10_30_09.keras"
            trained_model = load_model(path)

            sample_image_path = f"{CONFIG.path_to_store}/thyroid_9.png"  # Replace
            visualize_feature_maps(trained_model, sample_image_path, tuple(CONFIG.SIZE))

        case 'TUNE':
            "Hyperparameter tuning using Keras Tuner"
            from src.ML.tuning import TUNING

            model = models.LENET()
            train_data, val_data = model.MLU.load_dataset()

            tuner = TUNING(model.build_model, train_data, val_data)
            best_hps = tuner.run_tuning()
            print("Best hyperparameters found: ", best_hps.values)
 
        case 'HISTORY':
            "display training history after training"

            import pickle as pkl

            # PYOD results
            path = "/home/matheus.berbet001/code/AmpliVision/pyod_data_110.pkl"
            with open(path, "rb") as f:
                pyod_results = pkl.load(f)
                print(pyod_results)
        

            # Training history
            path = f"{os.getcwd()}/AmpliVision/data/ML_perform/histories/history_{CONFIG.TAG}.pkl"
            with open(path, "rb") as f:
                history = pkl.load(f)
                print( *[f"{k}: {v}" for k, v in history.items()], sep="\n\n")


if __name__ == '__main__':

    if len(sys.argv) < 3:
        print("ERROR: Not enough arguments, exiting...")
        print("Usage: python -m AmpliVision <use_case> <dataset> <TAG> <model_name>")
        print("Example: python -m AmpliVision TRAIN MARKER test_run LENET")
        exit(1)

    # ----------- USER PARAMETERS (via command line) ------------ #
    use_case = str(sys.argv[1])
    dataset = str(sys.argv[2])
    TAG = sys.argv[3]
    TAG = TAG if TAG else dataset
    model_name = sys.argv[4] if len(sys.argv) > 4 else CONFIG.model_name

    path_to_imgs = f"{os.getcwd()}/AmpliVision/data/{dataset}/*" #scanned/* #scanned_DENV/*"
    scanned_path = f"{os.getcwd()}/AmpliVision/data/{dataset}/"
    
    # assign correct targets to specific datasets
    # targets are the classes that the model will predict
    # the tag is the name you want to give to the trained model
    _TAG = "MARKER"
    TAG = TAG if TAG else _TAG
    
    import datetime
    model_save_name = f"{TAG}_{datetime.datetime.now().strftime('%Y_%m')}"
    dataset_size = len([name for name in os.listdir(CONFIG.path_to_store) if os.path.isfile(os.path.join(CONFIG.path_to_store, name))])
    
    
    # ----------- INITIALIZE CONFIG OBJECT ---------- #
    CONFIG.initialize(**{
        "use_case": use_case,           # Use case to run
        "dataset": dataset,             # Code name for target labels to be predicted  
        "TAG": TAG,                     # Name to be given to the trained model
        "model_name": model_name,       # Model architecture to be used (LENET, ALEXNET, EFFICIENTNETB0, etc)
        "path_to_imgs": path_to_imgs,   # Path to images to be loaded (Pre-Phase1 scanned images)
        "scanned_path": scanned_path,   # Path to scanned images (Phase1 scanned images)
        "SAVE_NAME": model_save_name,   # Name to save the trained model as
        "MODEL_PARAMS": {
            "optimizer": tf.keras.optimizers.Adam(learning_rate=0.0001),
            "learning_rate": 0.0001,
            "loss": tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            "metrics": [
                "accuracy", 
                "AUC", 
                tf.keras.metrics.F1Score(average='macro')
            ]
        },
        "STEPS_PER_EPOCH": int(dataset_size / CONFIG.BATCH_N)
        
    })
    CONFIG.display()

    main()
