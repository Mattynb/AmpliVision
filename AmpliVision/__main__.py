""" Entry point for multiple AmpliVision workflows """

import os
import sys
import tensorflow as tf

from src.ML import models, ML_Utils
from src.pyod_workflow import run_pyod_workflow
from src.config import CONFIG
from src.business import Business

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
            print("generating data")
            import time 
            t= time.time()
            Business.generate_data_parallel()
            print(f"{round(float(time.time()-t),3)} s  --> generate_data_parallel")

        case 'GENERATE_DATA_729':
            print("generating data")
            import time 
            t=time.time()
            Business.generate_all_729_classes()
            print(f"{round(float(time.time()-t),3)} s  --> generate_all_729_classes") 

            # possible_combinations = list(itertools.product(["r","g","b"], repeat=6))
            # targets = [f"class_{"".join(combo)}" for combo in possible_combinations]

    
            
        case 'CHECK_DATA':
            Business.check_data()

        case 'VIEW':
            Business.view(model_name)

        case 'TUNE':
            "Hyperparameter tuning using Keras Tuner"
            from src.ML.tuning import TUNING

            model = models.LENET()
            train_data, val_data = model.MLU.load_dataset()

            tuner = TUNING(model.build_model, train_data, val_data)
            best_hps = tuner.run_tuning()
            print("Best hyperparameters found: ", best_hps.values)
 
        case 'HISTORY':
            Business.history()


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
