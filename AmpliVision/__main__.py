""" Entry point for multiple AmpliVision workflows """

import os
import sys
import matplotlib.pyplot as plt

from src.ML import models, ML_Utils
from src.pyod_workflow import run_pyod_workflow
from src.config import CONFIG


def main() -> None:
    print(f" --- Running {use_case} --- ")

    match use_case.upper():

        # Scans AMPLI test images and extracts results
        case 'SCAN':
            ML_Utils(CONFIG.TAG).prepare_image_RBGen()
        
        # Trains LENET model using generated images
        case 'TRAIN':
            """ Train a CNN model using the model_name architecture to predict Ampli test diagnostics """

            model_class = getattr(models, CONFIG.model_name, None)
            if model_class is None:
                print(f"ERROR: Model {model_name} not found in models.py, exiting...")
                exit(1)

            model_class().run()

        case 'TEST':
            """ Test a trained CNN model in predicting never seen Ampli tests """
            
            models.LENET().test_model(TAG)

        case 'CHECK_DATA':
            "* Pottentially deprecated. Please test functionality before running *"
            "checking if data is correct by displaying one image of each class. Should be run in jupyter notebook"

            ds = models.LENET().MLU.build_dataset()

            for img, label in ds.take(1):
                print(img.shape, label.shape) 
                print("label: ", label)
                try: 
                    plt.imshow(img[0])
                    plt.show()
                except Exception as e:
                    print("ERROR: You may be attempting to plot a graph in a headless process. Error: ", e)
        
        case 'HISTORY':
            "display training history after training"

            import pickle as pkl

            path = f"{os.getcwd()}/AmpliVision/data/ML_perform/histories/history_{CONFIG.TAG}.pkl"
            with open(path, "rb") as f:
                history = pkl.load(f)
                print( *[f"{k}: {v}" for k, v in history.items()], sep="\n\n")
        

        case 'PYOD':
            "Outlier Detection"
            kwargs = {
                'TARGETS' : TARGETS,
                'path_to_imgs' : path_to_imgs,
                'scanned_path' : scanned_path,
                'SIZE' : CONFIG.SIZE,
                'BATCH_N' : CONFIG.BATCH_N,
                'EPOCHS' : CONFIG.EPOCHS,
                'BLACK' : CONFIG.BLACK
            }
            run_pyod_workflow(kwargs)

def manage_targets():
    """ assigns targets user wants the CNN to predict in specific datasets """
    
    CONFIG.dataset = dataset.upper()

    if "MARKER" in dataset:    
        TAG = 'MARKER'
        TARGETS = ['lung', 'thyroid', 'ovarian', 'prostate', 'skin', 'control', 'breast']
    
    elif "YOUR_TARGET" in dataset:
        # Here is an example to show where you can assign your own targets to your dataset     
        print(" YOUR_TARGET not implemented yet in manage_targets() [__main__.py file], exiting...")
        exit()

    elif dataset == "_":
        # bypassing for certain usecases that dont need targets
        TARGETS = []
        TAG = dataset

    else:
        print("ERROR: Unsupported Workflow Run, use \"MARKER\", \"_\" as sys.argv[2] or implement the new target, exiting...")
        exit(1) 

    print(f"------------- {TAG} Workflow Run --------------")
    print(f"{TAG} TARGETS: {TARGETS}")

    return TARGETS, TAG


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
    model_name = sys.argv[4]

    path_to_imgs = f"{os.getcwd()}/AmpliVision/data/{dataset}/*" #scanned/* #scanned_DENV/*"
    scanned_path = f"{os.getcwd()}/AmpliVision/data/{dataset}/"
    
    # assign correct targets to specific datasets
    # targets are the classes that the model will predict
    # the tag is the name you want to give to the trained model
    TARGETS, _TAG = manage_targets()
    TAG = TAG if TAG else _TAG


    # ----------- INITIALIZE CONFIG OBJECT ---------- #
    CONFIG.initialize(**{
        "use_case": use_case,           # Use case to run
        "dataset": dataset,             # Code name for target labels to be predicted  
        "TAG": TAG,                     # Name to be given to the trained model
        "model_name": model_name,       # Model architecture to be used (LENET, ALEXNET, EFFICIENTNETB0, etc)
        "path_to_imgs": path_to_imgs,   # Path to images to be loaded (Pre-Phase1 scanned images)
        "scanned_path": scanned_path,   # Path to scanned images (Phase1 scanned images)
        "TARGETS": TARGETS              # List of target labels to be predicted
    })
    CONFIG.display()

    main()
