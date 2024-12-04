""" Entry point for multiple AmpliVision workflows """

import os
import sys
import matplotlib.pyplot as plt

from src.ML import models, ML_Utils
from src.pyod_workflow import run_pyod_workflow


def main(
        use_case: str,          # Use case to run
        path_to_imgs: str,      # Path to images to be loaded (Pre-Phase1 scanned images)
        scanned_path: str,      # Path to scanned images (Phase1 scanned images)
        dataset:str = None,     # Code name for target labels to be predicted  
        TAG:str = None,         # Name to be given to the trained model
        display: bool = False, 
        **kwargs
    ) -> None:


    # -------- DEFAULT VALUES -------- #
    SIZE = kwargs.get("SIZE", [1024, 1024])
    EPOCHS = kwargs.get("EPOCHS", 64)
    BATCH_N = kwargs.get("BATCH_N", 64)
    # training steps are 7 training, 1 validation 

    # determines if generated images will show only the painted tests area 
    # (making everything else black) or not
    BLACK = kwargs.get("BLACK", False) 
    
    # assign correct targets to specific datasets
    # targets are the classes that the model will predict
    # the tag is the name you want to give to the trained model
    TARGETS, _TAG = manage_targets(dataset)
    TAG = TAG if TAG else _TAG

    print(f" --- Running {use_case} --- ")

    match use_case.upper():

        # Scans AMPLI test images and extracts results
        case 'SCAN':
            ML_Utils(
                path_to_imgs,
                scanned_path,
                TAG
            ).prepare_image_RBGen()
        
        # Trains LENET model using generated images
        case 'LENET':
            """ Train a CNN model using the LENET architecture to predict Ampli test diagnostics """

            kwargs = {  
                "tag": TAG,
            }

            models.LENET(
                TARGETS,
                path_to_imgs,
                scanned_path,
                SIZE,
                BATCH_N,
                EPOCHS,
                BLACK,
                **kwargs
            ).run()

        case 'TEST':
            """ Test a trained CNN model in predicting never seen Ampli tests """
            
            kwargs = {  
                "tag": TAG,
            }
            models.LENET(
                TARGETS,
                path_to_imgs,
                scanned_path,
                SIZE,
                BATCH_N,
                EPOCHS,
                BLACK,
                **kwargs
            ).test_model(TAG)

        case 'CHECK_DATA':
            "checking if data is correct by displaying one image of each class. Should be run in jupyter notebook"

            BATCH_N = len(TARGETS)
            kwargs = {  
                "tag": TAG,
            }

            ds = models.LENET(
                TARGETS,
                path_to_imgs,
                scanned_path,
                SIZE,
                BATCH_N,
                EPOCHS,
                BLACK,
                **kwargs
            ).MLU.build_dataset(TARGETS, BATCH_N, SIZE, BLACK)

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

            path = f"{os.getcwd()}/AmpliVision/data/ML_perform/histories/history_{TAG}.pkl"
            with open(path, "rb") as f:
                history = pkl.load(f)
                print( *[f"{k}: {v}" for k, v in history.items()], sep="\n\n")
        

        case 'PYOD':
            "Outlier Detection"
            kwargs = {
                'TARGETS' : TARGETS,
                'path_to_imgs' : path_to_imgs,
                'scanned_path' : scanned_path,
                'SIZE' : SIZE,
                'BATCH_N' : BATCH_N,
                'EPOCHS' : EPOCHS,
                'BLACK' : BLACK
            }
            run_pyod_workflow(kwargs)

def manage_targets(dataset):
    """ assigns targets user wants the CNN to predict in specific datasets """
    
    dataset = dataset.upper()

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
    use_case = str(sys.argv[1])
    dataset = str(sys.argv[2])
    TAG = sys.argv[3]
    TAG = TAG if TAG else dataset
    path_to_imgs = f"{os.getcwd()}/AmpliVision/data/{dataset}/*" #scanned/* #scanned_DENV/*"
    scanned_path = f"{os.getcwd()}/AmpliVision/data/{dataset}/"
    
    print("path_to_imgs: ", path_to_imgs) 
    main(use_case, path_to_imgs, scanned_path, dataset, TAG, display=False)
