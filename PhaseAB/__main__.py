""" """
from src.phaseA import phaseA1, phaseA2, phaseA3
from src.phaseB import phaseB
from src.generators.image_generation.RuleBasedGenerator import RuleBasedGenerator


def main(path_to_imgs: str, scanned_path: str, display: bool = False) -> None:
    """
    Workflow as of 8/2/2024

    og_images (one per class) -> PhaseA.1 (scan) -> PhaseA.2  -> PhaseB 
                                                    |               |                     
                                            (position graph)  (test results)
                                                    |               |
                                                    +-------+-------+
                                                            |
                                                            V
                                                   Rule Based Generator   
    """

    scientist_in_lab_workflow(path_to_imgs, scanned_path, display=display)

   


def scientist_in_lab_workflow(path_to_imgs: str, scanned_path: str, display: bool = False):
    """
    Phase A1 -> Phase A2 -> Phase B -> Rule Based Generator -> Train CNN 
                                                | 
                                                -------------> Phase A2 -> PCA / HCA Baseline 
    
    Scientists Goal:
    
    
    """

    # --- Phase A --- #
    # Phase A.1 - Scanning images
    print("--- Scanning Images ---")
    Images = phaseA1(
        path_to_imgs, scanned_path,
        display=display, do_white_balance=True,
        is_pre_scanned=True
    )
    print("--- Scanning Images Done ---")

    # Phase A.2 - Grids
    print(" --- Grids 1 --- ")
    Grids = phaseA2(Images, display=display)
    print(" --- Grids 1 Done --- ")

    
    # free memory
    del Images

    # save test results
    print(" --- Results 1 --- ")
    results = phaseB(Grids, display=display)
    print(" --- Results 1 Done --- ")
  

    # Phase A.3 - Position Graph
    print(" --- Graph 1 --- ")
    graphs = phaseA3(Grids, display=display)
    print(" --- Graph 1 Done --- ")

    # free memory
    del Grids

    # --- Image Generation --- #
    RBG = RuleBasedGenerator(graphs, results)
    RBG.generate(1)


def user_workflow(path_to_imgs: str, scanned_path: str, display: bool = False):
    """
    Phase A1 -> Trained CNN -> Pyod Outlier Detection -- Known  -------------------------------------------> result
                                        |                                                               |
                                        ---------------- Unknown -> Phase A2 -> Phase B -> PCA / HCA ----
    """

if __name__ == '__main__':
    path_to_imgs = "PhaseAB/data/scanned/*" #DENV_imgs/*"
    scanned_path = "PhaseAB/data/scanned/"
    main(path_to_imgs, scanned_path, display=False)

