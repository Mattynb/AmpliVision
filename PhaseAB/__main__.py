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

    # --- Phase A --- #
    # Phase A.1 - Scanning images
    Images = phaseA1(
        path_to_imgs, scanned_path,
        display=display, do_white_balance=True,
        is_pre_scanned=False
    )

    # Phase A.2 - Grids
    print(" --- Grids --- ")
    Grids = phaseA2(Images, display=display)

    # save test results
    print(" --- Results --- ")
    results = phaseB(Grids, display=display)

    # Phase A.3 - Position Graph
    print(" --- Graph --- ")
    graphs = phaseA3(Grids, display=display)

    # --- Image Generation --- #
    print(" --- Image Generation --- ")
    #RBG = RuleBasedGenerator(graphs, results)
    #RBG.generate()


if __name__ == '__main__':
    path_to_imgs = "PhaseAB/data/scanned/*" #DENV_imgs/*"
    scanned_path = "PhaseAB/data/scanned/"
    main(path_to_imgs, scanned_path, display=False)
    
    
    from src.backend.add_to_db.connect_to_db import Client
    Client.close()
