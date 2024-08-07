""" """
from src.phaseA import phaseA1, phaseA2, phaseA3
from src.phaseB import phaseB
from src.generators.image_generation.RuleBasedGenerator import RuleBasedGenerator


def main(path_to_imgs: str, scanned_path: str) -> None:
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
        display=False,
        is_pre_scanned=True, do_white_balance=False
    )

    # Phase A.2 - Grids
    Grids = phaseA2(Images, display=False)

    # save test results
    results = phaseB(Grids)

    # Phase A.3 - Position Graph
    graphs = phaseA3(Grids, display=True)

    # --- Image Generation --- #
    RBG = RuleBasedGenerator(graphs, results)
    # RBG.generate(1000)


if __name__ == '__main__':
    # r"C:\Users\Matheus\Desktop\NanoTechnologies_Lab\Phase A\data\New_images_06262024\*"
    path_to_imgs = "PhaseAB/data/scanned/*"
    scanned_path = "PhaseAB/data/scanned/"
    main(path_to_imgs, scanned_path)
