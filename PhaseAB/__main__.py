""" """
from src.phaseA import phaseA1, phaseA2
from src.phaseB import phaseB
from src.image_generation.RuleBasedGenerator import RuleBasedGenerator


def main(path_to_imgs: str) -> None:
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
    Images = phaseA1(path_to_imgs)

    # Phase A.2 - Grids + Position Graph
    Grids, graphs = phaseA2(Images)

    # save test results
    results = phaseB(Grids)

    # --- Image Generation --- #
    RBG = RuleBasedGenerator(graphs, results)
    RBG.generate(1000)


if __name__ == '__main__':
    path_to_imgs = r"C:\Users\Matheus\Desktop\NanoTechnologies_Lab\Phase A\data\New_images_06262024\*"
    main(path_to_imgs)
