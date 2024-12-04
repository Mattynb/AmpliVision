from .objs.utils import generate_csv_filename
from .backend import identify_block
from .objs import TestAnalyzer
from .objs.utils import write_to_csv
from .generators import DataExtractor


def phaseB(_Grids, display:bool = False, jupyter:bool = False) -> dict:
    """ Identifies the blocks in the grid and exports the results to a csv file. 

    Returns:
    --------
    results: dict
        A dictionary containing the fingerprints of each target.

    results = {
        target_name: {
            block_name: {
                "r" : {
                    [spot1_mean, spot1_std, spot2_mean, spot2_std],
                }
                "g" : [...]
                "b" : [...]
                }
            }
        }
    }
    """

    # identify the blocks in the grid and export the results to a csv file.
    fingerprints = {}
    for image_name, Grid_DS in _Grids.items():
        csv_filename, csv_rows = identify_blocks_in_grid(Grid_DS, image_name)

        # exports the results to a csv file.
        save_path = write_to_csv(csv_filename, csv_rows, jupyter)

        # Extract data from the results folder, getting the fingerprints
        target_name = image_name[:image_name.find("_")]

        folder_path = save_path[:save_path.rfind("/")]+"/"
        print(folder_path)

        # get fingerprints of each target
        RGB_extractor = DataExtractor(target_name, folder_path)

        fingerprints[target_name] = RGB_extractor.extract(display=display)

    return fingerprints


def identify_blocks_in_grid(Grid_DS, image_name):
    """ Identifies all blocks """

    csv_rows = []
    csv_filename = generate_csv_filename(image_name)

    print(f"len(Grid_DS.get_blocks()): {len(Grid_DS.get_blocks())}")
    for block in Grid_DS.get_blocks():
        block, csv_rows = identify_block_in_grid(block, csv_rows)

    return csv_filename, csv_rows


def identify_block_in_grid(block, csv_rows):
    """ Identifies single block """
    # identify block
    block.set_rgb_sequence()
    block = identify_block(block)

    # if block is a test block, save results to csv
    b_types = ("test", "control", "Unknown")
    b_type = block.get_block_type()
    if b_type.startswith(b_types):
        ta = TestAnalyzer(block)
        results = ta.analyze_test_result(display=False, double_thresh=True)
        csv_rows.append(results)

    return block, csv_rows
