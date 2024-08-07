from .objs.utils import generate_csv_filename
from .backend import identify_block
from .objs import TestAnalyzer
from .objs.utils import write_to_csv
from .generators.rgb_data_generator import DataExtractor


def phaseB(_Grids) -> list:
    """ Identifies the blocks in the grid and exports the results to a csv file. """

    # identify the blocks in the grid and export the results to a csv file.
    fingerprints = {}
    for image_name, Grid_DS in _Grids.items():
        csv_filename, csv_rows = identify_blocks_in_grid(Grid_DS, image_name)

        # exports the results to a csv file.
        save_path = write_to_csv(csv_filename, csv_rows)

        # Extract data from the results folder, getting the fingerprints
        target_name = image_name[:image_name.find("_")]

        # "TODO: CHANGE THIS TERIBLE LOGIC. Will need to tweak file structure"
        prefix = "PhaseAB/"
        folder_path = prefix + save_path[:save_path.rfind("/")]+"/"

        print(f"folder {folder_path} target {target_name}")
        # get fingerprints of each target
        RGB_extractor = DataExtractor(target_name, folder_path)
        fingerprints[target_name] = RGB_extractor.extract(display=0)

    print(fingerprints)

    return fingerprints


def identify_blocks_in_grid(Grid_DS, image_name):
    """ Identifies all blocks """

    csv_rows = []
    csv_filename = generate_csv_filename(image_name)

    for block in Grid_DS.get_blocks():
        block = identify_block_in_grid(block, csv_rows)

    return csv_filename, csv_rows


def identify_block_in_grid(block, csv_rows):
    """ Identifies single block """
    # identify block
    block.set_rgb_sequence()
    block = identify_block(block)

    # if block is a test block, save results to csv
    b_types = ("Test", "Control")
    b_type = block.get_block_type()
    if b_type.startswith(b_types):
        ta = TestAnalyzer(block)
        csv_rows.append(ta.analyze_test_result(display=False))

    return block
