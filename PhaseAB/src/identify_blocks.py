""" This module contains the function to identify the blocks in the grid and export the results to a csv file."""

from .objs import TestAnalyzer
from .backend import identify_block
from .objs.utils import generate_csv_filename


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
