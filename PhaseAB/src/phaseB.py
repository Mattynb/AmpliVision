from objs.utils import write_to_csv
from identify_blocks import identify_blocks_in_grid


def phaseB(Grids, image_name):
    """ Identifies the blocks in the grid and exports the results to a csv file. """
    for image_name, Grid_DS in Grids.items():
        csv_filename, csv_rows = identify_blocks_in_grid(Grid_DS, image_name)

        # exports the results to a csv file.
        write_to_csv(csv_filename, csv_rows)
