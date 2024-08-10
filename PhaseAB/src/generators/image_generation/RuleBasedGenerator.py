from networkx import DiGraph
from src.objs import Grid
from src.objs import Square
from src.objs import TestAnalyzer

import cv2 as cv
import re


class RuleBasedGenerator:
    def __init__(self, graphs: DiGraph, results: dict[dict[dict[list[int]]]], config=None):
        """"""
        self.components_path = "PhaseAB/data/test_components"
        self.save_path = "PhaseAB/data/generated_images"

        self.results = self.validate_results(results)
        self.graphs = self.validate_graphs(graphs)

    def validate_graphs(self, graphs):
        "graphs should all be the same. Warn user if different"
        return graphs

    def validate_results(self, results):
        "results should be all different. Warn user if same"
        return results

    def generate(self, image):
        # will need to be broken into functions but the idea is:
        """
        - using the structure graph
        - For each target disease
            - For each block in the structure
                - select the correct image
                - paint spots according to results
                - ex: block.paint_spot(mean, std)
                - place sample block in a randomish position,
                    place others accordingly. Check for rotation here.
        """

        di_graph = self.graphs[0]
        results = self.results

        for i, target in enumerate(results.keys()):

            grid_img = self.load_image('grid')
            Grid_DS = Grid(grid_img)
            img = grid_img
            geometry = True
            block_index = (0, 0)
            for block_name in di_graph.nodes:

                # get the transparent image of the block
                block_img = self.load_image(block_name, geometry)

                # add unpainted block to the image
                img = Grid_DS.add_artificial_block(block_index, img, block_img)

                # get the square and make it a block
                block = Grid_DS.get_square(block_index)
                block.img = img
                block.is_block = True
                block.block_type = block_name
                Grid_DS.set_square(block_index, block)

                # get the rgb values of the spots in block
                b_types = ("test", "control")
                if block_name.startswith(b_types):
                    block_spot_rgbs = results[target][block_name]

                    # paint the spots in the block
                    TA = TestAnalyzer(block)
                    TA.paint_spots(block_spot_rgbs)

                """

                block.paint_spot(block_img, block['r'])
                """
                # place block in a randomish position
                # place others accordingly. Check for rotation here.

                geometry = not geometry
                block_index = (block_index[0] + 1, block_index[1])

        return image

    def load_image(self, component_name, geometry=None):

        # if component_name has 2 numbers, remove the second one
        if not component_name.startswith('test'):
            component_name = re.sub(r'\d+', '', component_name)

        # convex, concave, or empty
        geometry = '_cnvx' if geometry == 1 else '_cncv' if geometry == 0 else ''
        component_name = f"{component_name + geometry}".replace('__', '_')
        path = f"{self.components_path}/{component_name}.png"

        image = cv.imread(path, cv.IMREAD_UNCHANGED)
        """ cv.imshow('image',image)
        cv.waitKey(0)
        cv.destroyAllWindows()"""
        return image


if __name__ == '__main__':
    pass
