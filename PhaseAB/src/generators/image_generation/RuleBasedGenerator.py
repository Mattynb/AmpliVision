from networkx import DiGraph
from src.objs import Grid
from src.objs import TestAnalyzer
from src.phaseA import phaseA1, phaseA2, phaseA3
from src.phaseB import phaseB

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

    def generate(self):
        # will need to be broken into functions but the idea is:
        """

        generate a bunch of test images with no spots 

        then pass them through phaseA/B and paint them there
        """

        di_graph = self.graphs[0]
        results = self.results

        # meant to avoid duplicate images after augmentation (rotation, flippin, etc)
        starting_indexes = [
            (0, 0), (1, 0),
            (0, 1), (1, 1),
            (0, 2), (1, 2),
            (0, 3), (1, 3),
            (0, 4), (1, 4),
        ]

        for block_index in starting_indexes:

            print(f"Generating blank spots image index {block_index}...")
            for i, target in enumerate(results.keys()):

                grid_img = self.load_image('grid')
                Grid_DS = Grid(grid_img)
                img = grid_img
                geometry = True
                for block_name in di_graph.nodes:

                    # get the transparent image of the block
                    block_img = self.load_image(block_name, geometry)

                    # add unpainted block to the image
                    img = Grid_DS.add_artificial_block(
                        block_index, img, block_img)

                    geometry = not geometry
                    block_index = (block_index[0] + 1, block_index[1])

                # save the image
                cv.imwrite(f"{self.save_path}/{target}_{block_index}.png", img)

            # pass the image through phaseA/B and paint the spots
            ...
        return

    def load_image(self, component_name, geometry=None):

        # if component_name has 2 numbers, remove the second one
        if not component_name.startswith('test'):
            component_name = re.sub(r'\d+', '', component_name)

        # convex, concave, or empty
        geometry = '_cnvx' if geometry == 1 else '_cncv' if geometry == 0 else ''
        component_name = f"{component_name + geometry}".replace('__', '_')
        path = f"{self.components_path}/{component_name}.png"

        image = cv.imread(path, cv.IMREAD_UNCHANGED)
        return image


if __name__ == '__main__':
    pass
