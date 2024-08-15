from networkx import DiGraph
from src.objs import Grid
from src.objs import TestAnalyzer
from src.phaseA import phaseA1, phaseA2, phaseA3
from src.phaseB import phaseB, identify_block_in_grid

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
        MAX_INDEX = 9
        starting_indexes = [
            (x, y)  
            for x in range(MAX_INDEX - len(di_graph.nodes))
            for y in range(MAX_INDEX//2)
        ]

        # generates one image per target where blocks start in different indexes
        for _index in starting_indexes:
            for i, target in enumerate(results.keys()):
                
                print(f"Generating blank spots image of {target} @ {_index}...")

                grid_img = self.load_image('grid')
                Grid_DS = Grid(grid_img)
                img = grid_img
                geometry = True
                block_index = _index
                for block_name in di_graph.nodes:

                    # get the transparent image of the block
                    block_img = self.load_image(block_name, geometry)

                    # add unpainted block to the image
                    img = Grid_DS.add_artificial_block(
                        block_index, img, block_img)

                    geometry = not geometry
                    block_index = (block_index[0] + 1, block_index[1])

                # save the image
                cv.imwrite(f"{self.save_path}/blank/{target}_{block_index}.png", img)

            
            # pass the image through phaseA/B and paint the spots
            images = phaseA1(
                f"{self.save_path}/blank/*", 
                f"{self.save_path}/blank/",
                do_white_balance=True
            )
            Grids = phaseA2(images)
            for image_name, grid in Grids.items():
                target_ = image_name.split("_")[0]
                for block in grid.get_blocks():
                    block, csv_rows = identify_block_in_grid(block, [])
                    
                    if block.block_type[:4] in ('test','cont'):
                        block_results = results[target_][block.block_type] 
                        
                        print(f"type = {block.block_type}")
                        ta = TestAnalyzer(block)
                        ta.paint_spots(block_results)
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
