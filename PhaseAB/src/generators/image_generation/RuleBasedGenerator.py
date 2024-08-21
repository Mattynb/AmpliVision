from networkx import DiGraph
from src.objs import Grid
from src.objs import TestAnalyzer
from src.phaseA import phaseA1, phaseA2, phaseA3
from src.phaseB import phaseB, identify_block_in_grid

import cv2 as cv
import numpy as np
import re


class RuleBasedGenerator:
    def __init__(self, graphs: DiGraph, results: dict[dict[dict[list[int]]]], config=None):
        """"""
        self.components_path = "data/test_components"
        self.save_path = "data/generated_images"

        self.results = self.validate_results(results)
        self.graphs = self.validate_graphs(graphs)

    def validate_graphs(self, graphs):
        "graphs should all be the same. Warn user if different"
        return graphs

    def validate_results(self, results):
        "results should be all different. Warn user if same"
        return results

    def generate(self, n):
        # will need to be broken into functions but the idea is:
        """
        generate a bunch of test images with no spots 
        then pass them through phaseA/B and paint them there
        """

        print("-"*50, "\nRule Based Generator\n", "-"*50)
        di_graph = self.graphs[0]
        results = self.results

        # meant to avoid duplicate images after augmentation 
        # (rotation, flippin, etc)  
        MAX_INDEX = 9
        starting_indexes = [
            (x, y)  
            for x in range(MAX_INDEX - len(di_graph.nodes))
            for y in range(MAX_INDEX//2)
        ]
        starting_indexes = [(1, 5)]

        # generates one image per target where blocks start in different indexes
        print("Generating blank images...")
        self.generate_blank(di_graph, results, starting_indexes)

        # get the unpainted scanned images from phaseA
        print("scanning blank images...")
        images = phaseA1(
            f"{self.save_path}/blank/*", 
            f"{self.save_path}/blank/",
            do_white_balance=True
        )

        # and their grids
        print("generating grids...")
        Grids = phaseA2(images)

        # paint the spots in the images
        # each image has its own grid
        print("painting spots...")
        Grids = self.paint_spots(Grids, results)
        
        # save the painted images in all possible orientations
        print("saving images...")
        self.save_augmented_images(Grids, n)

        return
    
    def save_augmented_images(self, Grids, num):
        for n in range(num):
            for image_name, grid in Grids.items():
                img = grid.img
                for i in range(4):
                    # rotate images all 4 ways
                    img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)

                    # save flipped image
                    for j in range(-1, 2):
                            
                        noisy_img = self.add_noise(img, percent=0.05)

                        if j == -1:
                            cv.imwrite(f"{self.save_path}/final/{image_name}_{i}__{n}.png", noisy_img)

                        # flip images. 
                        # 0 = x-axis, 1 = y-axis, -1 = both
                        noisy_img = cv.flip(noisy_img, j)

                        # save flipped image
                        cv.imwrite(f"{self.save_path}/final/{image_name}_{i}_{j}__{n}.png", noisy_img)


    def paint_spots(self, Grids, results):
        for image_name, grid in Grids.items():
            target_name = image_name.split("_")[0]

            # where each grid has multiple blocks
            for block in grid.get_blocks():

                # only painting test and control blocks for now
                block, _ = identify_block_in_grid(block, [])
                if block.block_type[:4] in ('test','cont'):
                    block_results = results[target_name][block.block_type] 
                    
                    # paint based on TestAnalyzer results (probed rgb values averaged across csvs)
                    # print(f"type = {block.block_type}")
                    ta = TestAnalyzer(block)
                    block = ta.paint_spots(block_results)

                    # "paste" the painted block back into the image
                    grid.paste_test_area(block)

        return Grids

    
    def generate_blank(self, di_graph, results, starting_indexes):
        for _index in starting_indexes:
            for i, target in enumerate(results.keys()):
                
                print(f"Generating blank spots image of {target} @ {_index}...")

                grid_img = self.load_image('grid')
                Grid_DS = Grid(grid_img)
                img = grid_img
                geometry = True
                block_index = _index
                for block_name in di_graph.nodes:

                    # get the image of the block with transparent bkg
                    block_img = self.load_image(block_name, geometry)

                    # add unpainted block to the image
                    img = Grid_DS.add_artificial_block(
                        block_index, img, block_img)

                    # alternate between convex and concave blocks
                    geometry = not geometry

                    # move to the next position
                    block_index = (block_index[0] + 1, block_index[1])

                # save the blank image
                cv.imwrite(f"{self.save_path}/blank/{target}_{block_index}.png", img)
    

    def add_noise(self, _image, percent = 0.05):
        image = _image.copy()

        # Get the dimensions of the image
        height, width, channels = image.shape

        # Calculate the number of pixels to be altered (5% of total pixels)
        total_pixels = height * width
        num_noise_pixels = int(percent * total_pixels)

        # Generate random noise
        for _ in range(num_noise_pixels):
            # Pick a random pixel in the image
            y_coord = np.random.randint(0, height)
            x_coord = np.random.randint(0, width)
            
            # Add noise by altering the pixel value
            noise = np.random.randint(0, 256)

            # r =g = b = noise
            noise = [noise, noise, noise]

            image[y_coord, x_coord] = noise

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
        return image


if __name__ == '__main__':
    pass
