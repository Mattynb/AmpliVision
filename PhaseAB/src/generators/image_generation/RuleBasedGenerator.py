from src.objs import Grid
from src.objs import TestAnalyzer
from src.phaseA import phaseA1, phaseA2, phaseA3
from src.phaseB import phaseB, identify_block_in_grid 

import tensorflow as tf
import cv2 as cv
import numpy as np
import re
import time

from networkx import DiGraph
from math import ceil
import random

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

    def validate_results(self, results:list[str]):
        return results

    def setup(self, starting_indexes: list[tuple[int]] = []):

        t = time.time()

        # taking the first graph as they should be the same
        di_graph = self.graphs[0]
        self.results

        # meant to avoid duplicate images after augmentation 
        # (rotation, flippin, etc)      
        MAX_INDEX = 9
        self.starting_indexes = [   # Index where the first block will be placed 
            (x, y)  
            for x in range(MAX_INDEX - len(di_graph.nodes))
            for y in range(ceil(MAX_INDEX/2))
        ] if starting_indexes is [] else starting_indexes
        print(starting_indexes)

        # clear the folders
        self.clear_folder(f"{self.save_path}/blank")
        #self.clear_folder(f"{self.save_path}/final")

        unique_labels = self.results.keys()
        print(f"unique_labels = {unique_labels}")
        self.label_mapping = {
            label: idx 
            for idx, label 
            in enumerate(sorted(unique_labels))
        }
        print(f"label_mapping = {self.label_mapping}")

        print(f"setup completed in {round(time.time()-t, 2)} seconds")

        
    def generate(
            self, 
            batch_size:int = 1, 
            targets: list[str] = None, 
            rotation: int = None, 
            noise: int = 0.05, 
            rgb: bool = False, 
            save: bool = False
        ):
        # will need to be broken into functions but the idea is:
        """
        generate a bunch of test images with no spots 
        then pass them through phaseA/B and paint them there
        """

        print("-"*50, "\nRule Based Generator\n", "-"*50)

        # generates one image per target where blocks start in different indexes
        t= time.time()
        targets = self.results.keys() if targets is None else targets
        rotation = random.randint(0, 3) if rotation is None else rotation
        
        for i in range(batch_size):
            images = self.generate_blank()
    
            #batch_images = []
            #batch_labels = []
            for image_content in images:
                print("here")
                for target in targets:
                    
                    print("-"*20, "GENENERATING SINGLE" ,"-"*20)
                    img = self.generate_single_image(
                        image_content, target, rotation, noise, rgb, save
                    )
                    #print("-"*20,"SINGLE DONE ","-"*20)
                    #batch_images.append(img[0])
                    #batch_labels.append(img[1])
            #yield np.array(batch_images), np.array(batch_labels)


    def generate_single_image(self, image_content, target, rotation, noise, rgb, save):
        # PhaseA2 expects a dictionary of images
        t = time.time()
        image = dict()
        image[target] = image_content.copy()
        #print(f"Time to make dict: {time.time()-t}")

        # get their virtual grids
        t = time.time()
        Grid = phaseA2(image, display=False)
        #print(f"Time to get virtual grids: {time.time()-t}")

        # paint the spots in the images
        # each image has its own gridsave
        t = time.time()
        Grid = self.paint_spots(Grid, self.results)
        print(f"Time to paint spots: {time.time()-t}")

        # save the painted images in all possible orientations
        self.save_augmented_images(Grid, 1) if save else None
        
        grid = list(Grid.values())[0]

        img = self.add_noise(grid.img, percent=noise)
        
        img = self.rotate_image(img, rotation)

        # bgr to rgb
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB) if rgb else img
        
        # one-hot encoding
        label = self.label_mapping[target]
        one_hot_label = tf.keras.utils.to_categorical(label, num_classes=len(self.label_mapping))       
        
        # resizing to proper input layer size
        img = tf.image.resize_with_crop_or_pad(img, 1202, 1202)
        img = tf.expand_dims(img, axis=0)

        return img, one_hot_label

    def rotate_image(self, image, r):
        if r == 0:
            return image
        
        rotations = ['', cv.ROTATE_90_CLOCKWISE, cv.ROTATE_180, cv.ROTATE_90_COUNTERCLOCKWISE]
        return cv.rotate(image, rotations[r])



    def save_augmented_images(self, Grids, num):
        for n in range(num):
            for image_name, grid in Grids.items():
                img = grid.img
                i = 0
                #for i in range(4):
                    # rotate images all 4 ways
                img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)        
                noisy_img = self.add_noise(img, percent=0.05)
                cv.imwrite(f"{self.save_path}/final/{image_name}_{i}__{n}_{random.randint(0,1000)}.png", noisy_img)

    def paint_spots(self, Grids, results):
        for image_name, grid in Grids.items():
            target_name = image_name.split("_")[0]

            # where each grid has multiple blocks
            for block in grid.get_blocks():

                # only painting test and control blocks for now

                #t = time.time()
                block, _ = identify_block_in_grid(block, [])
                #print(f"     idenitified block in {round(time.time() - t, 2)}")
               
                if block.block_type[:4] in ('test','cont'):
                    block_results = results[target_name][block.block_type] 
 
                    # paint based on TestAnalyzer results (probed rgb values averaged across csvs)
                    # print(f"type = {block.block_type}")
                    t = time.time()
                    ta = TestAnalyzer(block)
                    #print(f"        TestAnalyzer in {round(time.time() - t, 2)}")

                    t = time.time()
                    block = ta.paint_spots(block_results)
                    #print(f"        paint_spots in {round(time.time() - t, 2)}")

                    # "paste" the painted block back into the image
                    t = time.time()
                    grid.paste_test_area(block)
                    #print(f"        paste_test_area in {round(time.time() - t, 2)}")

                    

        return Grids


    def generate_blank(self, save:bool = False, target: str = None,) -> np.ndarray:
        
        targets = self.results.keys() if target is None else [target]
        
        grid_img = self.load_image('grid')  

        for _index in self.starting_indexes:
            
            Grid_DS = Grid(grid_img)

            # fill blank grid with blank blocks
            self._place_unpainted_blocks(_index, Grid_DS)

            # save the blank image
            if save:
                cv.imwrite(f"{self.save_path}/blank/{target}_{_index}.png", Grid_DS.img) 

            yield Grid_DS.img.copy()
    
    def _place_unpainted_blocks(self, block_index, Grid_DS):
        """ Places unpainted blocks in empty grid"""
        
        di_graph = self.graphs[0]
        geometry = True

        for block_name in di_graph.nodes:
            # get the image of the block with transparent bkg
            block_img = self.load_image(block_name, geometry)

            # add unpainted block to the image
            img = Grid_DS.add_artificial_block(
                block_index, Grid_DS.img, block_img)
            Grid_DS.img = img

            # alternate between convex and concave blocks
            geometry = not geometry

            # move to the next position
            block_index = (block_index[0] + 1, block_index[1])



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
        
        return cv.imread(path, cv.IMREAD_UNCHANGED)

    def clear_folder(self, path):
        import os
        import shutil

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
        path = os.path.join(project_root, path)

        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

        print( f"Folder {path} cleared.")

if __name__ == '__main__':
    pass
