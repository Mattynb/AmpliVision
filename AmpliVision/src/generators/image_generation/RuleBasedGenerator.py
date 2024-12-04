from src.objs import Grid
from src.objs import TestAnalyzer
from src.phaseA import phaseA1, phaseA2, phaseA3
from src.phaseB import phaseB, identify_block_in_grid 

import tensorflow as tf
import cv2 as cv
import numpy as np
import re
import time
import random
import os

from networkx import DiGraph
from math import ceil


class RuleBasedGenerator:
    def __init__(self, graphs: DiGraph, results: dict[dict[dict[list[int]]]], **kwargs):
        """"""
        self.components_path = f"{os.getcwd()}/AmpliVision/data/test_components"
        self.save_path = f"{os.getcwd()}/AmpliVision/data/generated_images"

        # if kwargs has jupyter bool set to True, then do different path. RBG(targets, graphs, results, jupyter=True)
        if 'jupyter' in kwargs:
            self.components_path = f"{os.getcwd()}/data/test_components" 
            self.save_path = f"{os.getcwd()}/data/generated_images"


        self.results = self.validate_results(results)
        self.graphs = self.validate_graphs(graphs)


    def validate_graphs(self, graphs):
        "TODO: graphs should all be the same. Warn user if different"
        return graphs

    def validate_results(self, results:list[str]):
        return results

    def setup(self, starting_indexes: list[tuple[int]] = None):
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
        ] if starting_indexes is None else starting_indexes

        # clear the folders
        #self.clear_folder(f"{self.save_path}/blank")
        #self.clear_folder(f"{self.save_path}/final")

        unique_labels = self.results.keys()
        print(f"unique_labels = {unique_labels}")
        self.label_mapping = {
            label: idx 
            for idx, label 
            in enumerate(sorted(unique_labels))
        }

    def generate_for_od(
            self, 
            targets: list[str], 
            noise: int = 0.05, 
            black_background: bool = False,
            rgb: bool = True,
            save: bool = False,
            contamination: float = 0.05
            ):
        
    
        # to play nice with tensorflow
        try:
            targets = [target.decode('utf-8') for target in targets]
        except AttributeError:
            pass

        
        #targets = self.results.keys() if targets is None else targets
        
        # add outlier samples
        targets_ = []
        targets_.extend(targets)
        #targets_.extend(["OUTLIER" for _ in targets]) 
        targets = targets_

        blank_images = self.generate_blank(save=False)
        # generates one image per target where blocks start in different indexes
        i = 0 # index of the target
        j = 0 # total number of images generated
        outlier_interval = int(1/contamination) +1
        while True:
            t= time.time()

            if j % outlier_interval == 0:
                target = "OUTLIER"
                i -= 1
            else:
                target = targets[i]

            rotation = random.randint(0, 3) 
            
            image_content = random.choice(blank_images)
            print(i,' - ', target)
            print("-"*20, "GENENERATING SINGLE" ,"-"*20)
            img = self.generate_single_image(
                image_content, target, rotation, noise, rgb, save, black_background, True
            )
            batch_images = tf.convert_to_tensor(img[0], dtype=tf.float32)
            batch_labels = tf.convert_to_tensor(img[1], dtype=tf.float32)
            yield batch_images, batch_labels

            del batch_images, batch_labels, img, image_content
           
            i = 0 if i == len(targets) - 1 else i + 1
            j += 1
            print("-"*20,"SINGLE DONE in ", f"{round(time.time() - t, 2)} s","-"*20)

        
    def generate(
            self, 
            targets: list[str] = None, 
            noise: int = 0.03, 
            black_background: bool = False,
            rotation: int = None, 
            rgb: bool = True,
            save: bool = False
        ):
        # will need to be broken into functions but the idea is:
        """
        generate a bunch of test images with no spots 
        then pass them through phaseA/B and paint them there
        """

        #print("-"*50, "\nRule Based Generator\n", "-"*50)

        # to play nice with tensorflow
        try:
            targets = [target.decode('utf-8') for target in targets]
        except AttributeError:
            pass

        # overriding label_mapping 
        self.label_mapping = {
            label: idx 
            for idx, label 
            in enumerate(sorted(targets))
        }

        #targets = self.results.keys() if targets is None else targets
      
        # generates one image per target where blocks start in different indexes
        i = 0
        blank_images = self.generate_blank()
        while True:
            target = targets[i] #random.choice(targets)
            
            rotation = random.randint(0, 3)
            image_content = random.choice(blank_images) # random index choice. Room for performance optimization here
            
            img = self.generate_single_image(
                image_content, target, rotation, noise, rgb, save, black_background
            )
            batch_images = tf.convert_to_tensor(img[0], dtype=tf.float32)
            batch_labels = tf.convert_to_tensor(img[1], dtype=tf.float32)
            yield batch_images, batch_labels
           
            i = 0 if i == len(targets)-1 else i + 1


    def generate_single_image(
            self, 
            image_content, 
            target, 
            rotation, 
            noise, 
            rgb, 
            save,
            black_background,
            for_outlier = False,
        ):
        # PhaseA2 expects a dictionary of images
        image = dict()
        image[target] = image_content.copy()

        # get their virtual grids
        Grid = phaseA2(image, display=False)

        # paint the spots in the images
        # each image has its own gridsave
        Grid = self.paint_spots(Grid, self.results, black_background)
        
        grid = list(Grid.values())[0]
        img = self.add_noise(grid.img, percent=noise)

        img = self.rotate_image(img, rotation)

        # bgr to rgb
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB) if rgb else img

        # save the painted image in all possible orientations
        self.save_augmented_images(Grid, 1, img=img) if save else None

        # one-hot encoding
        if for_outlier:
            label = 1 if target == 'OUTLIER' else 0
            one_hot_label = tf.keras.utils.to_categorical(label, num_classes=2)       
        else:
            label = self.label_mapping[target]
            one_hot_label = tf.keras.utils.to_categorical(label, num_classes=len(self.label_mapping))       

        return img, one_hot_label

    def rotate_image(self, image, r):
        if r == 0:
            return image
        
        rotations = ['', cv.ROTATE_90_CLOCKWISE, cv.ROTATE_180, cv.ROTATE_90_COUNTERCLOCKWISE]
        return cv.rotate(image, rotations[r])



    def save_augmented_images(self, Grids, num, img=None):
        for n in range(num):
            for image_name, grid in Grids.items():
                img = grid.img if img is None else img
                noisy_img = self.add_noise(img, percent=0.05)
                cv.imwrite(f"{self.save_path}/final/{image_name}_{random.randint(0,10000)}.png", noisy_img)

    def paint_spots(self, Grids, results, black_background = False):
        for image_name, grid in Grids.items():
            target_name = str(image_name).split("_")[0]
            
            if black_background:
                grid.img =  np.zeros((1242,1242,3), dtype='uint8')

            # where each grid has multiple blocks
            for block in grid.get_blocks():

                # only painting test and control blocks for now
                block, _ = identify_block_in_grid(block, [])
               
                if block.block_type[:4] in ('test','cont'):
                    match target_name:
                        case "OUTLIER":
                            block_results =  {
                                'r': [random.randint(0,254), 0, random.randint(0,254), 0], 
                                'g': [random.randint(0,254), 0, random.randint(0,254), 0], 
                                'b': [random.randint(0,254), 0, random.randint(0,254), 0]
                            }
                        case _:
                            try:
                                block_results = results[target_name][block.block_type] 
                            except KeyError:
                                raise KeyError(f"""
                                target_name = {target_name}, block_type = {block.block_type} not found in results.
                                Make sure the results csv files are in a folder named with todays date (e.g. DD-MM-YYYY)
                                """)
  
                    # paint based on TestAnalyzer results (probed rgb values averaged across csvs)
                    ta = TestAnalyzer(block)

                    block = ta.paint_spots(block_results)

                    # "paste" the painted block back into the image
                    grid.paste_test_area(block)

                    

        return Grids


    def generate_blank(self, save:bool = False, target: str = None,) -> np.ndarray:   
        " Generates blank images with no spots painted on them"   
        grid_img = self.load_image('grid')  

        output = []
        for _index in self.starting_indexes:
            
            Grid_DS = Grid(grid_img)

            # fill blank grid with blank blocks
            self._place_unpainted_blocks(_index, Grid_DS)

            # save the blank image
            if save:
                print(f"Saving blank image {target}_{_index} to {self.save_path}/blank")
                print(Grid_DS.img.shape)
                cv.imwrite(f"{self.save_path}/blank/{target}_{_index}.png", Grid_DS.img) 

            output.append(Grid_DS.img.copy())

        return output
    
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
        " Add grayscale random noise to the generated image "
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
        """ Load a no bkg image from the components folder """

        # if component_name has 2 numbers, remove the second one
        if not component_name.startswith('test'):
            component_name = re.sub(r'\d+', '', component_name)

        # convex, concave, or empty
        geometry = '_cnvx' if geometry == 1 else '_cncv' if geometry == 0 else ''
        component_name = f"{component_name + geometry}".replace('__', '_')
        path = f"{self.components_path}/{component_name}.png"

        return cv.imread(path, cv.IMREAD_UNCHANGED)

    def clear_folder(self, path):
        " Clear the folder at the given path "
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
