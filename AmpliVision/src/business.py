import concurrent.futures
import os
import matplotlib.pyplot as plt
import tensorflow as tf

from src.ML import models, ML_Utils
from src.config import CONFIG
from src.objs.utils.utils_csv import write_to_csv

import itertools
import random
from datetime import datetime

class Business:
    @staticmethod
    def generate_data_parallel():
        "Generate synthetic images using the configured parameters in CONFIG."

        # Utils.py is flipping R and B channels
        # The current expectation is that the generator yields bgr images
        # and it is up to load() functions to flip 

        # Change these: 
        # CONFIG.SAVE = True
        # CONFIG.TRAIN_DATASET = "GEN"
        # CONFIG.SIZE = [1242, 1242]
        # CONFIG.CROP_TO_TEST_AREA = False
        # CONFIG.NOISE = 0

        MAX_THREADS = 16
        n_images = len(CONFIG.TARGETS) * CONFIG.N_PER_CLASS         #CONFIG.EPOCHS * CONFIG.STEPS_PER_EPOCH * CONFIG.BATCH_N
        
        print(f"""
            Generating {n_images} synthetic images.
            {n_images / len(CONFIG.TARGETS)} per class approximately.
            Saving to: {CONFIG.path_to_store}"""
        )

        def generate_image_chunk(thread_id, chunk_size, targets, store_path, utils):
            # 1. Instantiate the dataset pipeline locally inside the thread
            local_gen = utils.build_dataset(BATCH_N=1, Keras_Preprocess=False)

            # 2. Advance the generator to this thread's unique starting point
            # (If your dataset is deterministic, skipping prevents every thread 
            # from generating the exact same first N images)
            local_gen = local_gen.skip(thread_id * chunk_size)

            completed = 0
            # 3. Each thread takes only its assigned chunk size
            for i, (img, label) in enumerate(local_gen.take(chunk_size)):
                
                true_label = targets[int(tf.argmax(label[0]).numpy())]
                
                # FIX: Calculate a global index so filenames don't collide!
                # If thread 1 and thread 2 both use 'i', they will overwrite each other's files.
                global_index = (thread_id * chunk_size) + i // len(targets)
                
                filename = f"{true_label}_{global_index}.png"
                filepath = os.path.join(store_path, filename)
                plt.imsave(filepath, img[0].numpy())
                completed += 1
                
            return f"Thread {thread_id} finished saving {completed} images."

        # Calculate the chunk size
        images_per_thread = n_images // MAX_THREADS

        print(f"Starting {MAX_THREADS} threads, each generating {images_per_thread} images...")
        
        utils = ML_Utils()  
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            futures = []
            

            for t_id in range(MAX_THREADS):
                print(f"submitted thread {t_id}")
                future = executor.submit(
                    generate_image_chunk, 
                    t_id, 
                    images_per_thread, 
                    CONFIG.TARGETS, 
                    CONFIG.path_to_store,
                    utils
                )
                futures.append(future)
                
            # Wait for all chunks to finish and print their return statements
            for future in concurrent.futures.as_completed(futures):
                print(future.result())

    @staticmethod
    def check_data():
        "checking if data is correct by displaying one image of each class."

        model_class = getattr(models, CONFIG.model_name, None)
        if model_class is None:
            print(f"ERROR: Model {CONFIG.model_name} not found in models.py, exiting...")
            exit(1)

        from src.ML.models import KerasModelBase
        import numpy as np # Make sure numpy is imported

        # Helper function to normalize images for saving
        def prep_for_save(img_tensor):
            img_arr = img_tensor.numpy()
            # Min-Max scaling to force the array into [0.0, 1.0]
            img_arr = (img_arr - np.min(img_arr)) / (np.max(img_arr) - np.min(img_arr) + 1e-8)
            return img_arr

        ds = model_class().MLU.build_dataset(
            BATCH_N = 1,
            Keras_Preprocess=isinstance(model_class, KerasModelBase)
        )
        
        print("\n --- Checking Data Generation --- \n")
        i = 0
        for img, label in ds.take(7):
            print(img.shape, label.shape) 
            print("image array head: ", img[0][:5, :5, 0])
            print("label: ", label[5:], "...")
            try: 
                # Applied normalization here
                plt.imsave(f"gen_sanity_test_img_{i}.png", prep_for_save(img[0]))
                i += 1
            except Exception as e:
                print("ERROR: You may be attempting to plot a graph in a headless process. Error: ", e)

        print("\n --- Checking Data Loading from TRAIN directory --- \n")
        CONFIG.BATCH_N = 1
        train_ds, _ = ML_Utils.load_dataset(
            Keras_Preprocess=isinstance(model_class, KerasModelBase),
            #skip_sentinel = True,
        )
        i=0
        for img, label in train_ds.take(7):
            print(img.shape, label.shape) 
            print("image array head: ", img[0][:5, :5, 0])
            print("label: ", label[5:], "...")
            try: 
                # Applied normalization here
                plt.imsave(f"load_sanity_test_img_{i}.png", prep_for_save(img[0]))
                i += 1
            except Exception as e:
                print("ERROR: You may be attempting to plot a graph in a headless process. Error: ", e)

        print("\n --- Checking Data Loading from TEST directory --- \n")
        CONFIG.BATCH_N = 1
        test_ds = ML_Utils.load_dataset(
            train_split= 1, 
            Keras_Preprocess = isinstance(model_class, tf.keras.Model), 
            data_path = f"{os.getcwd()}/AmpliVision/data/scanned_MARKER/test/",
            use_case = "Test",
            #skip_sentinel = True,
        )
        i=0
        # Fixed loop to iterate over test_ds instead of train_ds
        for img, label in test_ds.take(7): 
            print(img.shape, label.shape) 
            print("image array head: ", img[0][:5, :5, 0])
            print("label: ", label[5:], "...")
            try: 
                # Note: You might want to rename the output file to distinguish from the train files
                plt.imsave(f"test_load_sanity_img_{i}.png", prep_for_save(img[0]))
                i += 1
            except Exception as e:
                print("ERROR: You may be attempting to plot a graph in a headless process. Error: ", e)

    @staticmethod
    def history():
        "display training history after training"

        import pickle as pkl

        # PYOD results
        path = "/home/matheus.berbet001/code/AmpliVision/pyod_data_110.pkl"
        with open(path, "rb") as f:
            pyod_results = pkl.load(f)
            print(pyod_results)
    

        # Training history
        path = f"{os.getcwd()}/AmpliVision/data/ML_perform/histories/history_{CONFIG.TAG}.pkl"
        with open(path, "rb") as f:
            history = pkl.load(f)
            print( *[f"{k}: {v}" for k, v in history.items()], sep="\n\n")

    @staticmethod
    def view(model_name):
        """ Visualize feature maps of convolutional layers for a given image using a trained model """
        from src.ML.visuals import visualize_feature_maps
        
        model_class = getattr(models, CONFIG.model_name, None)
        if model_class is None:
            print(f"ERROR: Model {model_name} not found in models.py, exiting...")
            exit(1)

        #trained_model = model_class().build_model()

        #H, W = tuple(CONFIG.SIZE)
        #input_shape_with_batch = (None, H, W, 3) 

        # Explicitly build the Sequential model so 'model.input' is defined.
        #trained_model.build(input_shape=input_shape_with_batch)

        # load trained model from disk. the 
        from tensorflow.keras.models import load_model

        path = "/home/matheus.berbet001/code/AmpliVision/AmpliVision/data/ML_models/ALEXNET_2025_10_30_09.keras"
        trained_model = load_model(path)

        sample_image_path = f"{CONFIG.path_to_store}/thyroid_9.png"  # Replace
        visualize_feature_maps(trained_model, sample_image_path, tuple(CONFIG.SIZE))

    
    # Helper function to add random noise to an RGB value and keep it between 0 and 255
    def apply_noise(value, noise_range):
        noisy_value = value + random.uniform(-noise_range, noise_range)
        # Clip the value to stay within 0.0 - 255.0 and round to 1 decimal place
        return round(max(0.0, min(255.0, noisy_value)), 1)


    @staticmethod
    def generate_all_729_classes():
        
        colors = [
            (255.0, 0.0, 0.0),    # Red
            (0.0, 255.0, 0.0),    # Green
            (0.0, 0.0, 255.0)     # Blue
        ]

        base_bkg_r, base_bkg_g, base_bkg_b = 220.0, 220.0, 220.0
        SPOT_NOISE = 15.0
        BKG_NOISE = 5.0

        now = datetime.now()
        date_str = now.strftime("%m/%d/%Y")
        time_str = now.strftime("%H:%M:%S")

        # 6 spots total across the 3 blocks (3^6 = 729 combinations)
        possible_combinations = list(itertools.product(colors, repeat=6))
        targets = [f"class{i}" for i in range(len(possible_combinations))]
        print(f"{len(targets)} possible combinations")

        # Reverted back to just spot1 and spot2
        columns = [
            'date', ' time',
            ' grid_index',
            ' block_type ',
            ' spot1_r', ' spot1_g', ' spot1_b',
            ' spot2_r', ' spot2_g', ' spot2_b',
            ' bkg_r', ' bkg_g', ' bkg_b',
            ' spot1_corr_r', ' spot1_corr_g', ' spot1_corr_b',
            ' spot2_corr_r', ' spot2_corr_g', ' spot2_corr_b',
        ]

        grid_x, grid_y = 5, 4

        for i, combo in enumerate(possible_combinations):
            class_name = targets[i]
            data = []
            
            print(f"processing {class_name}")
            
            # Map the 6 spots generated by itertools to their respective blocks
            # combo is a tuple of 6 RGB states: (b1s1, b1s2, b2s1, b2s2, b3s1, b3s2)
            block_configs = [
                ('test_block1', combo[0], combo[1]),
                ('test_block2', combo[2], combo[3]),
                ('test_block3', combo[4], combo[5])
            ]
            
            for block_name, base_s1, base_s2 in block_configs:
                # Generate noisy colors for this specific block's spots
                s1 = [Business.apply_noise(c, SPOT_NOISE) for c in base_s1]
                s2 = [Business.apply_noise(c, SPOT_NOISE) for c in base_s2]
                
                bkg_r = Business.apply_noise(base_bkg_r, BKG_NOISE)
                bkg_g = Business.apply_noise(base_bkg_g, BKG_NOISE)
                bkg_b = Business.apply_noise(base_bkg_b, BKG_NOISE)
                
                # Calculate corrections
                s1_corr = [round(bkg - spot, 1) for bkg, spot in zip([bkg_r, bkg_g, bkg_b], s1)]
                s2_corr = [round(bkg - spot, 1) for bkg, spot in zip([bkg_r, bkg_g, bkg_b], s2)]
                
                grid_index = f"({grid_x}, {grid_y})"
                grid_y += 1
                if grid_y > 9:
                    grid_y = 0
                    grid_x += 1
                    
                data.append([
                    date_str, time_str, grid_index, block_name,
                    s1[0], s1[1], s1[2],
                    s2[0], s2[1], s2[2],
                    bkg_r, bkg_g, bkg_b,
                    s1_corr[0], s1_corr[1], s1_corr[2],
                    s2_corr[0], s2_corr[1], s2_corr[2]
                ])
                
            # Build DataFrame and save
            write_to_csv(f"{datetime.now().strftime("%m-%d-%Y")}/{class_name}.csv", data, False)
            
        print(f"Successfully generated {len(targets)} files (class0.csv to class728.csv).")