""" Functions used across machine learning workflows """

import os
import numpy as np
import pickle as pkl
import tensorflow as tf
import matplotlib.pyplot as plt

from src.phaseA import *
from src.phaseB import phaseB
from src.config import CONFIG
from src.generators.image_generation.RuleBasedGenerator import RuleBasedGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.backend import identify_block

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'


class  ML_Utils:
    def __init__(self):
        self.prepare_image_Gen()
        self.PlotCallback = self.PlotCallback()

        #GAN
        #self.sim2real_generator = tf.keras.models.load_model(CONFIG.GAN_SAVE_PATH)
        #self.sim2real_generator.trainable = False


    def prepare_image_Gen(self, display=False):
        """ Does initial setup needed to create GEN """

        # Phase A.1 - Scanning images
        # if path to images does not start with scanned path, 
        # then phaseA1 will scan the images and save them to 
        # a new folder named scanned_NAMEOFINPUTFOLDER
        Images = phaseA1(
            CONFIG.path_to_imgs, 
            CONFIG.scanned_path,
            display=display, 
            do_white_balance=False,
            is_pre_scanned= "scanned" in CONFIG.path_to_imgs
        )
    
        # Phase A.2 - Grids
        Grids = phaseA2(Images, display=display)
        del Images

        # save test results
        self.results = phaseB(Grids, display=display)
        print(len(Grids))
     
        # Phase A.3 - Position Graph
        self.graphs = phaseA3(Grids, display=display)
        del Grids


    def build_dataset(
            self,
            BATCH_N = CONFIG.BATCH_N,
            SIZE = CONFIG.SIZE,
            OUTLIER = False,
            contamination = CONFIG.NOISE,
            Keras_Preprocess = False,
            generator_only = False,
            RGB = True,
            BLACK = CONFIG.BLACK
        ):
        """ Creates a dataset using rule based generator to work with tensor flow """

        GEN = RuleBasedGenerator(self.graphs, self.results)
        GEN.setup()
        if generator_only:
            return GEN

        #save = True if OUTLIER else False # save

        _args = [ 
            CONFIG.TARGETS, # what TARGETS to generate
            CONFIG.NOISE, # noise
            BLACK, # black background or no
            RGB, # rgb
            False, #save
        ]

        _args.append(contamination) if OUTLIER else None


        # transform generator into dataset
        g_dataset = tf.data.Dataset.from_generator(
            GEN.generate_for_od if OUTLIER else GEN.generate,  
            output_shapes=(
                [1242, 1242, 3], 
                2 if OUTLIER else [len(CONFIG.TARGETS)]
            ), 
            output_types=(tf.float32, tf.float32),
            args = _args
        )

        def preprocess_image(image, label, size):
            """Resizes, rotates, and normalizes the image."""
            image = tf.image.resize(image, size)
            image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
            image = tf.cast(image, tf.float32)
            if not Keras_Preprocess:
                image /= 255.0
            label = tf.cast(label, tf.float32)
            return image, label

        g_dataset = g_dataset.map(
            lambda x, y: preprocess_image(x, y, SIZE),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        g_dataset = g_dataset.batch(batch_size=BATCH_N)
        g_dataset = g_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        # GAN
        def preprocess_and_translate(image, label, size):
            # 1. Resize/Format
            image = tf.image.resize(image, size)
            image = tf.cast(image, tf.float32) / 255.0
            
            # 2. Pass synthetic image through the GAN to make it look real
            # (Requires expanding dims for batching, then squeezing back)
            image = tf.expand_dims(image, 0)
            image = self.sim2real_generator(image)[0]
            
            label = tf.cast(label, tf.float32)
            return image, label

        # Map the GAN translation over the generated dataset
        if CONFIG.GAN_ON:
            g_dataset = g_dataset.map(
                lambda x, y: preprocess_and_translate(x, y, SIZE),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        return g_dataset
    
    @staticmethod
    def load_dataset(
        train_split=0.8, 
        Keras_Preprocess = False, 
        data_path = CONFIG.path_to_store,
        use_case = "Train"
    ):
        """ Loads folder with pre-generated png images and creates a tf dataset """

        def crop_path(data_path):
            # Define a new folder path for the cropped images
            cropped_path = os.path.join(data_path, "cropped")
            sentinel_file = os.path.join(cropped_path, ".cropped_sentinel")
        
            # Check if cropping has already been done
            if os.path.exists(sentinel_file):
                print(f"✅ Using cached cropped dataset from {cropped_path}")
                return cropped_path
        

            print(f"\n🚀 First-time setup: Cropping dataset offline. Saving to {cropped_path}...")
            os.makedirs(cropped_path, exist_ok=True)
            # Load chunk into memory dictionary format for phaseA2
            Images = phaseA1(
                data_path + "*", 
                data_path,
                display=False, 
                do_white_balance=False,
                is_pre_scanned=True
            )
            # Process the whole chunk through phaseA2
            grids = phaseA2(Images, display=False)
            # Crop and save to the new folder
            for img_name, grid in grids.items():
            
                for block in grid.blocks:
                    block.set_rgb_sequence()
                    block = identify_block(block)
                    
                cropped_img = RuleBasedGenerator.crop_to_test_areas(grid)
                if cropped_img is None:
                    print(f"skipping {img_name} since img cropping failed")
                    continue
            
                save_dest = os.path.join(cropped_path, img_name)
                print(f"saving cropped image to: {save_dest}")
                cv.imwrite(f"{save_dest}.png", cropped_img)

            # Create sentinel file to mark completion
            with open(sentinel_file, 'w') as f:
                f.write("cropped")

            print("✅ Offline cropping complete!\n")
        
            # Point the rest of the pipeline to load directly from the fast, cropped folder!
            return cropped_path
        # ==========================================
        if CONFIG.CROP_TO_TEST_AREA:
            data_path = crop_path(data_path)

        all_image_paths = [
            os.path.join(data_path, fname) 
            for fname in os.listdir(data_path) 
            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        all_image_paths.sort()
        
        print(f"image paths => {len(all_image_paths)}. Ex: {all_image_paths}")
        # valid_paths = []
        # for path in all_image_paths:
        #     if os.path.getsize(path) > 0:
        #         valid_paths.append(path)
        #     else:
        #         print(f"🚨 CRITICAL WARNING: Found and skipped empty file: {path}")
        
        valid_paths = [
            p for p in all_image_paths 
            if os.path.getsize(p) > 0
        ]
        if len(valid_paths) < len(all_image_paths):
            print(f"⚠️  Skipped {len(all_image_paths) - len(valid_paths)} empty files")

        all_image_paths = valid_paths

        def order_paths(all_image_paths):
            """
            folder contains images from all classes sorted alphabetically.
            this functions returns a list that sends a diffent class at a time.
            e.g. if there are 3 classes with 4 images each, the order is:
            class1_img1, class2_img1, class3_img1, class4_img1,
            class1_img2, class2_img2, class3_img2, class4_img2, ...
            """
            n_classes = len(CONFIG.TARGETS)
            images_per_class = len(all_image_paths) // n_classes
            ordered_paths = []
    
            for i in range(images_per_class):
                for j in range(n_classes):
                    index = j * images_per_class + i
                    ordered_paths.append(all_image_paths[index])
    
            return ordered_paths
    
        def define_label_from_path(file_path):
            "images are saved as label_etc.png. returns one-hot encoded label"
            filename = tf.strings.split(file_path, os.path.sep)[-1]
            label_str = tf.strings.split(filename, '_')[0]
            label = tf.cast(tf.equal(CONFIG.TARGETS, label_str), tf.float32)
            #print(f"Defined label {label} from file path {file_path}. label_str: {label_str}")
            return label

        def load_and_preprocess_image(path):
            "loads and preprocesses image"
            image = tf.io.read_file(path)
            image = tf.image.decode_png(image, channels=3)
            # bgr to rgb?
            #image = image[..., ::-1]
            image = tf.image.resize(image, CONFIG.SIZE)

            # salt and pepper noise 
            # ASSUMES that image has no noise already
            if CONFIG.NOISE > 0:
                noise = tf.random.uniform(shape=tf.shape(image), minval=0, maxval=1)
                salt_mask = noise < (CONFIG.NOISE / 2.0)  # half of the noise is salt
                pepper_mask = (noise >= (CONFIG.NOISE / 2.0)) & (noise < (CONFIG.NOISE))  # half of the noise is pepper
                image = tf.where(salt_mask, 255.0, image)  # Add salt noise
                image = tf.where(pepper_mask, 0.0, image)  # Add pepper noise


            # check if image is not already normalized
            if tf.reduce_max(image) > 1.0 and not Keras_Preprocess:
                image = tf.cast(image, tf.float32) / 255.0
            return image
        
        def load_image_and_label(path):
            "loads image and its label"
            label = define_label_from_path(path)
            image = load_and_preprocess_image(path)
            print("Loaded image shape: ", image.shape)
            print("Loaded label: ", label)
            return image, label
        
        ordered_paths = order_paths(all_image_paths)
        dataset_size = len(ordered_paths)

        # 1. Split the paths in Python FIRST
        train_count = int(train_split * dataset_size)
        train_paths = ordered_paths[:train_count]
        val_paths = ordered_paths[train_count:]
        print(f"Training dataset size: {len(train_paths)}, Validation dataset size: {len(val_paths)}")
        

        # 2. Create the TRAIN dataset
        train_ds = tf.data.Dataset.from_tensor_slices(train_paths)
        
        if use_case == "Test":
            train_ds = train_ds.map(load_image_and_label, num_parallel_calls=CONFIG.MAX_THREADS)
            train_ds = train_ds.batch(CONFIG.BATCH_N).prefetch(2)
            return train_ds

        # SHUFFLE STRINGS HERE! (Instantaneous)
        train_ds = train_ds.shuffle(buffer_size=len(train_paths)) 

        train_ds = train_ds.map(
            load_image_and_label, 
            num_parallel_calls=CONFIG.MAX_THREADS
        )
        train_ds = train_ds.batch(CONFIG.BATCH_N).repeat()
        
        # if CONFIG.CROP_TO_TEST_AREA:
        #     train_ds = train_ds.map(crop_batch_to_block, num_parallel_calls=tf.data.AUTOTUNE)
        
        train_dataset = train_ds.prefetch(1)
        
        # 3. Create the VALIDATION dataset (No shuffle needed)
        if len(val_paths) > 0:
            val_ds = tf.data.Dataset.from_tensor_slices(val_paths)
            val_ds = val_ds.map(load_image_and_label, num_parallel_calls=CONFIG.MAX_THREADS)
            val_dataset = val_ds.batch(CONFIG.BATCH_N).repeat().prefetch(1)
        
            return train_dataset, val_dataset
        
        print("Returning only train_dataset in load_dataset because train_split is 1")
        return train_dataset, None
    
            



    def test_dataset(self, BATCH_N = 1):
        """ Used to see if data is being generated correctly """

        # probably wrong order
        classes = ['lung', 'thyroid', 'ovarian', 'prostate', 'skin', 'control', 'breast']
        for i, (img, label) in enumerate(self.build_dataset(BATCH_N, CONFIG.SIZE).take(1)):
            print(img.shape)  
            print(f"\n{classes[np.where(label[i].numpy() == 1)[0][0]]}")
            #plt.imshow(im)
            #plt.show()


    def plot_model_performance(self, history, fig_name):
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(fig_name+"_acc.png")
    
        plt.clf()

        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(fig_name+"_loss.png")

        
    class PlotCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            super(ML_Utils.PlotCallback, self).__init__()
            self.path = f"{os.getcwd()}/AmpliVision/data/"
            self.id_str = CONFIG.TAG

        def on_train_begin(self, logs={}):
            
            # accuracy and loss for each epoch
            self.epoch_accuracy = []
            self.epoch_val_accuracy = []
            self.epoch_loss = []
            self.epoch_val_loss = []

            # confusion matrix for each epoch
            #self.epoch_confusion_matrix = []


        
        def on_epoch_end(self, epoch, logs={}):
            # plot the accuracy and loss
            self.plot_acc_loss(epoch, logs)

            # 3. Save history
            with open(self.path + "ML_models/" + f"history_{self.id_str}.pkl", 'wb') as file_pi:
                pkl.dump(self.model.history.history, file_pi)

        """
        def on_batch_end(self, batch, logs={}):
            # 2. Generate and Save Confusion Matrix
            print(f"\n\nOn batch end - batch: {batch}\nLogs: {logs}\n\n")
    
            #self.plot_cm_at_epoch(batch)
        """ 

        def plot_cm_at_epoch(self, epoch):
            # Extract images and true labels from the validation dataset
            # We handle this carefully to avoid shuffling issues
            all_y_true = []
            all_y_pred = []

            for images, labels in self.validation_data:
                preds = self.model.predict(images, verbose=0)
                all_y_true.append(np.argmax(labels.numpy(), axis=1))
                all_y_pred.append(np.argmax(preds, axis=1))
            
            y_true = np.concatenate(all_y_true)
            y_pred = np.concatenate(all_y_pred)

            # Calculate Matrix
            cm = confusion_matrix(y_true, y_pred, labels=range(len(self.target_names)))
            
            # Plotting
            plt.figure(figsize=(10, 8))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.target_names)
            disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
            plt.title(f'Confusion Matrix - Epoch {epoch + 1}')
            
            # Save CM plot
            cm_path = self.path + "ML_perform/" + f"cm_{self.id_str}_epoch_{epoch+1}.png"
            plt.savefig(cm_path)
            plt.close() # Close to free up memory


        def plot_acc_loss(self, epoch, logs={}):
            # Append the metrics for each epoch
            self.epoch_accuracy.append(logs.get('accuracy'))
            self.epoch_val_accuracy.append(logs.get('val_accuracy'))
            self.epoch_loss.append(logs.get('loss'))
            self.epoch_val_loss.append(logs.get('val_loss'))
            
            # Clear the current plot to start a new one
            plt.clf()
            
            # Plot accuracy
            plt.subplot(1, 2, 1)
            plt.plot(self.epoch_accuracy, label='Training Accuracy')
            plt.plot(self.epoch_val_accuracy, label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend(loc='upper left')
            
            # Plot loss
            plt.subplot(1, 2, 2)
            plt.plot(self.epoch_loss, label='Training Loss')
            plt.plot(self.epoch_val_loss, label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(loc='upper right')
            
            # Save the figure to a file after each epoch
            file_path = self.path + "ML_perform/" + f"{self.id_str}.png"
            plt.savefig(file_path)
            print(f"Saved plot for epoch {epoch+1} at {file_path}")

"""

class MyMetrics():
    "https://stackoverflow.com/questions/49005892/keras-confusion-matrix-at-every-epoch"
    
    @staticmethod
    def recall_m(y_true, y_pred): # TPR
        true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1))) # TP
        possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1))) # P
        recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
        return recall

    @staticmethod
    def precision_m(y_true, y_pred):
        true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1))) # TP
        predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1))) # TP + FP
        precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
        return precision
    
    @staticmethod
    def f1_m(y_true, y_pred):
        precision = MyMetrics.precision_m(y_true, y_pred)
        recall = MyMetrics.recall_m(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+tf.keras.backend.epsilon()))

    @staticmethod
    def TP(y_true, y_pred):
        tp = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1))) # TP
        y_pos = tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1))
        n_pos = tf.keras.backend.sum(y_pos)
        y_neg = 1 - y_pos
        n_neg = tf.keras.backend.sum(y_neg)
        n = n_pos + n_neg
        return tp/n

    @staticmethod
    def TN(y_true, y_pred):
        y_pos = tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1))
        n_pos = tf.keras.backend.sum(y_pos)
        y_neg = 1 - y_pos
        n_neg = tf.keras.backend.sum(y_neg)
        n = n_pos + n_neg
        y_pred_pos = tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1))
        y_pred_neg = 1 - y_pred_pos
        tn = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_neg * y_pred_neg, 0, 1))) # TN
        return tn/n

    @staticmethod
    def FP(y_true, y_pred):
        y_pos = tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1))
        n_pos = tf.keras.backend.sum(y_pos)
        y_neg = 1 - y_pos
        n_neg = tf.keras.backend.sum(y_neg)
        n = n_pos + n_neg
        tn = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_neg * y_pred, 0, 1))) # FP
        return tn/n

    @staticmethod
    def FN(y_true, y_pred):
        y_pos = tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1))
        n_pos = tf.keras.backend.sum(y_pos)
        y_neg = 1 - y_pos
        n_neg = tf.keras.backend.sum(y_neg)
        n = n_pos + n_neg
        y_pred_pos = tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1))
        y_pred_neg = 1 - y_pred_pos
        tn = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred_neg, 0, 1))) # FN
        return tn/n
                    
"""