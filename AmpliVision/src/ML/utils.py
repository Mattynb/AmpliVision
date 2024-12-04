""" Functions used across machine learning workflows """

from src.phaseA import *
from src.phaseB import phaseB
from src.generators.image_generation.RuleBasedGenerator import RuleBasedGenerator

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import pickle as pkl

import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'


class  ML_Utils:
    def __init__(
        self, 
        path_to_imgs,
        scanned_path,
        id_str
        ):
    
        self.path_to_imgs = path_to_imgs #"data/scanned/*" #DENV_imgs/*"
        self.scanned_path = scanned_path #"data/scanned/" 
        self.id_str = id_str

        self.prepare_image_RBGen()

        self.PlotCallback = self.PlotCallback( id_str )


    def prepare_image_RBGen(self, display= False):
        """ Does initial setup needed to create RBG """

        # Phase A.1 - Scanning images
        # if path to images does not start with scanned path, 
        # then phaseA1 will scan the images and save them to 
        # a new folder named scanned_NAMEOFINPUTFOLDER
        Images = phaseA1(
            self.path_to_imgs, 
            self.scanned_path,
            display=display, 
            do_white_balance=False,
            is_pre_scanned="scanned" in self.path_to_imgs
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
            TARGETS, 
            BATCH_N, 
            SIZE,
            BLACK = False,
            OUTLIER = False,
            contamination = 0.05
             
        ):
        """ Creates a dataset using rule based generator to work with tensor flow """

        RBG = RuleBasedGenerator(self.graphs, self.results)
        RBG.setup()
        #save = True if OUTLIER else False # save

        _args = [ 
            TARGETS, # what TARGETS to generate
            0.05, # noise
            BLACK, # black background or no
            True, # rgb
            False, #save
        ]

        _args.append(contamination) if OUTLIER else None


        # transform generator into dataset
        g_dataset = tf.data.Dataset.from_generator(
            RBG.generate_for_od if OUTLIER else RBG.generate,  
            output_shapes=(
                [1242, 1242, 3], 
                2 if OUTLIER else [len(TARGETS)]
            ), 
            output_types=(tf.float32, tf.float32),
            args = _args
        )
        
        # dataset is (x_batch / 255, y_batch), with some random rotation
        g_dataset = g_dataset.map(
            lambda x, y: (
                # x - Image
                tf.cast(
                    tf.image.rot90(
                        tf.image.resize(
                            x, 
                            SIZE
                        ),
                        k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
                    ),
                    tf.float32
                ) / 255, 

                # y - Label
                tf.cast(y, tf.float32)
            ),
            num_parallel_calls=1
        )
        g_dataset = g_dataset.batch(batch_size=BATCH_N)
        g_dataset = g_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return g_dataset
    

    def test_dataset(self):
        """ Used to see if data is being generated correctly """

        # probably wrong order
        classes = ['lung', 'thyroid', 'ovarian', 'prostate', 'skin', 'control', 'breast']
        for img, label in self.build_dataset(BATCH_N = 1, BLACK=True).take(1):
            print(img.shape) 
            for i, im in enumerate(img):   
                print(f"\n{classes[np.where(label[i].numpy() == 1)[0][0]]}")
                #plt.imshow(im)
                #plt.show()

        #for img, label in self.build_dataset(BATCH_N = 1).take(1):
        #    print(img.shape) 
        #    for i, im in enumerate(img):   
        #        print(f"\n{classes[np.where(label[i].numpy() == 1)[0][0]]}")
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
        def __init__(self, id_str):
            super(ML_Utils.PlotCallback, self).__init__()
            self.path = f"{os.getcwd()}/AmpliVision/data/"
            self.id_str = id_str

        def on_train_begin(self, logs={}):
            
            # accuracy and loss for each epoch
            self.epoch_accuracy = []
            self.epoch_val_accuracy = []
            self.epoch_loss = []
            self.epoch_val_loss = []

            # confusion matrix for each epoch
            self.epoch_confusion_matrix = []


        
        def on_epoch_end(self, epoch, logs={}):
            
            # plot the accuracy and loss
            self.plot_acc_loss(epoch, logs)

            # save the history of the model
            with open(self.path + "ML_models/" + f"history_{self.id_str}.pkl", 'wb') as file_pi:
                pkl.dump(self.model.history.history, file_pi)


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
