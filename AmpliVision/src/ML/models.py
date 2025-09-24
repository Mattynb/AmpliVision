import os
import tensorflow as tf
from .utils import ML_Utils
from .evaluator import test_model_generated
import pickle as pkl

class workflow:
    """ Workflow parent class for all implemented ML models """
    def __init__( 
            self, 
            TARGETS,
            path_to_imgs,
            scanned_path,
            SIZE,
            BATCH_N,
            EPOCHS,
            BLACK = False   
        ):
        
        self.SIZE = SIZE
        self.TARGETS = TARGETS
        self.BATCH_N = BATCH_N
        self.EPOCHS = EPOCHS
        self.BLACK = BLACK

        #self.MLU.test_dataset()

    def train_model(self, TAG):    
        import datetime

        model_save_name = f"{TAG}_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.abspath(f"{os.getcwd()}/AmpliVision/data/ML_models/{model_save_name}"),
            save_weights_only=False,
            monitor='val_accuracy',
            mode='auto',
            save_best_only=True, 
            period=1
        )

        callbacks = [
            self.MLU.PlotCallback,
            checkpoint
        ]

        g_dataset = self.MLU.build_dataset(self.TARGETS, self.BATCH_N, self.SIZE, self.BLACK)
        v_dataset = self.MLU.build_dataset(self.TARGETS, int((self.BATCH_N*0.2)) + 1, self.SIZE, self.BLACK)
        with tf.device('/GPU:0'):
            self.history = self.model.fit(
                g_dataset,
                epochs=self.EPOCHS, #EPOCHS,
                validation_data=v_dataset,
                steps_per_epoch=7,
                validation_steps=1,
                callbacks = callbacks
            )

        # Save model
        self.model.save(f"{os.getcwd()}/AmpliVision/data/ML_models/{model_save_name}")
    
        # Save history
        with open(f"{os.getcwd()}/AmpliVision/data/ML_models/history_{model_save_name}.pkl", 'wb') as file_pi:
            pkl.dump(self.history.history, file_pi)
    

    def test_model(self, TAG):

        path = f"{os.getcwd()}/AmpliVision/data/ML_models/{TAG}"
        model = tf.keras.models.load_model(path)

        dataset =  self.MLU.build_dataset(self.TARGETS, self.BATCH_N, self.SIZE, self.BLACK)
        
        test_model_generated(
            dataset,
            model,
            self.TARGETS,
            TAG
        )


    def run(self):
        self.build_model()
        self.train_model(self.id_str)
        return self # for chaining

    


import tensorflow_addons as tfa
class LENET(workflow):
    def __init__( 
            self,
            TARGETS,
            path_to_imgs,
            scanned_path,
            SIZE,
            BATCH_N,
            EPOCHS,
            BLACK = False,
            TAG = "_",
            **kwargs
        ):
        self.model_name = 'LENET'
        self.id_str = TAG
        super().__init__(TARGETS, path_to_imgs, scanned_path, SIZE, BATCH_N, EPOCHS, BLACK) 
        #f"{kwargs['TARGET_NAME']}{self.model_name}_Sz{self.SIZE}_Bn{self.BATCH_N}_Ep{self.EPOCHS}{'_BLACK' if self.BLACK else ''}"
        self.MLU = ML_Utils(path_to_imgs, scanned_path, self.id_str)


    def build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=tuple([self.SIZE[0], self.SIZE[1], 3])))
        model.add(tf.keras.layers.AveragePooling2D((2,2)))
        model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
        model.add(tf.keras.layers.AveragePooling2D((2,2)))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=120, activation='relu'))
        model.add(tf.keras.layers.Dense(units=84, activation='relu'))
        model.add(tf.keras.layers.Dense(units=len(self.TARGETS), activation = 'softmax'))
        model.compile(
            optimizer = "adam", #tf.keras.optimizers.Adam(learning_rate=0.001),
            loss = 'categorical_crossentropy',
            metrics = [
                'accuracy', 
                'AUC',
                tfa.metrics.F1Score(num_classes=len(self.TARGETS), average="macro")
            ]
        )
        model.summary()
        self.model = model  

# You can implement other models bellow based on the LENET model class

# Basic Architectures:
# AlexNet
class ALEXNET(workflow):
    def __init__( 
            self,
            TARGETS,
            path_to_imgs,
            scanned_path,
            SIZE,
            BATCH_N,
            EPOCHS,
            BLACK = False,
            TAG = "_",
            **kwargs
        ):
        self.model_name = 'ALEXNET'
        self.id_str = TAG
        
        SIZE = (227, 227)  # AlexNet standard input size

        super().__init__(TARGETS, path_to_imgs, scanned_path, SIZE, BATCH_N, EPOCHS, BLACK) 
        #f"{kwargs['TARGET_NAME']}{self.model_name}_Sz{self.SIZE}_Bn{self.BATCH_N}_Ep{self.EPOCHS}{'_BLACK' if self.BLACK else ''}"
        self.MLU = ML_Utils(path_to_imgs, scanned_path, self.id_str)


    def build_model(self):
        model = tf.keras.Sequential()
    
        #  Convolutional 11x11 kernel, 4 stride, 96 filters
        model.add(tf.keras.layers.Conv2D(
            filters=96, 
            kernel_size=(11, 11), 
            strides=(4,4),
            activation='relu', 
            input_shape=tuple([self.SIZE[0], self.SIZE[1], 3]))
        )
        model.add(tf.keras.layers.MaxPool2D((3, 3), strides=(2,2)))

        # Convolutional 5x5 kernel, 1 stride, 256 filters
        model.add(tf.keras.layers.Conv2D(
            filters=256, 
            kernel_size=(5, 5),
            strides=(1,1),
            activation='relu'),
            #padding='same' # same size as previous layer
        )
        model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2,2)))

        # Convolutional 3x3 kernel, 1 stride, 384 filters
        model.add(tf.keras.layers.Conv2D(
            filters=384, 
            kernel_size=(3, 3), 
            strides=(1, 1), 
            activation='relu', 
            #padding='same'
        ))

        model.add(tf.keras.layers.Conv2D(
            filters=384, 
            kernel_size=(3, 3), 
            strides=(1, 1), 
            activation='relu', 
            #padding='same'
        ))

        model.add(tf.keras.layers.Conv2D(
            filters=256, 
            kernel_size=(3, 3), 
            strides=(1, 1), 
            activation='relu', 
            #padding='same'
        ))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
        
        
        model.add(tf.keras.layers.Flatten())

        model.add(tf.keras.layers.Dense(units=4096, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))

        model.add(tf.keras.layers.Dense(units=4096, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))

        model.add(tf.keras.layers.Dense(units=len(self.TARGETS), activation = 'softmax'))
        
        model.compile(
            optimizer = "adam", #tf.keras.optimizers.Adam(learning_rate=0.001),
            loss = 'categorical_crossentropy',
            metrics = [
                'accuracy', 
                'AUC',
                tfa.metrics.F1Score(num_classes=len(self.TARGETS), average="macro")
            ]
        )
        model.summary()
        self.model = model  

# VGG16
class VGG16(workflow):
    def __init__( 
            self,
            TARGETS,
            path_to_imgs,
            scanned_path,
            SIZE,
            BATCH_N,
            EPOCHS,
            BLACK = False,
            TAG = "_",
            **kwargs
        ):
        self.model_name = 'VGG16'
        self.id_str = TAG

        SIZE = (224, 224)  # VGG16 standard input size

        # Initialize the base class with common parameters
        super().__init__(TARGETS, path_to_imgs, scanned_path, SIZE, BATCH_N, EPOCHS, BLACK) 
        
        # Initialize ML_Utils assuming it handles data loading/processing
        self.MLU = ML_Utils(path_to_imgs, scanned_path, self.id_str)


    def build_model(self):
        # NOTE ON INPUT SIZE: VGG16 is typically designed for 224x224x3 images.
        # This implementation uses self.SIZE for the input_shape.
        
        model = tf.keras.Sequential()
        
        # --- BLOCK 1: 2x Conv (64 filters) + MaxPool ---
        model.add(tf.keras.layers.Conv2D(
            filters=64, 
            kernel_size=(3, 3), 
            activation='relu', 
            padding='same',
            input_shape=tuple([self.SIZE[0], self.SIZE[1], 3])
        ))
        model.add(tf.keras.layers.Conv2D(
            filters=64, 
            kernel_size=(3, 3), 
            activation='relu', 
            padding='same'
        ))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        # --- BLOCK 2: 2x Conv (128 filters) + MaxPool ---
        model.add(tf.keras.layers.Conv2D(
            filters=128, 
            kernel_size=(3, 3), 
            activation='relu', 
            padding='same'
        ))
        model.add(tf.keras.layers.Conv2D(
            filters=128, 
            kernel_size=(3, 3), 
            activation='relu', 
            padding='same'
        ))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        # --- BLOCK 3: 3x Conv (256 filters) + MaxPool ---
        model.add(tf.keras.layers.Conv2D(
            filters=256, 
            kernel_size=(3, 3), 
            activation='relu', 
            padding='same'
        ))
        model.add(tf.keras.layers.Conv2D(
            filters=256, 
            kernel_size=(3, 3), 
            activation='relu', 
            padding='same'
        ))
        model.add(tf.keras.layers.Conv2D(
            filters=256, 
            kernel_size=(3, 3), 
            activation='relu', 
            padding='same'
        ))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        # --- BLOCK 4: 3x Conv (512 filters) + MaxPool ---
        model.add(tf.keras.layers.Conv2D(
            filters=512, 
            kernel_size=(3, 3), 
            activation='relu', 
            padding='same'
        ))
        model.add(tf.keras.layers.Conv2D(
            filters=512, 
            kernel_size=(3, 3), 
            activation='relu', 
            padding='same'
        ))
        model.add(tf.keras.layers.Conv2D(
            filters=512, 
            kernel_size=(3, 3), 
            activation='relu', 
            padding='same'
        ))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        # --- BLOCK 5: 3x Conv (512 filters) + MaxPool ---
        model.add(tf.keras.layers.Conv2D(
            filters=512, 
            kernel_size=(3, 3), 
            activation='relu', 
            padding='same'
        ))
        model.add(tf.keras.layers.Conv2D(
            filters=512, 
            kernel_size=(3, 3), 
            activation='relu', 
            padding='same'
        ))
        model.add(tf.keras.layers.Conv2D(
            filters=512, 
            kernel_size=(3, 3), 
            activation='relu', 
            padding='same'
        ))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        # --- Fully Connected Layers ---
        model.add(tf.keras.layers.Flatten())
        
        # FC-1: 4096 neurons
        model.add(tf.keras.layers.Dense(units=4096, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5)) 

        # FC-2: 4096 neurons
        model.add(tf.keras.layers.Dense(units=4096, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5)) 

        # Output Layer: num_classes neurons
        model.add(tf.keras.layers.Dense(units=len(self.TARGETS), activation = 'softmax'))
        
        # Compile the model using the standard metrics
        model.compile(
            optimizer = "adam", 
            loss = 'categorical_crossentropy',
            metrics = [
                'accuracy', 
                'AUC',
                tfa.metrics.F1Score(num_classes=len(self.TARGETS), average="macro")
            ]
        )
        
        model.summary()
        self.model = model

# Inception V1-V4

# Residual Learning:
# ResNet

# Portable Attention based:
# Senet
# EfficientNet
# RegNet
# CA

# Mobile Architectures:
# MobileNet V1-V3
# ShuffleNet V1-V2
# MnasNet

