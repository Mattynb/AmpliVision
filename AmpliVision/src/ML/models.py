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