import os
import tensorflow as tf
from .utils import ML_Utils
from .evaluator import test_model_generated
from ..config import Config
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
        
        self.SIZE = self.DEFAULT_SIZE if self.DEFAULT_SIZE else SIZE
        self.TARGETS = TARGETS
        self.BATCH_N = BATCH_N
        self.EPOCHS = EPOCHS
        self.BLACK = BLACK

        print(f"\n\n***********  IMAGE SIZE IS: {self.SIZE} ***********\n\n")

        #self.MLU.test_dataset()

    def train_model(self, TAG):    
        import datetime
        import time

        model_save_name = f"{TAG}_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"

        """
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.abspath(f"{os.getcwd()}/AmpliVision/data/ML_models/{model_save_name}"),
            save_weights_only=False,
            monitor='val_accuracy',
            mode='auto',
            save_best_only=True, 
            save_freq='epoch',
        )"""

        callbacks = [
            self.MLU.PlotCallback,
            #checkpoint
        ]

        g_dataset = self.MLU.build_dataset(self.TARGETS, Config.BATCH_N, self.SIZE, self.BLACK, Keras_Preprocess=isinstance(self, KerasModelBase))
        v_dataset = self.MLU.build_dataset(self.TARGETS, int((Config.BATCH_N*0.5)), self.SIZE, self.BLACK, Keras_Preprocess=isinstance(self, KerasModelBase))

        inference_time = time.time()
        with tf.device('/GPU:0'):
            self.history = self.model.fit(
                g_dataset,
                epochs=Config.EPOCHS,
                validation_data=v_dataset,
                steps_per_epoch=Config.STEPS_PER_EPOCH,
                validation_steps=Config.VALIDATION_STEPS,
                callbacks = callbacks
            )
        inference_time = time.time() - inference_time
        print(f"Training completed in: {inference_time/60:.2f} minutes")    

        # Save model
        #self.model.save(f"{os.getcwd()}/AmpliVision/data/ML_models/{model_save_name}")
    
        # Save history
        #with open(f"{os.getcwd()}/AmpliVision/data/ML_models/history_{model_save_name}.pkl", 'wb') as file_pi:
        #    pkl.dump(self.history.history, file_pi)


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

    

class LENET(workflow):
    
    DEFAULT_SIZE = (1024, 1024)  # Default size for LENET if not specified

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
                tf.keras.metrics.F1Score(average=None, threshold=None, name='f1_score', dtype=None)

            ]
        )
        model.summary()
        self.model = model  

# You can implement other models bellow based on the LENET model class

# Basic Architectures:
# AlexNet
class ALEXNET(workflow):

    DEFAULT_SIZE = (227, 227)  # AlexNet standard input size

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
                tf.keras.metrics.F1Score(average=None, threshold=None, name='f1_score', dtype=None)

            ]
        )
        model.summary()
        self.model = model  

#************#

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
class KerasModelBase(workflow):
    """
    Base class to dynamically load and implement any Keras Application model.
    Child classes only need to set MODEL_NAME and DEFAULT_SIZE.
    """
    
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
        self.model_name = kwargs.get("model_name")
        self.id_str = TAG

        # Initialize the base class with common parameters
        super().__init__(TARGETS, path_to_imgs, scanned_path, SIZE, BATCH_N, EPOCHS, BLACK) 
        
        # Initialize ML_Utils
        self.MLU = ML_Utils(path_to_imgs, scanned_path, self.id_str)

    def build_model(self, weights=None, include_top=False):
        """
        Builds the model.
        """
        # Dynamically determine the import path
        # Example: 'tf.keras.applications.EfficientNetB0'
        import_path = f'tf.keras.applications.{self.model_name}'
        
        try:
            eval(import_path)
        except AttributeError:
            raise ImportError(f"Could not find {self.model_name} model at {import_path}. Check TensorFlow version or model name.")

        input_shape = tuple([self.SIZE[0], self.SIZE[1], 3])

        model_constructor_kwargs = {
            "include_top": include_top,
            "weights": weights,
            "input_shape": input_shape,
            "classes": len(self.TARGETS)
        }

        base_model = elegant_keras_applications_constructor(
            model_str=self.model_name,
            model_constructor_kwargs=model_constructor_kwargs,
            input_shape=input_shape
        )
        
        x = base_model.output
        if not include_top:
            x = GlobalAveragePooling2D()(x)
            x = Dropout(0.5)(x)  # Optional dropout layer for regularization
            predictions = Dense(len(self.TARGETS), activation='softmax')(x)
            model = Model(inputs=base_model.input, outputs=predictions)
        else:
            model = base_model
        
        model.compile(
            optimizer = "adam",
            loss = 'categorical_crossentropy',
            metrics = [
                'accuracy',
                'AUC',
                tf.keras.metrics.F1Score(average=None, threshold=None, name='f1_score', dtype=None)
            ]
        )

        model.summary()
        self.model = model


#https://peterbbryan.medium.com/avoiding-a-keras-applications-pitfall-58156a01115f
from typing import Any, Callable, Dict
import inspect

def get_model(model_str: str) -> Callable[..., tf.keras.models.Model]:
    """
    Get Keras application classifier architecture.
    Args:
        model_str: String name of model.
    Returns:
        Model constructor.
    """

    # get Keras applications classes
    models = {
        name: obj
        for name, obj in inspect.getmembers(tf.keras.applications)
        if inspect.isfunction(obj)
    }
    assert (
        model_str in models.keys()
    ), f"Model name {model_str} not in Keras applications. Options are {list(models.keys())}"

    return models[model_str]


def get_model_with_preprocessing(
    model_str: str, model_constructor_kwargs: Dict[str, Any], input_shape: list[int, int, int]
) -> tf.keras.models.Model:
    """
    Keras helper function to get appropriate model-preprocessing pair.
    Args:
        model_str: Name of Keras application model.
        model_constructor_kwargs: Optional model constructor kwargs.
    """

    # build model
    model_constructor: Callable[..., tf.keras.models.Model] = get_model(
        model_str=model_str
    )

    # get the correct preprocess function
    preprocess_function = inspect.getmodule(model_constructor).preprocess_input

    model = model_constructor(**model_constructor_kwargs)

    input_layer = tf.keras.layers.Input(shape=input_shape) #model.input.shape[1:])
    preprocessing_layer = tf.keras.layers.Lambda(preprocess_function)(input_layer)
    output = model(preprocessing_layer)

    return tf.keras.models.Model(input_layer, output)


def elegant_keras_applications_constructor(
    model_str: str, model_constructor_kwargs: Dict[str, Any], input_shape: list[int, int, int] = {}
):
    """
    CLI interface to demonstrate functionality.
    Args:
        model_str: Name of Keras application model.
        model_constructor_kwargs: Optional model constructor kwargs.
    """

    model = get_model_with_preprocessing(
        model_str=model_str, model_constructor_kwargs=model_constructor_kwargs, input_shape=input_shape
    )

    return model

####

class EfficientNetB0(KerasModelBase):
    MODEL_NAME = 'EfficientNetB0'
    DEFAULT_SIZE = (224, 224)

class EfficientNetB1(KerasModelBase):
    MODEL_NAME = 'EfficientNetB1'
    DEFAULT_SIZE = (240, 240)

class EfficientNetB2(KerasModelBase):
    MODEL_NAME = 'EfficientNetB2'
    DEFAULT_SIZE = (260, 260)

class EfficientNetB3(KerasModelBase):
    MODEL_NAME = 'EfficientNetB3'
    DEFAULT_SIZE = (300, 300)

class EfficientNetB4(KerasModelBase):
    MODEL_NAME = 'EfficientNetB4'
    DEFAULT_SIZE = (380, 380)

class EfficientNetB5(KerasModelBase):
    MODEL_NAME = 'EfficientNetB5'
    DEFAULT_SIZE = (456, 456)

class EfficientNetB6(KerasModelBase):
    MODEL_NAME = 'EfficientNetB6'
    DEFAULT_SIZE = (528, 528)

class EfficientNetB7(KerasModelBase):
    MODEL_NAME = 'EfficientNetB7'
    DEFAULT_SIZE = (600, 600)

class EfficientNetV2B0(KerasModelBase):
    MODEL_NAME = 'EfficientNetV2B0'
    DEFAULT_SIZE = (224, 224)

class EfficientNetV2B1(KerasModelBase):
    MODEL_NAME = 'EfficientNetV2B1'
    DEFAULT_SIZE = (240, 240)

class EfficientNetV2B2(KerasModelBase):
    MODEL_NAME = 'EfficientNetV2B2'
    DEFAULT_SIZE = (260, 260)

class EfficientNetV2B3(KerasModelBase):
    MODEL_NAME = 'EfficientNetV2B3'
    DEFAULT_SIZE = (300, 300)

class EfficientNetV2S(KerasModelBase):
    MODEL_NAME = 'EfficientNetV2S'
    DEFAULT_SIZE = (384, 384)

class EfficientNetV2M(KerasModelBase):
    MODEL_NAME = 'EfficientNetV2M'
    DEFAULT_SIZE = (480, 480)

class EfficientNetV2L(KerasModelBase):
    MODEL_NAME = 'EfficientNetV2L'
    DEFAULT_SIZE = (480, 480)

class ConvNeXtTiny(KerasModelBase):
    MODEL_NAME = 'ConvNeXtTiny'
    DEFAULT_SIZE = (224, 224)

class ConvNeXtSmall(KerasModelBase):
    MODEL_NAME = 'ConvNeXtSmall'
    DEFAULT_SIZE = (224, 224)

class ConvNeXtBase(KerasModelBase):
    MODEL_NAME = 'ConvNeXtBase'
    DEFAULT_SIZE = (224, 224)

class ConvNeXtLarge(KerasModelBase):
    MODEL_NAME = 'ConvNeXtLarge'
    DEFAULT_SIZE = (224, 224)

class ConvNeXtXLarge(KerasModelBase):
    MODEL_NAME = 'ConvNeXtXLarge'
    DEFAULT_SIZE = (224, 224)

class VGG16(KerasModelBase):
    MODEL_NAME = 'VGG16'
    DEFAULT_SIZE = (224, 224
    )
class VGG19(KerasModelBase):
    MODEL_NAME = 'VGG19'
    DEFAULT_SIZE = (224, 224
    )
class ResNet50(KerasModelBase):
    MODEL_NAME = 'ResNet50'
    DEFAULT_SIZE = (224, 224)

class ResNet50V2(KerasModelBase):
    MODEL_NAME = 'ResNet50V2'
    DEFAULT_SIZE = (224, 224)

class MobileNet(KerasModelBase):
    MODEL_NAME = 'MobileNet'
    DEFAULT_SIZE = (224, 224)

class MobileNetV2(KerasModelBase):
    MODEL_NAME = 'MobileNetV2'
    DEFAULT_SIZE = (224, 224)

class MobileNetV3Small(KerasModelBase):
    MODEL_NAME = 'MobileNetV3Small'
    DEFAULT_SIZE = (224, 224)

class MobileNetV3Large(KerasModelBase):
    MODEL_NAME = 'MobileNetV3Large'
    DEFAULT_SIZE = (224, 224)

class DenseNet121(KerasModelBase):
    MODEL_NAME = 'DenseNet121'
    DEFAULT_SIZE = (224, 224)

class NASNetLarge(KerasModelBase):
    MODEL_NAME = 'NASNetLarge'
    DEFAULT_SIZE = (331, 331)

class NASNetMobile(KerasModelBase):
    MODEL_NAME = 'NASNetMobile'
    DEFAULT_SIZE = (224, 224)

class InceptionV3(KerasModelBase):
    MODEL_NAME = 'InceptionV3'
    DEFAULT_SIZE = (299, 299)

class InceptionResNetV2(KerasModelBase):
    MODEL_NAME = 'InceptionResNetV2'
    DEFAULT_SIZE = (299, 299)

class Xception(KerasModelBase):
    MODEL_NAME = 'Xception'
    DEFAULT_SIZE = (299, 299)
