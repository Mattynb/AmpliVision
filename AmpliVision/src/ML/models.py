import os
import pickle as pkl
import tensorflow as tf

from src.config import CONFIG

from .utils import ML_Utils
from .evaluator import test_model_generated


class workflow:
    """ Workflow parent class for all implemented ML models """
    def __init__(self):
        
        self.SIZE = self.DEFAULT_SIZE if self.DEFAULT_SIZE else CONFIG.SIZE

        print(f"\n\n***********  IMAGE SIZE IS: {self.SIZE} ***********\n\n")

        #self.MLU.test_dataset()

    def train_model(self):    
        import datetime
        import time

        model_save_name = f"{CONFIG.TAG}_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"

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

        g_dataset = self.MLU.build_dataset(CONFIG.BATCH_N, self.SIZE, Keras_Preprocess=isinstance(self, KerasModelBase))
        v_dataset = self.MLU.build_dataset(int((CONFIG.BATCH_N*0.5)), self.SIZE, Keras_Preprocess=isinstance(self, KerasModelBase))

        inference_time = time.time()
        with tf.device('/GPU:0'):
            self.history = self.model.fit(
                g_dataset,
                epochs=CONFIG.EPOCHS,
                validation_data=g_dataset,
                steps_per_epoch=CONFIG.STEPS_PER_EPOCH,
                validation_steps=CONFIG.VALIDATION_STEPS,
                callbacks = callbacks
            )
        inference_time = time.time() - inference_time
        print(f"Training completed in: {inference_time/60:.2f} minutes")    

        # Save model
        #self.model.save(f"{os.getcwd()}/AmpliVision/data/ML_models/{model_save_name}")
    
        # Save history
        #with open(f"{os.getcwd()}/AmpliVision/data/ML_models/history_{model_save_name}.pkl", 'wb') as file_pi:
        #    pkl.dump(self.history.history, file_pi)


    def test_model(self):
        """ Test a trained model """

        path = f"{os.getcwd()}/AmpliVision/data/ML_models/{CONFIG.TAG}"
        model = tf.keras.models.load_model(path)

        dataset =  self.MLU.build_dataset(CONFIG.TARGETS, CONFIG.BATCH_N, self.SIZE, CONFIG.BLACK)
        
        test_model_generated(
            dataset,
            model,
            CONFIG.TARGETS,
            CONFIG.TAG
        )


    def run(self):
        self.build_model()
        self.train_model()
        return self # for chaining

    

class LENET(workflow):
    
    DEFAULT_SIZE = (1024, 1024)  # Default size for LENET if not specified

    def __init__(self):
        super().__init__() 
        #f"{kwargs['TARGET_NAME']}{CONFIG.model_name}_Sz{self.SIZE}_Bn{CONFIG.BATCH_N}_Ep{CONFIG.EPOCHS}{'_BLACK' if CONFIG.BLACK else ''}"
        self.MLU = ML_Utils()
    

    def build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=tuple([self.SIZE[0], self.SIZE[1], 3])))
        model.add(tf.keras.layers.AveragePooling2D((2,2)))
        model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
        model.add(tf.keras.layers.AveragePooling2D((2,2)))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=120, activation='relu'))
        model.add(tf.keras.layers.Dense(units=84, activation='relu'))
        model.add(tf.keras.layers.Dense(units=len(CONFIG.TARGETS), activation = 'softmax'))
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

    def __init__(self):
        super().__init__() 
        #f"{kwargs['TARGET_NAME']}{CONFIG.model_name}_Sz{self.SIZE}_Bn{CONFIG.BATCH_N}_Ep{CONFIG.EPOCHS}{'_BLACK' if CONFIG.BLACK else ''}"
        self.MLU = ML_Utils()


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

        model.add(tf.keras.layers.Dense(units=len(CONFIG.TARGETS), activation = 'softmax'))
        
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
    Child classes only need to DEFAULT_SIZE.
    """
    
    def __init__(self):
        # Initialize the base class with common parameters
        super().__init__() 
        
        # Initialize ML_Utils
        self.MLU = ML_Utils()

    def build_model(self, weights='imagenet', include_top=False):
        """
        Builds the model.
        """
        # Dynamically determine the import path
        # Example: 'tf.keras.applications.EfficientNetB0'
        import_path = f'tf.keras.applications.{CONFIG.model_name}'
        
        try:
            eval(import_path)
        except AttributeError:
            raise ImportError(f"Could not find {CONFIG.model_name} model at {import_path}. Check TensorFlow version or model name.")

        input_shape = tuple([self.SIZE[0], self.SIZE[1], 3])

        model_constructor_kwargs = {
            "include_top": include_top,
            "weights": weights,
            "input_shape": input_shape,
            "classes": len(CONFIG.TARGETS)
        }

        base_model = elegant_keras_applications_constructor(
            model_str=CONFIG.model_name,
            model_constructor_kwargs=model_constructor_kwargs,
            input_shape=input_shape
        )
        
        x = base_model.output
        if not include_top:
            x = GlobalAveragePooling2D()(x)
            x = Dense(256, activation='relu')(x)
            x = Dropout(0.5)(x)  # Optional dropout layer for regularization
            predictions = Dense(len(CONFIG.TARGETS), activation='softmax')(x)
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

    input_layer = tf.keras.layers.Input(shape=model.input.shape[1:])
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
    
    # Freeze the base model (transfer learning)
    for layer in model.layers:
        layer.trainable = False

    return model

####

class EfficientNetB0(KerasModelBase):
   DEFAULT_SIZE = (224, 224)

class EfficientNetB1(KerasModelBase):
   DEFAULT_SIZE = (240, 240)

class EfficientNetB2(KerasModelBase):
   DEFAULT_SIZE = (260, 260)

class EfficientNetB3(KerasModelBase):
   DEFAULT_SIZE = (300, 300)

class EfficientNetB4(KerasModelBase):
   DEFAULT_SIZE = (380, 380)

class EfficientNetB5(KerasModelBase):
   DEFAULT_SIZE = (456, 456)

class EfficientNetB6(KerasModelBase):
   DEFAULT_SIZE = (528, 528)

class EfficientNetB7(KerasModelBase):
   DEFAULT_SIZE = (600, 600)

class EfficientNetV2B0(KerasModelBase):
    DEFAULT_SIZE = (224, 224)

class EfficientNetV2B1(KerasModelBase):
    DEFAULT_SIZE = (240, 240)

class EfficientNetV2B2(KerasModelBase):
    DEFAULT_SIZE = (260, 260)

class EfficientNetV2B3(KerasModelBase):
    DEFAULT_SIZE = (300, 300)

class EfficientNetV2S(KerasModelBase):
    DEFAULT_SIZE = (384, 384)

class EfficientNetV2M(KerasModelBase):
    DEFAULT_SIZE = (480, 480)

class EfficientNetV2L(KerasModelBase):
    DEFAULT_SIZE = (480, 480)

class ConvNeXtTiny(KerasModelBase):
    DEFAULT_SIZE = (224, 224)

class ConvNeXtSmall(KerasModelBase):
    DEFAULT_SIZE = (224, 224)

class ConvNeXtBase(KerasModelBase):
    DEFAULT_SIZE = (224, 224)

class ConvNeXtLarge(KerasModelBase):
    DEFAULT_SIZE = (224, 224)

class ConvNeXtXLarge(KerasModelBase):
    DEFAULT_SIZE = (224, 224)

class VGG16(KerasModelBase):
    DEFAULT_SIZE = (224, 224)

class VGG19(KerasModelBase):
    DEFAULT_SIZE = (224, 224)

class ResNet50(KerasModelBase):
    DEFAULT_SIZE = (224, 224)

class ResNet50V2(KerasModelBase):
    DEFAULT_SIZE = (224, 224)

class MobileNet(KerasModelBase):
    DEFAULT_SIZE = (224, 224)

class MobileNetV2(KerasModelBase):
    DEFAULT_SIZE = (224, 224)

class MobileNetV3Small(KerasModelBase):
    DEFAULT_SIZE = (224, 224)

class MobileNetV3Large(KerasModelBase):
    DEFAULT_SIZE = (224, 224)

class DenseNet121(KerasModelBase):
    DEFAULT_SIZE = (224, 224)

class NASNetLarge(KerasModelBase):
    DEFAULT_SIZE = (331, 331)

class NASNetMobile(KerasModelBase):
    DEFAULT_SIZE = (224, 224)

class InceptionV3(KerasModelBase):
    DEFAULT_SIZE = (299, 299)

class InceptionResNetV2(KerasModelBase):
    DEFAULT_SIZE = (299, 299)

class Xception(KerasModelBase):
    DEFAULT_SIZE = (299, 299)
