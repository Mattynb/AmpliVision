import os
import pickle as pkl
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from src.config import CONFIG

from .utils import ML_Utils
from .evaluator import test_model_generated, test_model_
from ..generators.image_generation.RuleBasedGenerator import RuleBasedGenerator as RBG


class workflow:
    """ Workflow parent class for all implemented ML models """
    def __init__(self):
        pass
        #ML_Utils()#.test_dataset()

    def train_model(self):
        """ Train the model """
        if self.PACKAGE == "tensorflow":
            self.train_tf_model()
        elif self.PACKAGE == "sklearn":
            self.train_sklearn_model()
        else:
            raise NotImplementedError(f"Training not implemented for package: {CONFIG.PACKAGE}")

    def train_sklearn_model(self): 
        """ Train a sklearn model """
        ds = self.MLU.build_dataset(CONFIG.BATCH_N, CONFIG.SIZE, generator_only=True)
        n = 25000 #CONFIG.BATCH_N * CONFIG.STEPS_PER_EPOCH * CONFIG.EPOCHS

        def _gen():
            X_list = []
            y_list = []
            print(f"\nGenerating {n} training data points...") 
            import time
            it = time.time()
            for i, (img, label) in enumerate(ds.generate(CONFIG.TARGETS)):

                X_list.append(img)
                y_list.append(label)
                if (i+1) % 10 == 0:
                    print(f"{i+1}/{n} samples generated. Time elapsed: {time.time() - it:.2f} seconds")
                    it = time.time()
                if i+1 >= n:
                    break

            print(f"Generated {len(X_list)} samples for training.\n")

        def _load():
            X_list = []
            y_list = []
            # loads PNG images as X_list. And assigns labels to y_list based on image names.
            print("\nLoading training data from disk...")
            import time
            it = time.time()
            for img_name in os.listdir(CONFIG.path_to_store):
                if img_name.endswith('.png'):
                    img_path = os.path.join(CONFIG.path_to_store, img_name)
                    img = plt.imread(img_path)
                    X_list.append(img)

                    target_label = img_name.split('_')[0]
                    y_list.append(CONFIG.TARGETS.index(target_label))
            print(f"Loaded images. Time elapsed: {time.time() - it:.2f} seconds")
            it = time.time()

            return X_list, y_list
            
                
        X_list, y_list = _load()


        # 0. Preprocess data if needed (flatten images for RandomForest)
        X = np.array(X_list)
        #X_flat = X.reshape(X.shape[0], -1)  
        # Flatten images with PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=100)  # Adjust n_components as needed
        X_flat = pca.fit_transform(X.reshape(X.shape[0], -1))
        y = np.array(y_list)
        #y = np.argmax(y, axis=1)
        
        print(f"training data shape: {X_flat.shape}, labels shape: {y.shape}")

        # 1. Split the data
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_flat, y, test_size=0.2, random_state=42
        )

        # 2. Fit the model
        self.model.fit(X_train, y_train)

        # 3. Evaluate
        val_accuracy = self.model.score(X_val, y_val)
        print(f"Validation Accuracy: {val_accuracy*100:.2f}%")
        print("Training completed.")


    def train_tf_model(self):    
        import time

        
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.abspath(f"{os.getcwd()}/AmpliVision/data/ML_models/{CONFIG.TAG}.keras"),
            save_weights_only=False,
            monitor='val_accuracy',
            mode='auto',
            save_best_only=True, 
            save_freq='epoch',
        )

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            mode='max',
            patience=5,
            min_delta=0.001,
            restore_best_weights=True
        )

        callbacks = [
            self.MLU.PlotCallback,
            early_stop,
            checkpoint
        ]

        if CONFIG.TRAIN_DATASET == "GEN":
            print("\nUsing generated dataset for training...")
            train_dataset = self.MLU.build_dataset(CONFIG.BATCH_N, CONFIG.SIZE, Keras_Preprocess=isinstance(self, KerasModelBase), BLACK=CONFIG.BLACK)
            validate_dataset = self.MLU.build_dataset(int((CONFIG.BATCH_N*0.5)), CONFIG.SIZE, Keras_Preprocess=isinstance(self, KerasModelBase))

        elif CONFIG.TRAIN_DATASET == "LOAD":
            print("\nUsing loaded dataset for training...")
            train_dataset, validate_dataset = self.MLU.load_dataset(Keras_Preprocess=isinstance(self, KerasModelBase))
        
        inference_time = time.time()
        with tf.device('/GPU:0'):
            #self.history = 
            self.model.fit(
                train_dataset,
                epochs=CONFIG.EPOCHS,
                validation_data=validate_dataset,
                steps_per_epoch=CONFIG.STEPS_PER_EPOCH,
                validation_steps=CONFIG.VALIDATION_STEPS,
                callbacks = callbacks
            )
        inference_time = time.time() - inference_time
        print(f"Training completed in: {inference_time/60:.2f} minutes")    

        # Save model
        self.model.save(f"{os.getcwd()}/AmpliVision/data/ML_models/{CONFIG.TAG}.keras")

        # Save history
        #with open(f"{os.getcwd()}/AmpliVision/data/ML_models/history_{model_save_name}.pkl", 'wb') as file_pi:
        #    pkl.dump(self.history.history, file_pi)

        return self.model

    def test_model(self):
        """ Test a trained model """

        path = f"{os.getcwd()}/AmpliVision/data/ML_models/{CONFIG.TAG}.keras"
        #model = tf.keras.models.load_model(path)
        #model = self.model

        # Check if model is in memory, otherwise safely load with the REAL function
        if hasattr(self, 'model') and self.model is not None:
            print("\n--- Using in-memory trained model ---")
            model = self.model
        else:
            print("\n--- Loading model from disk safely ---")
            # Fetch the actual Keras preprocessing function for this specific model
            import inspect
            model_constructor = get_model(CONFIG.model_name)
            real_preprocess_function = inspect.getmodule(model_constructor).preprocess_input
            
            # Inject it into the custom_objects registry during load
            model = tf.keras.models.load_model(
                path, 
                custom_objects={'preprocess_input': real_preprocess_function}
            )

        print(model.summary())

        # real data
        print("\n\n--- Testing model with SCANNED images ---\n")
        CONFIG.TEST_DATASET = "MARKER"
        test_model_(model)

        # synthetic 
        CONFIG.BATCH_N = 7 *10
        dataset =  self.MLU.build_dataset(CONFIG.BATCH_N, CONFIG.SIZE, Keras_Preprocess=isinstance(self, KerasModelBase))

        print(f"\n\n--- Testing model with {CONFIG.BATCH_N} GENERATED images ---\n")
        CONFIG.TEST_DATASET = "GENERATED"
        test_model_generated(dataset, model)


    def run(self):
        self.build_model()
        self.train_model()
        self.test_model()
        return self.model # for chaining

    

class LENET(workflow):
    PACKAGE = "tensorflow"
    DEFAULT_SIZE = (1024, 1024)  # Default size for LENET if not specified

    def __init__(self):
        super().__init__() 
        #f"{kwargs['TARGET_NAME']}{CONFIG.model_name}_Sz{CONFIG.SIZE}_Bn{CONFIG.BATCH_N}_Ep{CONFIG.EPOCHS}{'_BLACK' if CONFIG.BLACK else ''}"
        self.MLU = ML_Utils()
    

    def build_model(self):
        model = tf.keras.Sequential()

        # Layer 1: Convolutional + Average Pooling (square)
        model.add(tf.keras.layers.Conv2D(
            filters=6, 
            kernel_size=(3, 3), 
            activation='relu', 
            input_shape=tuple([CONFIG.SIZE[0], CONFIG.SIZE[1], 3]))
        )
        model.add(tf.keras.layers.AveragePooling2D((2,2)))
        
        # Layer 2: Convolutional + Average Pooling ()
        model.add(tf.keras.layers.Conv2D(filters=15, kernel_size=(3, 3), activation='relu'))
        model.add(tf.keras.layers.AveragePooling2D((2,2)))
        
        model.add(tf.keras.layers.GlobalAveragePooling2D())
        
        model.add(tf.keras.layers.Dense(units=64, activation='relu'))
        model.add(tf.keras.layers.Dense(units=96, activation='relu'))
        model.add(tf.keras.layers.Dense(units=len(CONFIG.TARGETS), activation = 'softmax'))
        
        model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
            loss = 'categorical_crossentropy',
            metrics = [
                'accuracy', 
                'AUC',
                tf.keras.metrics.F1Score(average='macro', threshold=None, name='f1_score', dtype=None)
            ]
        )
        model.summary()
        self.model = model  
        return model

    def model_builder(self, hp):
        """
        Builds and compiles a Keras Sequential model for Keras Tuner hyperparameter search.
        
        Args:
            hp (kerastuner.HyperParameters): The hyperparameter search space object."""
        
        model = tf.keras.Sequential()

        # --- 1. Tunable Convolutional Block 1 ---
        model.add(tf.keras.layers.Conv2D(
            # Tune the number of filters in the first layer (6 in original)
            filters=hp.Int('filters_1', min_value=1, max_value=12, step=1),
            kernel_size=(3, 3), 
            activation='relu', 
            input_shape=tuple([CONFIG.SIZE[0], CONFIG.SIZE[1], 3])
        ))
        model.add(tf.keras.layers.AveragePooling2D((2,2)))
        
        # --- 2. Tunable Convolutional Block 2 ---
        model.add(tf.keras.layers.Conv2D(
            # Tune the number of filters in the second layer (16 in original)
            filters=hp.Int('filters_2', min_value=3, max_value=15, step=3), 
            kernel_size=(3, 3),
            activation='relu'
        ))
        model.add(tf.keras.layers.AveragePooling2D((2,2)))
        
        model.add(tf.keras.layers.GlobalAveragePooling2D())

        # --- 3. Tunable Dense Layers ---
        model.add(tf.keras.layers.Dense(
            # Tune the size of the first dense layer (120 in original)
            units=hp.Int('dense_units_1', min_value=32, max_value=256, step=32), 
            activation='relu')
        )   
        model.add(tf.keras.layers.Dense(units=hp.Int('dense_units_2', min_value=32, max_value=120, step=32), activation='relu'))
        model.add(tf.keras.layers.Dense(units=len(CONFIG.TARGETS), activation='softmax'))
        
        # --- 4. Tunable Optimization Parameters ---
        # Tune the learning rate for the Adam optimizer
        learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy', 
                'AUC',
                tf.keras.metrics.F1Score(average='macro', threshold=None, name='f1_score', dtype=None)
            ]
        )
        
        return model
    


# You can implement other models bellow based on the LENET model class

# Basic Architectures:
# AlexNet
class ALEXNET(workflow):

    PACKAGE="tensorflow"
    DEFAULT_SIZE = (227, 227)  # AlexNet standard input size

    def __init__(self):
        super().__init__() 
        #f"{kwargs['TARGET_NAME']}{CONFIG.model_name}_Sz{CONFIG.SIZE}_Bn{CONFIG.BATCH_N}_Ep{CONFIG.EPOCHS}{'_BLACK' if CONFIG.BLACK else ''}"
        self.MLU = ML_Utils()


    def build_model(self):
        model = tf.keras.Sequential()

        # 1% 
        model.add(tf.keras.layers.Conv2D(
            filters=3, 
            kernel_size=(3,3),
            padding='valid',
            activation='relu', 
            input_shape=tuple([CONFIG.SIZE[0], CONFIG.SIZE[1], 3]))
        )
        model.add(tf.keras.layers.MaxPool2D((2, 2)))

        model.add(tf.keras.layers.Conv2D(
            filters=3, 
            kernel_size=(3, 3),
            strides=(1,1),
            activation='relu',
            padding='same') # same size as previous layer
        )
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))

        model.add(tf.keras.layers.Flatten())
        # conect to the flattened layer shape
        model.add(tf.keras.layers.Dense(units=120, activation='relu'))
        #model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(units=84, activation='relu'))
        #model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(units=len(CONFIG.TARGETS), activation = 'softmax'))
        
        model.compile(
            optimizer = "adam", #tf.keras.optimizers.Adam(learning_rate=0.001),
            loss = 'categorical_crossentropy',
            metrics = [
                'accuracy', 
                'AUC',
                tf.keras.metrics.F1Score(average='macro', threshold=None, name='f1_score', dtype=None)

            ]
        )
        model.summary()
        self.model = model  
        return model


    def model_builder(self, hp):
        """
        Builds and compiles a Keras Sequential model for Keras Tuner hyperparameter search.
        
        Args:
            hp (kerastuner.HyperParameters): The hyperparameter search space object.
            input_shape (tuple): The shape of the input image (H, W, 3).
            
        Returns:
            tf.keras.Model: The compiled Keras model.
        """
        model = tf.keras.Sequential()

        # --- 1. Tunable Convolutional Block 1 ---
        model.add(tf.keras.layers.Conv2D(
            # Tune the number of filters in the first layer (96 in original)
            filters=hp.Int('filters_1', min_value=64, max_value=128, step=32),
            kernel_size=(11, 11), 
            strides=(4,4),
            activation='relu', 
            input_shape=tuple([CONFIG.SIZE[0], CONFIG.SIZE[1], 3])
        ))
        model.add(tf.keras.layers.MaxPool2D((2, 2)))

        # --- 2. Tunable Convolutional Block 2 ---
        model.add(tf.keras.layers.Conv2D(
            # Tune the number of filters in the second layer (16 in original)
            filters=hp.Int('filters_2', min_value=16, max_value=64, step=16), 
            kernel_size=(3, 3),
            activation='relu'
        ))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))

        # --- 3. Fixed/Semi-Tunable Deeper Convs (Example of fixed architecture) ---
        # The original model structure is kept, but could be made tunable if desired
        model.add(tf.keras.layers.Conv2D(filters=384, kernel_size=(2, 2), strides=(1, 1), activation='relu'))
        model.add(tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
        model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Flatten())

        # --- 4. Tunable Dense Layers ---
        model.add(tf.keras.layers.Dense(
            # Tune the size of the first dense layer (120 in original)
            units=hp.Int('dense_units_1', min_value=64, max_value=256, step=32), 
            activation='relu')
        )
        # Add optional/tunable Dropout
        if hp.Boolean("dropout_1"):
            model.add(tf.keras.layers.Dropout(
                hp.Float('dropout_rate_1', min_value=0.2, max_value=0.5, step=0.1))
            )
            
        model.add(tf.keras.layers.Dense(units=84, activation='relu'))
        model.add(tf.keras.layers.Dense(units=len(CONFIG.TARGETS), activation='softmax'))
        
        # --- 5. Tunable Optimization Parameters ---
        # Tune the learning rate for the Adam optimizer
        learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy', 
                'AUC',
                tf.keras.metrics.F1Score(average='macro', threshold=None, name='f1_score', dtype=None)
            ]
        )
        
        return model


class RandomForest(workflow):
    """ Random Forest Classifier Model """

    PACKAGE = "sklearn"
    DEFAULT_SIZE = CONFIG.SIZE

    def __init__(self):
        super().__init__() 
        self.MLU = ML_Utils()
         
    def build_model(self):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(
            n_estimators=100,
            criterion='gini',
            max_depth=None,
            random_state=42
        )

#************#
"""
@tf.keras.utils.register_keras_serializable()
def preprocess_input(x):
    #Placeholder function required for Keras Lambda layer deserialization.
    # Note: This specific function is never executed. 
    # The actual preprocessing is handled by the dynamic function inside 
    # the elegant_keras_applications_constructor logic.
    return x
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
class KerasModelBase(workflow):
    """
    Base class to dynamically load and implement any Keras Application model.
    Child classes only need to DEFAULT_SIZE.
    """

    PACKAGE = "tensorflow"
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

        input_shape = tuple([CONFIG.SIZE[0], CONFIG.SIZE[1], 3])

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
        
        if not include_top:
            x = GlobalAveragePooling2D()(base_model.output)
            x = Dense(120, activation='relu')(x)
            x = Dropout(0.2)(x)
            x = Dense(84, activation='relu')(x)
            x = Dropout(0.2)(x)
            predictions = Dense(len(CONFIG.TARGETS), activation='softmax')(x)
            model = Model(inputs=base_model.input, outputs=predictions)
        else:
            model = base_model
        
        # remove learning_rate 
        model_params = CONFIG.MODEL_PARAMS.copy()
        model_params.pop("learning_rate", None)
        print(f"Compiling model with parameters: {model_params}")
        model.compile(**model_params)

        model.summary()
        self.model = model
        return model


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
        layer.trainable = True

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
