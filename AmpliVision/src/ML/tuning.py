
import keras_tuner as kt
from src.config import CONFIG
import tensorflow as tf
import os

    


class TUNING:
    "Hyperparameter tuning using Keras Tuner"

    def __init__(self, build_model_func, train_data, val_data):
        self.build_model_func = build_model_func
        self.train_data = train_data
        self.val_data = val_data

    def run_tuning(self):
        tuner = kt.Hyperband(
            self.build_model_func,
            objective='val_accuracy',
            max_epochs=3,
            factor=3,
            directory='tuner_results',
            project_name='amplivision_tuning',
            #distribution_strategy=tf.distribute.MirroredStrategy(),
        )

        hp_ = kt.HyperParameters()
        hp_.Int('batch_size', min_value=16, max_value=128, step=16)
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
        
        tuner.search(
            self.train_data,
            epochs=10,
            verbose=1,
            initial_epoch=0,
            validation_data=self.val_data,
            callbacks=[stop_early],
            batch_size=hp_.get('batch_size')
        )

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(f"Best hyperparameters: {best_hps.values}")

        tuner.results_summary()
        return best_hps
    