
import keras_tuner as kt
from src.ML.models import get_model_builder
from src.config import CONFIG
import tensorflow as tf
import os

    


class TUNING:
    "Hyperparameter tuning using Keras Tuner"

    def __init__(self, build_model_func, train_data, val_data):
        self.build_model_func = build_model_func
        self.train_data = train_data
        self.val_data = val_data

    def run_tuning(self, max_trials=10, executions_per_trial=1):
        tuner = kt.Hyperband(
            self.build_model_func,
            objective='val_accuracy',
            max_epochs=10,
            factor=3,
            directory='tuner_results',
            project_name='amplivision_tuning'
        )

        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

        tuner.search(self.train_data,
                     epochs=10,
                     validation_data=self.val_data,
                     callbacks=[stop_early],
                     max_trials=max_trials,
                     executions_per_trial=executions_per_trial)

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(f"Best hyperparameters: {best_hps.values}")
        return best_hps
    