import tensorflow as tf
from tensorflow.keras import layers
import os

from src.config import CONFIG
from .models import workflow

# ---------------------------------------------------------
# 1. Component Builders (Generators & Discriminators)
# ---------------------------------------------------------

def build_generator(input_shape):
    """
    U-Net style generator with skip connections. 
    Excellent for preserving spatial details (like the test spots) 
    while changing the textures (background plastic).
    """
    inputs = layers.Input(shape=input_shape)

    # Downsampling
    d1 = layers.Conv2D(64, (4, 4), strides=2, padding='same')(inputs)
    d1 = layers.LeakyReLU(negative_slope=0.2)(d1)
    
    d2 = layers.Conv2D(128, (4, 4), strides=2, padding='same')(d1)
    d2 = layers.BatchNormalization()(d2)
    d2 = layers.LeakyReLU(negative_slope=0.2)(d2)

    d3 = layers.Conv2D(256, (4, 4), strides=2, padding='same')(d2)
    d3 = layers.BatchNormalization()(d3)
    d3 = layers.LeakyReLU(negative_slope=0.2)(d3)

    # Bottleneck
    b = layers.Conv2D(512, (4, 4), strides=2, padding='same')(d3)
    b = layers.ReLU()(b)

    # Upsampling
    u1 = layers.Conv2DTranspose(256, (4, 4), strides=2, padding='same')(b)
    u1 = layers.BatchNormalization()(u1)
    u1 = layers.ReLU()(u1)
    u1 = layers.Concatenate()([u1, d3]) # Skip connection

    u2 = layers.Conv2DTranspose(128, (4, 4), strides=2, padding='same')(u1)
    u2 = layers.BatchNormalization()(u2)
    u2 = layers.ReLU()(u2)
    u2 = layers.Concatenate()([u2, d2]) # Skip connection

    u3 = layers.Conv2DTranspose(64, (4, 4), strides=2, padding='same')(u2)
    u3 = layers.BatchNormalization()(u3)
    u3 = layers.ReLU()(u3)
    u3 = layers.Concatenate()([u3, d1]) # Skip connection

    # Output Layer (3 channels for RGB, Tanh activation for [-1, 1] scaling)
    outputs = layers.Conv2DTranspose(3, (4, 4), strides=2, padding='same', activation='tanh')(u3)
    
    return tf.keras.Model(inputs, outputs)


def build_discriminator(input_shape):
    """
    PatchGAN Discriminator. 
    Instead of outputting 1 scalar (Real/Fake), it outputs a grid.
    This forces the generator to make the image realistic at a local texture level.
    """
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, (4, 4), strides=2, padding='same')(inputs)
    x = layers.LeakyReLU(negative_slope=0.2)(x)

    x = layers.Conv2D(128, (4, 4), strides=2, padding='same')(x)
    x = layers.GroupNormalization(groups=1)(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)

    x = layers.Conv2D(256, (4, 4), strides=2, padding='same')(x)
    x = layers.GroupNormalization(groups=1)(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)

    # Output a 1-channel grid of Real/Fake predictions
    outputs = layers.Conv2D(1, (4, 4), strides=1, padding='same')(x)

    return tf.keras.Model(inputs, outputs)


# ---------------------------------------------------------
# 2. The Custom Keras Model for CycleGAN Training Loop
# ---------------------------------------------------------

class CycleGAN(tf.keras.Model):
    def __init__(self, gen_G, gen_F, disc_X, disc_Y, lambda_cycle=10.0, lambda_identity=5.0):
        super(CycleGAN, self).__init__()
        self.gen_G = gen_G # Synth -> Real
        self.gen_F = gen_F # Real -> Synth
        self.disc_X = disc_X # Discriminator for Real images
        self.disc_Y = disc_Y # Discriminator for Synth images
        
        # Loss weights
        self.lambda_cycle = lambda_cycle
        
        # CRITICAL: Identity loss weight. This forces the generator to NOT change
        # the color of the spots. If it is given an image that already looks real,
        # it shouldn't modify the colors.
        self.lambda_identity = lambda_identity 

    def compile(self, gen_G_opt, gen_F_opt, disc_X_opt, disc_Y_opt, gen_loss_fn, disc_loss_fn):
        super(CycleGAN, self).compile()
        self.gen_G_optimizer = gen_G_opt
        self.gen_F_optimizer = gen_F_opt
        self.disc_X_optimizer = disc_X_opt
        self.disc_Y_optimizer = disc_Y_opt
        self.gen_loss_fn = gen_loss_fn
        self.disc_loss_fn = disc_loss_fn
        self.cycle_loss_fn = tf.keras.losses.MeanAbsoluteError()
        self.identity_loss_fn = tf.keras.losses.MeanAbsoluteError()

    def train_step(self, batch_data):
        # Unpack the unpaired datasets (batch_data is a tuple of (real_images, synth_images))
        real_x, synth_y = batch_data 

        with tf.GradientTape(persistent=True) as tape:
            # 1. Forward passes
            fake_y = self.gen_G(real_x, training=True) # Synth -> Real
            cycled_x = self.gen_F(fake_y, training=True) # Real -> Synth -> Real

            fake_x = self.gen_F(synth_y, training=True) # Real -> Synth
            cycled_y = self.gen_G(fake_x, training=True) # Synth -> Real -> Synth

            # Identity mapping (CRITICAL FOR PRESERVING TEST SPOT COLORS)
            same_x = self.gen_F(real_x, training=True)
            same_y = self.gen_G(synth_y, training=True)

            # 2. Discriminator outputs
            disc_real_x = self.disc_X(real_x, training=True)
            disc_fake_x = self.disc_X(fake_x, training=True)
            
            disc_real_y = self.disc_Y(synth_y, training=True)
            disc_fake_y = self.disc_Y(fake_y, training=True)

            # 3. Calculate Generator Losses
            gen_G_loss = self.gen_loss_fn(tf.ones_like(disc_fake_y), disc_fake_y)
            gen_F_loss = self.gen_loss_fn(tf.ones_like(disc_fake_x), disc_fake_x)
            
            cycle_loss_G = self.cycle_loss_fn(real_x, cycled_x) * self.lambda_cycle
            cycle_loss_F = self.cycle_loss_fn(synth_y, cycled_y) * self.lambda_cycle
            
            id_loss_G = self.identity_loss_fn(real_x, same_x) * self.lambda_cycle * self.lambda_identity
            id_loss_F = self.identity_loss_fn(synth_y, same_y) * self.lambda_cycle * self.lambda_identity

            total_gen_G_loss = gen_G_loss + cycle_loss_G + id_loss_G
            total_gen_F_loss = gen_F_loss + cycle_loss_F + id_loss_F

            # 4. Calculate Discriminator Losses
            smoothed_real_target_x = tf.ones_like(disc_real_x) * 0.9
            smoothed_real_target_y = tf.ones_like(disc_real_y) * 0.9

            disc_X_loss = self.disc_loss_fn(smoothed_real_target_x, disc_real_x) + \
                          self.disc_loss_fn(tf.zeros_like(disc_fake_x), disc_fake_x)
            
            disc_Y_loss = self.disc_loss_fn(smoothed_real_target_y, disc_real_y) + \
                          self.disc_loss_fn(tf.zeros_like(disc_fake_y), disc_fake_y)

        # 5. Calculate Gradients
        gen_G_grads = tape.gradient(total_gen_G_loss, self.gen_G.trainable_variables)
        gen_F_grads = tape.gradient(total_gen_F_loss, self.gen_F.trainable_variables)
        disc_X_grads = tape.gradient(disc_X_loss, self.disc_X.trainable_variables)
        disc_Y_grads = tape.gradient(disc_Y_loss, self.disc_Y.trainable_variables)

        # 6. Apply Gradients
        self.gen_G_optimizer.apply_gradients(zip(gen_G_grads, self.gen_G.trainable_variables))
        self.gen_F_optimizer.apply_gradients(zip(gen_F_grads, self.gen_F.trainable_variables))
        self.disc_X_optimizer.apply_gradients(zip(disc_X_grads, self.disc_X.trainable_variables))
        self.disc_Y_optimizer.apply_gradients(zip(disc_Y_grads, self.disc_Y.trainable_variables))

        return {
            "G_loss": total_gen_G_loss,
            "F_loss": total_gen_F_loss,
            "D_X_loss": disc_X_loss / 2,
            "D_Y_loss": disc_Y_loss / 2
        }

# ---------------------------------------------------------
# 3. Your Interface (Integrating into your workflow)
# ---------------------------------------------------------

class Sim2RealWorkflow(workflow):
    """
    Workflow wrapper for training the CycleGAN. 
    Inherits from your base workflow class so it plays nicely with your ecosystem.
    """

    def build_model(self):
        # Use GAN_SIZE instead of SIZE
        input_shape = (CONFIG.GAN_SIZE[0], CONFIG.GAN_SIZE[1], 3)
        
        # Instantiate networks
        gen_G = build_generator(input_shape)
        gen_F = build_generator(input_shape)
        disc_X = build_discriminator(input_shape)
        disc_Y = build_discriminator(input_shape)

        self.model = CycleGAN(gen_G, gen_F, disc_X, disc_Y)
        
        self.model.compile(
            gen_G_opt=tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
            gen_F_opt=tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
            disc_X_opt=tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5),
            disc_Y_opt=tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5),
            gen_loss_fn=tf.keras.losses.MeanSquaredError(), 
            disc_loss_fn=tf.keras.losses.MeanSquaredError()
        )
        return self.model

    def _load_gan_domain(self, folder_path):
        """
        Loads images from a folder, resizes to GAN_SIZE, and scales to [-1, 1].
        """
        # Load images directly from the directory (no labels)
        ds = tf.keras.utils.image_dataset_from_directory(
            folder_path,
            labels=None, 
            color_mode='rgb',
            batch_size=CONFIG.GAN_BATCH_N,
            image_size=tuple(CONFIG.GAN_SIZE),
            shuffle=True
        )
        
        # CycleGAN requires inputs scaled to [-1, 1]
        def preprocess(img):
            img = tf.cast(img, tf.float32)
            img = (img / 127.5) - 1.0
            return img
            
        ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Repeat infinitely so the zip function never runs out of pairings
        ds = ds.repeat().prefetch(tf.data.AUTOTUNE)
        return ds

    def train_tf_model(self):
        print("\n--- Preparing Unpaired Datasets from Disk ---")
        
        # 1. Load Real Data (Domain X)
        print(f"Loading Real from: {CONFIG.gan_path_real}")
        real_ds = self._load_gan_domain(CONFIG.gan_path_real)

        # 2. Load Synthetic Data (Domain Y)
        print(f"Loading Synthetic from: {CONFIG.gan_path_synth}")
        synth_ds = self._load_gan_domain(CONFIG.gan_path_synth)

        # 3. Zip them together 
        # The dataset will now yield batches of: (real_batch, synthetic_batch)
        gan_dataset = tf.data.Dataset.zip((real_ds, synth_ds))

        monitor = GANMonitor(
            test_synth_image_path=f"{CONFIG.gan_path_synth}breast_18.png", 
            save_dir=f"{os.getcwd()}/AmpliVision/data/gan_progress/"
        )

        print("\n--- Starting CycleGAN Training ---")
        self.model.fit(
            gan_dataset,
            epochs=CONFIG.GAN_EPOCHS,
            steps_per_epoch=CONFIG.GAN_STEPS_PER_EPOCH,
            callbacks=[monitor]
        )

        self.model.gen_G.save(CONFIG.GAN_SAVE_PATH)

class GANMonitor(tf.keras.callbacks.Callback):
    def __init__(self, test_synth_image_path, save_dir):
        super().__init__()
        self.test_img = self.load_test_image(test_synth_image_path)
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def load_test_image(self, path):
        # Load a single synthetic image, scale to [-1, 1], add batch dimension
        import cv2 as cv
        import numpy as np
        from src.config import CONFIG
        
        img = cv.imread(path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(img, tuple(CONFIG.GAN_SIZE))
        img = (img.astype(np.float32) / 127.5) - 1.0
        return np.expand_dims(img, axis=0)

    def on_epoch_end(self, epoch, logs=None):
        import matplotlib.pyplot as plt
        
        # Pass the synthetic image through the Generator
        prediction = self.model.gen_G(self.test_img, training=False)[0].numpy()
        
        # Rescale back to [0, 1] for saving
        prediction = (prediction + 1.0) / 2.0
        
        # Save the image
        plt.imsave(f"{self.save_dir}/epoch_{epoch+1}.png", prediction)