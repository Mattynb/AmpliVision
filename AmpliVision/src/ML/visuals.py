import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model # Import Model for layer output extraction
from src.config import CONFIG

def visualize_feature_maps(model: tf.keras.Model, img_path: str, target_size: tuple):
    """
    Visualizes the feature maps (filter outputs) from the convolutional layers
    of a Keras model when a single image is passed through it.

    Args:
        model (tf.keras.Model): The built and trained Keras model.
        img_path (str): File path to the image to visualize.
        target_size (tuple): The (height, width) size the image should be
                             resized to before inputting to the model (e.g., (1024, 1024)).
    """
    # --- 1. Preprocess the Input Image ---
    try:
        # Load and preprocess the image
        img = tf.keras.utils.load_img(img_path, target_size=target_size)
        img_array = tf.keras.utils.img_to_array(img)
        # Add a batch dimension (1, H, W, C)
        img_tensor = np.expand_dims(img_array, axis=0)
        # Normalize the image (if the model was trained with normalized inputs)
        # Assuming normalization to [0, 1] for typical models
        img_tensor = img_tensor / 255.0 
    except Exception as e:
        print(f"Error loading or preprocessing image: {e}")
        return

    # --- 2. Extract Outputs of Convolutional Layers ---
    # Find all Conv2D layer names
    layer_names = [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    
    if not layer_names:
        print("No Conv2D layers found in the model to visualize.")
        return

    # Create a new model that outputs the activation maps for all selected layers
    feature_map_outputs = [model.get_layer(name).output for name in layer_names]
    activation_model = Model(inputs=model.inputs, outputs=feature_map_outputs)
    
    # Get the feature maps for the input image
    feature_maps = activation_model.predict(img_tensor)
    
    # Ensure feature_maps is a list even if there's only one Conv2D layer
    if not isinstance(feature_maps, list):
        feature_maps = [feature_maps]

    # --- 3. Visualize the Feature Maps ---
    plt.figure(figsize=(15, 15))
    
    for i, fmaps in enumerate(feature_maps):
        layer_name = layer_names[i]
        # fmaps has shape (1, H, W, F), where F is the number of filters
        n_features = fmaps.shape[-1]
        
        # Calculate grid size for display (e.g., 6 filters -> 2x3 grid)
        # Find the smallest square-like grid
        size = fmaps.shape[1] # H
        display_grid = int(np.ceil(np.sqrt(n_features)))
        
        fig, axes = plt.subplots(display_grid, display_grid, figsize=(display_grid * 2.5, display_grid * 2.5))
        plt.suptitle(f"Feature Maps for Layer: **{layer_name}** ({n_features} filters)\nOutput Size: {size}x{size}", 
                     fontsize=16, y=1.02)
        
        for filter_index in range(n_features):
            # Extract the activation for the specific filter
            ax = axes.flat[filter_index]
            
            # The map is of shape (H, W), removing the batch and filter dimensions
            channel_image = fmaps[0, :, :, filter_index]
            
            # Post-process for visualization:
            # Normalize to avoid extreme values and make visualization better
            # You might need to adjust this depending on the activation function
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std() if channel_image.std() > 0 else 1e-6
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            
            # Display the activation map
            ax.imshow(channel_image, cmap='viridis') # 'viridis' is a good default color map
            ax.set_title(f"Filter {filter_index + 1}", fontsize=10)
            ax.axis('off')
            
        # Hide any unused subplots
        for j in range(n_features, display_grid * display_grid):
            axes.flat[j].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout for suptitle
        fig.savefig(f"{CONFIG.model_name}_2_feature_maps_{layer_name}.png", dpi=150)

# Example of how to use the function (REQUIRES a built/trained model and an image)
# Note: You need to have the LENET class, a trained model instance, and an image file.

# Assuming you have an instance of your trained model:
# lenet_instance = LENET()
# lenet_instance.SIZE = (1024, 1024) 
# lenet_instance.build_model()
# model = lenet_instance.model # Get the Keras Sequential model

# # Load weights if necessary (model.load_weights(...))

# # Example Usage:
# # visualize_feature_maps(
# #     model=model, 
# #     img_path='/path/to/your/test_image.jpg', 
# #     target_size=(1024, 1024)
# # )