
import os
import cv2 as cv    
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from .config import CONFIG
from .ML.utils import ML_Utils
from .OD.outlier_detection import OutlierDetector


def run_pyod_workflow(model, keras_preprocess=False):
    # 1. Increase batch size for better PyOD clustering
    CONFIG.BATCH_N = 500
    print("Changed BATCH_N to: ", CONFIG.BATCH_N, " for PyOD workflow")

    ds = ML_Utils().build_dataset(
        BATCH_N=CONFIG.BATCH_N, 
        OUTLIER=True, 
        contamination=0.1,
        Keras_Preprocess=keras_preprocess
    )

    # 2. Safely grab the penultimate layer (features before softmax)
    #penultimate_layer = model.layers[-2]
    print(f"Extracting features from layer: global_average_pooling2d") #{penultimate_layer.name}")
    
    clf_features = tf.keras.Model(
        inputs=model.inputs, 
        outputs=model.get_layer('global_average_pooling2d').output        #penultimate_layer.output
    )

    vectors = []
    labels = []
    
    # 3. FIX RETRACING: Predict on the whole batch at once directly from the dataset
    for x_batch, y_batch in ds.take(1):
        # Passes the (500, W, H, C) tensor directly to Keras
        features = clf_features.predict(x_batch, batch_size=CONFIG.BATCH_N)
        vectors.extend(features)
        labels.extend(y_batch.numpy())

    # Format vectors as a numpy array
    vectors = np.array(vectors)
    
    # 4. FIX SKLEARN ERROR: Convert one-hot labels ([1,0]) back to flat 1D integers (0 or 1)
    labels = np.array([np.argmax(label) for label in labels])

    # Scale the features for distance-based algorithms
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    vectors = scaler.fit_transform(vectors)

    # Reduce dimensionality
    from sklearn.decomposition import PCA
    # 15-20 components is usually the sweet spot for PyOD over Keras features
    pca = PCA(n_components=15, random_state=42) 
    vectors = pca.fit_transform(vectors)

    data = [
        vectors,
        labels
    ]

    # Save data
    import pickle as pkl
    with open(f"{os.getcwd()}/AmpliVision/data/pyod/pyod_data_{CONFIG.BATCH_N}{CONFIG.SAVE_NAME}.pkl", "wb") as f:
        pkl.dump(data, f)

    # Run outlier detection
    OutlierDetector(
        run_id = 1,
        data = data,
        features = None,
        bad_ids = None,
        number_bad = None,
        exclude = None,
        timing = False,
    )
    """
    # generate a dataset of 110 images with 10% outliers
    CONFIG.BATCH_N = 110
    print("Changed BATCH_N to: ", CONFIG.BATCH_N, " for PyOD workflow")

    ds = ML_Utils().build_dataset(
        BATCH_N=CONFIG.BATCH_N, 
        OUTLIER=True, 
        contamination=0.1,
        Keras_Preprocess=keras_preprocess
    )
    
    imgs = []
    labels = []
    for img, label in ds.take(1):
        imgs.append(img.numpy())
        labels.extend(label.numpy())
    print("# of imgs: ", len(imgs[0]))

    # 4. Safely grab the penultimate layer (the last feature layer before softmax)
    penultimate_layer = model.layers[-2]
    print(f"Extracting features from layer: {penultimate_layer.name}")
    
    clf = tf.keras.Model(
        inputs=model.inputs, 
        outputs=penultimate_layer.output
    )

    # get features for each image 
    vectors = []
    for i in range(len(imgs)):
        img = imgs[i]
        features = clf.predict(img)
        vectors.extend(features)
    
    ""
    # split dataset into images and labels
    imgs = []
    labels = []
    for img, label in ds.take(1):
        imgs.append(img.numpy())
        labels.extend(label.numpy())
        #print("img: ", img.shape)
        #print("label: ", label)
    print("# of imgs: ", len(imgs))

    # load trained model
    #model_path = f"{os.getcwd()}/AmpliVision/data/ML_models/" + CONFIG.TAG + ".keras"
    #model = tf.keras.models.load_model(model_path)
    print("model: ", model.summary())

    # get the last 3 dense layers of the model 
    layer_names = [layer.name for layer in model.layers[ -3 :  ]]
    print(layer_names)
    clf = tf.keras.Model(
        inputs=model.inputs, 
        outputs=[
            model.get_layer(layer_names[0]).output, 
            model.get_layer(layer_names[1]).output,
            model.get_layer(layer_names[2]).output
        ]
    )

    # get features for each image using the dense layers
    vectors = []
    for i in range(len(imgs)):
        img = imgs[i]
        
        features = clf.predict(img)
        ##vectors.extend(features[0])
        vectors.extend(features[1])
        ##vectors.extend(features[2])
        
        ## uncoment below if using more than one dense layer
        
        #vectors.append(
        #    np.concatenate([
                #features[0].flatten(), 
                #features[1].flatten(),
        #        features[2]#.flatten()
        #    ])
        #)

    # format vectors and labels as numpy arrays
    vectors = np.array(vectors)
    labels = np.array([array.argmax() for array in labels])


    ## uncoment below if using more than one dense layer
    # use PCA to reduce dimensionality
    #from sklearn.decomposition import PCA
    #pca = PCA(n_components=7)
    #vectors = pca.fit_transform(vectors)
    
    
    data = [
        vectors,
        labels
    ]

    # save data
    import pickle as pkl
    with open(f"{os.getcwd()}/AmpliVision/data/pyod/pyod_data_110{CONFIG.SAVE_NAME}.pkl", "wb") as f:
        pkl.dump(data, f)

    # run outlier detection
    OutlierDetector(
        run_id = 1,
        #algorithms = ["PCA"],
        data = data,
        features = None,
        bad_ids = None,
        number_bad = None,
        exclude = None,
        timing = False,
    )
    """


def generate_dendrogram(self, ds):
    """
    Work in progress - Meant to create a dendrogram of feature vectors where the closest vectors are closer together
    
    Right now it can create dendragram but needs better features to be useful.
    """

    from scipy.cluster.hierarchy import dendrogram, linkage

    # Ensure that vectors are in numpy array format
    vectors = np.array(vectors)

    # Step 1: Apply hierarchical clustering using linkage
    # Here we use 'ward' method, which minimizes the variance within clusters
    linked = linkage(vectors, method='ward')

    # Step 2: Plot the dendrogram
    plt.figure(figsize=(10, 7))
    dendrogram(
        linked,
        orientation='right',
        labels=[k.removesuffix("png").strip("_0123456789") for k in ds.keys()],  # Use labels for each sample if available
        distance_sort='descending',
        show_leaf_counts=True
    )
    plt.title("Dendrogram of Feature Vectors")
    plt.xlabel("Sample Index")
    plt.ylabel("Distance")
    plt.savefig("dendra")
    