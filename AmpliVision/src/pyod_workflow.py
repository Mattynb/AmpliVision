
import os
import cv2 as cv    
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from .ML import models
from .OD.outlier_detection import OutlierDetector


def run_pyod_workflow(**kwargs):
   
    # generate a dataset of 110 images with 10% outliers
    BATCH_N = 110
    ds = models.LENET(
        kwargs['TARGETS'],
        kwargs['path_to_imgs'],
        kwargs['scanned_path'],
        kwargs['SIZE'],
        kwargs['BATCH_N'],
        kwargs['EPOCHS'],
        kwargs['BLACK'],
        **kwargs
    ).MLU.build_dataset(kwargs['TARGETS'], BATCH_N, kwargs['SIZE'], kwargs['BLACK'], OUTLIER=True, contamination=0.1)

    # split dataset into images and labels
    imgs = []
    labels = []
    for img, label in ds.take(1):
        imgs.append(img.numpy())
        labels.extend(label.numpy())
        print("img: ", img.shape)
        print("label: ", label)
    print("# of imgs: ", len(imgs))

    # load trained model
    model_path = f"{os.getcwd()}/AmpliVision/data/ML_models/MARKER_FINAL__2024_10_30_23_12_23"
    model = tf.keras.models.load_model(model_path)
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
        """
        vectors.append(
            np.concatenate([
                #features[0].flatten(), 
                #features[1].flatten(),
                features[2]#.flatten()
            ])
        )
        """

    # format vectors and labels as numpy arrays
    vectors = np.array(vectors)
    labels = np.array([array.argmax() for array in labels])


    ## uncoment below if using more than one dense layer
    """
    # use PCA to reduce dimensionality
    from sklearn.decomposition import PCA
    pca = PCA(
        n_components=7,
    )
    vectors = pca.fit_transform(vectors)
    #"""
    
    data = [
        vectors,
        labels
    ]

    # save data
    import pickle as pkl
    with open("pyod_data_110.pkl", "wb") as f:
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
    