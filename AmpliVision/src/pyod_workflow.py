import os
import pickle as pkl
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from .config import CONFIG
from .ML.utils import ML_Utils
from .OD.outlier_detection import OutlierDetector


def generate_dendrogram(vectors, labels=None):
    """
    Creates a dendrogram of feature vectors using Ward hierarchical clustering.
    """
    from scipy.cluster.hierarchy import dendrogram, linkage

    vectors = np.array(vectors)
    linked = linkage(vectors, method='ward')

    plt.figure(figsize=(12, 8))
    
    # Optional sample labeling
    sample_labels = (
        [f"({'Outlier' if l == 1 else 'Inlier'})" for i, l in enumerate(labels)]
        if labels is not None
        else None
    )

    print("sample_labels:", sample_labels)

    dendrogram(
        linked,
        orientation='right',
        labels=sample_labels,
        distance_sort='descending',
        show_leaf_counts=True
    )
    plt.title("Dendrogram of Feature Vectors")
    plt.xlabel("Distance")
    plt.ylabel("Sample Index")
    plt.tight_layout()
    plt.savefig("dendrogram.png")
    plt.close()
    print("Saved dendrogram plot to dendrogram.png")


def run_pyod_workflow(model, keras_preprocess=False):
    # 1. Increase batch size for better PyOD clustering
    print("Changed BATCH_N to: ", CONFIG.PYOD_BATCH_N, " for PyOD workflow")

    ds = ML_Utils().build_dataset(
        BATCH_N=CONFIG.PYOD_BATCH_N, 
        OUTLIER=True, 
        contamination=CONFIG.CONTAMINATION,
        Keras_Preprocess=keras_preprocess
    )

    print("Extracting features from layer: global_average_pooling2d")
    clf_features = tf.keras.Model(
        inputs=model.inputs, 
        outputs=model.get_layer('global_average_pooling2d').output
    )

    vectors = []
    labels = []
    
    # Predict on the whole batch at once directly from dataset
    for x_batch, y_batch in ds.take(1):
        features = clf_features.predict(x_batch, batch_size=CONFIG.PYOD_BATCH_N)
        vectors.extend(features)
        labels.extend(y_batch.numpy())

    vectors = np.array(vectors)
    labels = np.array([np.argmax(label) for label in labels])

    # Scale the features for distance-based algorithms
    scaler = StandardScaler()
    vectors = scaler.fit_transform(vectors)

    # --- GENERATE DENDROGRAM ---
    # Called on scaled feature vectors before PCA squashing
    generate_dendrogram(vectors, labels=labels)

    # Dynamic PCA Component selection based on explained variance ratio (~92%)
    # pca_full = PCA(random_state=42).fit(vectors)
    # cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    # optimal_components = min(int(np.argmax(cum_var >= 0.92) + 1), 64)
    # print(f"PCA reduced components to {optimal_components} (retaining >= {cum_var * 100}% explained variance) or {64}, whichever is smaller")

    # pca = PCA(n_components=optimal_components, random_state=42) 
    # vectors = pca.fit_transform(vectors)

    data = [vectors, labels]

    # Save data reliably using absolute paths relative to script root
    base_dir = Path(__file__).resolve().parent.parent
    save_path = base_dir / "data" / "pyod" / f"pyod_data_{CONFIG.PYOD_BATCH_N}{CONFIG.SAVE_NAME}.pkl"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "wb") as f:
        pkl.dump(data, f)

    # Run outlier detection
    OutlierDetector(
        run_id=1,
        data=data,
        features=None,
        bad_ids=None,
        number_bad=None,
        exclude=None,
        timing=False,
        contamination=CONFIG.CONTAMINATION,
    )