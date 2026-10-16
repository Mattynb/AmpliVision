""" CODE SOURCED AND/OR BASED ON: https://github.com/mpsych/ODM/ """

import time
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    f1_score,
    accuracy_score,
    jaccard_score,
    precision_score,
    average_precision_score,
    recall_score,
    hamming_loss,
    log_loss,
)

# Debug flag moved to the top of the module
DEBUG = False

# Pruned slow/weak performers (e.g. AnoGAN) to focus on distance/neighbor-based methods
ALGORITHMS = [
    "KNN",
    "AvgKNN",
    "MedKNN",
    "COF",
    "CBLOF",
    "LOF",
    "IForest",
    "HBOS",
    "COPOD",
    "ECOD",
    "PCA",
    "LODA",
    "Sampling",
]

BAD_IMAGE_ID_PATH = ""


class OutlierDetector:
    def __init__(
        self,
        run_id,
        algorithms=None,
        data=None,
        features=None,
        bad_ids=None,
        number_bad=None,
        exclude=None,
        timing=False,
        contamination=0.1,
        **kwargs,
    ):
        data_x = data[0]  # imgs or vectors
        data_y = data[1]  # labels

        t0 = time.time()
        self.__run_id = run_id
        self.__algorithms = ALGORITHMS if algorithms is None else algorithms
        self.__bad_ids = bad_ids if bad_ids is not None else BAD_IMAGE_ID_PATH
        self.__number_bad = number_bad if number_bad is not None else None
        self.__exclude = exclude if exclude is not None else []
        self.__contamination = contamination

        accuracy_scores = {}
        errors = {}
        verbose = kwargs.get("verbose", False)

        for alg in self.__algorithms:
            if alg not in self.__exclude:
                print("-" * 20, f" Running {alg} ", "-" * 20)
                try:
                    t_scores, t_labels, ev = self._detect_outliers(
                        data_x,
                        data_y,
                        pyod_algorithm=alg,
                        verbose=verbose,
                        contamination=self.__contamination,
                        **kwargs,
                    )
                    # Leaderboard tracking: storing F1-Score as primary ranking metric
                    accuracy_scores[alg] = ev["f1_score"]
                except Exception as e:
                    print(f"Error running {alg}: {e}")
                    errors[alg] = str(e)

                print("-" * 20, f" {alg} DONE ", "-" * 20)

        # Leaderboard sorting and printing
        accuracy_scores = dict(
            sorted(accuracy_scores.items(), key=lambda item: item[1], reverse=True)
        )

        print("\n" + "=" * 20 + " LEADERBOARD (Ranked by F1-Score) " + "=" * 20)
        print(f"{'Rank':<6}{'Algorithm':<20}{'F1-Score':<10}")
        print("-" * 36)
        for rank, (alg, f1) in enumerate(accuracy_scores.items(), 1):
            print(f"{rank:<6}{alg:<20}{f1 * 100:.2f}%")
        print("=" * 56 + "\n")

    @staticmethod
    def _detect_outliers(features, labels, pyod_algorithm, contamination=0.1, **kwargs):
        """Detect outliers using PyOD algorithm."""

        # Reshape array helper function if inputs arrive improperly dimensioned
        if isinstance(features, np.ndarray):
            if len(features.shape) == 1:
                features = features.reshape(-1, 1)
            elif len(features.shape) == 3:
                features = features.reshape(features.shape[0], -1)

        # Split data for evaluation
        x_train, x_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )

        return_decision_function = kwargs.get("return_decision_function", False)

        match pyod_algorithm:
            case "ECOD":
                from pyod.models.ecod import ECOD
                n_jobs = kwargs.get("n_jobs", 1)
                clf = ECOD(n_jobs=n_jobs, contamination=contamination)

            case "LOF":
                from pyod.models.lof import LOF
                if "LOF" in kwargs:
                    clf = LOF(**kwargs["LOF"])
                else:
                    n_neighbors = kwargs.get("n_neighbors", 20)
                    algorithm = kwargs.get("algorithm", "auto")
                    leaf_size = kwargs.get("leaf_size", 30)
                    metric = kwargs.get("metric", "minkowski")
                    p = kwargs.get("p", 2)
                    novelty = kwargs.get("novelty", True)
                    n_jobs = kwargs.get("n_jobs", 1)
                    metric_params = kwargs.get("metric_params", None)
                    clf = LOF(
                        n_neighbors=n_neighbors,
                        algorithm=algorithm,
                        leaf_size=leaf_size,
                        metric=metric,
                        p=p,
                        metric_params=metric_params,
                        contamination=contamination,
                        n_jobs=n_jobs,
                        novelty=novelty,
                    )

            case "OCSVM":
                from pyod.models.ocsvm import OCSVM
                if "OCSVM" in kwargs:
                    clf = OCSVM(**kwargs["OCSVM"])
                else:
                    kernel = kwargs.get("kernel", "rbf")
                    degree = kwargs.get("degree", 3)
                    gamma = kwargs.get("gamma", "auto")
                    coef0 = kwargs.get("coef0", 0.0)
                    tol = kwargs.get("tol", 1e-3)
                    nu = kwargs.get("nu", 0.5)
                    shrinking = kwargs.get("shrinking", True)
                    cache_size = kwargs.get("cache_size", 200)
                    verbose = kwargs.get("verbose", False)
                    max_iter = kwargs.get("max_iter", -1)
                    clf = OCSVM(
                        kernel=kernel,
                        degree=degree,
                        gamma=gamma,
                        coef0=coef0,
                        tol=tol,
                        nu=nu,
                        shrinking=shrinking,
                        cache_size=cache_size,
                        verbose=verbose,
                        max_iter=max_iter,
                        contamination=contamination,
                    )

            case "IForest":
                from pyod.models.iforest import IForest
                if "IForest" in kwargs:
                    clf = IForest(**kwargs["IForest"])
                else:
                    n_estimators = kwargs.get("n_estimators", 100)
                    max_samples = kwargs.get("max_samples", "auto")
                    max_features = kwargs.get("max_features", 1.0)
                    bootstrap = kwargs.get("bootstrap", False)
                    n_jobs = kwargs.get("n_jobs", 1)
                    behaviour = kwargs.get("behaviour", "old")
                    verbose = kwargs.get("verbose", 0)
                    random_state = kwargs.get("random_state", None)
                    clf = IForest(
                        n_estimators=n_estimators,
                        max_samples=max_samples,
                        contamination=contamination,
                        max_features=max_features,
                        bootstrap=bootstrap,
                        n_jobs=n_jobs,
                        behaviour=behaviour,
                        random_state=random_state,
                        verbose=verbose,
                    )

            case "CBLOF":
                from pyod.models.cblof import CBLOF
                if "CBLOF" in kwargs:
                    clf = CBLOF(**kwargs["CBLOF"])
                else:
                    n_clusters = kwargs.get("n_clusters", 8)
                    alpha = kwargs.get("alpha", 0.9)
                    beta = kwargs.get("beta", 5)
                    use_weights = kwargs.get("use_weights", False)
                    check_estimator = kwargs.get("check_estimator", False)
                    n_jobs = kwargs.get("n_jobs", 1)
                    random_state = kwargs.get("random_state", None)
                    clustering_estimator = kwargs.get("clustering_estimator", None)
                    clf = CBLOF(
                        n_clusters=n_clusters,
                        contamination=contamination,
                        clustering_estimator=clustering_estimator,
                        alpha=alpha,
                        beta=beta,
                        use_weights=use_weights,
                        check_estimator=check_estimator,
                        random_state=random_state,
                        n_jobs=n_jobs,
                    )

            case "COPOD":
                from pyod.models.copod import COPOD
                if "COPOD" in kwargs:
                    clf = COPOD(**kwargs["COPOD"])
                else:
                    n_jobs = kwargs.get("n_jobs", 1)
                    clf = COPOD(contamination=contamination, n_jobs=n_jobs)

            case "SOS":
                from pyod.models.sos import SOS
                if "SOS" in kwargs:
                    clf = SOS(**kwargs["SOS"])
                else:
                    perplexity = kwargs.get("perplexity", 4.5)
                    metric = kwargs.get("metric", "euclidean")
                    eps = kwargs.get("eps", 1e-5)
                    clf = SOS(
                        contamination=contamination,
                        perplexity=perplexity,
                        metric=metric,
                        eps=eps,
                    )

            case "KDE":
                from pyod.models.kde import KDE
                if "KDE" in kwargs:
                    clf = KDE(**kwargs["KDE"])
                else:
                    bandwidth = kwargs.get("bandwidth", 1.0)
                    algorithm = kwargs.get("algorithm", "auto")
                    leaf_size = kwargs.get("leaf_size", 30)
                    metric = kwargs.get("metric", "minkowski")
                    metric_params = kwargs.get("metric_params", None)
                    clf = KDE(
                        contamination=contamination,
                        bandwidth=bandwidth,
                        algorithm=algorithm,
                        leaf_size=leaf_size,
                        metric=metric,
                        metric_params=metric_params,
                    )

            case "Sampling":
                from pyod.models.sampling import Sampling
                if "Sampling" in kwargs:
                    clf = Sampling(**kwargs["Sampling"])
                else:
                    subset_size = kwargs.get("subset_size", 20)
                    metric = kwargs.get("metric", "minkowski")
                    random_state = kwargs.get("random_state", None)
                    metric_params = kwargs.get("metric_params", None)
                    clf = Sampling(
                        contamination=contamination,
                        subset_size=subset_size,
                        metric=metric,
                        metric_params=metric_params,
                        random_state=random_state,
                    )

            case "PCA":
                from pyod.models.pca import PCA
                if "PCA" in kwargs:
                    clf = PCA(**kwargs["PCA"])
                else:
                    copy = kwargs.get("copy", True)
                    whiten = kwargs.get("whiten", False)
                    svd_solver = kwargs.get("svd_solver", "auto")
                    tol = kwargs.get("tol", 0.0)
                    iterated_power = kwargs.get("iterated_power", "auto")
                    weighted = kwargs.get("weighted", True)
                    standardization = kwargs.get("standardization", True)
                    random_state = kwargs.get("random_state", None)
                    n_selected_components = kwargs.get("n_selected_components", None)
                    n_components = kwargs.get("n_components", None)
                    clf = PCA(
                        n_components=n_components,
                        n_selected_components=n_selected_components,
                        contamination=contamination,
                        copy=copy,
                        whiten=whiten,
                        svd_solver=svd_solver,
                        tol=tol,
                        iterated_power=iterated_power,
                        random_state=random_state,
                        weighted=weighted,
                        standardization=standardization,
                    )

            case "LMDD":
                from pyod.models.lmdd import LMDD
                if "LMDD" in kwargs:
                    clf = LMDD(**kwargs["LMDD"])
                else:
                    n_iter = kwargs.get("n_iter", 50)
                    dis_measure = kwargs.get("dis_measure", "aad")
                    random_state = kwargs.get("random_state", None)
                    clf = LMDD(
                        contamination=contamination,
                        n_iter=n_iter,
                        dis_measure=dis_measure,
                        random_state=random_state,
                    )

            case "COF":
                from pyod.models.cof import COF
                if "COF" in kwargs:
                    clf = COF(**kwargs["COF"])
                else:
                    n_neighbors = kwargs.get("n_neighbors", 20)
                    method = kwargs.get("method", "fast")
                    clf = COF(
                        contamination=contamination, n_neighbors=n_neighbors, method=method
                    )

            case "HBOS":
                from pyod.models.hbos import HBOS
                if "HBOS" in kwargs:
                    clf = HBOS(**kwargs["HBOS"])
                else:
                    n_bins = kwargs.get("n_bins", 10)
                    alpha = kwargs.get("alpha", 0.1)
                    tol = kwargs.get("tol", 0.5)
                    clf = HBOS(
                        contamination=contamination, n_bins=n_bins, alpha=alpha, tol=tol
                    )

            case "KNN":
                from pyod.models.knn import KNN
                if "KNN" in kwargs:
                    clf = KNN(**kwargs["KNN"])
                else:
                    n_neighbors = kwargs.get("n_neighbors", 5)
                    method = kwargs.get("method", "largest")
                    radius = kwargs.get("radius", 1.0)
                    algorithm = kwargs.get("algorithm", "auto")
                    leaf_size = kwargs.get("leaf_size", 30)
                    metric = kwargs.get("metric", "minkowski")
                    p = kwargs.get("p", 2)
                    n_jobs = kwargs.get("n_jobs", 1)
                    metric_params = kwargs.get("metric_params", None)
                    clf = KNN(
                        contamination=contamination,
                        n_neighbors=n_neighbors,
                        method=method,
                        radius=radius,
                        algorithm=algorithm,
                        leaf_size=leaf_size,
                        metric=metric,
                        p=p,
                        metric_params=metric_params,
                        n_jobs=n_jobs,
                    )

            case "AvgKNN":
                from pyod.models.knn import KNN
                if "AvgKNN" in kwargs:
                    clf = KNN(**kwargs["AvgKNN"])
                else:
                    n_neighbors = kwargs.get("n_neighbors", 5)
                    method = kwargs.get("method", "mean")
                    radius = kwargs.get("radius", 1.0)
                    algorithm = kwargs.get("algorithm", "auto")
                    leaf_size = kwargs.get("leaf_size", 30)
                    metric = kwargs.get("metric", "minkowski")
                    p = kwargs.get("p", 2)
                    n_jobs = kwargs.get("n_jobs", 1)
                    metric_params = kwargs.get("metric_params", None)
                    clf = KNN(
                        contamination=contamination,
                        n_neighbors=n_neighbors,
                        method=method,
                        radius=radius,
                        algorithm=algorithm,
                        leaf_size=leaf_size,
                        metric=metric,
                        p=p,
                        metric_params=metric_params,
                        n_jobs=n_jobs,
                    )

            case "MedKNN":
                from pyod.models.knn import KNN
                if "MedKNN" in kwargs:
                    clf = KNN(**kwargs["MedKNN"])
                else:
                    n_neighbors = kwargs.get("n_neighbors", 5)
                    method = kwargs.get("method", "median")
                    radius = kwargs.get("radius", 1.0)
                    algorithm = kwargs.get("algorithm", "auto")
                    leaf_size = kwargs.get("leaf_size", 30)
                    metric = kwargs.get("metric", "minkowski")
                    p = kwargs.get("p", 2)
                    n_jobs = kwargs.get("n_jobs", 1)
                    metric_params = kwargs.get("metric_params", None)
                    clf = KNN(
                        contamination=contamination,
                        n_neighbors=n_neighbors,
                        method=method,
                        radius=radius,
                        algorithm=algorithm,
                        leaf_size=leaf_size,
                        metric=metric,
                        p=p,
                        metric_params=metric_params,
                        n_jobs=n_jobs,
                    )

            case "LODA":
                from pyod.models.loda import LODA
                if "LODA" in kwargs:
                    clf = LODA(**kwargs["LODA"])
                else:
                    n_bins = kwargs.get("n_bins", 10)
                    n_random_cuts = kwargs.get("n_random_cuts", 100)
                    clf = LODA(
                        contamination=contamination,
                        n_bins=n_bins,
                        n_random_cuts=n_random_cuts,
                    )

            case _:
                raise ValueError(f"Algorithm {pyod_algorithm} not supported")

        # Fit model on full feature set (Transductive evaluation)
        clf.fit(features)
        decision_scores = clf.decision_scores_
        predicted_labels = clf.labels_

        # Predict probability estimates bounded in [0, 1]
        try:
            probs = clf.predict_proba(features)[:, 1]
        except Exception:
            # Fallback min-max normalization if predict_proba is not implemented
            min_s, max_s = decision_scores.min(), decision_scores.max()
            probs = (decision_scores - min_s) / (max_s - min_s + 1e-12)

        ev = OutlierDetector.evaluate(labels, predicted_labels, decision_scores, probs)

        print(f"Accuracy for {pyod_algorithm}: {ev['acc_score'] * 100:.2f}%")
        print("F1-Score: ", round(ev["f1_score"] * 100, 2))
        print("PR-AUC (Average Precision): ", round(ev["average_precision"] * 100, 2))
        print("ROC-AUC: ", round(ev["roc_auc"] * 100, 2))

        if return_decision_function:
            return decision_scores, predicted_labels, getattr(clf, "decision_function", None), ev

        return decision_scores, predicted_labels, ev

    @staticmethod
    def evaluate(groundtruth, pred_labels, decision_scores, probs):
        """Evaluates the results using appropriate metrics for continuous scores vs binary outputs."""
        cm = confusion_matrix(groundtruth, pred_labels)
        return {
            "groundtruth_indices": np.where(np.array(groundtruth) > 0)[0],
            "pred_indices": np.where(np.array(pred_labels) > 0)[0],
            # Continuous ranking metrics (raw decision scores)
            "roc_auc": roc_auc_score(groundtruth, decision_scores),
            "average_precision": average_precision_score(groundtruth, decision_scores),
            # Bounded probability metrics [0, 1]
            "log_loss": log_loss(groundtruth, probs),
            # Thresholded binary metrics
            "f1_score": f1_score(groundtruth, pred_labels, zero_division=0),
            "acc_score": accuracy_score(groundtruth, pred_labels),
            "jaccard_score": jaccard_score(groundtruth, pred_labels, zero_division=0),
            "precision_score": precision_score(groundtruth, pred_labels, zero_division=0),
            "recall_score": recall_score(groundtruth, pred_labels, zero_division=0),
            "hamming_loss": hamming_loss(groundtruth, pred_labels),
            "tn": cm[0, 0],
            "fp": cm[0, 1],
            "fn": cm[1, 0],
            "tp": cm[1, 1],
        }