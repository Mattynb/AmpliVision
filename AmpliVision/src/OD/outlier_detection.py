""" CODE SOURCED AND/OR BASED ON: https://github.com/mpsych/ODM/ """


import time
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

ALGORITHMS = [
    #AE",
    "AvgKNN",
    #"VAE",
    #"SOGAAL",
    #"DeepSVDD",
    "HBOS",
    "LOF",
    "OCSVM",
    "IForest",
    "CBLOF",
    "COPOD",
    "SOS",
    "KDE",
    "PCA",
    "LMDD",
    "COF",
    "ECOD",
    "KNN",
    "MedKNN",
    "SOD",
    "INNE",
    #"FB",
    "LODA",
    "Sampling",
    "AnoGAN",
    #"SUOD",
]

BAD_IMAGE_ID_PATH = ""

class OutlierDetector():
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
        **kwargs,
    ):

        data_x = data[0] # imgs or vectors
        data_y = data[1] # labels

        t0 = time.time()
        self.__run_id = run_id
        self.__algorithms = ALGORITHMS if algorithms is None else algorithms
        self.__bad_ids = bad_ids if bad_ids is not None else BAD_IMAGE_ID_PATH
        self.__number_bad = number_bad if number_bad is not None else None
        self.__exclude = exclude if exclude is not None else []

        # run the outlier detection algorithms
        accuracy_scores = {}
        errors = {}
        cache_key = ""
        t0 = time.time()
        verbose = kwargs.get("verbose", False)
        for alg in self.__algorithms:
            if alg not in self.__exclude:
                print("-"*20, f" Running {alg} ", "-"*20)
                t_scores, t_labels = self._detect_outliers(
                    data_x,
                    data_y,
                    pyod_algorithm=alg,
                    verbose=verbose,
                    **kwargs,
                )

            #print(f"{alg} t_scores: {t_scores} t_labels: {t_labels}")
            print("-"*20, f" {alg} DONE ", "-"*20)
        
        accuracy_scores = dict(
            sorted(accuracy_scores.items(), key=lambda item: item[1], reverse=True)
        )

    @staticmethod
    def _detect_outliers(features, labels, pyod_algorithm, **kwargs):
        """Detect outliers using PyOD algorithm. Default algorithm is HBOS.
        See PYOD documentation to see which arguments are available for each
        algorithm and to get description of each algorithm and how the
        arguments work with each algorithm.
        link: https://pyod.readthedocs.io/en/latest/pyod.html
        """

        # split data into training and testing
        x_train, x_test  = train_test_split(
            features, test_size=0.2, random_state=42
        )

        return_decision_function = kwargs.get("return_decision_function", False)

        # make sure data_x is a 2d array and not a 1d array or 3D array
        def reshape(data_x):
            if isinstance(data_x, np.ndarray):
                if len(data_x.shape) == 1:
                    data_x = data_x.reshape(-1, 1)
                elif len(data_x.shape) == 3:
                    data_x = data_x.reshape(data_x.shape[0], -1)
            elif isinstance(data_x, list):
                for i in range(len(data_x)):
                    if len(data_x[i]) == 1:
                        data_x[i] = np.pad(data_x[i], (0, len(data_x[0]) - 1), "constant")
                    if len(data_x[i]) == 3:
                        data_x[i] = np.pad(data_x[i], (0, len(data_x[0]) - 3), "constant")
                data_x = np.array(data_x)
            return data_x
        
        match pyod_algorithm:
            case "ECOD":
                if DEBUG:
                    print("In ECOD algorithm")
                from pyod.models.ecod import ECOD

                n_jobs = kwargs.get("n_jobs", 1)
                contamination = kwargs.get("contamination", 0.1)
                clf = ECOD(n_jobs=n_jobs, contamination=contamination)

            case "LOF":
                if DEBUG:
                    print("In LOF algorithm")
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
                    contamination = kwargs.get("contamination", 0.1)
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
                if DEBUG:
                    print("In OCSVM algorithm")
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
                    contamination = kwargs.get("contamination", 0.1)
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
                if DEBUG:
                    print("In IForest algorithm")
                from pyod.models.iforest import IForest

                if "IForest" in kwargs:
                    clf = IForest(**kwargs["IForest"])
                else:
                    n_estimators = kwargs.get("n_estimators", 100)
                    max_samples = kwargs.get("max_samples", "auto")
                    contamination = kwargs.get("contamination", 0.1)
                    max_features = kwargs.get("max_features", 1.0)
                    bootstrap = kwargs.get("bootstrap", False)  # had  this as True
                    n_jobs = kwargs.get("n_jobs", 1)
                    behaviour = kwargs.get("behaviour", "old")
                    verbose = kwargs.get("verbose", 0)
                    random_state = kwargs.get("random_state", None)
                    if DEBUG:
                        print("n_estimators: ", n_estimators)
                        print("max_samples: ", max_samples)
                        print("contamination: ", contamination)
                        print("max_features: ", max_features)
                        print("bootstrap: ", bootstrap)
                        print("n_jobs: ", n_jobs)
                        print("behaviour: ", behaviour)
                        print("verbose: ", verbose)
                        print("random_state: ", random_state)
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
                if DEBUG:
                    print("In CBLOF algorithm")
                from pyod.models.cblof import CBLOF

                if "CBLOF" in kwargs:
                    clf = CBLOF(**kwargs["CBLOF"])
                else:
                    n_clusters = kwargs.get("n_clusters", 8)
                    contamination = kwargs.get("contamination", 0.1)
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
                if DEBUG:
                    print("In COPOD algorithm")
                from pyod.models.copod import COPOD

                if "COPOD" in kwargs:
                    clf = COPOD(**kwargs["COPOD"])
                else:
                    contamination = kwargs.get("contamination", 0.1)
                    n_jobs = kwargs.get("n_jobs", 1)
                    clf = COPOD(contamination=contamination, n_jobs=n_jobs)

            case "MCD":
                if DEBUG:
                    print("In MCD algorithm")
                from pyod.models.mcd import MCD

                if "MCD" in kwargs:
                    clf = MCD(**kwargs["MCD"])
                else:
                    contamination = kwargs.get("contamination", 0.1)
                    store_precision = kwargs.get("store_precision", True)
                    assume_centered = kwargs.get("assume_centered", False)
                    support_fraction = kwargs.get("support_fraction", None)
                    random_state = kwargs.get("random_state", None)
                    clf = MCD(
                        contamination=contamination,
                        store_precision=store_precision,
                        assume_centered=assume_centered,
                        support_fraction=support_fraction,
                        random_state=random_state,
                    )

            case "SOS":
                if DEBUG:
                    print("In SOS algorithm")
                from pyod.models.sos import SOS

                if "SOS" in kwargs:
                    clf = SOS(**kwargs["SOS"])
                else:
                    contamination = kwargs.get("contamination", 0.1)
                    perplexity = kwargs.get("perplexity", 4.5)
                    metric = kwargs.get("metric", "euclidean")
                    eps = kwargs.get("eps", 1e-5)
                    if DEBUG:
                        print("contamination: ", contamination)
                    clf = SOS(
                        contamination=contamination,
                        perplexity=perplexity,
                        metric=metric,
                        eps=eps,
                    )

            case "KDE":
                if DEBUG:
                    print("In KDE algorithm")
                from pyod.models.kde import KDE

                if "KDE" in kwargs:
                    clf = KDE(**kwargs["KDE"])
                else:
                    contamination = kwargs.get("contamination", 0.1)
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
                if DEBUG:
                    print("In Sampling algorithm")
                from pyod.models.sampling import Sampling

                if "Sampling" in kwargs:
                    clf = Sampling(**kwargs["Sampling"])
                else:
                    contamination = kwargs.get("contamination", 0.1)
                    subset_size = kwargs.get("subset_size", 20)
                    metric = kwargs.get("metric", "minkowski")
                    random_state = kwargs.get("random_state", None)
                    metric_params = kwargs.get("metric_params", None)
                    print(f"random_state: {random_state}")
                    clf = Sampling(
                        contamination=contamination,
                        subset_size=subset_size,
                        metric=metric,
                        metric_params=metric_params,
                        random_state=random_state,
                    )

            case "GMM":
                if DEBUG:
                    print("In GMM algorithm")
                from pyod.models.gmm import GMM

                if "GMM" in kwargs:
                    clf = GMM(**kwargs["GMM"])
                else:
                    n_components = kwargs.get("n_components", 1)
                    covariance_type = kwargs.get("covariance_type", "full")
                    tol = kwargs.get("tol", 1e-3)
                    reg_covar = kwargs.get("reg_covar", 1e-6)
                    max_iter = kwargs.get("max_iter", 100)
                    n_init = kwargs.get("n_init", 1)
                    init_params = kwargs.get("init_params", "kmeans")
                    contamination = kwargs.get("contamination", 0.1)
                    weights_init = kwargs.get("weights_init", None)
                    warm_start = kwargs.get(
                        "warm_start", False
                    )  # warm start was None not False
                    random_state = kwargs.get("random_state", None)
                    precisions_init = kwargs.get("precisions_init", None)
                    means_init = kwargs.get("means_init", None)
                    clf = GMM(
                        n_components=n_components,
                        covariance_type=covariance_type,
                        tol=tol,
                        reg_covar=reg_covar,
                        max_iter=max_iter,
                        n_init=n_init,
                        init_params=init_params,
                        contamination=contamination,
                        weights_init=weights_init,
                        means_init=means_init,
                        precisions_init=precisions_init,
                        random_state=random_state,
                        warm_start=warm_start,
                    )

            case "PCA":
                if DEBUG:
                    print("In PCA algorithm")
                from pyod.models.pca import PCA

                if "PCA" in kwargs:
                    clf = PCA(**kwargs["PCA"])
                else:
                    contamination = kwargs.get("contamination", 0.1)
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
                if DEBUG:
                    print("In LMDD algorithm")
                from pyod.models.lmdd import LMDD

                if "LMDD" in kwargs:
                    clf = LMDD(**kwargs["LMDD"])
                else:
                    contamination = kwargs.get("contamination", 0.1)
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
                if DEBUG:
                    print("In COF algorithm")
                from pyod.models.cof import COF

                if "COF" in kwargs:
                    clf = COF(**kwargs["COF"])
                else:
                    contamination = kwargs.get("contamination", 0.1)
                    n_neighbors = kwargs.get("n_neighbors", 20)
                    method = kwargs.get("method", "fast")
                    clf = COF(
                        contamination=contamination, n_neighbors=n_neighbors, method=method
                    )

            case "HBOS":
                if DEBUG:
                    print("In HBOS algorithm")
                from pyod.models.hbos import HBOS

                if "HBOS" in kwargs:
                    clf = HBOS(**kwargs["HBOS"])
                else:
                    contamination = kwargs.get("contamination", 0.1)
                    n_bins = kwargs.get("n_bins", 10)
                    alpha = kwargs.get("alpha", 0.1)
                    tol = kwargs.get("tol", 0.5)
                    clf = HBOS(
                        contamination=contamination, n_bins=n_bins, alpha=alpha, tol=tol
                    )

            case "KNN":
                if DEBUG:
                    print("In KNN algorithm")
                from pyod.models.knn import KNN

                # look if kwargs["KNN"] is a dictionary value and if so use it
                if "KNN" in kwargs:
                    clf = KNN(**kwargs["KNN"])
                else:
                    contamination = kwargs.get("contamination", 0.1)
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
                if DEBUG:
                    print("In AvgKNN algorithm")
                from pyod.models.knn import KNN

                if "AvgKNN" in kwargs:
                    clf = KNN(**kwargs["AvgKNN"])
                else:
                    contamination = kwargs.get("contamination", 0.1)
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
                if DEBUG:
                    print("In MedKNN algorithm")
                from pyod.models.knn import KNN

                if "MedKNN" in kwargs:
                    clf = KNN(**kwargs["MedKNN"])
                else:
                    contamination = kwargs.get("contamination", 0.1)
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

            case "SOD":
                if DEBUG:
                    print("In SOD algorithm")
                from pyod.models.sod import SOD

                if "SOD" in kwargs:
                    clf = SOD(**kwargs["SOD"])
                else:
                    contamination = kwargs.get("contamination", 0.1)
                    n_neighbors = kwargs.get("n_neighbors", 20)
                    ref_set = kwargs.get("ref_set", 10)
                    if ref_set >= n_neighbors:
                        ref_set = n_neighbors - 1
                    alpha = kwargs.get("alpha", 0.8)
                    clf = SOD(
                        contamination=contamination,
                        n_neighbors=n_neighbors,
                        ref_set=ref_set,
                        alpha=alpha,
                    )

            case "ROD":
                if DEBUG:
                    print("In ROD algorithm")
                from pyod.models.rod import ROD

                if "ROD" in kwargs:
                    clf = ROD(**kwargs["ROD"])
                else:
                    contamination = kwargs.get("contamination", 0.1)
                    parallel_execution = kwargs.get("parallel_execution", False)
                    clf = ROD(
                        contamination=contamination, parallel_execution=parallel_execution
                    )

            case "INNE":
                if DEBUG:
                    print("In INNE algorithm")
                from pyod.models.inne import INNE

                if "INNE" in kwargs:
                    clf = INNE(**kwargs["INNE"])
                else:
                    n_estimators = kwargs.get("n_estimators", 200)
                    max_samples = kwargs.get("max_samples", "auto")
                    contamination = kwargs.get("contamination", 0.1)
                    random_state = kwargs.get("random_state", None)
                    clf = INNE(
                        n_estimators=n_estimators,
                        max_samples=max_samples,
                        contamination=contamination,
                        random_state=random_state,
                    )

            case "FB":
                if DEBUG:
                    print("In FB algorithm")
                from pyod.models.feature_bagging import FeatureBagging

                if "FB" in kwargs:
                    clf = FeatureBagging(**kwargs["FB"])
                else:
                    n_estimators = kwargs.get("n_estimators", 10)
                    contamination = kwargs.get("contamination", 0.1)
                    max_features = kwargs.get("max_features", 1.0)
                    bootstrap_features = kwargs.get("bootstrap_features", False)
                    check_detector = kwargs.get(
                        "check_detector", True
                    )  # was False when should be True
                    check_estimator = kwargs.get("check_estimator", False)
                    n_jobs = kwargs.get("n_jobs", 1)
                    combination = kwargs.get("combination", "average")
                    verbose = kwargs.get("verbose", 0)
                    random_state = kwargs.get("random_state", None)
                    estimator_params = kwargs.get("estimator_params", None)
                    base_estimator = kwargs.get("base_estimator", None)
                    clf = FeatureBagging(
                        base_estimator=base_estimator,
                        n_estimators=n_estimators,
                        contamination=contamination,
                        max_features=max_features,
                        bootstrap_features=bootstrap_features,
                        check_detector=check_detector,
                        check_estimator=check_estimator,
                        n_jobs=n_jobs,
                        random_state=random_state,
                        combination=combination,
                        verbose=verbose,
                        estimator_params=estimator_params,
                    )

            case "LODA":
                if DEBUG:
                    print("In LODA algorithm")
                from pyod.models.loda import LODA

                if "LODA" in kwargs:
                    clf = LODA(**kwargs["LODA"])
                else:
                    contamination = kwargs.get("contamination", 0.1)
                    n_bins = kwargs.get("n_bins", 10)
                    n_random_cuts = kwargs.get("n_random_cuts", 100)
                    clf = LODA(
                        contamination=contamination,
                        n_bins=n_bins,
                        n_random_cuts=n_random_cuts,
                    )

            case "SUOD":
                if DEBUG:
                    print("In SUOD algorithm")
                from pyod.models.suod import SUOD

                if "SUOD" in kwargs:
                    clf = SUOD(**kwargs["SUOD"])
                else:
                    contamination = kwargs.get("contamination", 0.1)
                    combination = kwargs.get("combination", "average")
                    rp_flag_global = kwargs.get("rp_flag_global", True)
                    target_dim_frac = kwargs.get("target_dim_frac", 0.5)
                    jl_method = kwargs.get("jl_method", "basic")
                    bps_flag = kwargs.get("bps_flag", True)
                    approx_flag_global = kwargs.get("approx_flag_global", True)
                    verbose = kwargs.get("verbose", False)
                    rp_ng_clf_list = kwargs.get("rp_ng_clf_list", None)
                    rp_clf_list = kwargs.get("rp_clf_list", None)
                    n_jobs = kwargs.get("n_jobs", None)
                    cost_forecast_loc_pred = kwargs.get("cost_forecast_loc_pred", None)
                    cost_forecast_loc_fit = kwargs.get("cost_forecast_loc_fit", None)
                    base_estimators = kwargs.get("base_estimators", None)
                    approx_ng_clf_list = kwargs.get("approx_ng_clf_list", None)
                    approx_clf_list = kwargs.get("approx_clf_list", None)
                    approx_clf = kwargs.get("approx_clf", None)
                    if base_estimators is not None:
                        base_estimators = OutlierDetector._init_base_detectors(
                            base_estimators
                        )
                    clf = SUOD(
                        base_estimators=base_estimators,
                        contamination=contamination,
                        combination=combination,
                        n_jobs=n_jobs,
                        rp_clf_list=rp_clf_list,
                        rp_ng_clf_list=rp_ng_clf_list,
                        rp_flag_global=rp_flag_global,
                        target_dim_frac=target_dim_frac,
                        jl_method=jl_method,
                        bps_flag=bps_flag,
                        approx_clf_list=approx_clf_list,
                        approx_ng_clf_list=approx_ng_clf_list,
                        approx_flag_global=approx_flag_global,
                        approx_clf=approx_clf,
                        cost_forecast_loc_fit=cost_forecast_loc_fit,
                        cost_forecast_loc_pred=cost_forecast_loc_pred,
                        verbose=verbose,
                    )

            case "AE":
                if DEBUG:
                    print("In AE algorithm")
                from pyod.models.auto_encoder import AutoEncoder
                from keras.losses import mean_squared_error

                if "AE" in kwargs:
                    clf = AutoEncoder(**kwargs["AE"])
                else:
                    hidden_activation = kwargs.get("hidden_activation", "relu")
                    output_activation = kwargs.get("output_activation", "sigmoid")
                    loss = kwargs.get("loss", mean_squared_error)
                    optimizer = kwargs.get("optimizer", "adam")
                    epochs = kwargs.get("epochs", 100)
                    batch_size = kwargs.get("batch_size", 32)
                    dropout_rate = kwargs.get("dropout_rate", 0.2)
                    l2_regularizer = kwargs.get("l2_regularizer", 0.1)
                    validation_size = kwargs.get("validation_size", 0.1)
                    preprocessing = kwargs.get("preprocessing", True)
                    verbose = kwargs.get("verbose", 1)
                    contamination = kwargs.get("contamination", 0.1)
                    random_state = kwargs.get("random_state", None)
                    #hidden_neurons = kwargs.get("hidden_neurons", None)
                    clf = AutoEncoder(
                        #hidden_neurons=hidden_neurons,
                        hidden_activation=hidden_activation,
                        output_activation=output_activation,
                        loss=loss,
                        optimizer=optimizer,
                        epochs=epochs,
                        batch_size=batch_size,
                        dropout_rate=dropout_rate,
                        l2_regularizer=l2_regularizer,
                        validation_size=validation_size,
                        preprocessing=preprocessing,
                        verbose=verbose,
                        random_state=random_state,
                        contamination=contamination,
                    )

            case "VAE":
                if DEBUG:
                    print("In VAE algorithm")
                from pyod.models.vae import VAE
                from keras.losses import mean_squared_error as mse

                if "VAE" in kwargs:
                    clf = VAE(**kwargs["VAE"])
                else:
                    latent_dim = kwargs.get("latent_dim", 2)
                    hidden_activation = kwargs.get("hidden_activation", "relu")
                    output_activation = kwargs.get("output_activation", "sigmoid")
                    #loss = kwargs.get("loss", mse)  # Assuming this will be replaced by the actual loss function later
                    optimizer_name = kwargs.get("optimizer", "adam")
                    epochs = kwargs.get("epochs", 100)
                    batch_size = kwargs.get("batch_size", 32)
                    dropout_rate = kwargs.get("dropout_rate", 0.2)
                    optimizer_params = kwargs.get("optimizer_params", {"weight_decay": 1e-5})
                    validation_size = kwargs.get("validation_size", 0.1)
                    preprocessing = kwargs.get("preprocessing", True)
                    verbose = kwargs.get("verbose", 1)
                    contamination = kwargs.get("contamination", 0.1)
                    beta = kwargs.get("gamma", 1.0)  # `gamma` appears to correspond to `beta` in VAE
                    capacity = kwargs.get("capacity", 0.0)
                    random_state = kwargs.get("random_state", 42)  # Default is 42 in VAE
                    encoder_neurons = kwargs.get("encoder_neurons", [128, 64, 32])
                    decoder_neurons = kwargs.get("decoder_neurons", [32, 64, 128])

                    # Initialize the VAE model with adapted parameters
                    clf = VAE(
                        contamination=contamination,
                        preprocessing=preprocessing,
                        lr=kwargs.get("lr", 1e-3),
                        epoch_num=epochs,
                        batch_size=batch_size,
                        optimizer_name=optimizer_name,
                        random_state=random_state,
                        use_compile=kwargs.get("use_compile", False),
                        compile_mode=kwargs.get("compile_mode", "default"),
                        verbose=verbose,
                        optimizer_params=optimizer_params,
                        beta=beta,
                        capacity=capacity,
                        encoder_neuron_list=encoder_neurons,
                        decoder_neuron_list=decoder_neurons,
                        latent_dim=latent_dim,
                        hidden_activation_name=hidden_activation,
                        output_activation_name=output_activation,
                        batch_norm=kwargs.get("batch_norm", False),
                        dropout_rate=dropout_rate
                    )

            case "ABOD":
                if DEBUG:
                    print("In ABOD algorithm")
                from pyod.models.abod import ABOD

                if "ABOD" in kwargs:
                    clf = ABOD(**kwargs["ABOD"])
                else:
                    contamination = kwargs.get("contamination", 0.1)
                    n_neighbors = kwargs.get("n_neighbors", 5)
                    method = kwargs.get("method", "fast")
                    clf = ABOD(
                        contamination=contamination, n_neighbors=n_neighbors, method=method
                    )

            case "SOGAAL":
                if DEBUG:
                    print("In SOGAAL algorithm")
                from pyod.models.so_gaal import SO_GAAL

                if "SOGAAL" in kwargs:
                    clf = SO_GAAL(**kwargs["SOGAAL"])
                else:
                    stop_epochs = kwargs.get("stop_epochs", 20)
                    lr_d = kwargs.get("lr_d", 0.01)
                    lr_g = kwargs.get("lr_g", 0.0001)
                    decay = kwargs.get("decay", 1e-6)
                    momentum = kwargs.get("momentum", 0.9)
                    contamination = kwargs.get("contamination", 0.1)
                    clf = SO_GAAL(
                        stop_epochs=stop_epochs,
                        lr_d=lr_d,
                        lr_g=lr_g,
                        decay=decay,
                        momentum=momentum,
                        contamination=contamination,
                    )

            case "MOGAAL":
                if DEBUG:
                    print("In MOGAAL algorithm")
                from pyod.models.mo_gaal import MO_GAAL

                if "MOGAAL" in kwargs:
                    clf = MO_GAAL(**kwargs["MOGAAL"])
                else:
                    stop_epochs = kwargs.get("stop_epochs", 20)
                    lr_d = kwargs.get("lr_d", 0.01)
                    lr_g = kwargs.get("lr_g", 0.0001)
                    decay = kwargs.get("decay", 1e-6)
                    momentum = kwargs.get("momentum", 0.9)
                    contamination = kwargs.get("contamination", 0.1)
                    clf = MO_GAAL(
                        stop_epochs=stop_epochs,
                        lr_d=lr_d,
                        lr_g=lr_g,
                        decay=decay,
                        momentum=momentum,
                        contamination=contamination,
                    )

            case "DeepSVDD":
                if DEBUG:
                    print("In DeepSVDD algorithm")
                from pyod.models.deep_svdd import DeepSVDD

                if "DeepSVDD" in kwargs:
                    clf = DeepSVDD(**kwargs["DeepSVDD"])
                else:
                    use_ae = kwargs.get("use_ae", False)
                    hidden_activation = kwargs.get("hidden_activation", "relu")
                    output_activation = kwargs.get("output_activation", "sigmoid")
                    optimizer = kwargs.get("optimizer", "adam")
                    epochs = kwargs.get("epochs", 100)
                    batch_size = kwargs.get("batch_size", 32)
                    dropout_rate = kwargs.get("dropout_rate", 0.2)
                    l2_regularizer = kwargs.get("l2_regularizer", 0.1)
                    validation_size = kwargs.get("validation_size", 0.1)
                    preprocessing = kwargs.get("preprocessing", True)
                    verbose = kwargs.get("verbose", 1)
                    contamination = kwargs.get("contamination", 0.1)
                    random_state = kwargs.get("random_state", None)
                    hidden_neurons = kwargs.get("hidden_neurons", None)
                    c = kwargs.get("c", None)
                    clf = DeepSVDD(
                        c=c,
                        use_ae=use_ae,
                        hidden_neurons=hidden_neurons,
                        hidden_activation=hidden_activation,
                        output_activation=output_activation,
                        optimizer=optimizer,
                        epochs=epochs,
                        batch_size=batch_size,
                        dropout_rate=dropout_rate,
                        l2_regularizer=l2_regularizer,
                        validation_size=validation_size,
                        preprocessing=preprocessing,
                        verbose=verbose,
                        random_state=random_state,
                        contamination=contamination,
                    )

            case "AnoGAN":
                if DEBUG:
                    print("In AnoGAN algorithm")
                from pyod.models.anogan import AnoGAN

                if "AnoGAN" in kwargs:
                    clf = AnoGAN(**kwargs["AnoGAN"])
                else:
                    activation_hidden = kwargs.get("activation_hidden", "tanh")
                    dropout_rate = kwargs.get("dropout_rate", 0.2)
                    latent_dim_G = kwargs.get("latent_dim_G", 2)
                    G_layers = kwargs.get("G_layers", [20, 10, 3, 10, 20])
                    verbose = kwargs.get("verbose", 0)
                    D_layers = kwargs.get("D_layers", [20, 10, 5])
                    index_D_layer_for_recon_error = kwargs.get(
                        "index_D_layer_for_recon_error", 1
                    )
                    epochs = kwargs.get("epochs", 500)
                    preprocessing = kwargs.get("preprocessing", False)
                    learning_rate = kwargs.get("learning_rate", 0.001)
                    learning_rate_query = kwargs.get("learning_rate_query", 0.01)
                    epochs_query = kwargs.get("epochs_query", 20)
                    batch_size = kwargs.get("batch_size", 32)
                    contamination = kwargs.get("contamination", 0.1)
                    output_activation = kwargs.get("output_activation", None)
                    clf = AnoGAN(
                        activation_hidden=activation_hidden,
                        dropout_rate=dropout_rate,
                        latent_dim_G=latent_dim_G,
                        G_layers=G_layers,
                        verbose=verbose,
                        D_layers=D_layers,
                        index_D_layer_for_recon_error=index_D_layer_for_recon_error,
                        epochs=epochs,
                        preprocessing=preprocessing,
                        learning_rate=learning_rate,
                        learning_rate_query=learning_rate_query,
                        epochs_query=epochs_query,
                        batch_size=batch_size,
                        output_activation=output_activation,
                        contamination=contamination,
                    )

            case "CD":
                if DEBUG:
                    print("In CD algorithm")
                from pyod.models.cd import CD

                if "CD" in kwargs:
                    clf = CD(**kwargs["CD"])
                else:
                    contamination = kwargs.get("contamination", 0.1)
                    whitening = kwargs.get("whitening", True)
                    rule_of_thumb = kwargs.get("rule_of_thumb", False)
                    clf = CD(
                        contamination=contamination,
                        whitening=whitening,
                        rule_of_thumb=rule_of_thumb,
                    )

            case "XGBOD":
                if DEBUG:
                    print("In XGBOD algorithm")
                from pyod.models.xgbod import XGBOD

                if "XGBOD" in kwargs:
                    clf = XGBOD(**kwargs["XGBOD"])
                else:
                    max_depth = kwargs.get("max_depth", 3)
                    learning_rate = kwargs.get("learning_rate", 0.1)
                    n_estimators = kwargs.get("n_estimators", 100)
                    silent = kwargs.get("silent", True)
                    objective = kwargs.get("objective", "binary:logistic")
                    booster = kwargs.get("booster", "gbtree")
                    n_jobs = kwargs.get("n_jobs", 1)
                    gamma = kwargs.get("gamma", 0)
                    min_child_weight = kwargs.get("min_child_weight", 1)
                    max_delta_step = kwargs.get("max_delta_step", 0)
                    subsample = kwargs.get("subsample", 1)
                    colsample_bytree = kwargs.get("colsample_bytree", 1)
                    colsample_bylevel = kwargs.get("colsample_bylevel", 1)
                    reg_alpha = kwargs.get("reg_alpha", 0)
                    reg_lambda = kwargs.get("reg_lambda", 1)
                    scale_pos_weight = kwargs.get("scale_pos_weight", 1)
                    base_score = kwargs.get("base_score", 0.5)
                    standardization_flag_list = kwargs.get(
                        "standardization_flag_list", None
                    )
                    random_state = kwargs.get("random_state", None)
                    nthread = kwargs.get("nthread", None)
                    estimator_list = kwargs.get("estimator_list", None)
                    clf = XGBOD(
                        estimator_list=estimator_list,
                        standardization_flag_list=standardization_flag_list,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        n_estimators=n_estimators,
                        silent=silent,
                        objective=objective,
                        booster=booster,
                        n_jobs=n_jobs,
                        nthread=nthread,
                        gamma=gamma,
                        min_child_weight=min_child_weight,
                        max_delta_step=max_delta_step,
                        subsample=subsample,
                        colsample_bytree=colsample_bytree,
                        colsample_bylevel=colsample_bylevel,
                        reg_alpha=reg_alpha,
                        reg_lambda=reg_lambda,
                        scale_pos_weight=scale_pos_weight,
                        base_score=base_score,
                        random_state=random_state,
                    )

            case _:
                raise ValueError("Algorithm not supported")

        clf.fit(features)
        #y_train_pred = clf.labels_
        #y_train = clf.decision_scores_
        #y_test = clf.predict(x_test)
        #y_test = clf.decision_function(x_test)

        ev = OutlierDetector.evaluate(labels, clf.labels_)

        print(f"accuracy for {pyod_algorithm}:", ev["acc_score"]*100)
    

        print("f1_score: ", round(ev["f1_score"] * 100, 2))
        print("groundtruth_indices: ", ev["groundtruth_indices"])
        print("pred_indices: ", ev["pred_indices"])
        print("common outlier indices: ", np.intersect1d(ev["groundtruth_indices"], ev["pred_indices"]))


        # visualize the results
        """ visualize(
            pyod_algorithm,
            x_train,
            y_train,    
            x_test,
            y_test,
            y_train_pred,
            y_train_pred,
            show_figure = False,
            save_figure = True
        )"""

        if return_decision_function:
            return clf.decision_scores_, clf.labels_, clf.decision_function

        return clf.decision_scores_, clf.labels_

    @staticmethod
    def _init_base_detectors(base_detectors):
        """Initializes the base detectors
        Parameters
        ----------
        base_detectors : list
            The base detectors to use
        **kwargs : dict
            The kwargs to pass to the base detectors
        Returns
        -------
        base_detectors : list
            The initialized base detectors
        """
        base_temp = []
        for detector in base_detectors:
            if detector == "PCA":
                from pyod.models.pca import PCA
                base_temp.append(PCA())
            continue

        return base_temp
    
    @staticmethod
    def evaluate(groundtruth, pred):
        """Evaluates the results of the outlier detection algorithm
        Parameters
        ----------
        groundtruth : list
          The groundtruth labels
        pred : list
          The predicted labels
        Returns
        -------
        evaluation : dict
          The evaluation metrics
        """

        cm = sklearn.metrics.confusion_matrix(groundtruth, pred)
        return {
            "groundtruth_indices": np.where(np.array(groundtruth) > 0),
            "pred_indices": np.where(np.array(pred) > 0),
            "roc_auc": sklearn.metrics.roc_auc_score(groundtruth, pred),
            "f1_score": sklearn.metrics.f1_score(groundtruth, pred),
            "acc_score": sklearn.metrics.accuracy_score(groundtruth, pred),
            "jaccard_score": sklearn.metrics.jaccard_score(groundtruth, pred),
            "precision_score": sklearn.metrics.precision_score(groundtruth, pred),
            "average_precision": sklearn.metrics.average_precision_score(
                groundtruth, pred
            ),
            "recall_score": sklearn.metrics.recall_score(groundtruth, pred),
            "hamming_loss": sklearn.metrics.hamming_loss(groundtruth, pred),
            "log_loss": sklearn.metrics.log_loss(groundtruth, pred),
            "tn": cm[0, 0],
            "fp": cm[0, 1],
            "fn": cm[1, 0],
            "tp": cm[1, 1],
        }

    
    @staticmethod
    def accuracy(imgs, train_scores, bad_ids, number_bad, verbose=False, timing=False):
        """Calculates the accuracy of a pyod algorithm
        Parameters
        ----------
        imgs : list
            The images to be analyzed
        train_scores : list, np.ndarray
            The anomaly scores of the training data
        bad_ids : str, list
            The path to a text file with a list of bad ids or a list of bad ids
        number_bad : int
            The number of bad images
        verbose : bool, optional (default=False)
            Whether to display the results
        timing : bool, optional (default=False)
            If True, the time to calculate the accuracy is returned
        Returns
        -------
        accuracy : float
            The accuracy of the algorithm
        """
        t0 = time.time()
        if isinstance(bad_ids, str):
            with open(bad_ids, "r") as f:
                bad_ids = f.read().splitlines()
        elif not isinstance(bad_ids, list):
            raise ValueError("bad_ids must be a string or a list")

        img_list = [
            SimpleNamespace(image=img, train_scores=train_scores[0][i])
            for i, img in enumerate(imgs)
        ]
        # set the counter to 0
        counter = 0

        # loop through the first number_bad images in the list
        for i in range(number_bad):
            # get the SOPInstanceUID of the image
            uid = img_list[i].image.SOPInstanceUID
            # check if the uid is in the bad_ids list
            if uid in bad_ids:
                # if it is, increment the counter
                counter += 1

        # calculate the accuracy as a decimal
        accuracy = counter / number_bad

        if verbose:
            print("Accuracy: {:.10f}".format(accuracy))
        if timing:
            print(f"accuracy ...took: {time.time() - t0}s")
        return accuracy
    



import time
from types import SimpleNamespace
import mahotas as mh
import matplotlib.pyplot as plt
import numpy as np


class Features:
    @staticmethod
    def histogram(pixels, norm_type=None, timing=False, **kwargs):
        """Create histogram of data
        Returns
        -------
        np.ndarray
            List of histograms
        """
        t0 = time.time()
        histograms = []
        # if pixels is a list of images get list of histograms for each image
        if isinstance(pixels, list):
            for i in range(len(pixels)):
                # if pixels is a list of SimpleNamespace use the pixels attribute
                if isinstance(pixels[i], SimpleNamespace):
                    tmp_pixels = pixels[i].pixels.copy()
                else:  # else assume pixels is a list of np.ndarray
                    tmp_pixels = pixels[i].copy()
                # if normalization is specified normalize pixels before histogram
                if norm_type is not None:
                    tmp_pixels = Normalize.get_norm(
                        tmp_pixels, norm_type=norm_type, timing=timing, **kwargs
                    )[0]
                # append histogram to list
                histograms.append(mh.fullhistogram(tmp_pixels.astype(np.uint8)))
        # if pixels is a single image get histogram
        else:
            tmp_pixels = pixels.copy()
            if norm_type is not None:
                tmp_pixels = Normalize.get_norm(
                    tmp_pixels, norm_type=norm_type, timing=timing, **kwargs
                )[0]
            histograms = mh.fullhistogram(tmp_pixels.astype(np.uint8))

        if timing:
            print("Histogram: ", time.time() - t0)
        return histograms

    @staticmethod
    def orb(
        imgs,
        norm_type=None,
        timing=False,
        downsample=True,
        return_pixel_values=True,
        **kwargs
    ):
        """Create ORB features of data
        Parameters:
        ----------
        imgs: np.ndarray | list of np.ndarray | SimpleNamespace
            array of pixels
        norm_type : str, optional
            Type of normalization. The default is minmax.
        downsample: bool, optional
            Whether to downsample the image to 128x128. The default is True.
        return_pixel_values: bool, optional
            Whether to return the keypoints or pixel values. The default is
            True.
        timing: bool, optional
            Whether to time the function. The default is False.
        **kwargs : dict
            Additional arguments for ORB
        Parameters passable through kwargs:
        ----------
        n_keypoints: int, optional
            Number of keypoints to be returned. The function will return the
            best n_keypoints according to the Harris corner response if more
            than n_keypoints are detected. If not, then all the detected
            keypoints are returned.
        fast_n: int, optional
            The n parameter in skimage.feature.corner_fast. Minimum number of
            consecutive pixels out of 16 pixels on the circle that should all be
            either brighter or darker w.r.t test-pixel. A point c on the circle
            is darker w.r.t test pixel p if Ic < Ip - threshold and brighter
            if Ic > Ip + threshold. Also stands for the n in FAST-n corner
            detector.
        fast_threshold: float, optional
            The threshold parameter in feature.corner_fast. Threshold used to
            decide whether the pixels on the circle are brighter, darker or
            similar w.r.t. the test pixel. Decrease the threshold when more
            corners are desired and vice-versa.
        harris_k: float, optional
            The k parameter in skimage.feature.corner_harris. Sensitivity factor
            to separate corners from edges, typically in range [0, 0.2]. Small
            values of k result in detection of sharp corners.
        downscale: float, optional
            Downscale factor for the image pyramid. Default value 1.2 is chosen
            so that there are more dense scales which enable robust scale
            invariance for a subsequent feature description.
        n_scales: int, optional
            Maximum number of scales from the bottom of the image pyramid to
            extract the features from.
        timing: bool, optional
            Whether to time the function. The default is False.
        Returns
        -------
        np.ndarray | list of np.ndarray
            List of keypoints for each image or list of pixel values for each
            image if return_pixel_values is True.
        """
        t0 = time.time()

        from skimage.feature import ORB

        n_keypoints = kwargs.get("n_keypoints", 50)
        fast_n = kwargs.get("fast_n", 9)
        fast_threshold = kwargs.get("fast_threshold", 0.08)
        harris_k = kwargs.get("harris_k", 0.04)
        downscale = kwargs.get("downscale", 1.2)
        n_scales = kwargs.get("n_scales", 8)
        output_shape = kwargs.get("output_shape", (512, 512))

        pixels = Normalize.extract_pixels(imgs)
        if downsample:
            pixels = Normalize.downsample(pixels, output_shape=output_shape)[0]

        if norm_type is not None:
            pixels = Normalize.get_norm(
                pixels, norm_type=norm_type, timing=timing, **kwargs
            )[0]

        descriptor_extractor = ORB(
            n_keypoints=n_keypoints,
            fast_n=fast_n,
            fast_threshold=fast_threshold,
            harris_k=harris_k,
            downscale=downscale,
            n_scales=n_scales,
        )
        keypoints = []
        if isinstance(pixels, (list, np.ndarray)):
            for i in range(len(pixels)):
                descriptor_extractor.detect_and_extract(pixels[i])
                keypoints.append(descriptor_extractor.keypoints)
        else:
            descriptor_extractor.detect_and_extract(pixels)
            keypoints.append(descriptor_extractor.keypoints)

        if return_pixel_values:
            intensities = [
                Features._extract_pixel_intensity_from_keypoints(
                    keypoints[i], pixels[i]
                )
                for i in range(len(keypoints))
            ]
            keypoints = intensities

        if timing:
            print("ORB: {:.2f} s".format(time.time() - t0))
        return keypoints

    @staticmethod
    def sift(
        imgs,
        norm_type=None,
        timing=False,
        downsample=True,
        return_pixel_values=True,
        **kwargs
    ):
        """Create SIFT features of data
        Parameters:
        ----------
        imgs : np.ndarray | list of np.ndarray | SimpleNamespace
            array of pixels
        norm_type : str, optional
            Type of normalization. The default is minmax.
        downsample : bool, optional
            Downsample image. The default is False.
        return_pixel_values : bool, optional
            Return pixel values. The default is True.
        timing : bool, optional
            Print timing. The default is False.
        **kwargs : dict
            Additional arguments for sift
        Parameters passable through kwargs:
        ----------
        upsampling: int, optional
            Prior to the feature detection the image is upscaled by a factor of
            1 (no upscaling), 2 or 4. Method: Bi-cubic interpolation.
        n_octaves: int, optional
            Maximum number of octaves. With every octave the image size is
            halved and the sigma doubled. The number of octaves will be
            reduced as needed to keep at least 12 pixels along each dimension
            at the smallest scale.
        n_scales: int, optional
            Maximum number of scales in every octave.
        sigma_min: float, optional
            The blur level of the seed image. If upsampling is enabled
            sigma_min is scaled by factor 1/upsampling
        sigma_in: float, optional
            The assumed blur level of the input image.
        c_dog: float, optional
            Threshold to discard low contrast extrema in the DoG. It’s final
            value is dependent on n_scales by the relation:
            final_c_dog = (2^(1/n_scales)-1) / (2^(1/3)-1) * c_dog
        c_edge: float, optional
            Threshold to discard extrema that lie in edges. If H is the Hessian
            of an extremum, its “edgeness” is described by tr(H)²/det(H).
            If the edgeness is higher than (c_edge + 1)²/c_edge, the extremum
            is discarded.
        n_bins: int, optional
            Number of bins in the histogram that describes the gradient
            orientations around keypoint.
        lambda_ori: float, optional
            The window used to find the reference orientation of a keypoint has
            a width of 6 * lambda_ori * sigma and is weighted by a standard
            deviation of 2 * lambda_ori * sigma.
        c_max: float, optional
            The threshold at which a secondary peak in the orientation histogram
            is accepted as orientation
        lambda_descr: float, optional
            The window used to define the descriptor of a keypoint has a width
            of 2 * lambda_descr * sigma * (n_hist+1)/n_hist and is weighted by
            a standard deviation of lambda_descr * sigma.
        n_hist: int, optional
            The window used to define the descriptor of a keypoint consists of
            n_hist * n_hist histograms.
        n_ori: int, optional
            The number of bins in the histograms of the descriptor patch.
        timing: bool, optional
            Whether to time the function. The default is False.
        Returns
        -------
        np.ndarray | list of np.ndarray
            List of keypoints for each image or list of pixel values for each
            image if return_pixel_values is True.
        """
        t0 = time.time()

        from skimage.feature import SIFT

        upsampling = kwargs.get("upsampling", 1)
        n_octaves = kwargs.get("n_octaves", 1)
        n_scales = kwargs.get("n_scales", 1)
        sigma_min = kwargs.get("sigma_min", 1.3)
        sigma_in = kwargs.get("sigma_in", 0.5)
        c_dog = kwargs.get("c_dog", 0.7)
        c_edge = kwargs.get("c_edge", 0.05)
        n_bins = kwargs.get("n_bins", 10)
        lambda_ori = kwargs.get("lambda_ori", 0.5)
        c_max = kwargs.get("c_max", 1.5)
        lambda_descr = kwargs.get("lambda_descr", 0.5)
        n_hist = kwargs.get("n_hist", 1)
        n_ori = kwargs.get("n_ori", 1)
        output_shape = kwargs.get("output_shape", (256, 256))

        pixels = Normalize.extract_pixels(imgs)
        if downsample:
            pixels = Normalize.downsample(pixels, output_shape=output_shape)[0]

        if norm_type is not None:
            pixels = Normalize.get_norm(
                pixels, norm_type=norm_type, timing=timing, **kwargs
            )[0]

        descriptor_extractor = SIFT(
            upsampling=upsampling,
            n_octaves=n_octaves,
            n_scales=n_scales,
            sigma_min=sigma_min,
            sigma_in=sigma_in,
            c_dog=c_dog,
            c_edge=c_edge,
            n_bins=n_bins,
            lambda_ori=lambda_ori,
            c_max=c_max,
            lambda_descr=lambda_descr,
            n_hist=n_hist,
            n_ori=n_ori,
        )

        keypoints = []
        if isinstance(pixels, (list, np.ndarray)):
            for i in range(len(pixels)):
                descriptor_extractor.detect_and_extract(pixels[i])
                keypoints.append(descriptor_extractor.keypoints)
        else:
            descriptor_extractor.detect_and_extract(pixels)
            keypoints.append(descriptor_extractor.keypoints)

        if return_pixel_values:
            intensities = [
                Features._extract_pixel_intensity_from_keypoints(
                    keypoints[i], pixels[i]
                )
                for i in range(len(keypoints))
            ]
            intensities = Features._fix_jagged_keypoint_arrays(intensities)
            keypoints = intensities

        if timing:
            print("SIFT: {:.2f} s".format(time.time() - t0))

        return keypoints

    @staticmethod
    def downsample(
        images, output_shape=None, flatten=False, normalize=None, timing=False, **kwargs
    ):
        """Downsample images to a given shape.
        Parameters
        ----------
        images : numpy.ndarray, list of numpy.ndarray
            Array of images to be downsampled
        output_shape : tuple
            Shape of the output images
        flatten : bool
            If true, the images are flattened to a 1D array
        normalize : str | bool
            If not None, the images are normalized using the given method
        timing : bool
            If true, the time needed to perform the downsampling is printed
        Returns
        -------
        numpy.ndarray | list of numpy.ndarray
            Downsampled images
        """
        t0 = time.time()
        from skimage.transform import resize

        if output_shape is None:
            output_shape = (128, 128)
        pixels = Normalize.extract_pixels(images)
        if isinstance(pixels, list):
            resized = [resize(img, output_shape) for img in pixels]
        else:
            resized = resize(pixels, output_shape)

        if flatten:
            resized = [img.flatten() for img in resized]

        if normalize is not None:
            resized = Normalize.get_norm(
                resized, norm_type=normalize, timing=timing, **kwargs
            )[0]

        if timing:
            print(f"downsample: {time.time() - t0}")
        return resized

    @staticmethod
    def _extract_pixel_intensity_from_keypoints(keypoints, img):
        """Extract pixel intensities from keypoints
        Parameters
        ----------
        keypoints : np.ndarray
            array of keypoints
        img : np.ndarray
            image to extract intensities from
        Returns
        -------
        np.ndarray
            array of pixel intensities
        """
        intensities = []
        for i in range(len(keypoints)):
            x = int(keypoints[i][0])
            y = int(keypoints[i][1])
            intensities.append(img[x, y])
        return np.array(intensities)

    @staticmethod
    def _fix_jagged_keypoint_arrays(keypoints):
        """Normalize keypoint lengths by finding the smallest length of
        keypoints and then selecting a random distribution of keypoints of
        similar length in the rest of the keypoints.
        Parameters
        ----------
        keypoints : list of np.ndarray
            list of keypoints
        Returns
        -------
        np.ndarray
            array of keypoints
        """
        min_len = min(len(kp) for kp in keypoints)
        new_keypoints = []
        for kp in keypoints:
            if len(kp) > min_len:
                new_keypoints.append(
                    kp[np.random.choice(len(kp), min_len, replace=False)]
                )
            else:
                new_keypoints.append(kp)
        return np.array(new_keypoints)

    @staticmethod
    def get_features(data, feature_type="hist", norm_type=None, timing=False, **kwargs):
        """Get features of data
        Parameters
        ----------
        data : SimpleNamespace, np.ndarray, list of np.ndarray, any
            array of pixels
        feature_type : str, optional
            Type of feature to extract. The default is "histogram".
        norm_type : str, optional
            Type of normalization. The default is None.
        timing: bool, optional
            Whether to time the function. The default is False.
        Returns
        -------
        np.ndarray, ski.feature.FeatureDetector
        """
        t0 = time.time()
        if feature_type in ["hist", "histogram"]:
            features = Features.histogram(
                data, norm_type=norm_type, timing=timing, **kwargs
            )
        elif feature_type == "sift":
            rpv = kwargs.get("return_pixel_values", True)
            ds = kwargs.get("downsample", False)
            features = Features.sift(
                data,
                norm_type=norm_type,
                return_pixel_values=rpv,
                downsample=ds,
                timing=timing,
                **kwargs
            )
        elif feature_type == "orb":
            rpv = kwargs.get("return_pixel_values", True)
            ds = kwargs.get("downsample", True)
            features = Features.orb(
                data,
                norm_type=norm_type,
                return_pixel_values=rpv,
                downsample=ds,
                timing=timing,
                **kwargs
            )
        elif feature_type == "downsample":
            output_shape = kwargs.get("output_shape", (256, 256))
            flatten = kwargs.get("flatten", True)
            features = Normalize.downsample(
                data,
                output_shape=output_shape,
                flatten=flatten,
                norm_type=norm_type,
                timing=timing,
                **kwargs
            )[0]
        else:
            raise ValueError("Feature type not supported")
        if timing:
            print("Features: ", time.time() - t0)
        return features

    @staticmethod
    def show_image_and_feature(
        image,
        features=None,
        feature_types=None,
        norm_type="min-max",
        downsample=False,
        output_shape=None,
        train_scores=None,
        label=None,
        log=False,
        **kwargs
    ):
        """Displays an image next to the specified features.
        Parameters
        ----------
        image : SimpleNamespace
            array of pixel values
        features : list
            (default is None) list of features to display
        feature_types : list
            (default is 'hist') type of feature to display
        norm_type : str
            type of normalization to use, options are:
            'min-max' : normalize the image using the min and max values
            'max' : normalize the image using the max value
            'guassian' : normalize the image using a guassian distribution
            'z-score' : normalize the image using a z-score
            'robost' : normalize the image using a robust distribution
            'downsample' : downsample the image to 64x64
            (default is 'min-max')
        downsample : bool
            (default is False) downsample the image to 64x64 or to the shape
            specified by output_shape
        output_shape : tuple
            (default is None) shape of the output image
        train_scores : list
            (default is None) train scores of the features
        label : str
            (default is None) label of the image
        log : bool
            (default is False) whether to log the image
        """
        if feature_types is None:
            feature_types = []

        pixels = Normalize.extract_pixels(image)

        if output_shape is not None:
            downsample = True

        # normalize the image
        img = Normalize.get_norm(
            pixels,
            norm_type=norm_type,
            downsample=downsample,
            output_shape=output_shape,
            **kwargs
        )

        fig, ax = plt.subplots(1, len(feature_types) + 1, figsize=(10, 5))
        if label is not None:
            fig.suptitle(f"SOPInstanceUID: {image.SOPInstanceUID} Label: {label}")
        else:
            fig.suptitle(f"SOPInstanceUID: {image.SOPInstanceUID}")
        # add extra width between plots
        fig.subplots_adjust(wspace=0.4)

        ax[0].imshow(img, cmap="gray")
        # ax[0].set_title(image.SOPInstanceUID, size=8-len(feature_types))

        if "sift" in feature_types:
            idx = feature_types.index("sift") + 1
            if features is None:
                kp = Features.sift(img)
                keypoints = kp.keypoints
            else:
                keypoints = features[idx - 1]
            ax[idx].imshow(img)
            x_points = keypoints[:, 1]
            y_points = keypoints[:, 0]
            ax[idx].scatter(x_points, y_points, facecolors="none", edgecolors="r")
            label = Features._get_train_score("sift", feature_types, train_scores)
            ax[idx].set_title(label, size=8)

        if "orb" in feature_types:
            idx = feature_types.index("orb") + 1
            if features is None:
                kp = Features.orb(img)
                keypoints = kp.keypoints
            else:
                keypoints = features[idx - 1]
            keypoints = keypoints[0].astype(int)
            img_ds = Normalize.downsample(img)
            ax[idx].imshow(img_ds[0])
            x_points = keypoints[:, 1]
            y_points = keypoints[:, 0]
            ax[idx].scatter(x_points, y_points, facecolors="none", edgecolors="r")
            label = Features._get_train_score("orb", feature_types, train_scores)
            ax[idx].set_title(label, size=8)

        if "hist" in feature_types or "histogram" in feature_types:
            idx = feature_types.index("hist") + 1
            if features is None:
                y_axis = Features.histogram(img)
                idx = feature_types.index("hist") + 1
                y_axis = Features.histogram(img)
                if len(y_axis) < 256:
                    y_axis = np.append(y_axis, np.zeros(256 - len(y_axis)))
                x_axis = np.arange(0, 256, 1)
                ax[idx].set_ylim(0, 1)
                ax[idx].bar(x_axis, y_axis, color="b", log=True, width=10)
                ax[idx].set_xlim(0.01, 255)
                ax[idx].set_ylim(0.01, 10**8)
            else:
                y_axis = features[idx - 1]
                print(y_axis)
                ax[idx].set_ylim(0, np.max(y_axis))
                ax[idx].bar(np.arange(0, len(y_axis)), y_axis, log=log)
            label = Features._get_train_score("hist", feature_types, train_scores)
            ax[idx].set_title(label, size=8)
        if "downsample" in feature_types:
            idx = feature_types.index("downsample") + 1
            img_ds = Normalize.downsample(img) if features is None else features[idx - 1]
            ax[idx].imshow(img_ds[0], cmap="gray")
            label = Features._get_train_score("downsample", feature_types, train_scores)
            ax[idx].set_title(label, size=8)

        plt.show()

    @staticmethod
    def view_image_and_features(
        images, feature_types=None, norm_type="min-max", train_scores=None
    ):
        """Displays an image next to its histogram.
        Parameters
        ----------
        images : list
            list of SimpleNamespace
        feature_types : list
            (default is 'hist') type of feature to display
        feature_labels : list
            (default is 'Histogram') label of the feature
        norm_type : str
            type of normalization to use, options are:
            'min-max' : normalize the image using the min and max values
            'max' : normalize the image using the max value
            'guassian' : normalize the image using a guassian distribution
            'z-score' : normalize the image using a z-score
            'robost' : normalize the image using a robust distribution
            (default is 'min-max')
        train_scores : list of lists
            (default is None) train scores of the features
        """
        # add the images and train scores a image list
        if feature_types is None:
            feature_types = []
        images_list = []
        for i in range(len(images)):
            ds = SimpleNamespace()
            ds.image = images[i]
            ds.train_scores = train_scores[i]
            images_list.append(ds)

        # loop over the images and use the show image and feature function
        for i in range(len(images)):
            ds = images_list[i]
            Features.show_image_and_feature(
                ds.image,
                feature_types=feature_types,
                norm_type=norm_type,
                train_scores=ds.train_scores,
            )

    @staticmethod
    def _get_train_score(feature, feature_types, train_scores):
        """Get train scores of features
        Parameters
        ----------
        feature : str
            feature to get train scores of
        feature_types : list
           type of features to display
        train_scores : list
            (default is None) train scores of the feature
        Returns
        -------
        list
            train scores of the features
        """
        if train_scores is None:
            return feature
        if len(train_scores) == 1:
            return str(train_scores[0])
        else:
            return str(train_scores[feature_types.index(feature)])
        

import time
import types
import pydicom as dicom
import numpy as np
import mahotas as mh

DEBUG = False


# class to represent different normalization techniques


class Normalize:
    @staticmethod
    def extract_pixels(images, timing=False):
        """Extract pixels from images
        Parameters
        ----------
        images : numpy.ndarray | list of numpy.ndarray
            Array of images to be normalized
        timing : bool, optional
            If true, the time needed to perform the normalization is printed.
            The default is False.
        """
        t0 = time.time()
        pixels = []
        if isinstance(images, list):
            if isinstance(images[0], np.ndarray):
                return images
            elif isinstance(images[0], types.SimpleNamespace):
                pixels.extend(image.pixels for image in images)
            elif isinstance(images[0], dicom.dataset.FileDataset):
                pixels.extend(image.pixel_array for image in images)
            else:
                raise TypeError("Unknown type of images")
        elif isinstance(images, np.ndarray):
            pixels = images  # was returning this as list before
        elif isinstance(images, types.SimpleNamespace):
            pixels = images.pixels
        else:
            raise TypeError("Unknown type of images")
        if timing:
            print(f"Extract pixels: {time.time() - t0}")
        return pixels

    @staticmethod
    def _minmax_helper(pixels, **kwargs):
        """Helper function to normalize data using minmax method"""
        bins = kwargs.get("bins", 256)
        max_val = np.max(pixels)
        min_val = np.min(pixels)
        normalized_pixels = pixels.astype(np.float32).copy()
        normalized_pixels -= min_val
        normalized_pixels /= max_val - min_val
        normalized_pixels *= bins - 1

        return normalized_pixels

    @staticmethod
    def minmax(pixels, timing=False, **kwargs):
        """The min-max approach (often called normalization) rescales the
        feature to a fixed range of [0,1] by subtracting the minimum value
        of the feature and then dividing by the range, which is then multiplied
        by 255 to bring the value into the range [0,255].
        Parameters
        ----------
        pixels : numpy.ndarray | list of numpy.ndarray
            Array of pixels to be normalized
        timing : bool
            If true, the time needed to perform the normalization is printed
        Returns
        -------
        numpy.ndarray
            Normalized pixels
        """
        t0 = time.time()
        if isinstance(pixels, list):
            normalized_pixels = [Normalize._minmax_helper(p, **kwargs) for p in pixels]
        else:
            normalized_pixels = Normalize._minmax_helper(pixels, **kwargs)

        if timing:
            print(f"minmax: {time.time() - t0}")
        return normalized_pixels, None

    @staticmethod
    def _max_helper(pixels, **kwargs):
        """Helper function to normalize data using max method"""
        bins = kwargs.get("bins", 256)
        max_val = np.max(pixels)
        normalized_pixels = pixels.astype(np.float32).copy()
        normalized_pixels /= abs(max_val)
        normalized_pixels *= bins - 1
        return normalized_pixels

    @staticmethod
    def max(pixels, timing=False, **kwargs):
        """The maximum absolute scaling rescales each feature between -1 and 1
        by dividing every observation by its maximum absolute value.
        Parameters
        ----------
        pixels : numpy.ndarray | list of numpy.ndarray
            Array of pixels to be normalized
        downsample : bool
            If true, the image is downsampled to a smaller size
        output_shape : tuple of int
            The shape of the output image if downsampling is used
        timing : bool
            If true, the time needed to perform the normalization is printed
        Returns
        -------
        numpy.ndarray
            Normalized pixels
        """
        t0 = time.time()
        temp_pixels = pixels
        if isinstance(temp_pixels, list):
            normalized_pixels = [Normalize._max_helper(p, **kwargs) for p in temp_pixels]
        else:
            normalized_pixels = Normalize._max_helper(temp_pixels, **kwargs)

        if timing:
            print(f"max: {time.time() - t0}")
        return normalized_pixels, None

    @staticmethod
    def _gaussian_helper(pixels, **kwargs):
        """Helper function to normalize data using gaussian blur"""
        sigma = kwargs.get("sigma", 20)
        normalized_pixels = mh.gaussian_filter(pixels, sigma=sigma)
        normalized_pixels /= normalized_pixels.max()
        return normalized_pixels

    @staticmethod
    def gaussian(pixels, timing=False, **kwargs):
        """Normalize by gaussian
        Parameters
        ----------
        pixels : numpy.ndarray | list of numpy.ndarray
            Array of pixels to be normalized
        downsample : bool
            If true, the image is downsampled to a smaller size
        output_shape : tuple of int
            The shape of the output image if downsampling is used
        timing : bool
            If true, the time needed to perform the normalization is printed
        Returns
        -------
        numpy.ndarray
            Normalized pixels
        """
        t0 = time.time()
        temp_pixels = pixels
        bins = kwargs.get("bins", 256)
        if isinstance(temp_pixels, list):
            filtered = [Normalize._gaussian_helper(p, **kwargs) for p in temp_pixels]
        else:
            filtered = Normalize._gaussian_helper(temp_pixels, **kwargs)

        normalized_pixels = filtered.copy()
        normalized_pixels *= bins - 1

        if timing:
            print(f"gaussian: {time.time() - t0}")
        return normalized_pixels, filtered

    @staticmethod
    def _zscore_helper(pixels, **kwargs):
        """Helper function to normalize data using zscore method"""
        bins = kwargs.get("bins", 255)
        normalized_pixels = pixels.astype(np.float32).copy()
        normalized_pixels -= np.mean(normalized_pixels)
        normalized_pixels /= np.std(normalized_pixels)
        normalized_pixels *= bins

        return normalized_pixels

    @staticmethod
    def z_score(pixels, timing=False, **kwargs):
        """The z-score method (often called standardization) transforms the data
        into a distribution with a mean of 0 and a standard deviation of 1.
        Each standardized value is computed by subtracting the mean of the
        corresponding feature and then dividing by the standard deviation.
        Parameters
        ----------
        pixels : numpy.ndarray | list of numpy.ndarray
            Array of pixels to be normalized
        downsample : bool
            If true, the image is downsampled to a smaller size
        output_shape : tuple of int
            The shape of the output image if downsampling is used
        timing : bool
            If true, the time needed to perform the normalization is printed
        Returns
        -------
        numpy.ndarray
            Normalized pixels
        """
        t0 = time.time()
        temp_pixels = pixels
        if isinstance(temp_pixels, list):
            normalized_pixels = [
                Normalize._zscore_helper(p, **kwargs) for p in temp_pixels
            ]
        else:
            normalized_pixels = Normalize._zscore_helper(temp_pixels, **kwargs)

        if timing:
            print(f"zscore: {time.time() - t0}")
        return normalized_pixels, None

    @staticmethod
    def _robust_helper(pixels, **kwargs):
        """
        Robust Scalar transforms x to x’ by subtracting each value of features
        by the median and dividing it by the interquartile range between the
        1st quartile (25th quantile) and the 3rd quartile (75th quantile).
        """
        bins = kwargs.get("bins", 256)
        normalized_pixels = pixels.astype(np.float32).copy()
        normalized_pixels -= np.median(normalized_pixels)
        normalized_pixels /= np.percentile(normalized_pixels, 75) - np.percentile(
            normalized_pixels, 25
        )
        normalized_pixels *= bins - 1

        return normalized_pixels

    @staticmethod
    def robust(pixels, timing=False, **kwargs):
        """In robust scaling, we scale each feature of the data set by subtracting
        the median and then dividing by the interquartile range. The interquartile
        range (IQR) is defined as the difference between the third and the first
        quartile and represents the central 50% of the data.
        Parameters
        ----------
        pixels : numpy.ndarray | list of numpy.ndarray
            Array of pixels to be normalized
        downsample : bool
            If true, the image is downsampled to a smaller size
        output_shape : tuple of int
            The shape of the output image if downsampling is used
        timing : bool
            If true, the time needed to perform the normalization is printed
        Returns
        -------
        numpy.ndarray
            Normalized pixels
        """
        t0 = time.time()
        temp_pixels = pixels
        if isinstance(temp_pixels, list):
            normalized_pixels = [
                Normalize._robust_helper(p, **kwargs) for p in temp_pixels
            ]
        else:
            normalized_pixels = Normalize._robust_helper(temp_pixels, **kwargs)

        if timing:
            print(f"robust: {time.time() - t0}")

        return normalized_pixels, None

    @staticmethod
    def downsample(
        images, output_shape=None, flatten=False, normalize=None, timing=False, **kwargs
    ):
        """Downsample images to a given shape.
        Parameters
        ----------
        images : numpy.ndarray, list of numpy.ndarray
            Array of images to be downsampled
        output_shape : tuple
            Shape of the output images
        flatten : bool
            If true, the images are flattened to a 1D array
        normalize : str
            The type of normalization to perform on the images
        timing : bool
            If true, the time needed to perform the downsampling is printed
        Returns
        -------
        numpy.ndarray | list of numpy.ndarray
            Downsampled images
        """
        t0 = time.time()
        from skimage.transform import resize

        if output_shape is None:
            output_shape = (256, 256)
        images_copy = Normalize.extract_pixels(images)
        if isinstance(images_copy, list):
            resized = []
            for img in images_copy:
                if flatten:
                    resized.append(np.array(resize(img, output_shape)).reshape(-1))
                else:
                    resized.append(np.array(resize(img, output_shape)))
                if timing:
                    print(f"downsample: {time.time() - t0}")
        elif flatten:
            resized = np.array(resize(images_copy, output_shape)).reshape(-1)
        else:
            resized = np.array(resize(images_copy, output_shape))

        if normalize is not None:
            if isinstance(normalize, bool):
                normalize = "minmax"
            if isinstance(resized, list):
                for i in range(len(resized)):
                    resized[i] = Normalize.get_norm(resized[i], normalize)[0]
            else:
                resized = Normalize.get_norm(resized, normalize)[0]

        if timing:
            print(f"downsample: {time.time() - t0}")
        return np.asarray(resized), None

    @staticmethod
    def get_norm(pixels, norm_type, timing=False, **kwargs):
        """Normalize pixels
        Parameters
        ----------
        pixels : numpy.ndarray | list of numpy.ndarray
            Array of pixels to be normalized
        norm_type : str
            Type of normalization. The default is 'min-max'.
            options -> attributes possible:
                'min-max',
                'max',
                'gaussian',
                'z-score',
                'robust',
                'downsample' -> output_shape
        timing : bool, optional
            If true, the time needed to perform the normalization is printed.
            The default is False.
        Returns
        -------
        numpy.ndarray
            Normalized pixels
        """
        t0 = time.time()
        pixels = Normalize.extract_pixels(pixels)
        if norm_type.lower() == "max":
            normalized, filtered = Normalize.max(pixels, timing, **kwargs)
        elif norm_type.lower() in ["minmax", "min-max"]:
            normalized, filtered = Normalize.minmax(pixels, timing, **kwargs)
        elif norm_type.lower() == "gaussian":
            normalized, filtered = Normalize.gaussian(pixels, timing, **kwargs)
        elif norm_type.lower() in ["zscore", "z-score"]:
            normalized, filtered = Normalize.z_score(pixels, timing, **kwargs)
        elif norm_type.lower() == "robust":
            output_shape = kwargs.get("output_shape", (256, 256))
            normalized, filtered = Normalize.robust(pixels, timing, **kwargs)
        elif norm_type.lower() == "downsample":
            output_shape = kwargs.get("output_shape", (256, 256))
            flatten = kwargs.get("flatten", True)
            normalized, filtered = Normalize.downsample(
                pixels,
                output_shape=output_shape,
                flatten=flatten,
                normalize=norm_type,
                **kwargs
            )
        else:
            raise ValueError("Invalid normalization type")

        if timing:
            print(f"get_norm: {time.time() - t0}")

        return normalized, filtered