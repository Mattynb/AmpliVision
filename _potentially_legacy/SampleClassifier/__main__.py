
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import LabelBinarizer

from src.visuals import Visuals
from src.evaluate import ClassifierEvaluator
from src.io import save_results, load_results

from sklearn.model_selection import train_test_split


def main(path: str):

    # Load the data as a pandas DataFrame
    df = pd.read_csv(path)

    """
    # drop duplicates across subsets of 3 columns
    for i in range(1, len(df.columns), 3):
        # get subset of 3 columns + class
        SUB = df.columns[i:i+3].append(pd.Index(['class']))
        df = df.drop_duplicates(subset=SUB)
    print(f"Total dataset size after dropping duplicates: {len(df)}")

    # check for duplicates across rgb subsets
    for i in range(1, len(df.columns), 3):

        try:
            assert len(df[df.columns[i:i+3]].duplicated()) == 0
        except AssertionError:
            print(f"Found duplicates in subset {i//3}")
    """

    X = df.drop('class', axis=1)
    y = df['class']
    """
    # One-hot encode the target
    lb = LabelBinarizer()
    y = lb.fit_transform(y)"""

    # Splitting data into training (70%), testing(15%), and validation (15%) sets with stratification
    # Stratified sampling ensures that the distribution of classes in the training and test sets
    # is the same as the distribution of classes in the original dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.60, random_state=42, stratify=y)

    X_train, X_val, y_train, y_val = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42)

    print("Evaluating Classifiers", sep='-'*30)

    # Create a ClassifierEvaluator object
    Evaluator = ClassifierEvaluator(
        X_train, y_train, X_test, y_test, X_val, y_val
    )

    # Evaluate classifiers
    Evaluator.evaluate_classifiers(
        X_train, X_test, y_train, y_test
    )

    # Plot learning curves
    # Visuals.plot_learning_curve(Evaluator, scorer='accuracy', scaled_y=False)
    Visuals.plot_learning_curve(
        Evaluator, scorer='neg_log_loss', scaled_y=False)

    # Plot confusion matrices
    # Visuals.plot_confusion_matrix(Evaluator)

    # Plot classification reports
    # Visuals.plot_classification_report(Evaluator)

    # save to file
    save_results(Evaluator)

    plt.show()


'roc_auc_ovo', 'average_precision', 'positive_likelihood_ratio', 'balanced_accuracy', 'jaccard_micro', 'f1', 'mutual_info_score', 'neg_median_absolute_error', 'recall', 'roc_auc_ovr_weighted', 'neg_negative_likelihood_ratio', 'adjusted_rand_score', 'd2_absolute_error_score', 'f1_macro', 'neg_mean_squared_log_error', 'matthews_corrcoef', 'roc_auc_ovo_weighted', 'normalized_mutual_info_score', 'v_measure_score', 'top_k_accuracy', 'precision_micro', 'jaccard', 'f1_samples', 'recall_macro', 'neg_brier_score', 'neg_log_loss', 'neg_mean_squared_error', 'neg_mean_absolute_percentage_error', 'precision_samples', 'roc_auc', 'precision', 'jaccard_weighted', 'completeness_score', 'precision_weighted', 'recall_micro', 'recall_weighted', 'neg_root_mean_squared_log_error', 'jaccard_macro', 'neg_mean_absolute_error', 'r2', 'jaccard_samples', 'neg_mean_gamma_deviance', 'neg_mean_poisson_deviance', 'recall_samples', 'explained_variance', 'f1_micro', 'fowlkes_mallows_score', 'max_error', 'f1_weighted', 'adjusted_mutual_info_score', 'neg_root_mean_squared_error', 'precision_macro', 'roc_auc_ovr', 'rand_score', 'accuracy', 'homogeneity_score'

if __name__ == '__main__':
    path = r'C:\Users\Matheus\Desktop\NanoTechnologies_Lab\Phase A\data\generated_results\07-11-2024\COMBINED_generated.csv'
    main(path)

    """
    Eval = load_results()
    Visuals.plot_learning_curve(
        Eval, title='Loaded Learning Curves', scaled_y=True)
    # Visuals.plot_learning_curve(Eval, title='Loaded Learning Curves')
    # Visuals.plot_confusion_matrix(Eval, title='Loaded Confusion Matrix')
    # plot_classification_report(cm, title='Loaded Classification Report')

    # """
    plt.show()
