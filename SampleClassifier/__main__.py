
import pandas as pd
import matplotlib.pyplot as plt

from src.visuals import Visuals
from src.evaluate import ClassifierEvaluator
from src.io import save_results, load_results


from sklearn.model_selection import train_test_split


LABELS = [
    "brst", "ctrl", "lung",
    "ovrn", "prst", "skin", "tyrd"
]


def main(path: str):

    # Load the data as a pandas DataFrame
    df = pd.read_csv(path)
    X = df.drop('class', axis=1)
    y = df['class']

    # Splitting data into training (70%), testing(15%), and validation (15%) sets with stratification
    # Stratified sampling ensures that the distribution of classes in the training and test sets
    # is the same as the distribution of classes in the original dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.8, random_state=42, stratify=y)

    # X_train, X_val, y_train, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42, stratify=y_test)

    print("Evaluating Classifiers", sep='-'*30)

    # Create a ClassifierEvaluator object
    Evaluator = ClassifierEvaluator(df)

    # Evaluate classifiers
    Evaluator.evaluate_classifiers(
        X_train, X_test, y_train, y_test
    )

    # Plot learning curves
    Visuals.plot_learning_curve(Evaluator)
    Visuals.plot_learning_curve(Evaluator, scaled_y=True)

    # Plot confusion matrices
    Visuals.plot_confusion_matrix(Evaluator)

    # Plot classification reports
    Visuals.plot_classification_report(Evaluator)

    plt.show()
    # save to file
    save_results(Evaluator)


if __name__ == '__main__':
    path = r'C:\Users\Matheus\Desktop\NanoTechnologies_Lab\Phase A\data\generated_results\07-11-2024\COMBINED_generated.csv'
    main(path)

    """
    Eval = load_results()
    Visuals.plot_learning_curve(
        Eval, title='Loaded Learning Curves', scaled_y=True)
    Visuals.plot_learning_curve(Eval, title='Loaded Learning Curves')
    # plot_confusion_matrix(cm, title='Loaded Confusion Matrix')
    # plot_classification_report(cm, title='Loaded Classification Report')

    # """
    plt.show()
