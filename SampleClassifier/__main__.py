from tkinter import N
import pandas as pd
import pickle as pkl
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


LABELS = [
    "brst", "ctrl", "lung",
    "ovrn", "prst", "skin", "tyrd"
]


def main(path: str):

    # Load the data as a pandas DataFrame
    df = pd.read_csv(path)

    # Splitting data into features and labels
    X = df.drop('class', axis=1)
    y = df['class']

    # Splitting data into training (70%), testing(15%), and validation (15%) sets with stratification
    # Stratified sampling ensures that the distribution of classes in the training and test sets
    # is the same as the distribution of classes in the original dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.8, random_state=42, stratify=y)

    X_val, y_val = None, None
    # train_test_split(X_test, y_test, test_size=0.5, random_state=42, stratify=y_test)

    # List of classifiers to compare
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000, solver='liblinear'),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5, weights='distance'),
        'LDA': LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
        'SVM linear': SVC(kernel='linear', C=1.0),
        'SVM rbf': SVC(kernel='rbf', C=1.0, gamma='auto'),
        'SVM poly': SVC(kernel='poly', C=1.0, degree=3, gamma='auto'),
        'SVM sigmoid': SVC(kernel='sigmoid', C=1.0, gamma='auto'),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(max_depth=3, learning_rate=0.2718, random_state=42)
    }

    print("Evaluating Classifiers", sep='-'*30)
    # Evaluate classifiers
    results = evaluate_classifiers(
        classifiers, X_train, X_test, X_val, y_train, y_test, y_val, df)

    # Plot learning curves
    plot_learning_curve(results[0])

    # Plot confusion matrices
    plot_confusion_matrix(results[1])

    # Plot classification reports
    plot_classification_report(results[1])

    # save to file
    save_results(results)


# Evaluate classifiers


def evaluate_classifiers(classifiers, X_train, X_test, X_val, y_train, y_test, Y_val, df):

    lc_inputs = []  # learning curve inputs
    cm_cr_inputs = []   # confusion matrix and classification report inputs

    n = len(classifiers)
    for i, (name, clf) in enumerate(classifiers.items()):

        print(f"Evaluating {name} ({i+1}/{len(classifiers)})")
        model = make_pipeline(MinMaxScaler(), clf)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred, normalize=True)
        name = name + f' (Accuracy: {accuracy:.2f}%)'

        # learning curves
        train_sizes = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
        train_scores, validation_scores = calc_learning_curves(
            model, df, df.columns[:-1], df.columns[-1],
            train_sizes, cv=5)
        lc_inputs.append([name, train_sizes, train_scores, validation_scores])

        # confusion matrix
        cm_cr_inputs.append([name, y_test, y_pred])

    return lc_inputs, cm_cr_inputs


def calc_learning_curves(estimator, data, features, target, train_sizes, cv):

    train_sizes, train_scores, validation_scores = learning_curve(
        estimator, data[features], data[target], train_sizes=train_sizes,
        cv=cv, scoring='accuracy')

    train_scores_mean = train_scores.mean(axis=1)
    validation_scores_mean = validation_scores.mean(axis=1)

    return train_scores_mean, validation_scores_mean


def add_learning_curve_subplot(name, i, ax1, train_sizes, train_scores_mean, validation_scores_mean, scaled_y=False):
    ax1[i//3, i % 3].plot(train_sizes, train_scores_mean,
                          label='Training')
    ax1[i//3, i % 3].plot(train_sizes, validation_scores_mean,
                          label='Cross-validation')
    ax1[i//3, i % 3].set_title(name)
    ax1[i//3, i % 3].set_xlabel('Training Size')
    ax1[i//3, i % 3].set_ylabel('Error')
    ax1[i//3, i % 3].legend(loc='best')

    if scaled_y:
        ax1[i//3, i % 3].set_ylim([0, 1])

    return ax1


def add_confusion_matrix_subplot(name, i, ax2, y_test, y_pred):
    cm = pd.crosstab(
        y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])

    sn.heatmap(cm, annot=True,
               ax=ax2[i//3, i % 3],
               xticklabels=LABELS, yticklabels=LABELS)

    ax2[i//3, i % 3].set_title(name)

    ax2[i//3, i % 3].set_xlabel('')
    ax2[i//3, i % 3].set_ylabel('')

    return ax2


def add_classification_report_subplot(name, i, ax3, y_test, y_pred):
    cr = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        target_names=LABELS)

    sn.heatmap(pd.DataFrame(cr).iloc[:-1, :-2].T,
               annot=True, ax=ax3[i//3, i % 3])
    ax3[i//3, i % 3].set_title(name)
    ax3[i//3, i % 3].set_xlabel('Metrics')
    ax3[i//3, i % 3].set_ylabel('Classes')

    # set the spacing between subplots to zero
    plt.subplots_adjust(wspace=0, hspace=0)

    return ax3


def save_results(results):
    cm, lc = results

    with open(r'lc_inputs.pkl', 'wb') as f:
        pkl.dump(lc, f)

    with open(r'cr_cm_inputs.pkl', 'wb') as f:
        pkl.dump(cm, f)


def load_results():
    lc, cm = None, None

    with open(r'lc_inputs.pkl', 'rb') as f:
        lc = pkl.load(f)

    with open(r'cr_cm_inputs.pkl', 'rb') as f:
        cm = pkl.load(f)

    return lc, cm


def plot_learning_curve(lc_results, title='Learning Curves', scaled_y=False):
    n = len(lc_results)
    fig, ax = plt.subplots(n//3, 3)

    for i, (name, train_sizes, train_scores, validation_scores) in enumerate(lc_results):
        add_learning_curve_subplot(
            name, i, ax, train_sizes, train_scores, validation_scores, scaled_y)

    fig.suptitle(title)
    fig.tight_layout()


def plot_confusion_matrix(cm_results, title='Confusion Matrix'):
    n = len(cm_results)
    fig, ax = plt.subplots(n//3, 3)
    for i, (name, y_test, y_pred) in enumerate(cm_results):
        add_confusion_matrix_subplot(name, i, ax, y_test, y_pred)

    fig.suptitle(title)
    fig.tight_layout()


def plot_classification_report(cr_results, title='Classification Report'):
    n = len(cr_results)
    fig, ax = plt.subplots(n//3, 3, figsize=(8, 8))
    for i, (name, y_test, y_pred) in enumerate(cr_results):
        add_classification_report_subplot(name, i, ax, y_test, y_pred)

    fig.suptitle(title)
    fig.tight_layout(rect=[1, 1, 1, 1])


if __name__ == '__main__':
    # path = r'C:\Users\Matheus\Desktop\NanoTechnologies_Lab\Phase A\data\generated_results\07-11-2024\COMBINED_generated.csv'
    # main(path)

    # """
    cm, lc = load_results()
    plot_learning_curve(lc, title='Loaded Learning Curves', scaled_y=True)
    plot_learning_curve(lc, title='Loaded Learning Curves')
    plot_confusion_matrix(cm, title='Loaded Confusion Matrix')
    plot_classification_report(cm, title='Loaded Classification Report')
    plt.show()
    # """
