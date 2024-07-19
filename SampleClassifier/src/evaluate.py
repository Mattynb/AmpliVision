from numpy import mean
from pandas import crosstab, DataFrame

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score, classification_report

# classifiers
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


class ClassifierEvaluator:

    def __init__(self, df):
        self.classifiers = {
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

        self.TRAINING_SIZES = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
        self.DATA = df

        self.calc_inputs = None

    def evaluate_classifiers(self, X_train, X_test, y_train, y_test):
        """ Evaluate classifiers using learning curves, confusion matrix, and classification report """

        t = len(self.classifiers)
        calc_inputs = []

        # iterate through the classifiers
        for i, (name, clf) in enumerate(self.classifiers.items()):

            print(f"Evaluating {name} ({i+1}/{t})")

            # scale the data, fit the model, and test
            model = make_pipeline(MinMaxScaler(), clf)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # adding accuracy to the name
            accuracy = accuracy_score(y_test, y_pred, normalize=True)
            name = name + f' (Accuracy: {accuracy:.2f}%)'

            # inputs
            calc_inputs.append({
                'i': i,
                'name': name,
                'model': model,
                'y_pred': y_pred,
                'y_test': y_test,
            })

        self.calc_inputs = calc_inputs

    def calc_learning_curves(self, i, scorer='accuracy', cv=None):
        """ Calculate learning curves for a given model to be used in the plot_learning_curve method """

        # get the model, target, and features
        model = self.calc_inputs[i]['model']
        target = self.DATA.columns[0]
        features = self.DATA.columns[:-1]

        # calculate learning curves
        train_sizes, train_scores, validation_scores = learning_curve(
            model, self.DATA[features], self.DATA[target], cv=cv, scoring=scorer, train_sizes=self.TRAINING_SIZES
        )

        # calculate the mean of the scores
        train_scores_mean = mean(train_scores, axis=1)
        validation_scores_mean = mean(validation_scores, axis=1)

        return train_sizes, train_scores_mean, validation_scores_mean

    def calc_confusion_matrix(self, i):
        """ Calculate confusion matrix for a given model to be used in the plot_confusion_matrix method """

        y_test = self.calc_inputs[i]['y_test']
        y_pred = self.calc_inputs[i]['y_pred']

        cm = crosstab(
            y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])

        return DataFrame(cm)

    def calc_classification_report(self, i):
        """ Calculate classification report for a given model to be used in the plot_classification_report method """
        LABELS = [
            "brst", "ctrl", "lung",
            "ovrn", "prst", "skin", "tyrd"
        ]

        y_test = self.calc_inputs[i]['y_test']
        y_pred = self.calc_inputs[i]['y_pred']

        cr = classification_report(
            y_test, y_pred, output_dict=True, target_names=LABELS)

        return DataFrame(cr)
