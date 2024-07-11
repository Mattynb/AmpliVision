import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def main(path: str):

    # Load the data as a pandas DataFrame
    df = pd.read_csv(path)

    # Splitting data into features and labels
    X = df.drop('class', axis=1)
    y = df['class']

    # Splitting data into training and testing sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # List of classifiers to compare
    classifiers = {     
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        #'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, solver='liblinear'),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5, weights='distance'),
        'LDA': LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
        'SVM linear': SVC(kernel='linear', C=1.0),
        'SVM rbf': SVC(kernel='rbf', C=1.0, gamma='scale'),
        'SVM poly': SVC(kernel='poly', C=1.0, degree=3, gamma='scale'),
        'SVM sigmoid': SVC(kernel='sigmoid', C=1.0, gamma='auto')
    }

    print("Evaluating Classifiers", sep='-'*30)
    # Evaluate classifiers
    results = evaluate_classifiers(classifiers, X_train, X_test, y_train, y_test)

    print_results(results)

    results

# Evaluate classifiers
def evaluate_classifiers(classifiers, X_train, X_test, y_train, y_test):
    results = {}
    for i, (name, clf) in enumerate(classifiers.items()):

        results[name] = []

        print(f"Evaluating {name} ({i+1}/{len(classifiers)})")
        model = make_pipeline(MinMaxScaler(), clf)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred, normalize=True)
        results[name].append(accuracy)

        # cross validation
        xv_scores = cross_val_score(model, X_train, y_train, cv=6)
        results[name].append(xv_scores.mean())

        # confusion matrix
        cl_confusion_matrix(y_test, y_pred)

    return results

# Print confusion matrix
def cl_confusion_matrix(y_test, y_pred):
    print(pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted']))
    print("\n")

# predict point
def predict_point(X, name, model):
    # Test data
    test_data_2 = [3.27,5.58,3.4,53.12,97.32,73.52]
    test_data_2 = pd.DataFrame([test_data_2], columns=X.columns)
    
    #print(f"{name}: {model.predict(test_data_2)}")

# Print the accuracy of each classifier
def print_results(results):
    print("\n")
    print('-'*30, "Results", '-'*30)
    if results == None:
        print("No results to display")
        return
    for classifier, results in results.items():
        print(f"{classifier}:\nAccuracy = {results[0]:.2f} | Cross Validation = {results[1]:.2f}\n")
    print('-'*70)

if __name__ == '__main__':
    path = r'C:\Users\Matheus\Desktop\NanoTechnologies_Lab\Phase A\data\generated_results\07-11-2024\COMBINED_generated.csv'
    main(path)