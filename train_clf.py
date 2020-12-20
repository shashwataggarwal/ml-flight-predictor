import preprocess
import numpy as np
import pandas as pd
import pickle

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import validation_curve

from tqdm.auto import tqdm
tqdm.pandas()


def trainAndGenerateReport(X_train, X_test, y_train, y_test, clf, model_name):
    # Train Model
    clf.fit(X_train, y_train)

    # Calculate Accuracy
    print('Accuracy using {}'.format(model_name))
    print('Training Accuracy: ', clf.score(X_train, y_train))
    print('Testing Accuracy : ', clf.score(X_test, y_test))
    print()

    # Precict on Train
    y_pred = clf.predict(X_test)
    print(metrics.classification_report(
        y_test, y_pred, target_names=['Wait', 'Buy']))

    y_score = clf.predict_proba(X_test)[:, 1]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    fpr, tpr, _ = metrics.roc_curve(y_test, y_score)
    roc_auc = metrics.auc(fpr, tpr)

    # plotting
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for {}'.format(model_name))
    plt.legend(loc="lower right")
    plt.show()

    return clf


def save_model(model, file_name="model"):
    pickle.dump(model, open(f'./Weights/{file_name}.pkl', 'wb'))
    print("Model Saved!")


def load_model(file_name="model"):
    model = pickle.load(open(f'./Weights/{file_name}', 'rb'))
    print("Model Loaded!")
    return model


def get_X_series(airline, days_to_depart, path, day_time):
    print(airline, path, day_time, days_to_depart)
    return X.loc[(X[airline] == 1) & (X[path] == 1) & (X[day_time] == 1) & (X['days_to_depart'] == days_to_depart)]


random_state = 69
df = preprocess.preprocessClf()

X = df.loc[:, df.columns != 'logical']
X = X.loc[:, X.columns != 'std']
X = X.loc[:, X.columns != 'logical_new']

y = df['logical_new']

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state)

    # Logistic Regression
    model_name = 'logistic_regression'
    clf = LogisticRegression(random_state=random_state, n_jobs=-1)
    clf = make_pipeline(StandardScaler(), clf, verbose=True)

    clf = trainAndGenerateReport(
        X_train, X_test, y_train, y_test, clf, model_name)
    save_model(clf, model_name)

    # Random Forest Classifier
    model_name = 'random_forest_classifier'
    clf = RandomForestClassifier(
        max_depth=10, random_state=random_state, n_estimators=100)
    clf = make_pipeline(StandardScaler(), clf, verbose=True)

    clf = trainAndGenerateReport(
        X_train, X_test, y_train, y_test, clf, model_name)
    save_model(clf, model_name)

    # Gaussian NB Classifier
    model_name = 'gaussian_nb_classifier'
    clf = GaussianNB()
    clf = make_pipeline(StandardScaler(), clf, verbose=True)

    clf = trainAndGenerateReport(
        X_train, X_test, y_train, y_test, clf, model_name)
    save_model(clf, model_name)

    # MLP Classifier
    model_name = 'mlp_classifier'
    clf = MLPClassifier(hidden_layer_sizes=[
                        64, 32, 16], learning_rate_init=0.01, random_state=random_state, max_iter=500)
    clf = make_pipeline(StandardScaler(), clf, verbose=True)

    clf = trainAndGenerateReport(
        X_train, X_test, y_train, y_test, clf, model_name)
    save_model(clf, model_name)

    # SVM Classifier/SVC
    model_name = 'svm_classifier'
    clf = SVC(gamma='auto', probability=True)
    clf = make_pipeline(StandardScaler(), clf, verbose=True)

    clf = trainAndGenerateReport(
        X_train, X_test, y_train, y_test, clf, model_name)
    save_model(clf, model_name)
