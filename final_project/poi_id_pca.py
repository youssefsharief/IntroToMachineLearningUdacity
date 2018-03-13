
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score


def determine_the_optimum_number_of_pac_components(model, features, labels):
    from sklearn.cross_validation import train_test_split
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42)

    pipe = Pipeline(steps=[('pca', PCA(n_components=12)), ('model', model)])

    pipe.fit(features_train, labels_train)
    preds = pipe.predict(features_test)
    print("Recall Score:", recall_score(labels_test, preds))
    print("Precision Score:", precision_score(labels_test, preds))
    print("F1 Score:", f1_score(labels_test, preds))
