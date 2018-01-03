import numpy as np
import pandas as pd

from document_classification.nbsvm import NBSVM
from document_classification.rf import TextRF
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from document_classification.tools import diagnostics

# Pairs a sequence of binary values with their opposites
def opposites(targets):
    opps = np.zeros(targets.shape, dtype='int64')
    for i in range(len(targets)):
        opps[i] = (targets[i] - 1) * -1
    return np.array(zip(opps, targets))

# Importing the data
docs = pd.read_csv('~/data/addm/corpus_with_lemmas_clean.csv').iloc[:, 1:]
targets = np.array(docs['aucaseyn'])

# Converting phrases to BoF with TF-IDF normalization
vec = CountVectorizer(binary=True, max_features=None, ngram_range=(1, 2))
features = vec.fit_transform(docs['dx']).toarray()

# Getting seeds for the train-test loops
np.random.seed(10221983)
folds = KFold().split(features)

# Making an empty data frame to hold the statistics from each run
nbsvm_probs = np.zeros([features.shape[0], 2])
rf_probs = np.zeros([features.shape[0], 2])
mnb_probs = np.zeros([features.shape[0], 2])

# Running the cross-validation loop
for train, test in folds:
    #Splitting the data
    X_train, y_train = features[train, :], targets[train]
    X_test, y_test = features[test, :], targets[test]
    
    # Instantiating the models
    nbsvm = NBSVM()
    rf = TextRF()
    mnb = MultinomialNB()
    
    # Fitting the models
    nbsvm.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    mnb.fit(X_train, y_train)
    
    # Getting the predicted probabilities
    nbsvm_probs[test, 1] = nbsvm.predict_proba(X_test, y_test)
    rf_probs[test, 1] = rf.predict_proba(X_test)[:, 1]
    mnb_probs[test, 1] = mnb.predict_proba(X_test)[:, 1]

    
    