'''A script for testing the accuracy of different classifiers'''
from models.tools import *
from models.lsa import decompose
from models.nbsvm import NBSVM, TextMNB

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import roc_curve, auc

# Quick function for doing reverse lookup on a dict
def key_from_value(value, dict):
    return dict.keys()[dict.values().index(value)]

'''Prepping the data'''
# Setting the parameters for the loop
n_runs = 10

# Bringing in the data
corpus = pd.read_csv(BOW FILE GOES HERE)

# Making a full-featuerd version of the corpus with counts and tfidf
d = TextData()
d.process(corpus, 'dx', 'aucaseyn', method='counts', max_features=None, binary=False, ngrams=2)
vocab = d.vocabulary_

# Making the TF-IDF data for the random forest and LSA classifiers
didf = TextData()
didf.process(corpus, 'dx', 'aucaseyn', method='tfidf', max_features=None, ngrams=2)

# Making the LDA data
LDA = LatentDirichletAllocation
dlda = TextData()
dlda.process(corpus, 'dx', 'aucaseyn', method='counts', max_features=None, binary=False, ngrams=2)
dlda.set_xy(LDA(learning_method='online', n_topics=20).fit_transform(dlda.X), d.y)

# Making the LSA data
dlsa = TextData()
dlsa.set_xy(decompose(didf.X), didf.y)

'''Comparing the classifiers'''
# Holders for model-specific statistics
stat_columns = ['se', 'sp', 'ppv', 'f1', 'acc']
mnb_stats = pd.DataFrame(np.zeros([n_runs, 5]), columns=stat_columns)
nbsvm_stats = pd.DataFrame(np.zeros([n_runs, 5]), columns=stat_columns)
svm_stats = pd.DataFrame(np.zeros([n_runs, 5]), columns=stat_columns)
lda_stats = pd.DataFrame(np.zeros([n_runs, 5]), columns=stat_columns)
lsa_stats = pd.DataFrame(np.zeros([n_runs, 5]), columns=stat_columns)
rf_stats = pd.DataFrame(np.zeros([n_runs, 5]), columns=stat_columns)

# Holders for the random forest and NBSVM top features
nbsvm_top_features = pd.DataFrame(np.zeros([20, n_runs]))
rf_top_features = pd.DataFrame(np.zeros([20, n_runs]))

# Getting seeds for the train-test loops
np.random.seed(10221983)
seeds = np.random.randint(10e3, size=n_runs)

# Looping through the train-test splits, prioritizing coherence over elegance
for i, seed in enumerate(seeds):
	# Insantiating the models
	nbsvm = NBSVM(C=.1, beta=.95)
	svm = LinearSVC()
	mnb = TextMNB()
	lda = LinearSVC()
	lsa = LinearSVC()
	
	# Splitting the data
	seed = np.random.randint(5e3)
	d.split(seed=seed)
	didf.split(seed=seed)
	dlda.split(seed=seed)
	dlsa.split(seed=seed)
	
	# Fitting the models
	mnb.fit(d.X_train, d.y_train)
	nbsvm.fit(d.X_train, d.y_train)
	svm.fit(d.X_train, d.y_train)
	lsa = lsa.fit(dlsa.X_train, dlsa.y_train)
	lda = lda.fit(dlda.X_train, dlda.y_train)
	
	# Pulling the features for the NBSVM
	w_abs = np.absolute(np.ndarray.flatten(nbsvm.int_coef_))
	nbsvm_top_features.iloc[:, i] = pd.Series([key_from_value(val, vocab) for val in np.argsort(w_abs)[-20:]])
	
	# Running the tuning procedure for the random forest
	rf = RandomForestClassifier(n_estimators=1000, class_weight='balanced', n_jobs=-1)
	rf.fit(didf.X_train, didf.y_train)
	top_features = np.argsort(rf.feature_importances_)[-100:]
	rf_top_features.iloc[:, i] = pd.Series([key_from_value(val, vocab) for val in top_features[-20:]])
	
	# Training a new random forest on only the top features
	best_rf = RandomForestClassifier(n_estimators=3000, class_weight='balanced', n_jobs=-1)
	best_rf.fit(didf.X_train[:, top_features], didf.y_train)
	best_rf.score(didf.X_test[:, top_features], didf.y_test)
	
	# Scoring the models
	mnb_stats.ix[i] = model_diagnostics(mnb, d.X_test, d.y_test).values
	rf_stats.loc[i] = model_diagnostics(best_rf, didf.X_test[:, top_features], didf.y_test).values
	nbsvm_stats.ix[i] = model_diagnostics(nbsvm, d.X_test, d.y_test).values
	svm_stats.ix[i] = model_diagnostics(svm, d.X_test, d.y_test).values
	lda_stats.ix[i] = model_diagnostics(lda, dlda.X_test, dlda.y_test).values
	lsa_stats.ix[i] = model_diagnostics(lsa, dlsa.X_test, dlsa.y_test).values
