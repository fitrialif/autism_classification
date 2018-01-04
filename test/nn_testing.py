'''A not-so-fast implementation of fastText in Tensorflow'''
import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from models.tools import diagnostics
from text_helpers import opposites, IndexBatchGenerator

# Quick function for resetting the graph
def reset():
    tf.reset_default_graph()

# Importing the data
docs = pd.read_csv(CORPUS GOES HERE).iloc[:, 1:]
targets = opposites(np.array(docs['aucaseyn']))

# Converting phrases to BoF with TF-IDF normalization
vec = CountVectorizer(binary=True, max_features=None, ngram_range=(1, 2))
features = vec.fit_transform(docs['dx']).toarray()

# Getting seeds for the train-test loops
np.random.seed(10221983)
folds = KFold().split(features)

# Making an empty data frame to hold the statistics from each run
predicted_probs = np.zeros([features.shape[0], 2])

for train, test in folds:
    # Selecting the fold
    X_train, y_train = features[train, :], targets[train, :]
    X_test, y_test = features[test, :], targets[test, :]
        
    # Clearing the graph
    reset()
    
    # Settting the model parameters
    num_features  = features.shape[1]
    embedding_size = 200
    batch_size = 128
    epochs = 15
    learning_rate = 0.1
    num_classes = 2
    display_step = 10
    l2_weight = 0.001
    
    # Initializers to use for the variables
    norm = tf.random_normal_initializer()
    unif = tf.random_uniform_initializer(minval=-.01, maxval=.01)
    zeros = tf.zeros_initializer()
    
    # Placeholders for the feature vectors and the targets
    x = tf.placeholder(tf.float32, [None, num_features])
    y = tf.placeholder(tf.float32, [None, num_classes])
    
    # Initializing the embedding matrix for the features
    embeddings = tf.get_variable('embeddings', [num_features, embedding_size], dtype=tf.float32)
    averaged_features = tf.matmul(x, embeddings)
    
    # Adding the biases and weights for the linear transall_stats.loc[i] = np.array(test_stats)formation
    dense_weights = tf.get_variable('dense_weights', [embedding_size, num_classes], dtype=tf.float32, initializer=unif)
    dense_biases = tf.get_variable('dense_biases', [num_classes], dtype=tf.float32, initializer=zeros)
    dense = tf.matmul(averaged_features, dense_weights) + dense_biases
    
    # Getting the predictions to evaluate accuracy outside of the graph
    probs = tf.nn.softmax(dense)
    preds = tf.argmax(probs, 1)
    
    # Calculating weighted cross-entropy loss; squared weights may be used for L2 regularization
    squared_weights = tf.reduce_sum(tf.square(dense_weights)) + tf.reduce_sum(tf.square(embeddings))
    l2_reg = l2_weight * squared_weights
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense, labels=y) + l2_reg)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    
    # Making a saver for the checkpoints
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
    # Training the model in batches
    with tf.Session() as sess:
        sess.run(init)
        test_dict = {x: X_test, y: y_test}
        best_f1 = 0
        for e in range(epochs):
            print('\nStarting epoch number %s' %e)
            epoch_loss = 0
            X = X_train
            y_ = y_train
            bg = IndexBatchGenerator(range(X.shape[0]), batch_size, shuff=True)
            step = 1
            for batch in bg.batches:
                batch_dict = {x:X[batch], y: y_[batch]}
                sess.run(optimizer, feed_dict=batch_dict)
                cost = sess.run(loss, feed_dict=batch_dict)
                epoch_loss += cost
                step += 1
            mean_epoch_loss = np.true_divide(epoch_loss, step)
            print('Mean epoch loss=%.6f' %mean_epoch_loss)
            print ('Validation stats:')
            val_stats = diagnostics(preds.eval(feed_dict=test_dict), y_test[:, 1])
            print val_stats
            if val_stats['f1'][0] > best_f1:
                best_f1 = val_stats['f1'][0]
                saver.save(sess, 'checkpoints/addm/best_model')
        saver.restore(sess, 'checkpoints/addm/best_model')
        print('\nBest model test stats:')
        test_stats = diagnostics(preds.eval(feed_dict=test_dict), y_test[:, 1])
        print test_stats
        predicted_probs[test, :] = probs.eval(test_dict)
