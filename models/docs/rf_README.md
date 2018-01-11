# Random forests

This script will train a random forest on the corpus and then return its accuracy on the test data. 

## Arguments

The script has 3 positional arguments you'll need to enter for it to run:

  1. ```data```: The path for the CSV file holding the document text and the target variable.
  2. ```y_name```: The name of the column holding the target variable, like 'outcome' or 'sentiment'.
  3. ```x_name```: The name of the column holding the text, like 'docs' or 'evaluations'.

There are also a number of optional arguments that allow for model customization and tuning:

  1. ```-lm, --limit_features```: If 'yes', this will limit the size of the feature vectors for each document to whatever is  specified by the -ft, --max_features argument. Most authors will use the full feature set (i.e. corpus vocabulary) when reporting benchmark results, but reducing it will make training faster and will in some cases improve your accuracy. The default is 'yes'.
  2. ```-ft, --features```: As above; the number of maximum features for the vectorizer to calculate (default is 10,000).
  3. ```-vc, --vec_meth```: Method for vectorizing the text (count or tfidf).
  4. ```-tr, --n_trees```: Number of trees to use in the RF.
  5. ```-ng, --ngrams```: The maximum size of ngrams to consider. Note that the minimum size is always 1, and the default is 2.
  6. ```-sm, --split_method```: The method for splitting the data into training, validation, and test sets. The default is 'train-test', which calls sklearn's train_test_split() function, but 'cross-val' and 'other' maybe also be used. The second option will perform the same train-test split but report the mean cross-validation accuracy during training; and the third will split the data according to the levels of the user-specified column variable, like 'year' or 'sex'.
  7. ```-sv, --split_variable```: The column variable to be used for splitting the data when -sm, --split_method is 'other'.
  8. ```-tv, --test_val```: The level of -sv, --split_var to use for the test data, with all other levels being used for training and, when selected, cross-validation.
