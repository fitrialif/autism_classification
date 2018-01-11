# Random forests

This script will train a random forest on the corpus and then return its accuracy on the test data. 

## Arguments

The script has 3 positional arguments you'll need to enter for it to run:

  1. ```data```: The path for the CSV file holding the document text and the target variable.
  2. ```y_name```: The name of the column holding the target variable, like 'outcome' or 'sentiment'.
  3. ```x_name```: The name of the column holding the text, like 'docs' or 'evaluations'.

The script also has a number of optional arguments you can use to fine-tune the model's performance:

  1. ```-lm``` or ```--limit_features```: Limit the number of features for the RF? (yes or no).
  2. ```-ft``` or ```--features```: Number of features for the RF, if limited.
  3. ```-vc``` or ```--vec_meth```: Method for vectorizing the text (count or tfidf).
  4. ```-tr``` or ```--n_trees```: Number of trees to use in the RF.
  5. ```-ng``` or ```--ngrams```: Maximum ngram size.
  6. ```-sm``` or ```--split_method```: Split the data by year, train-test, or cross-validation.
  7. ```-sv``` or ```--split_variable```: Variable to use for splitting the data.
  8. ```-tv``` or ```--test_val```: Which value of split_variable to use for the test data.
