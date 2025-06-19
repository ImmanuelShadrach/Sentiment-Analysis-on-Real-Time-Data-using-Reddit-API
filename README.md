1. dataset.csv contains the raw data obtained from Kaggle  
2. data_prep.py does the preprocessing on this raw data and saves preprocessed data to preprocessed.csv  
3. model.py takes the preprocessed data and trains the model on it using tf-idf vectorizer and logistic regression for classification  
4. joblib files contain the trained model and vectorizer  
5. app.py is the main script that loads the trained model and vectorizer to make predictions and visualize the results  
