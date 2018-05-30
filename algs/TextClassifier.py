"""Takes unclassified corpus and classifies it"""
import os
import pickle

from sklearn.externals import joblib


def predict(trained_model_path, test_text_path, result_path):
    print("Loading trained model...")
    predicting_model = joblib.load(open(trained_model_path, 'rb'))
    x_test = []

    print("Parsing data")
    for line in open(os.path.join(os.path.dirname(__file__), '..', 'sources', test_text_path),
                     encoding='utf-8'):
        x_test.append(line)

    print("Predicting")
    y_pred = predicting_model.predict(x_test)

    print("Printing results")
    f1 = open(os.path.join(os.path.dirname(__file__), '..', 'results', result_path), 'w+',
              encoding='utf-8')
    mylist = list(zip(y_pred, x_test))
    print('\n'.join([str(line) for line in mylist]), file=f1)


predict("rus_1.sav", "rus.txt", "result_predicted")
