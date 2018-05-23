import os
import pickle

from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report

"""Gets classified test corpus and makes report with metrics"""


def count_metrics(trained_model_path, test_file):
    print("Loading trained model...")
    predicting_model = pickle.load(open(trained_model_path, 'rb'))

    x_train = []
    y_train = []
    print("Parsing data")
    for line in open(os.path.join(os.path.dirname(__file__), '..', 'sources', test_file),
                     encoding='utf-8'):
        fields = line.rstrip().split('\t')
        x_train.append(fields[1])
        y_train.append(fields[0])
    x_train_new, x_test, y_train_new, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=101)

    print("Predicting")
    y_pred = predicting_model.predict(x_test)

    f2 = open(os.path.join(os.path.dirname(__file__), '..', 'results', 'report.txt'), 'w+',
              encoding='utf-8')
    print(classification_report(y_test, y_pred), file=f2)


count_metrics('trained_model.sav', 'loaded_tweets_parsed.txt')
