import os

from sklearn.externals import joblib
from sklearn.metrics import classification_report

"""Gets classified test corpus and makes report with metrics"""


def count_metrics(trained_model_path, test_file):
    print("Loading trained model...")
    predicting_model = joblib.load(open(trained_model_path, 'rb'))

    x_train = []
    y_train = []
    print("Parsing data")
    for line in open(os.path.join(os.path.dirname(__file__), '..', 'sources', test_file),
                     encoding='utf-8'):
        try:
            fields = line.rstrip().split('\t')
            x_train.append(fields[1])
            y_train.append(fields[0])
        except IndexError:
            print(fields)

    print("Predicting")
    y_pred = predicting_model.predict(x_train)

    f2 = open(os.path.join(os.path.dirname(__file__), '..', 'results', 'report.txt'), 'w+',
              encoding='utf-8')
    print(classification_report(y_train, y_pred), file=f2)


count_metrics('trained_model_rus.sav', 'tweets_set_predict.txt')
