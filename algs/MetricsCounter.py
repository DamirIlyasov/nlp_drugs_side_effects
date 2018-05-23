import os
import pickle

from sklearn.metrics import classification_report

"""Gets classified test corpus and makes report with metrics"""


def count_metrics(trained_model_path, test_file):
    print("Loading trained model...")
    predicting_model = pickle.load(open(trained_model_path, 'rb'))

    x_test = []
    print("Parsing data")
    for line in open(os.path.join(os.path.dirname(__file__), '..', 'sources', test_file),
                     encoding='utf-8'):
        fields = line.rstrip().split('\t')
        x_test.append(fields[1])

    print("Predicting")
    x_pred = predicting_model.predict(x_test)

    f2 = open(os.path.join(os.path.dirname(__file__), '..', 'results', 'report.txt'), 'w+',
              encoding='utf-8')
    print(classification_report(x_test, x_pred), file=f2)
