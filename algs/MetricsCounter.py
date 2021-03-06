import os

from sklearn.externals import joblib
from sklearn.metrics import classification_report

"""Gets classified test corpus and makes report with metrics"""


def count_metrics(trained_model_path, test_file):
    print("Loading trained model...")
    predicting_model = joblib.load(open(trained_model_path, 'rb'))

    x_test = []
    y_test = []
    print("Parsing data")
    for line in open(os.path.join(os.path.dirname(__file__), '..', 'sources', test_file),
                     encoding='utf-8'):
        fields = line.rstrip().split('\t')
        x_test.append(fields[1])
        y_test.append(fields[0])

    print("Predicting")
    y_pred = predicting_model.predict(x_test)

    f2 = open(os.path.join(os.path.dirname(__file__), '..', 'results', 'report.txt'), 'w+',
              encoding='utf-8')
    print(classification_report(y_test, y_pred), file=f2)


count_metrics('rus_5.sav', 'rus_classified_4_6\\5\\test.txt')
