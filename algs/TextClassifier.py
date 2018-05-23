import os
import pickle

"""Takes unclassified corpus and classifies it"""


def predict(trained_model_path, test_text_path, result_path):
    print("Loading trained model...")
    predicting_model = pickle.load(open(trained_model_path, 'rb'))
    x_test = []

    print("Parsing data")
    for line in open(os.path.join(os.path.dirname(__file__), '..', 'sources', test_text_path),
                     encoding='utf-8'):
        x_test.append(line)

    print("Predicting")
    x_pred = predicting_model.predict(x_test)

    print("Printing results")
    f1 = open(os.path.join(os.path.dirname(__file__), '..', 'results', result_path), 'w+',
              encoding='utf-8')
    print("\n".join('%s' % x for x in list(zip(x_pred, x_test))), file=f1)

