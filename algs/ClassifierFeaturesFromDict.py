from sklearn.feature_extraction.text import CountVectorizer
import os
import pickle

from gensim.models import Word2Vec
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from analysis import analizer
from algs.Util import text_process


def classify_with_features(train_file):
    print("Loading w2v")
    model = Word2Vec.load(os.path.join(os.path.dirname(__file__), '..', 'sources', 'model_w2v.txt'))
    w2v = dict(zip(model.wv.index2word, model.wv.syn0))
    x_train = []
    y_train = []

    print("Parsing data")
    for line in open(os.path.join(os.path.dirname(__file__), '..', 'sources', 'loaded_tweets_parsed.txt'),
                     encoding='utf-8'):
        fields = line.rstrip().split('\t')
        x_train.append(fields[1])
        y_train.append(fields[0])

    X_train_new, X_test, y_train_new, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=101)

    svm_w2v_tfidf = Pipeline([('feats', FeatureUnion([
        ('vocab', CountVectorizer(vocabulary=set(
            analizer.__parseDictionary(
                dictionary=os.path.join(os.path.dirname(__file__), '..', 'data', 'SideEffectsEng.txt'),
                allowCompositeValues=False)))),
        ('tf_idf_vect', TfidfVectorizer(analyzer=text_process)),

    ])),
                              ("linear svc", LinearSVC())
                              ])

    # print("Training SVM..")
    # svm_w2v_tfidf.fit(X_train_new, y_train_new)
    filename = 'trained_model_features.sav'

    # print('Saving trained model..')
    # pickle.dump(svm_w2v_tfidf, open(filename, 'wb'))  # - this worked

    print("Loading trained model...")
    # load the model from disk
    svm_w2v_tfidf = pickle.load(open(filename, 'rb'))

    x_pred = svm_w2v_tfidf.predict(X_test)

    print("Printing results")
    f1 = open(os.path.join(os.path.dirname(__file__), '..', 'results', 'features_predicted_tweets.txt'), 'w+',
              encoding='utf-8')

    print("\n".join('%s' % x for x in list(zip(x_pred, X_test))), file=f1)
    # print(list(zip(x_pred, X_test)), sep='\n', file=f1)

    f2 = open(os.path.join(os.path.dirname(__file__), '..', 'results', 'features_report.txt'), 'w+',
              encoding='utf-8')
    print(classification_report(y_test, x_pred), file=f2)

    f3 = open(os.path.join(os.path.dirname(__file__), '..', 'results', 'features_score.txt'), 'w+',
              encoding='utf-8')
    print(svm_w2v_tfidf.score(X_test, y_test), file=f3)
