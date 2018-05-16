import os
import pickle

from gensim.models import Word2Vec
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC

from algs.AvgFeatureVec import AvgFeatureVec
from algs.Util import text_process

# data = Word2Vec.load('C:\Users\Dmitry\PycharmProjects\\nlp_drugs_side_effects\classificator\model\model.txt')
model = Word2Vec.load(os.path.join(os.path.dirname(__file__), '..', 'sources', 'model_w2v.txt'))
w2v = dict(zip(model.wv.index2word, model.wv.syn0))
x_train = []
y_train = []

for line in open(os.path.join(os.path.dirname(__file__), '..', 'sources', 'loaded_tweets_parsed.txt'),
                 encoding='utf-8'):
    fields = line.rstrip().split('\t')
    x_train.append(fields[1])
    y_train.append(fields[0])

vectorizer = AvgFeatureVec(w2v)

X_train_new, X_test, y_train_new, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=101)

svm_w2v_tfidf = Pipeline([('feats', FeatureUnion([
    ('tf_idf_vect', TfidfVectorizer(analyzer=text_process)),
    ('words_avg_vec', AvgFeatureVec(model, 100))
])),
                          ("linear svc", LinearSVC())
                          ])

svm_w2v_tfidf.fit(X_train_new, y_train_new)
# save trained model
filename = 'trained_model.sav'
pickle.dump(model, open(filename, 'wb'))

# load the model from disk
# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, Y_test)

x_pred = svm_w2v_tfidf.predict(X_test)

f1 = open(os.path.join(os.path.dirname(__file__), '..', 'results', 'predicted_tweets.txt'), 'w+')
print(x_pred, file=f1)
f2 = open(os.path.join(os.path.dirname(__file__), '..', 'results', 'svm_with_word2vec_report.txt'), 'w+')
print(classification_report(X_test, x_pred), file=f2)
