import os

import gensim
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC

from algs.AvgFeatureVec import AvgFeatureVec
from algs.Util import text_process

if __name__ == '__main__':
    print('Reading...')
    yelp = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'sources', 'stackoverflow_sample_125k.tsv'),
                       sep='\t',
                       header=-1)
    yelp = yelp[:5000]
    X = yelp[0]
    y_common = yelp[1]
    y = []
    for tags in y_common:
        y.append(tags.split(' ')[0])

    X_tokens = []
    for text in X:
        X_tokens.append(nltk.word_tokenize(text))

    print('Training w2v...')
    model = gensim.models.Word2Vec(X_tokens, workers=4, size=150)
    f1 = open(os.path.join(os.path.dirname(__file__), '..', 'results', 'word2vec.txt'), 'w+')
    print(model.most_similar("android"), file=f1)
    print(model.most_similar("java"), file=f1)
    print(model.most_similar("program"), file=f1)
    f1.close()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101
                                                        )
    feat_union = FeatureUnion([
        ('count_vect', TfidfVectorizer(analyzer=text_process)),
        ('words_avg_vec', AvgFeatureVec(model, 150))
    ])
    clf = Pipeline([('feats', feat_union),
                    ('clf', LinearSVC(multi_class='ovr'))
                    ])

print('Training Linear SVC...')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
f2 = open(os.path.join(os.path.dirname(__file__), '..', 'sources', 'svm_with_word2vec.txt'), 'w+')
print(classification_report(y_test, y_pred), file=f2)
f2.close()
print('Done')
