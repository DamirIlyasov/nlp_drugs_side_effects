import os
import pickle

from gensim.models import Word2Vec
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from algs.FeatureUnionBuilder import FeatureUnionBuilder, getFeatureUnion

"""Trains model with given parameters and saves it"""


def train_and_save_model(train_file, text_encoding, word_type, n_gram_range, features, laplace, unknown_word_freq,
                         trained_model_file, language):
    # setting default parameters
    if text_encoding is None:
        text_encoding = 'utf-8'
    if word_type is None:
        word_type = 'stem'
    if n_gram_range is None:
        n_gram_range = 1
    if features is None:
        features = False
    if laplace is None:
        laplace = False
    if unknown_word_freq is None:
        unknown_word_freq = 1

    x_train = []
    y_train = []

    print("Parsing data")
    for line in open(os.path.join(os.path.dirname(__file__), '..', 'sources', train_file),
                     encoding=text_encoding):
        fields = line.rstrip().split('\t')
        x_train.append(fields[1])
        y_train.append(fields[0])

    x_train_new, x_test, y_train_new, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=101)

    feature_union_builder = FeatureUnionBuilder(language, n_gram_range, word_type, unknown_word_freq)
    tfidf_vectorizer = feature_union_builder.getTfidfVectorizer()

    if features:
        print("Loading w2v")
        model = Word2Vec.load(os.path.join(os.path.dirname(__file__), '..', 'sources', 'model_w2v.txt'))
        word2vec_model = feature_union_builder.getWord2VecModel(model, 100)
        vocab_model = feature_union_builder.getVocabModel(
            os.path.join(os.path.dirname(__file__), '..', 'data', 'SideEffectsEng.txt'))
        feature_union = getFeatureUnion(tfidf_vectorizer, vocab_model, word2vec_model)
    else:
        feature_union = getFeatureUnion(tfidf_vectorizer)

    if not laplace:
        svm_w2v_tfidf = Pipeline([('feats', feature_union),
                                  ("linear svc", LinearSVC())
                                  ])
    else:
        svm_w2v_tfidf = Pipeline([('feats', feature_union),
                                  ("multinomialNB", MultinomialNB())
                                  ])

    print("Training SVM..")
    svm_w2v_tfidf.fit(x_train_new, y_train_new)

    print('Saving trained model..')
    pickle.dump(svm_w2v_tfidf, open(trained_model_file, 'wb'))
    print('Model is saved!')
