import os
import pickle

from gensim.models import Word2Vec, KeyedVectors
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from gensim.models.wrappers.fasttext import FastText

from algs.FeatureUnionBuilder import FeatureUnionBuilder, getFeatureUnion, getFeatureUnionWithVocab
from algs.VocabModel import VocabModel
from analysis.analizer import stemVocab

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

    feature_union_builder = FeatureUnionBuilder(language, n_gram_range, word_type, unknown_word_freq)
    tfidf_vectorizer = feature_union_builder.getTfidfVectorizer()

    if features:
        print("Loading w2v")
        # model = Word2Vec.load(os.path.join(os.path.dirname(__file__), '..', 'sources', 'word2vec_eng_big'))
        # model = KeyedVectors.load_word2vec_format(os.path.join(os.path.dirname(__file__), '..', 'sources', 'wikipedia-pubmed-and-PMC-w2v.bin'), binary=True)
        # model = KeyedVectors.load_word2vec_format(os.path.join(os.path.dirname(__file__), '..', 'sources', 'ruwikiruscorpora_upos_skipgram_300_2_2018_parsed_vec.vec'))
        # word2vec_model = feature_union_builder.getWord2VecModel(model, 200)
        # vocab_model = feature_union_builder.getVocabModel(
        #     os.path.join(os.path.dirname(__file__), '..', 'data', 'EngFromADR.txt'))
        # feature_union = getFeatureUnion(tfidf_vectorizer, vocab_model, word2vec_model)

        feature_union = getFeatureUnionWithVocab(tfidf_vectorizer, VocabModel(stemVocab(
                os.path.join(os.path.dirname(__file__), '..', 'data', 'SideEffectsRus.txt'), 'english')))
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
    svm_w2v_tfidf.fit(x_train, y_train)

    print('Saving trained model..')
    joblib.dump(svm_w2v_tfidf, open(trained_model_file, 'wb'), compress=3, protocol=2)
    print('Model is saved!')


train_and_save_model('rus_classified_4_6/1/train.txt', None, None, 1, False, False, None, 'rus_1.sav', 'russian')
