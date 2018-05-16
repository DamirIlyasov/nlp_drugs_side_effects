import nltk
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class AvgFeatureVec(BaseEstimator, TransformerMixin):

    def __init__(self, model, num_features=100):
        self.model = model
        self.num_features = num_features

    def get_feature_names(self):
        feature_names = []
        for i in range(100):
            feature_names.append('feature_%s' % i)
        return feature_names

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        features = []
        for text in X:
            features.append(
                self.makeFeatureVec(nltk.word_tokenize(text),
                                    self.model,
                                    self.num_features))
        return features

    def makeFeatureVec(self, words, model, num_features):
        featureVec = np.zeros((num_features,), dtype="float32")
        nwords = 0.
        index2word_set = set(model.wv.vocab)
        for word in words:
            if word in index2word_set:
                nwords += 1.
                featureVec = np.add(featureVec, model[word])

        featureVec = np.divide(featureVec, nwords)
        return featureVec