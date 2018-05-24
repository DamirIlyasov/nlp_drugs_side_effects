import nltk
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from analysis.analizer import getNGrammVocabulary


class AvgFeatureVec(BaseEstimator, TransformerMixin):

    def __init__(self, model, num_features):
        self.model = model
        self.num_features = num_features
        self.language = None

    def get_feature_names(self):
        feature_names = []
        for i in range(self.num_features):
            feature_names.append('feature_%s' % i)
        return feature_names

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        features = []
        for text in X:
            features.append(
                self.makeFeatureVec(getNGrammVocabulary(text, 'stem', self.language, 1),
                                    self.model,
                                    self.num_features))
        return features

    def makeFeatureVec(self, words, model, num_features):
        featureVec = np.zeros((num_features,), dtype="float64")
        nwords = 0.
        index2word_set = set(model.wv.vocab)
        for word in words:
            if word in index2word_set:
                nwords += 1.
                featureVec = np.add(featureVec, model[word])
        if nwords > 0:
            featureVec = np.divide(featureVec, nwords)
        return featureVec

    def setLanguage(self, language):
        self.language = language
