import numpy
from sklearn.base import BaseEstimator, TransformerMixin

from analysis.analizer import surfaceNoPm


class VocabModel(BaseEstimator, TransformerMixin):

    def __init__(self, vocab):
        self.vocab = vocab

    def fit(self, x, y=None):
        return self

    def transform(self, text):
        features = []
        for lemma in text:
            count = 0
            lemma = lemma.split(" ")
            for l in lemma:
                if l in self.vocab:
                    count += 1
            feature = [count]
            features.append(feature)
        return numpy.true_divide(features, numpy.amax(features))