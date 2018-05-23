from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from algs.AvgFeatureVec import AvgFeatureVec
from analysis.analizer import getNGrammVocabulary, stemVocab


def getFeatureUnion(tfidf_vectorizer, vocab_model):
    return FeatureUnion([("tfidf_vectorizer", tfidf_vectorizer),
                         ("vocab", vocab_model)])


def getFeatureUnion(tfidf_vectorizer):
    return FeatureUnion([("tfidf_vectorizer", tfidf_vectorizer)])


def getFeatureUnion(tfidf_vectorizer, vocab_model, word2vec_model):
    return FeatureUnion([("tfidf_vectorizer", tfidf_vectorizer),
                         ("vocab", vocab_model),
                         ("word2vec", word2vec_model)])


class FeatureUnionBuilder:
    def __init__(self, language, ngramm, word_type):
        self.language = language
        self.ngramm = ngramm
        self.wordType = word_type

    def getTfidfVectorizer(self):
        return TfidfVectorizer(analyzer=self.text_analyzer)

    def getVocabModel(self, vocab_path):
        return CountVectorizer(vocabulary=stemVocab(vocab_path, self.language))

    def getWord2VecModel(self, model, size):
        return AvgFeatureVec(model, size)

    def text_analyzer(self, text):
        return getNGrammVocabulary(text, self.wordType, self.language, self.ngramm)
