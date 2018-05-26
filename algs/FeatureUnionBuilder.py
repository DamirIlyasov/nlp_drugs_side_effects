from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from algs.AvgFeatureVec import AvgFeatureVec
from analysis.analizer import getNGrammVocabulary, stemVocab


def getFeatureUnionWithVocab(tfidf_vectorizer, vocab_model):
    return FeatureUnion([("tfidf_vectorizer", tfidf_vectorizer),
                         ("vocab", vocab_model)])


def getFeatureUnion(tfidf_vectorizer):
    return FeatureUnion([("tfidf_vectorizer", tfidf_vectorizer)])


def getFeatureUnion(tfidf_vectorizer, vocab_model, word2vec_model):
    return FeatureUnion([("tfidf_vectorizer", tfidf_vectorizer),
                         ("vocab", vocab_model),
                         ("word2vec", word2vec_model)])


class FeatureUnionBuilder:
    def __init__(self, language, ngramm, word_type, unknown_word_freq):
        self.language = language
        self.ngramm = ngramm
        self.wordType = word_type
        self.minDf = unknown_word_freq

    def getTfidfVectorizer(self):
        return TfidfVectorizer(preprocessor=self.text_analyzer, min_df=self.minDf, ngram_range=(1, self.ngramm))

    def getVocabModel(self, vocab_path):
        return CountVectorizer(vocabulary=set(stemVocab(vocab_path, self.language)), min_df=self.minDf)

    def getWord2VecModel(self, model, size):
        avg_feature_vec = AvgFeatureVec(model, size)
        avg_feature_vec.setLanguage(self.language)
        return avg_feature_vec

    def text_analyzer(self, text):
        return getNGrammVocabulary(text, self.wordType, self.language, self.ngramm)
