import gensim
# let X be a list of tokenized texts (i.e. list of lists of tokens)

from gensim.models import Word2Vec
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from TfIdfEmbeddingVectorizer import TfidfEmbeddingVectorizer

def getTrainCoupus(path):
    corpus = []
    for line in open(path):
        fields = line.rstrip().split('\t')
        doc_string = fields[3] + '\t' + fields[2]
        corpus.append(doc_string)
    return corpus

# data = Word2Vec.load('C:\Users\Dmitry\PycharmProjects\\nlp_drugs_side_effects\classificator\model\model.txt')
model = Word2Vec.load('C:\Users\Dmitry\PycharmProjects\\nlp_drugs_side_effects\classificator\model\model.txt')
w2v = dict(zip(model.wv.index2word, model.wv.syn0))
x_train = []
y_train = []
corpust = getTrainCoupus('C:\Users\Dmitry\PycharmProjects\\nlp_drugs_side_effects\parser\sources\loaded_tweets.txt')
for line in corpust:
    fields = line.rstrip().split('\t')
    x_train.append(fields[0])
    y_train.append(fields[1])
vectorizer = TfidfEmbeddingVectorizer(w2v)
etree_w2v_tfidf = Pipeline([
    ("word2vec vectorizer", vectorizer),
    ("linear svc", LinearSVC())])
etree_w2v_tfidf.fit(x_train, y_train)