import os

from gensim.models import KeyedVectors
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import numpy

from algs.AvgFeatureVec import AvgFeatureVec
from algs.FeatureUnionBuilder import FeatureUnionBuilder

feature_union_builder = FeatureUnionBuilder('english', 1, 'stem', 1)
model = KeyedVectors.load_word2vec_format(os.path.join(os.path.dirname(__file__), '..', 'sources', 'PubMed-w2v.bin'), binary=True)
x_train = []
y_train = []

print("Parsing data")
for line in open('C:\\Users\Dmitry\PycharmProjects\\nlp_drugs_side_effects\sources\eng_data\\4\\4_train.txt', encoding="utf-8"):
    fields = line.rstrip().split('\t')
    x_train.append(fields[1])
    y_train.append(fields[0])
avgFeatVec = AvgFeatureVec(model, 200)
w2v_train = avgFeatVec.transform(x_train)
svm_model = LinearSVC()
svm_model.fit(w2v_train,y_train)
x_test = []
y_test = []
for line in open('C:\\Users\Dmitry\PycharmProjects\\nlp_drugs_side_effects\sources\eng_data\\4\\4_test.txt', encoding="utf-8"):
    fields = line.rstrip().split('\t')
    x_test.append(fields[1])
    y_test.append(fields[0])
w2v_test = avgFeatVec.transform(x_test)
y_pred = svm_model.predict(w2v_test)
vectorizer = TfidfVectorizer()
vectorizer_train = vectorizer.fit_transform(x_train, y_train)
vectorizer_test = vectorizer.transform(x_test)
train = numpy.hstack((vectorizer_train.toarray(),w2v_train))
test = numpy.hstack((vectorizer_test.toarray(),w2v_test))
svm_model.fit(train,y_train)
y_pred1 = svm_model.predict(test)
report = classification_report(y_test,y_pred)
report1 = classification_report(y_test,y_pred1)