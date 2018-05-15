import nltk
from nltk.corpus import stopwords
from nltk.stem import snowball
from scipy import *
from scipy.sparse import *
import scipy.sparse as sps
import time


def analyze(language, input, output='', isMapped=True):
    docCount = 0
    firstClassCount = 0
    secondClassCount = 0

    with open(input, "r", encoding='utf-8') as infile:
        totalWords = []
        for line in infile:
            docCount = docCount + 1

            words = getPreProcessedWordsFromDocument(line, language)

            if isMapped:
                if words[0].__eq__('0'):
                    firstClassCount = firstClassCount + 1
                if words[0].__eq__('1'):
                    secondClassCount = secondClassCount + 1
                words.__delitem__(0)

            totalWords.extend(words)
        uniqueWords = set(totalWords)
        print(totalWords)
    wordCount = len(totalWords)
    uniqueWordsCount = len(uniqueWords)

    result = {
        'number of words': wordCount,
        'number of docs': docCount,
        'number of unique words': uniqueWordsCount,
        'docs in fist class': firstClassCount,
        'docs in second class': secondClassCount
    }
    if not output.__eq__(''):
        open(output, 'w', encoding='utf-8') \
            .write(result.__str__())

    return result


def getStemmer(language):
    return {
        'russian': snowball.RussianStemmer(),
        'english': snowball.EnglishStemmer()
    }.get(language)


def getStopWords(language):
    options = {
        'russian': set(stopwords.words('russian')),
        'english': set(stopwords.words('english'))
    }
    if not language in options:
        raise ValueError('unknown language')
    stop = options.get(language)
    stop.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'
                    , '%', '$', '#', '№', '*', '^', '@', '+', '-', '\'s', '\'m', '\'', '...', '\"'])
    return stop


def getPreProcessedWordsFromDocument(document, language='english'):
    stop = getStopWords(language)
    stemmer = getStemmer(language)
    return [stemmer.stem(word.lower()) for word in nltk.word_tokenize(document)
            if not stop.__contains__(word)]


def parseDictionary(dictionary, language='english'):
    dictionaryList = []
    with open(dictionary, 'r', encoding='utf-8') as difile:
        for line in difile:
            dictionaryList.append(getPreProcessedWordsFromDocument(line, language))
    return dictionaryList


def getEntranceMatrix(input, dictionary, language='english', output=''):
    start_time = time.time()
    parsedDictionary = parseDictionary(dictionary, language)
    rowCount = 0
    columnCount = parsedDictionary.__len__()
    i = 0

    elementaryShape = [[0 for x in range(columnCount)] for y in range(rowCount)]
    entranceArray = elementaryShape

    with open(input, 'r', encoding='utf-8') as infile:
        for line in infile:

            if entranceArray == []:
                entranceArray = [[0 for x in range(columnCount)] for y in range(1)]
            else:
                entranceArray = addRow(array=entranceArray)

            words = getPreProcessedWordsFromDocument(line, language)
            adrNumber = 0
            for adr in parsedDictionary:
                if not (set(words) & set(adr)).__len__() == 0:
                    entranceArray[i][adrNumber] = entranceArray[i][adrNumber] + 1
                adrNumber = adrNumber + 1
            i = i + 1

    if not output.__eq__(''):
        open(output, 'w', encoding='utf-8') \
            .write(entranceArray)

    print("--- %s seconds ---" % (time.time() - start_time))
    return csr_matrix(entranceArray)


def addRow(array, amount=1):
    if len(array) == 1:
        return [[0 for x in range(len(array[0]))] for y in range(2)]
    rowArray = zeros([amount, len(array[0])])
    res = sps.vstack((array, rowArray))
    return res.toarray()


getEntranceMatrix("test/loaded_tweets_parsed.txt", "dictionaries/ADR_dictionary_en.txt")
# print(getEntranceMatrix("1", "2"))
# print(analyze("english", "test/loaded_tweets_parsed.txt"))
