import nltk
from nltk.corpus import stopwords
from nltk.stem import snowball
from scipy import *
from scipy.sparse import *
import scipy.sparse as sps
import time
from sklearn.feature_extraction.text import CountVectorizer


punctuation = {'.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', '%', '$', '#', '№', '*', '^', '@',
               '+', '-', '\'s', '\'m', '\'', '...', '\"'}


"""""
    Функция для анализа корпуса
    
    Parameters
    ----------
    language : str = язык, используемый в корпусе (2 возможных параметра : 'russian', 'english')
    input : str = путь до файла с корпусом
    output : str = путь до файла для выгрузки данных (не обязательно)
    isMapped : boolean = размечен(классифицирован) ли корпус
    
    Returns
    -------
    result : dict = словарь с результатами анализа
"""


def analyze(language, input, output='', isMapped=True):
    docCount = 0
    firstClassCount = 0
    secondClassCount = 0

    with open(input, "r", encoding='utf-8') as infile:
        totalWords = []
        for line in infile:
            docCount = docCount + 1

            words = __getPreProcessedWordsFromDocument(line, language)

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


"""""
    Функция для аполучения матрицы вхождения из словаря методом дяди Руслана

    Parameters
    ----------
    language : str = язык, используемый в корпусе (2 возможных параметра : 'russian', 'english')
    input : str = путь до файла с корпусом
    output : str = путь до файла для выгрузки данных (не обязательно)
    dictionary : str = путь до файла со словарем

    Returns
    -------
    result : csr_matrix = матрица вхождений
"""


def getEntranceMatrix(input, dictionary, language='english', output=''):
    start_time = time.time()
    parsedDictionary = __parseDictionary(dictionary, language)
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
                entranceArray = __addRow(array=entranceArray)

            words = __getPreProcessedWordsFromDocument(line, language)
            adrNumber = 0
            for adr in parsedDictionary:
                if not (set(words) & set(adr)).__len__() == 0:
                    entranceArray[i][adrNumber] = entranceArray[i][adrNumber] + 1
                adrNumber = adrNumber + 1
            i = i + 1

    if not output.__eq__(''):
        __write2dArrayToFile(entranceArray, output)

    print("--- %s seconds ---" % (time.time() - start_time))
    return csr_matrix(entranceArray)


"""""
    Функция для аполучения матрицы вхождения из словаря с помощью CountVectorizer
    
    Parameters
    ----------
    language : str = язык, используемый в корпусе (2 возможных параметра : 'russian', 'english')
    input : str = путь до файла с корпусом
    output : str = путь до файла для выгрузки данных (не обязательно)
    dictionary : str = путь до файла со словарем

    Returns
    -------
    result : csr_matrix = матрица вхождений
    
    Notes
    -----
    Не берет во внимание составные признаки из словаря
"""


def getEntranceMatrixCountVectorizer(input, dictionary, language='english', output=''):
    start_time = time.time()
    corpus = []
    vocab = __parseDictionary(dictionary, language, False)
    with open(input, 'r', encoding='utf-8') as infile:
        for line in infile:
            corpus.append(line)

    analyzer = {
        'english': __getPreProcessedWordsFromDocument,
        'russian': __getPreProcessedWordsFromDocumentRus
    }.get(language)

    stop = __getStopWords(language)
    vectorizer = CountVectorizer(input=corpus, vocabulary=set(vocab), analyzer=analyzer, stop_words=stop)
    result = vectorizer.transform(corpus)

    if not output.__eq__(''):
        __write2dArrayToFile(output, result.toarray())
    print("--- %s seconds ---" % (time.time() - start_time))
    return csr_matrix(result)


def getStatisticFromEntranceMatrix(input):
    totalLines = 0
    linesWithADR = 0
    with open(input, 'r', encoding='utf-8') as infile:
        for line in infile:
            if line.__contains__('1'):
                linesWithADR = linesWithADR + 1
            totalLines = totalLines + 1
    result = linesWithADR / totalLines
    print("процент покрытия : " + str(result))


"""""
    Функция парсит данный корпус, стеммит каждое слово и убирает слова

    Parameters
    ----------
    language : str = язык, используемый в корпусе (2 возможных параметра : 'russian', 'english')
    input : str = путь до файла с корпусом
    output : str = путь до файла для выгрузки данных (не обязательно)

    Returns
    -------
    documents : list = лист разделенных на слова документов (лист листов)
"""


def parseDocument(input, language='english', output=''):
    documents = []
    with open(input, 'r', encoding='utf-8') as infile:
        for line in infile:
            documents.append(__getPreProcessedWordsFromDocument(line, language))
    if not output.__eq__(''):
        __write2dArrayToFile(output, documents)
    return documents


# функции  для обслуживания параметра функции --word-type
"""""
    surface_all
    В качестве слов берутся все токены как есть
"""


def surfaceAll(document):
    return [word.lower() for word in nltk.word_tokenize(document)]


"""""
    surface_no_pm
    Все токены, кроме знаков пунктуаций
"""


def surfaceNoPm(document):
    return [word.lower() for word in nltk.word_tokenize(document)
            if not punctuation.__contains__(word)]
    pass


"""""
    stem
    каждый токен подвергается стемму
"""


def stem(document, language):
    return __getPreProcessedWordsFromDocument(document, language)


# Служебные функции


def __divideMarkedCorpus(input, output1, output2):
    firstDocs = []
    secondDocs = []
    with open(input, 'r', encoding='utf-8') as infile:
        for line in infile:
            if nltk.word_tokenize(line)[0] == '0':
                firstDocs.append(line)
            if nltk.word_tokenize(line)[0] == '1':
                secondDocs.append(line)
    __writeArrayToFile('test/1_class_tweets.txt', firstDocs)
    __writeArrayToFile('test/2_class_tweets.txt', secondDocs)


def __writeArrayToFile(output, arrayData):
    with open(output, 'w', encoding='utf-8') as outfile:
            for value in arrayData:
                outfile.write(str(value))
            outfile.write("\n")


def __getStemmer(language):
    return {
        'russian': snowball.RussianStemmer(),
        'english': snowball.EnglishStemmer()
    }.get(language)


def __getStopWords(language):
    options = {
        'russian': set(stopwords.words('russian')),
        'english': set(stopwords.words('english'))
    }
    if not language in options:
        raise ValueError('unknown language')
    stop = options.get(language)
    stop.update(punctuation)
    return stop


def __getPreProcessedWordsFromDocument(document, language='english'):
    stop = __getStopWords(language)
    stemmer = __getStemmer(language)
    return [stemmer.stem(word.lower()) for word in nltk.word_tokenize(document)
            if not stop.__contains__(word)]


def __getPreProcessedWordsFromDocumentRus(document):
    stop = __getStopWords('russian')
    stemmer = __getStemmer('russian')
    return [stemmer.stem(word.lower()) for word in nltk.word_tokenize(document)
            if not stop.__contains__(word)]


def __parseDictionary(dictionary, language='english', allowCompositeValues=True):
    dictionaryList = []
    stemer = __getStemmer(language)
    with open(dictionary, 'r', encoding='utf-8') as difile:
        if allowCompositeValues:
            for line in difile:
                dictionaryList.append(__getPreProcessedWordsFromDocument(line, language))
        else:
            for line in difile:
                if not __isValueComposite(line):
                    line = line.replace("\n", "")
                    dictionaryList.append(stemer.stem(line))
                else:
                    continue
    return dictionaryList


def __isValueComposite(value):
    if value.split().__len__() > 1:
        return True
    else:
        return False


def __write2dArrayToFile(output, arrays):
    with open(output, 'w', encoding='utf-8') as outfile:
        for row in arrays:
            for value in row:
                outfile.write(str(value) + " ")
            outfile.write("\n")


def __addRow(array, amount=1):
    if len(array) == 1:
        return [[0 for x in range(len(array[0]))] for y in range(2)]
    rowArray = zeros([amount, len(array[0])])
    res = sps.vstack((array, rowArray))
    return res.toarray()


# Тестовые запуски


print("процент покрытия из класаа 0 : ")
getStatisticFromEntranceMatrix("test/entrance_mat_1.txt")

print("процент покрытия из класаа 1 : ")
getStatisticFromEntranceMatrix("test/entrance_mat_2.txt")

# parseDocument("test/loaded_tweets_parsed.txt", output="test/entrance_matrix.txt")
# getEntranceMatrix("test/loaded_tweets_parsed.txt", "dictionaries/ADR_dictionary_en.txt", "english", "test/entrance_matrix.txt")
# print(getEntranceMatrix("1", "2"))
# print(analyze("english", "test/loaded_tweets_parsed.txt"))
