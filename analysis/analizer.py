import nltk
from nltk.corpus import stopwords
from nltk.stem import snowball


def analyze(language, input, output='', isMapped=True):
    docCount = 0
    firstClassCount = 0
    secondClassCount = 0

    stop = getStopWords(language)
    stop.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'
                    , '%', '$', '#', '№', '*', '^', '@', '+', '-', '\'s', '\'m', '\'', '...', '\"'])

    stemmer = getStemmer(language)

    with open(input, "r", encoding='utf-8') as infile:
        totalWords = []
        for line in infile:
            docCount = docCount + 1

            words = [stemmer.stem(word.lower()) for word in nltk.word_tokenize(line, language)
                     if not stop.__contains__(word)]

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
    return {
        'russian': set(stopwords.words('russian')),
        'english': set(stopwords.words('english'))
    }.get(language)


# print(analyze("english", "test/loaded_tweets_parsed.txt"))
