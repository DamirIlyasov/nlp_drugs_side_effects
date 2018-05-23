import sys

from algs.ClassifierModelTrainer import train_and_save_model

if __name__ == '__main__':
    global pathToCorpus, \
        textEncoding, \
        wordType, \
        n_gram_range, \
        features, \
        unknownWordFrequency, \
        pathToSerializedModel, \
        language

    for argument in sys.argv:
        nextArgument = None
        if sys.argv.index(argument) < (len(sys.argv) - 1):
            nextArgument = sys.argv[sys.argv.index(argument) + 1]
        if argument == '--src-train-texts':
            # путь к корпусу, обязательный аргумент
            pathToCorpus = nextArgument
            print('pathToCorpus:', pathToCorpus)

        if argument == '--text-encoding':
            # кодировка-текста в файлах корпуса
            textEncoding = nextArgument
            print('textEncoding:', textEncoding)

        if argument == '--word-type':
            # возможные значения:
            # <surface_all | surface_no_pm | stem | suffix_X>, где
            # в случае surface_all в качестве слов берутся все токены как есть,
            # в случае surface_no_pm – все токены, кроме знаков пунктуаций,
            # в случае stem – “стемма” (см. http://snowball.tartarus.org/ ),
            # в случае suffix_X – окончания слов длиной X
            wordType = nextArgument
            print('wordType:', wordType)

        if argument == '-n':
            # n-грамность
            n_gram_range = nextArgument
            print('grammar:', n_gram_range)

        if argument == '--features':
            #   <false|true>
            # использовать дополнительные hand-crafted признаки, указанные в задании
            features = nextArgument
            print('features:', features)

        if argument == '--unknown-word-freq':
            # частота, ниже которой слова в обуч. множестве считаются неизвестными
            unknownWordFrequency = nextArgument
            print('unknownWordFrequency:', unknownWordFrequency)

        if argument == '-o':
            # путь-куда-сохранить-сериализованную-языковую-модель, обязательный аргумент
            pathToSerializedModel = nextArgument
            print('pathToSerializedModel:', pathToSerializedModel)

        if argument == '-language':
            # путь-куда-сохранить-сериализованную-языковую-модель, обязательный аргумент
            language = nextArgument
            print('language:', language)

    if (argument is None) or (pathToSerializedModel is None) or (language is None):
        print("Not all mandatory parameters filled!")
        sys.exit(0)

    train_and_save_model(pathToCorpus, textEncoding, wordType, n_gram_range, features, unknownWordFrequency,
                         pathToSerializedModel, language)
