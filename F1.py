import sys

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
        grammar = nextArgument
        print('grammar:', grammar)

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
        pathToSaveSerializedModel = nextArgument
        print('pathToSaveSerializedModel:', pathToSaveSerializedModel)

    if argument == '--lm':
        # путь-к-сериализованной-модели
        serializedModelPath = nextArgument
        print('serializedModelPath:', serializedModelPath)

    if argument == "--src-test-texts":
        # путь-к-тестовой-коллекции
        testTextPath = nextArgument
        print('testTextPath:', testTextPath)

    if argument == '--src-texts':
        # путь-к-коллекции
        textPath = nextArgument
        print('textPath:', textPath)

    if argument == '--o-texts':
        # путь-куда-сохранить-размеченную-коллекцию
        pathToSaveCollection = nextArgument
        print('pathToSaveCollection:', pathToSaveCollection)


def work(pathToCorpus, textEncoding, wordType, grammar,
         features, unknownWordFrequencym, pathToSaveSerializedModel,
         serializedModelPath, testTextPath, textPath, PathToSaveCollection):
    #main logic
