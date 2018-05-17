import sys

for argument in sys.argv:
    nextArgument = sys.argv[sys.argv.index(argument) + 1]
    if argument == '--src-train-texts':
        # путь к корпусу, обязательный аргумент
        pathToCorpus = nextArgument

    if argument == '--text-encoding':
        # кодировка-текста в файлах корпуса
        textEncoding = nextArgument

    if argument == '--word-type':
        # возможные значения:
        # <surface_all | surface_no_pm | stem | suffix_X>, где
        # в случае surface_all в качестве слов берутся все токены как есть,
        # в случае surface_no_pm – все токены, кроме знаков пунктуаций,
        # в случае stem – “стемма” (см. http://snowball.tartarus.org/ ),
        # в случае suffix_X – окончания слов длиной X
        wordType = nextArgument

    if argument == '-n':
        # n-грамность
        grammar = nextArgument

    if argument == '--features':
        #   <false|true>
        # использовать дополнительные hand-crafted признаки, указанные в задании
        features = nextArgument

    if argument == '--unknown-word-freq':
        # частота, ниже которой слова в обуч. множестве считаются неизвестными
        unknownWordFrequency = nextArgument

    if argument == '-o':
        # путь-куда-сохранить-сериализованную-языковую-модель, обязательный аргумент
        pathToSerializedModel = nextArgument
