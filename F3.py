import sys

for argument in sys.argv:
    nextArgument = sys.argv[sys.argv.index(argument) + 1]
    if argument == '--lm':
        # путь-к-сериализованной-модели
        pathToSerializedModel = nextArgument

    if argument == '--src-texts':
        # путь-к-коллекции
        textPath = nextArgument

    if argument == '--o-texts':
        # путь-куда-сохранить-размеченную-коллекцию
        pathToSave = nextArgument
