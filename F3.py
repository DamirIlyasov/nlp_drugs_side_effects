import sys

for argument in sys.argv:
    nextArgument = None
    if sys.argv.index(argument) < (len(sys.argv) - 1):
        nextArgument = sys.argv[sys.argv.index(argument) + 1]

    if argument == '--lm':
        # путь-к-сериализованной-модели
        pathToSerializedModel = nextArgument
        print('pathToSerializedModel:', pathToSerializedModel)

    if argument == '--src-texts':
        # путь-к-коллекции
        textPath = nextArgument
        print('textPath:', textPath)

    if argument == '--o-texts':
        # путь-куда-сохранить-размеченную-коллекцию
        pathToSave = nextArgument
        print('pathToSave:', pathToSave)
