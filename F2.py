import sys

for argument in sys.argv:
    nextArgument = None
    if sys.argv.index(argument) < (len(sys.argv) - 1):
        nextArgument = sys.argv[sys.argv.index(argument) + 1]

    if argument == '--lm':
        # путь-к-сериализованной-модели
        pathToSerializedModel = nextArgument
        print('pathToSerializedModel:', pathToSerializedModel)

    if argument == "--src-test-texts":
        # путь-к-тестовой-коллекции
        testTextPath = nextArgument
        print('testTextPath:', testTextPath)
