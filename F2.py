import sys

for argument in sys.argv:
    nextArgument = sys.argv[sys.argv.index(argument) + 1]
    if argument == '--lm':
        # путь-к-сериализованной-модели
        pathToSerializedModel = nextArgument

    if argument == '--src-test-texts':
        # путь-к-тестовой-коллекции
        testTextPath = nextArgument
