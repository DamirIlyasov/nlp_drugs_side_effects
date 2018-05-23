import sys

from algs.MetricsCounter import count_metrics

if __name__ == '__main__':
    global pathToSerializedModel, testTextPath

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

    if (pathToSerializedModel is None) or (testTextPath is None):
        print("Not all mandatory parameters filled!")
        sys.exit(0)

    count_metrics(pathToSerializedModel, testTextPath)
