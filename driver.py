import pickle
import sys

def printHelp():
    print('usage: python driver.py -l <map_name> [-h] [-p <inputfile>] [-d <inputfile>]\n')

    print('Options')
    print('\t-h : help')
    print('\t-p : path prediction')
    print('\t-d : drop location prediction')


def predict(model, input_data):
    return model.predict(input_data)


def main():
    argv = sys.argv[1:]

    path = False
    drop = False

    if '-h' in argv:
        printHelp()
        exit()


    if '-l' not in argv:
        print 'error: need map name'
        exit()

    map_name = argv[argv.index('-l') + 1]

    if '-p' in argv:
        pathInputFile = argv[argv.index('-p') + 1]
        path = True
    
    if '-d' in argv:
        dropInputFile = argv[argv.index('-d') + 1]
        drop = True

    if drop:
        #TODO: Input data format; current assumption -> pickle df
        dropInput = pickle.load(dropInputFile)
        modelDrop = pickle.load('''model''')
        dropPrediction = predict(modelDrop, dropInput)
        print "Suggested Drop Location: {!r}".format(dropPrediction)


    if path:
        #TODO: Input data format; current assumption -> pickle df
        pathInput = pickle.load(pathInputFile)
        modelPath = pickle.load('''model''')
        pathPrediction = predict(modelPath, pathInput)
        print "Suggested Path: {!r}".format(pathPrediction)

    return 0


if __name__ == '__main__':
    main()
