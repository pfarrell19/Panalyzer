import pickle
import sys

from sklearn.externals import joblib

import recommender

def printHelp():
    print('usage: python driver.py -l <map_name> [-h] [-p <inputfile>] [-d]\n')

    print('Options')
    print('\t-h : help')
    print('\t-p : path prediction')
    print('\t-d : drop location prediction. Enter flight path as one of the cardinal directions.')

def predict(model, input_data):
    return model.predict(input_data)


def get_drop_model_file_path(map_name, flight_path):
    '''
    :param map_name: string
    :param flight_path: string-> flight path of the plane
    :returns: the path of the trained model given a map name and a flight path
    '''


    if map_name != 'Savage_Main':
        return "./drop_models/model_" + map_name + "_" + flight_path + ".pkl"
    return "./drop_models/model_Savage_Main.pkl"



def main():
    argv = sys.argv[1:]

    path = False
    drop = False
    flight_path = None
    dropAll = True

    if '-h' in argv:
        printHelp()
        exit()


    if '-l' not in argv:
        print ('error: need map name')
        exit()

    map_name = argv[argv.index('-l') + 1]

    if '-p' in argv:
        pathInputFile = argv[argv.index('-p') + 1]
        path = True
    
    if '-d' in argv:
        drop = True



    if drop:
        maps = ['Desert_Main', 'Erangel_Main', 'DihorOtok_Main', 'Savage_Main']
        directions = ['nn', 'ne', 'ee', 'se', 'ss', 'sw', 'ww', 'nw']
        for map in maps:
            for direction in directions:
                map_name = map
                flight_path = direction
                #TODO: Input data format; current assumption -> pickle df
                model_file_path = get_drop_model_file_path(map_name, flight_path)
                print(model_file_path)
                model = joblib.load(model_file_path)
                recommender.get_drop_predictions(map_name, flight_path, model)


    if path:
        #TODO: Input data format; current assumption -> pickle df
        pathInput = pickle.load(pathInputFile)
        modelPath = pickle.load('''model''')
        pathPrediction = predict(modelPath, pathInput)
        print ("Suggested Path: {!r}".format(pathPrediction))

    return 0


if __name__ == '__main__':
    main()
