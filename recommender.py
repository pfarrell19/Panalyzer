from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, KFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer

import jsonparser
import player_info
import logging
import os
import downloader

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imageio import imread


def gen_candidate_locations(curr_x, curr_y, next_safe_x, next_safe_y, next_safe_r):
    candidates = []
    for x in range(curr_x - 10000, curr_x + 20000, 10000):
        for y in range(curr_y - 10000, curr_y + 20000, 10000):
            if player_info.in_zone(x, y, next_safe_x, next_safe_y, next_safe_r):
                candidates.append((x, y))
    return candidates


def predict_rank(game_state, x, y, poison_x, poison_y, poison_r, safe_x, safe_y, safe_r, model):
    return model.predict(np.array([game_state, x, y, poison_x, poison_y, poison_r, safe_x, safe_y, safe_r]))


def get_next_loc(game_state, x, y, poison_x, poison_y, poison_r, safe_x, safe_y, safe_r, model):
    candidates = gen_candidate_locations(x, y, poison_x, poison_y, poison_r)

    opt_cand_x = None
    opt_cand_y = None
    min_rank = 99

    for cand_x, cand_y in candidates:
        rank = predict_rank(game_state, cand_x, cand_y, poison_x, poison_y, poison_r, safe_x, safe_y, safe_r, model)
        if rank < min_rank:
            opt_cand_x = cand_x
            opt_cand_y = cand_y
            min_rank = rank

    return opt_cand_x, opt_cand_y


def gen_path(drop_x, drop_y):
    return None

# note that I am assuming the target is in the last position of the dataframe
# additionally, I am assuming that the list has already been filtered(ie. we are only training on the top players)
# additionally, my current assumption is the data has already been transformed into non-categorical data
def train_model(df, max_k):
    X = df.iloc[:, :-1].copy()#assuming our target is at the very end of the dataframe
    y = df.iloc[:, -1].copy()
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size= .2)
    k_scores = []
    for k in range(1,max_k):
        knn = KNeighborsClassifier(n_neighbors=k)
        cv_scores = cross_val_score(knn, x_train, y_train, cv=10, scoring='accuracy')
        k_scores.append(cv_scores.mean())

    max_score = max(k_scores)

    #     get the arg max of the score array to determine the optimnal value for k
    k_opt = None
    for i in range(len(k_scores)):
        if k_scores[i] == max_score:
            k_opt = i + 1

    knn = KNeighborsClassifier(n_neighbors=k_opt)
    kf = KFold(n_splits=10)

    # split the training data into different validation folds and run a training loop on it

    kf.get_n_splits(x_train)
    training_scores = []


    knn.fit(x_train, y_train)#we get about -2.something places per map accuracy
    # testing_accuracy = (knn.predict(x_test) - y_test).mean()

    predictions = knn.predict(x_test)
    correct_predictions = predictions[predictions == y_test]

    print("TESTING ACCURACY: ", len(correct_predictions) / len(predictions))
    return knn


# preprocess the dataframe
def preprocess_data(drop_data):
    drop_data = drop_data.dropna()
    drop_data = drop_data.drop(columns = ['player'])#probably don't need to include the player in the model
    drop_data = drop_data.drop(columns = ['drop_loc_raw'])#probably don't need to include the player in the model
    drop_data = drop_data.dropna()
    drop_data = drop_data[drop_data['rank'] <= 5]
    labelencoder_X=LabelEncoder()
    X = drop_data.iloc[:,:].values
    drop_data['flight_path'] = labelencoder_X.fit_transform(X[:, 1])
    drop_data['map'] = labelencoder_X.fit_transform(X[:, 2])
    drop_data = drop_data.drop(columns = ['rank'])
    drop_data = drop_data[['flight_path', 'map', 'drop_loc_cat']]
    scaler = MinMaxScaler()
    drop_data.loc[:,:-1] = scaler.fit_transform(drop_data[drop_data.columns[:-1]])
    return drop_data


def tune_player_path_model(position_df, max_k):
    """ Get the optimal k value for predicting rank based on player position and the locations of the two zones

    :param position_df: Output of player_info.join_player_and_zone(...)
    :param max_k:       max K value to test
    :return:            the model that resulted in the highest accuracy when predicting rank
    """
    # Zone and player data
    x = position_df.drop(['name', 'ranking'], axis=1)

    # Player rank
    y = position_df['ranking']

    best_score = 0
    best_model = None

    # Hyperparameter Tuning
    for k in range(1, max_k):
        scaler = StandardScaler()
        model = KNeighborsClassifier(n_neighbors=k)
        pipeline = Pipeline([('scaler', scaler),
                             ('fit', model)])

        score = cross_val_score(pipeline,
                                x,
                                y,
                                cv=5, scoring='accuracy').mean()

        #print("\tacc: ", score)
        if score > best_score or best_model is None:
            best_score = score
            best_model = pipeline

    print("Best Accuracy Score: " + str(best_score))
    return best_model

def ranking_to_bin(ranking):
    if ranking >= 5:
        return 1
    else:
        return 0

if __name__ == "__main__":
    data_dir = ".\\data\\"
    match_files = []
    telemetry_files = []

    downloader.setup_logging(show_debug=False)
    logging.info("Scanning for match and telemetry files in %s to parse", data_dir)
    for file in os.listdir(data_dir):
        if "_match" in file:
            logging.debug("Match file %s found, adding as match", file)
            match_files.append(file)
        elif "_telemetry" in file:
            logging.debug("Telemetry file %s found, adding as match", file)
            telemetry_files.append(file)

    map_data = dict()

    for i, telemetry_file in enumerate(telemetry_files):
        print("\tMatch {} of {}".format(i, len(telemetry_files)))
        telemetry = jsonparser.load_pickle(data_dir + telemetry_file)
        flight_cat = jsonparser.get_flight_cat_from_telemetry(telemetry)
        map_name = jsonparser.get_map(telemetry)

        if flight_cat is not None:
            print(map_name, " : ", flight_cat)
            player_loc_info = player_info.get_player_paths(telemetry)
            zone_info = jsonparser.getZoneStates(telemetry)
            combined = player_info.join_player_and_zone(player_loc_info, zone_info).dropna()
            combined["ranking"] = combined["ranking"].apply(ranking_to_bin)

            if (map_name, flight_cat) not in map_data.keys():
                map_data[(map_name, flight_cat)] = []
            map_data[(map_name, flight_cat)].append(combined)

    for key, data in map_data.items():
        print(key, " : ", len(data))
        optimal = tune_player_path_model(pd.concat(data), 15)
        with open("{}_{}-model.pkl".format(key[0], key[1]), "wb") as model_f:
            pickle.dump(optimal, model_f)
            model_f.close()

    #optimal = tune_player_path_model(all, 15)

