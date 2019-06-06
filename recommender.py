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
import data_visualization

import pickle
from math import sqrt, pi, cos, sin, atan2
from random import random, randint
import pandas as pd
import numpy as np


def in_zone(x, y, zone_x, zone_y, zone_r):
    """ Given (x, y) of a position, return True if it is within the zone boundaries
        and False if it is not
    """
    dist_to_center = sqrt((x - zone_x)**2 + (y - zone_y)**2)
    return dist_to_center < zone_r


def gen_new_safezone(curr_x, curr_y, curr_r, rad_decrease):
    """
    Given the current safe zone properties and the proportion to decrease the next one by, generate the next safe zone

    :param curr_x: current x coordinate of the safe zone center
    :param curr_y: current y coordinate of the safe zone center
    :param curr_r: current radius of the safe zone
    :param rad_decrease: the ratio to decrease the circle radius by. Typically 0.5
    :return: x, y and radius of the new safe zone
    """
    new_r = curr_r * rad_decrease

    # Get random radius to new point within new_r
    r_ = new_r * sqrt(random())
    # Get random angle
    theta = random() * 2 * pi

    new_x = curr_x + r_ * cos(theta)
    new_y = curr_y + r_ * sin(theta)

    return new_x, new_y, new_r


def get_closest_to_safezone(x, y, safe_x, safe_y, safe_r):
    """
    Get the point in the safe zone that is closest to the player location
    (Assumed that the player location is OUTSIDE the safe zone)

    :param x: player x coordinate
    :param y: player y coordinate
    :param safe_x: x coordinate of safe zone center
    :param safe_y: y coordinate of safe zone center
    :param safe_r: safe zone radius
    :return: the point (rounded like the player locations) closest to the safe zone
    """
    distance = sqrt((x - safe_x)**2 + (y - safe_y)**2)
    to_move = distance - safe_r
    angle = atan2(safe_y - y, safe_x - x)
    x_ = player_info.round_raw(x + to_move * cos(angle))
    y_ = player_info.round_raw(y + to_move * sin(angle))
    return x_, y_


def gen_candidate_locations(curr_x, curr_y, next_safe_x, next_safe_y, next_safe_r):
    """
    Given a player location and where the next safe zone is, calculate the neighboring locations to use
    as candidates for path generation. Only neighboring locations in the safe zone are considered.
     --> (return may be empty list)

    :param curr_x: player x coordinate
    :param curr_y: player y coordinate
    :param next_safe_x: safe zone x coordinate
    :param next_safe_y: safe zone y coordinate
    :param next_safe_r: safe zone radius
    :return: a list of [(x1, y1), (x2, y2), ...] of each neighboring location
    """
    candidates = []
    for x in range(curr_x - 10000, curr_x + 20000, 10000):
        for y in range(curr_y - 10000, curr_y + 20000, 10000):
            if in_zone(x, y, next_safe_x, next_safe_y, next_safe_r):
                candidates.append((x, y))
    return candidates


def get_next_loc(game_state, x, y, safe_x, safe_y, safe_r, model):
    """
    Given the current game_state, player location, safe zone location, and model, get the next location to add to path

    :param game_state: current game state
    :param x: player x coordinate
    :param y: player y coordinate
    :param safe_x: x coordinate of safe zone center
    :param safe_y: y coordinate of safe zone center
    :param safe_r: safe zone radius
    :param model: model used to predict whether locations will result in win or not
    :return:
    """
    candidates = gen_candidate_locations(x, y, safe_x, safe_y, safe_r)
    if len(candidates) == 0:    # No usual candidates were in the zone
        return get_closest_to_safezone(x, y, safe_x, safe_y, safe_r)

    winning_locs = []
    for cand_x, cand_y in candidates:
        rank = predict_rank(game_state, cand_x, cand_y, safe_x, safe_y, safe_r, model)
        if rank == 1:
            winning_locs.append((cand_x, cand_y))

    if len(winning_locs) > 0:
        return winning_locs[randint(0, len(winning_locs) - 1)]
    else:
        return -1, -1


def predict_rank(game_state, x, y, safe_x, safe_y, safe_r, model):
    """
    Given information about a location, time, and where the safe zone is, predict whether
    the location is likely to result in a win or loss

    :param game_state: float representing the time of game: 0.5, 1.0... etc
    :param x: x coordinate of location to predict
    :param y: y coordinate of location to predict
    :param safe_x: x coordinate of the center of the safe zone
    :param safe_y: y coordinate of the center of the safe zone
    :param safe_r: radius of the safe zone
    :param model: the model to predict with
    :return: 1 if the location is predicted to be a winning location, 0 if it is not
    """
    predicted = model.predict(np.array([game_state, x, y, safe_x, safe_y, safe_r]).reshape(1, -1))
    return int(predicted[0].item())


def gen_path(drop_x, drop_y, possible_safe_zones, end_state, model):
    """
    Given a drop location, potential locations for the first safe zone, and a model, generate a path
    starting at the drop location.

    Path is generated by looking at all neighboring locations that are predicted to result in a win and selecting a
    random one as the next location. If there are no neighboring locations predicted to result in a win, the location
    does not change and the path ends. Otherwise path generation continues until the end_state (a game_state) is reached.
    For each game_state (0.5, 1, 1.5,...) there are two locations generated, and on each even game_state the safe zone
    is updated. This results in 4 locations for each safe zone, except for the first zone which only has 2 generated
    (+ the original drop location)

    :param drop_x: x coordinate of the drop location to start from
    :param drop_y: y coordinate of the drop location to start from
    :param possible_safe_zones: a DataFrame where the columns are:
                                        [x, y, radius]
                                and each row represents a possible first safe zone
    :param end_state: the game_state to generate paths up to. Typical values from observed games are in the range of
                        6.5 to 9.5
    :param model: the model to use for predicting whether a location will result in a win
    :return: a DataFrame with columns:
                    [x, y, game_state, safe_x, safe_y, safe_r]
             where the first row is the drop location and each subsequent row is the next location in the path
    """
    safe_zone = possible_safe_zones.sample(n=1)
    safe_x = safe_zone["x"].values[0].item()
    safe_y = safe_zone["y"].values[0].item()
    safe_r = safe_zone["radius"].values[0].item()

    curr_x = drop_x
    curr_y = drop_y
    path = list()

    game_state = 0.5
    path.append({"x": curr_x, "y": curr_y,
                 "game_state": game_state,
                 "safe_x": safe_x,
                 "safe_y": safe_y,
                 "safe_r": safe_r})

    print("SAFE ZONE STARTING AT {}, {} : {}".format(safe_x, safe_y, safe_r))

    # While the end_state has not been reached
    while game_state < end_state:

        # Get the next position to move
        curr_x, curr_y = get_next_loc(game_state, curr_x, curr_y, safe_x, safe_y, safe_r, model)

        if curr_x == -1 and curr_y == -1:   # No candidate locations were predicted to be winning locations, path ends
            game_state = end_state
        else:
            # Add to path
            path.append({"x": curr_x, "y": curr_y,
                         "game_state": game_state,
                         "safe_x": safe_x,
                         "safe_y": safe_y,
                         "safe_r": safe_r})

            game_state += 0.25

        # Update safe zone if the game_state is a whole number
        if int(game_state) == game_state:
            safe_x, safe_y, safe_r = gen_new_safezone(safe_x, safe_y, safe_r, 0.5)
            print("NEW SAFE ZONE AT {}, {} : {}".format(safe_x, safe_y, safe_r))
    return pd.DataFrame(path)


# note that I am assuming the target is in the last position of the dataframe
# additionally, I am assuming that the list has already been filtered(ie. we are only training on the top players)
# additionally, my current assumption is the data has already been transformed into non-categorical data
def train_model(df, max_k):
    x = df.iloc[:, :-1].copy()  # assuming our target is at the very end of the dataframe
    y = df.iloc[:, -1].copy()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)
    x_train = scale(x_train)
    x_test = scale(x_test)
    k_scores = []
    for k in range(1, max_k):
        knn = KNeighborsClassifier(n_neighbors=k)
        cv_scores = cross_val_score(knn, x_train, y_train, cv=10, scoring='accuracy')
        k_scores.append(cv_scores.mean())

    max_score = max(k_scores)

    #  get the arg max of the score array to determine the optimal value for k
    k_opt = None
    for i in range(len(k_scores)):
        if k_scores[i] == max_score:
            k_opt = i + 1

    knn = KNeighborsClassifier(n_neighbors=k_opt)
    kf = KFold(n_splits=10)

    # split the training data into different validation folds and run a training loop on it

    kf.get_n_splits(x_train)
    training_scores = []

    knn.fit(x_train, y_train)  # we get about -2.something places per map accuracy
    # testing_accuracy = (knn.predict(x_test) - y_test).mean()

    predictions = knn.predict(x_test)
    correct_predictions = predictions[predictions == y_test]

    print("TESTING ACCURACY: ", len(correct_predictions) / len(predictions))
    return knn


def player_path_model(df, max_k):
    # TODO: get x and y from df

    # Zone Data
    x = None

    # Player Position Category
    y = None

    best_score = 0
    best_model = None

    # Hyperparameter Tuning
    for k in range(max_k):
        vec = DictVectorizer(sparse=False)
        scaler = StandardScaler()
        model = KNeighborsClassifier(n_neighbors=k)
        pipeline = Pipeline([('vec', vec),
                             ('scaler', scaler),
                             ('fit', model)])

        score = cross_val_score(pipeline,
                                x,
                                y,
                                cv=5, scoring='accuracy').mean()

        if score > best_score or best_model is None:
            best_score = score
            best_model = pipeline

    print("Best Accuracy Score: " + str(best_score))
    return best_model


# preprocess the dataframe
def preprocess_data(drop_data):
    drop_data = drop_data.dropna()
    drop_data = drop_data.drop(columns=['player'])  # probably don't need to include the player in the model
    drop_data = drop_data.drop(columns=['drop_loc_raw'])  # probably don't need to include the player in the model
    drop_data = drop_data.dropna()
    drop_data = drop_data[drop_data['rank'] <= 5]
    labelencoder_x = LabelEncoder()
    x = drop_data.iloc[:, :].values
    drop_data['flight_path'] = labelencoder_x.fit_transform(x[:, 1])
    drop_data['map'] = labelencoder_x.fit_transform(x[:, 2])
    drop_data = drop_data.drop(columns=['rank'])
    drop_data = drop_data[['flight_path', 'map', 'drop_loc_cat']]
    scaler = MinMaxScaler()
    drop_data.loc[:, :-1] = scaler.fit_transform(drop_data[drop_data.columns[:-1]])
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


def get_map_data(telemetry_files):
    """
    Given a list of telemetry file names, extract the player location and safe zone info to aggregate by map and flight
    path

    :param telemetry_files: list of telemetry file names
    :return: dict: {(map_name, flight_path): DataFrame of locations with safe zone, ...}
    """
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
            print("MAX STATE: ", combined['gameState'].max())

            if (map_name, flight_cat) not in map_data.keys():
                map_data[(map_name, flight_cat)] = []
            map_data[(map_name, flight_cat)].append(combined.dropna())

    for key, data in map_data.items():
        map_data[key] = pd.concat(data)

    return map_data


def train_models(map_data):
    """ Given the data for each map, train models for that data, fit them to the data, and pickle them

    :param map_data: dict: {(map_name, flight_path): DataFrame of player location, ...}
    :return:         dict: {(map_name, flight_path): DataFrame of models, ...}
    """
    models = dict()
    for key, data in map_data.items():
        print(key, " : ", len(data))
        optimal = tune_player_path_model(data, 15)

        data_x = data.drop(['name', 'ranking'], axis=1)
        data_y = data['ranking']
        optimal.fit(data_x, data_y)

        models[key] = optimal

        with open(".\\models\\{}_{}-model.pickle".format(key[0], key[1]), "wb") as model_f:
            pickle.dump(optimal, model_f)
            model_f.close()

    return models


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

    # Just a test telemetry object
    t = jsonparser.load_pickle(data_dir + telemetry_files[0])
    zone_info = jsonparser.getZoneStates(t)
    blue_zones = zone_info[["safetyZonePosition_x", "safetyZonePosition_y", "safetyZoneRadius"]]
    blue_zones.columns = blue_zones.columns.map({"safetyZonePosition_x": "x",
                                                 "safetyZonePosition_y": "y",
                                                 "safetyZoneRadius": "radius"})

    # Get map name, flight path, and location info from telemetry
    map_n = jsonparser.get_map(t)
    fp = jsonparser.get_flight_cat_from_telemetry(t)
    drop = player_info.get_player_paths(t)
    total = player_info.join_player_and_zone(drop, zone_info)

    # Load the model (NOTE, must have pickled models that are fit to the data already)
    model = jsonparser.load_pickle(".\\models\\Savage_Main_nn-model.pickle")

    # Get a random location to use as the drop location
    total.dropna(inplace=True)
    rand_pos = total.sample(n=1)
    x_ = rand_pos['x'].values[0].item()
    y_ = rand_pos['y'].values[0].item()
    print(x_, y_)

    # Generate a path (DataFrame)
    path = gen_path(int(x_), int(y_), blue_zones, 8.5, model)
    print(path)

    # Display the path
    data_visualization.display_player_path(pd.DataFrame(path), None, map_n)


