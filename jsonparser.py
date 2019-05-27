import pickle
import os
import matplotlib.pyplot as plt
from imageio import imread
import math
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, KFold
import math
import logging
import downloader

# TODO: remove global...
CM_TO_KM = 100000           # CM in a KM
MAX_MAP_SIZE = 8            # In KM


# search for specific attributes 
# player_id :   players acount id (eg. 'acount.*****')
# start_range : starting time (eg. 'YYYY-MM-DDTHH:MM:SS.SSSZ')
# end_range:    ending time 
# event_type:   type of event (eg. 'LogParachutepiLanding')
def search(json_object, player_id=None, start_range=None, end_range=None, event_type=None):
    events = []
    i = 0
    for entry in json_object: 
        if (is_event(entry, event_type) and
                in_range(entry['_D'], start_range, end_range) and
                has_player(entry, player_id)):
            
            events.append(entry)
    return events


def in_range(time, start, end):
    if start is None and end is None:
        return True
    elif start is None:
        return is_after(end, time)
    elif end is None:
        return is_after(time, start)
    else:
        return is_after(time, start) and is_after(end, time)


def is_after(time1, time2):
    t_index = time1.find('T')
    date1 = time1[:t_index].encode('ascii', 'ignore')
    date1 = date1.split('-')
    time1 = time1[t_index + 1:][:-1].encode('ascii', 'ignore')
    time1 = time1.split(':')

    t_index = time2.find('T')
    date2 = time2[:t_index].encode('ascii', 'ignore')
    date2 = date2.split('-')
    time2 = time2[t_index + 1:][:-1].encode('ascii', 'ignore')
    time2 = time2.split(':')

    equals = True

    for x in range(3):
        if int(date1[x]) > int(date2[x]):
                return True
    for x in range(2):
        if int(time1[x]) > int(time2[x]):
                return True

    if float(time1[2]) > float(time2[2]):
        return True

    if time1 != time2:
        return False

    return equals


def has_player(event, player_id):
    if player_id is None:
        return True

    for key in ['character', 'attacker', 'victim', 'assistant', 'reviver']:
        if key in event.keys():
            if event[key]['accountId'] == player_id:
                return True
        else:
            return False
    return False


def is_event(event, event_type):
    if event_type is None:
        return True
    else:
        return event_type == event['_T']


# Return the dict from the pickle file name
def load_pickle(pickle_file):
    with open(pickle_file, 'rb') as f:
        telemetry_data = pickle.load(f)
    f.close()
    return telemetry_data


# Get the location the plane starts in (not quite sure it's useful yet...)
def get_plane_start(telemetry):
    start_loc = None
    for log_entry in telemetry:
        if log_entry["_T"] == "LogMatchStart":
            # All players start at the exact same location, so we only need first
            start_loc = log_entry["characters"][0]['location']
    return start_loc


# Get the map name from telemetry
def get_map(telemetry):
    for log_entry in telemetry:
        if log_entry["_T"] == "LogMatchStart":
            return log_entry["mapName"]
    return None


# Get the result for each player in the match from the telemetry data
# Return the results as a list of dicts:
#       [{'name' : playerName, 'ranking': ranking},
#        ...]
def get_rankings(telemetry):
    results = []
    for log_entry in telemetry:
        if log_entry["_T"] == "LogMatchEnd":
            for character in log_entry['characters']:
                results.append({'name': character['name'],
                                'ranking': character['ranking']})
    return results


# Get the location of each player's landing from the telemetry data
# Return as a dict:
#           {playerName: [x, y],
#            ...}
def get_all_landings(telemetry):
    landings = {}
    for log_entry in telemetry:
        if log_entry["_T"] == "LogParachuteLanding":
            landing_loc = log_entry['character']['location']
            x = landing_loc['x']
            y = landing_loc['y']
            landings[log_entry['character']['name']] = [x, y]
    return landings


# Returns the first  and last locations someone jumped out of the plane
def get_flight_data(telemetry):
    first_coordinate = None # First player exit event from plane
    current_coordinate = None # Last player exist even from plane
    for log_entry in telemetry:
        if log_entry.get("_T") == "LogVehicleLeave" \
                and log_entry.get("vehicle").get("vehicleId") == "DummyTransportAircraft_C":
            current_coordinate = log_entry.get("character").get("location")
            if first_coordinate is None:
                first_coordinate = current_coordinate
    return first_coordinate, current_coordinate


# Returns the unit vector for the flight path given the first and last location someone exited plane
def get_flight_vector(first_drop, last_drop):
    flight_vector = np.array([last_drop['x'] - first_drop['x'], last_drop['y'] - first_drop['y']])
    vector_length = np.linalg.norm(flight_vector)
    flight_vector_norm = flight_vector / vector_length
    return flight_vector_norm


# Returns the angle between the two flight vectors u and v (squared)
# Note: u and v must be unit vectors already
def flight_diff(u, v):
    return np.arccos(np.clip(np.dot(u, v), -1, 1))**2


# Given a flight path (unit vectorized), get the direction that it is closest to
# from one of the eight recognized cardinal directions
def get_flight_category(flight_vector):
    # Set up unit vectors for the 8 directions to classify flight paths as
    between_mag = 1 / math.sqrt(2)
    nn = [0, 1]
    ne = [between_mag, between_mag]
    ee = [1, 0]
    se = [between_mag, -between_mag]
    ss = [0, -1]
    sw = [-between_mag, -between_mag]
    ww = [-1, 0]
    nw = [-between_mag, between_mag]

    # Put em in a DataFrame for processing
    directions = np.array([nn, ne, ee, se, ss, sw, ww, nw])
    dirs_df = pd.DataFrame(directions)

    # Combine the x, y component columns, add direction names, and get rid of the excess
    dirs_df['direction_vec'] = list(zip(dirs_df[0], dirs_df[1]))
    dirs_df['direction'] = np.array(['nn', 'ne', 'ee', 'se', 'ss', 'sw', 'ww', 'nw'])
    dirs_df.drop([0, 1], axis=1, inplace=True)

    # Calculate the angle between the flight_vector and each of the 8 directions
    # and store it under a new column, 'angle_from_path'
    dirs_df['angle_from_path'] = dirs_df['direction_vec'].apply(flight_diff, args=(flight_vector,))

    # And return the direction (string) where the angle between the flight_vector
    # and the direction is minimized (i.e. the closest direction to the flight_vector)
    return dirs_df['direction'].loc[dirs_df['angle_from_path'].idxmin()]


# Convert raw (x, y) coordinates to the category
# Maps are divided into a 20x20 grid of square_size x square_size blocks
# where each square can be represented by a letter for the x and y position of that
# square in the map, e.g. AA for the square containing (0, 0)
def get_loc_cat(x, y, map_dim):
    square_size = 0.05 * map_dim
    x_ = x // square_size
    y_ = y // square_size

    # add the number square along each access to 'A's ascii code and return the characters
    # as the category
    return chr(65 + int(x_)) + chr(65 + int(y_))


# Plot the drop locations of each player (in blue), with opacity in relation to their rank in that match
# (more opaque = lower rank), along with the location the first person left the plane (in green)
# and the last person to leave the plane (in red)
def display_drop_locations(telemetry, fig, fig_x, fig_y, fig_num, match_num):
    landings = get_all_landings(telemetry)
    rankings = get_rankings(telemetry)
    map_name = get_map(telemetry)

    # Set up plot scale
    if map_name == "Savage_Main":                        # 4km map
        x_max = MAX_MAP_SIZE * (1/2)
        y_max = x_max
        map_img = imread("savage.png")
        plt.imshow(map_img, zorder=0, extent=[0.0, 4.0, 0.0, 4.0])
    elif map_name == "Erangel_Main":                     # 8km map
        x_max = MAX_MAP_SIZE
        y_max = MAX_MAP_SIZE
        map_img = imread("erangel.png")
        plt.imshow(map_img, zorder=0, extent=[0.0, 8.0, 0.0, 8.0])
    elif map_name ==  "Desert_Main":                     # 8km map
        x_max = MAX_MAP_SIZE
        y_max = x_max
        map_img = imread("miramar.png")
        plt.imshow(map_img, zorder=0, extent=[0.0, 8.0, 0.0, 8.0])
    elif map_name == "DihorOtok_Main":                   # 6km map
        x_max = MAX_MAP_SIZE * (3/4)
        y_max = x_max
        map_img = imread("vikendi.png")
        plt.imshow(map_img, zorder=0, extent=[0.0, 6.0, 0.0, 6.0])

    first_launch, last_launch = get_flight_data(telemetry)

    if first_launch is not None:
        launch_x = [first_launch['x'], last_launch['x']]
        launch_y = [first_launch['y'], last_launch['y']]

        ax = fig.add_subplot(fig_x, fig_y, fig_num)

        # plot first and last jump locations
        ax.scatter(launch_x[0] / CM_TO_KM, launch_y[0] / CM_TO_KM, s=100,
                   color='green', edgecolors='black', zorder=1)
        ax.scatter(launch_x[1] / CM_TO_KM, launch_y[1] / CM_TO_KM, s=100,
                   color='red', edgecolors='black', zorder=1)

        # plot line between them
        ax.plot([x_ / CM_TO_KM for x_ in launch_x],
                [y_ / CM_TO_KM for y_ in launch_y], 'grey', linestyle='--', marker='', zorder=1)

        # plot each player according to their ranking
        for ranking in rankings:
            landing_loc = landings[ranking['name']]
           # print("Player {} landing at position\t ({}, {}) and ended up rank : {}".format(ranking['name'],
            #                                                                           landing_loc[0],
             #                                                                          landing_loc[1],
             #                                                                          ranking['ranking']))
            if ranking['ranking'] == 1:
                ax.scatter(landing_loc[0] / CM_TO_KM, landing_loc[1] / CM_TO_KM,
                           color='yellow', edgecolors='black', zorder=1)
            else:
                ax.scatter(landing_loc[0] / CM_TO_KM, landing_loc[1] / CM_TO_KM,
                           color='blue', alpha=1/ranking['ranking'], zorder=1)
        plt.ylim(0, y_max)
        plt.xlim(0, x_max)
        plt.xlabel('km')
        plt.ylabel('km')
        plt.title(map_name)
        plt.savefig('.\\match_landings\\match_{}.png'.format(match_num))
        plt.show()
    else:
        logging.error("Could not get launch data")



def build_drop_data(telemetry_files):  # TODO: Separate by map and patch version
    data_dir = ".\\data\\"
    drop_data = []

    # Plots each match landing locations on a new plot
    for match_num in range(0, len(telemetry_files)):
        logging.debug("Building match %i of %i", match_num, len(telemetry_files) - 1)
        # If the filesize is greater than 0, ie there is actual data in it
        if os.path.getsize(data_dir + telemetry_files[match_num]) > 0:
            try:
                telemetry = load_pickle(data_dir + telemetry_files[match_num])
            except EOFError:
                logging.error("Match file terminated unexpectedly, skipping")
                continue  # Skip processing files that terminate early
            first, last = get_flight_data(telemetry)
            map_name = get_map(telemetry)

            # Set map_size (in cm, like player locations)
            if map_name == "Savage_Main":  # 4km map
                map_size = 400000
            elif map_name == "Erangel_Main":  # 8km map
                map_size = 800000
            elif map_name == "Desert_Main":  # 8km map
                map_size = 800000
            elif map_name == "DihorOtok_Main":  # 6km map
                map_size = 600000

            # Get the flight direction
            if first is not None:
                flight_vec = get_flight_vector(first, last)
                dir = get_flight_category(flight_vec)

            # Get the landings
            landing_locs = get_all_landings(telemetry)
            rankings = get_rankings(telemetry)

            for player, loc in landing_locs.items():
                # For each player who dropped, get their rank from the rankings array
                loc_category = get_loc_cat(loc[0], loc[1], map_size)
                for ranking in rankings:
                    if player in ranking.values():
                        player_rank = ranking['ranking']

                drop_data.append({'player': player,
                                  'drop_loc_raw': loc,
                                  'drop_loc_cat': loc_category,
                                  'rank': player_rank,
                                  'flight_path': dir,
                                  'map': map_name})
                #print("{} ==> {}".format(flight_vec, dir))
            #display_drop_locations(telemetry, plt.figure(), 1, 1, 1, match_num)

    return pd.DataFrame(drop_data)

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

    # get the arg max of the score array to determine the optimnal value for k
    k_opt = [i for i in k_scores if i == max_score][0]

    knn = KNeighborsClassifier(n_neighbors=k_opt)
    kf = KFold(n_splits=10)

   # split the training data into different validation folds and run a training loop on it

    kf.get_n_splits(x_train)
    training_scores = []
    for train_index, test_index in kf.split(x_train):
        X_train = x_train[train_index]
        X_test = x_train[test_index]

        Y_train = y_train[train_index]
        Y_test = y_train[test_index]
        knn.fit(X_train, Y_train)

        training_scores.append((knn.predict(X_test) - Y_test).mean())
    print("TRAINING ACCURACY SCORES: ", training_scores)
    print("MEAN TRAINING ACCURACY: ", sum(training_scores) / len(training_scores))
    testing_accuracy = (knn.predict(x_test) - y_test).mean()
    print("TESTING ACCURACY: ", testing_accuracy)
    return knn









# Get safe zone and poison zone states (location and radius) throughout the game
# Returns list of dictionaries representing each time & states
# [ { '_D': str,
#     'safetyZonePosition' : {location},
#     'safetyZoneRadius' : int, 
#       ... }, ... ]
def getZoneStates(json_object):
    logGameStates = search(json_object, None, None, None, 'LogGameStatePeriodic')
    allStates = []
    for gameState in logGameStates:
        timestamp = gameState['_D']
        state = gameState['gameState']
        newStateObj = {k : state[k] for k in ('safetyZonePosition', 
                                                    'safetyZoneRadius', 
                                                    'poisonGasWarningPosition',
                                                    'poisonGasWarningRadius')}
        newStateObj['_D'] = timestamp
        allStates.append(newStateObj)
    return allStates

def main():
    data_dir = ".\\data\\"
    match_files = []
    telemetry_files = []

    downloader.setup_logging()
    logging.info("Scanning for match and telemetry files in %s to parse", data_dir)
    for file in os.listdir(data_dir):
        if "_match" in file:
            logging.debug("Match file %s found, adding as match", file)
            match_files.append(file)
        elif "_telemetry" in file:
            logging.debug("Telemetry file %s found, adding as match", file)
            telemetry_files.append(file)

    drop_data = build_drop_data(telemetry_files)
    print(drop_data.head())


if __name__ == "__main__":
    main()
