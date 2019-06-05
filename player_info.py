import jsonparser
import downloader
from data_visualization import display_player_paths
import logging
import os
import pandas as pd
import numpy as np
from pandas.io.json import json_normalize


def in_zone(x, y, zone_x, zone_y, zone_r):
    """ Given (x, y) of a position, return True if it is within the zone boundaries
        and False if it is not
    """
    dist_to_center = np.linalg.norm(np.array(x, y) - np.array(zone_x, zone_y))
    return dist_to_center < zone_r


def round_raw(raw):
    """
    Convert raw position value to halfway in the nearest 100m interval (Center of the grid square for that interval)
        e.g. position (126903, 341005) == (1.27km, 3.41km) -> (125000, 345000) == (1.25km, 3.45km)

    :param raw: raw value of coordinate (in cm)
    :return: rounded value to the nearest .05km
    """
    return (raw // 10000) * 10000 + 5000


def get_player_paths(telemetry):
    """
    Get all logged locations of the players from the telemetry data, along with the character name
    (for joining purposes with rank) and the game state (for joining purposes with the zone data)

    :param telemetry: json object of all the telemetry events for a given match
    :return: DataFrame with columns:

            [game_state, name, x, y]

            where each row represents a log of an individual player position.
            Players have multiple logs per game, i.e. multiple rows in the DataFrame
    """
    telemetry_df = json_normalize(telemetry)
    positions_df = (telemetry_df[telemetry_df['_T'] == 'LogPlayerPosition']
        [['common.isGame', 'character.name', 'character.location.x', 'character.location.y']])
    positions_df.columns = positions_df.columns.map({
        'common.isGame': 'game_state',
        'character.name': 'name',
        'character.location.x': 'x',
        'character.location.y': 'y'
    })

    rankings = jsonparser.get_rankings(telemetry)
    rankings_df = pd.DataFrame(rankings)
    return positions_df.set_index("name").join(rankings_df.set_index("name"))


def join_player_and_zone(player_paths, zone_info):
    """
    Join a DataFrame with player location info and a DataFrame with the zones location info

    :param player_paths: DataFrame containing information about player locations throughout the game
    :param zone_info: DataFrame containing information about the poison gas zone and safe zone throughout the game
    :return: DataFrame indexed on the game state (0.0, 0.1, 0.5, 1.0,...) with columns:

            [name, x, y, x_, y_, rank, poisonGasWarningPosition_x, poisonGasWarningPosition_y, poisonGasWarningRadius,
             safetyZonePosition_x, safetyZonePosition_y, safetyZoneRadius]

            where x_ and y_ are the rounded values of the raw locations.
            Each row represents a single player's location at a given time.
            Poison gas and safety zone values may be NaN for logs prior to match start.
    """
    player_paths = player_paths.dropna().reset_index()
    player_paths['x_'] = player_paths['x'].apply(round_raw)
    player_paths['y_'] = player_paths['y'].apply(round_raw)
    return player_paths.set_index("game_state").join(zone_info.set_index("isGame"))


if __name__ == "__main__":
    data_dir = ".\\data\\"
    match_files = []
    telemetry_files = []

    #downloader.setup_logging()
    logging.info("Scanning for match and telemetry files in %s to parse", data_dir)
    for file in os.listdir(data_dir):
        if "_match" in file:
            logging.debug("Match file %s found, adding as match", file)
            match_files.append(file)
        elif "_telemetry" in file:
            logging.debug("Telemetry file %s found, adding as match", file)
            telemetry_files.append(file)

    telemetry = jsonparser.load_pickle(data_dir + telemetry_files[0])

    paths = get_player_paths(telemetry).dropna().reset_index()
    blue_zones = jsonparser.getZoneStates(telemetry)

    all = join_player_and_zone(paths, blue_zones)
    display_player_paths(all[(all.name == 'AmuseTown')], "Savage_Main")
    print(all.columns, "\n", all.head())
