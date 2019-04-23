import pandas as pd
from pandas.io.json import json_normalize
import requests
import gzip 
import shutil
import os 
import json 

apikey = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJqdGkiOiIyY2IxM2ZhMC0zYzQ1LTAxMzctODYwNC0wMWI0YTFkMWRiOWEiLCJpc3MiOiJnYW1lbG9ja2VyIiwiaWF0IjoxNTU0NzM5MTUxLCJwdWIiOiJibHVlaG9sZSIsInRpdGxlIjoicHViZyIsImFwcCI6Ii02MTA3ZjhlYS1kZjNlLTQzYTctYTU5Ni1hMzljYTY5NDQ2NzUifQ.Aqxif7hL74A6LxrnpO7LNZYMCACAf4HHYTatEs3UWYs"


# Get a list of random match IDs from the PUBG API
def get_matches():
    # Header for auth
    header = {"Authorization" : "Bearer %s" % apikey,
              "Accept" : "application/vnd.api+json"}

    # Get the API call data
    apireq = requests.get("https://api.pubg.com/shards/steam/samples", headers=header)

    # Dump json data into pandas for better formatting
    samples_df = pd.DataFrame(apireq.json())

    # Extract just the match data
    samples_norm = json_normalize(samples_df.loc['relationships'])
    matches = samples_norm['matches.data'].apply(pd.Series).T[0].apply(pd.Series)

    return matches['id']


# Extract the telemetry json url from the match as well as the winning player's IDs (written to winner_outfile)
def get_match_data(matchID, winner_outfile):

    header = {"accept": "application/vnd.api+json"}
    apireq = requests.get("https://api.pubg.com/shards/steam/matches/%s" % matchID, headers=header)

    # API response is structured: { data : {...}
    #                               included: [...] }
    # where data has information about the IDs of the match objects (players, rosters, etc) and
    # included contains links to the actual objects about the match (look up by ID from data)

    # get the data
    match_data = pd.DataFrame(apireq.json()['data'])
    match_included = pd.DataFrame(apireq.json()['included'])

    # Get the id of the telemetry object to look up in the included array
    assets = pd.DataFrame(match_data.loc['assets', 'relationships'])['data'].apply(pd.Series)
    telemetry_id = assets[assets.type == "asset"]['id'].values[0]

    # Get the winning roster
    rosters = pd.DataFrame(match_data.loc['rosters', 'relationships'])['data'].apply(pd.Series)
    rosters_included = match_included[match_included.type == 'roster']
    combined = rosters_included.merge(rosters, on='id')
    combined = pd.concat([combined.drop(['attributes', 'relationships'], axis=1),
                                combined['attributes'].apply(pd.Series),
                                combined['relationships'].apply(pd.Series)], axis=1)

    winning_team = combined[combined.won == 'true']
    winning_participants = json_normalize(winning_team['participants'].values[0], record_path='data')

    # Now get the data (stats, etc) related to the winning players from the included array
    winning_participants_data = match_included[match_included.id.isin(winning_participants.id)]
    # And write it to disk
    winning_participants_data.to_csv(winner_outfile, header=False, mode="a")

    # Get the telemetry object from the included array
    telemetry_object = match_included[match_included.id == telemetry_id]

    # Return the url of the telemetry object
    return telemetry_object['attributes'].apply(pd.Series)['URL']


# Given location of gzips and location of where to extract to -> unzips gzip files
def extract_gzip(gzip_indir, gzip_outdir): 
    
    # Check if directories exist
    if not os.path.isdir(gzip_indir): 
        print("Cannot find directory '" + gzip_indir + "'")
        exit()
    elif not os.path.isdir(gzip_outdir):
        print("Cannot find directory '" + gzip_outdir + "'")
        exit()

    # Get name of all files to unzip
    files = os.listdir(gzip_indir)
    for f in files: 
        outfile  = f.replace(".gz", "")
        try: 
            with gzip.open(gzip_indir + f, 'rb') as g:
                print("Copying '" + f + "'")

                # Copy files to output directory
                with open(gzip_outdir + outfile, 'wb') as g_copy: 
                    shutil.copyfileobj(g, g_copy)

        except: 
            print("Unable to copy file '" + gzip_indir + f + "'\n")

    print("Copying data - done")

#tel_file (json file path) -> pandas.DataFrame
#Returns a dataframe of logs on parachute landing
def telemetry_pland(tel_file): 
    tel_df = None
    drop_events = []
    with open(tel_file) as f: 
        json_file = json.load(f)
        for entry in json_file: 
            if 'LogParachuteLanding' == entry['_T']:
                drop_events.append(entry)

    drop_table = json_normalize(drop_events)
    return drop_table
