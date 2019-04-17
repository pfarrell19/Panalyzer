import pandas as pd
from pandas.io.json import json_normalize
import requests
import gzip 
import shutil
import os 

def main():
    # Check API key
    api_keys = ["eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJqdGkiOiIyZmZlNTYxMC0zYTAxLTAxMzctOWEwNi0wZjk4N2JkNjRmNzEiLCJpc3MiOiJnYW1lbG9ja2VyIiwiaWF0IjoxNTU0NDkwMDQ5LCJwdWIiOiJibHVlaG9sZSIsInRpdGxlIjoicHViZyIsImFwcCI6InBhbmFseXplciJ9.gXZ-7_3E8BPBapLWixVd9NCoSuEyJn80bdx87sIXHco"]
    # with open('api_keys.txt') as api_keys_file:
    #     for key in api_keys_file:
    #         api_keys.append(key)

    # Get list of matches (also checks key limits)
    sample_matches = get_sample_matches(api_keys[0])

    # Survey all downloaded files to make list of all downloaded matches
    # TODO

    # For all matches not yet downloaded, put them in an array
    # TODO

    # Query each match on the API and dump its info, logging telemetry URL in an array
    for match in sample_matches:
        get_match_stats(match)

    # Download matches from the array
    # TODO

    # Decompress matches
    # TODO


# Get a list of random match IDs from the PUBG API
def get_sample_matches(api_key):
    print("Getting sample matches with API key %s", api_key)
    # Header for auth
    header = {"Authorization" : "Bearer %s" % api_key,
              "Accept" : "application/vnd.api+json"}

    # Get the API call data
    api_result = requests.get("https://api.pubg.com/shards/steam/samples", headers=header)
    if api_result.status_code == 403:
        return None

    # Dump json data into pandas for better formatting
    samples_df = pd.DataFrame(api_result.json())

    # Extract just the match data
    samples_norm = json_normalize(samples_df.loc['relationships'])
    matches = samples_norm['matches.data'].apply(pd.Series).T[0].apply(pd.Series)

    return matches['id']


# Given a match ID, pull the data for that match and return it as a pandas DataFrame
def get_match_stats(matchID):
    header = {"accept" : "application/vnd.api+json"}
    apireq = requests.get("https://api.pubg.com/shards/steam/matches/%s" % matchID, headers=header)

    # API response is structured: { data : {...}
    #                               included: [...] }
    # where data has information about the IDs of the match objects (players, rosters, etc) and
    # included contains links to the actual objects about the match (look up by ID from data)

    # get the data
    match_df = pd.DataFrame(apireq.json()['data'])

    # Get the id of the telemetry object to look up in the included array
    assets = pd.DataFrame(match_df.loc['assets', 'relationships'])['data'].apply(pd.Series)
    telemetry_id = assets[assets.type == "asset"]['id'].values[0]

    # Get the telemetry object from the included array
    match_data = pd.DataFrame(apireq.json()['included'])
    telemetry_object = match_data[match_data.id == telemetry_id]

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

if __name__== "__main__":
    main()
