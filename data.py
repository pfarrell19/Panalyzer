import pandas as pd
from pandas.io.json import json_normalize
import requests

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


# Given a match ID, pull the data for that match and return it as a pandas DataFrame
def get_match_stats(matchID):
    header = {"accept" : "application/vnd.api+json"}
    apireq = requests.get("https://api.pubg.com/shards/steam/matches/%s" % matchID, headers=header)
    print(apireq)


    match_df = pd.DataFrame(apireq.json()['data'])
    match_data = pd.DataFrame(apireq.json()['included'])
    assets = pd.DataFrame(match_df.loc['assets', 'relationships'])
    rosters = pd.DataFrame(match_df.loc['rosters', 'relationships'])
    #print(match_df.to_string())
    #print(assets['data'].apply(pd.Series).head().to_string())
    #print(rosters['data'].apply(pd.Series).head().to_string())
    print(match_data.head(), match_data.shape)
    #print(pd.DataFrame(norm).to_string())

    return


if __name__ == "__main__":
    ids = get_matches()
    get_match_stats(ids[0])
