import datetime

class match:
    def __init__(self, match_id, game_mode, map, start_time, duration, telemetry_url):
        self.match_id = match_id
        self.teams = []
        self.game_mode = game_mode
        self.map = map
        self.start_time = start_time
        self.duration = duration
        self.telemetry_url = telemetry_url

class team:
    def __init__(self, team_id):
        self.team_id = team_id
        self.members = []

class member:
    def __init__(self, member_id, username):
        self.member_id = member_id
        self.username = username