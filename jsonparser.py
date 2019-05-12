def search(json_object, player_id, start_range, end_range, event_type):
    # TODO

def get_flight_data(telemetry):
    first_coordinate = None # First player exit event from plane
    current_coordinate = None # Last player exist even from plane
    for log_entry in telemetry:
        if log_entry.get("_T") == "LogVehicleLeave" and log_entry.get("vehicle").get("vehicleId") == "DummyTransportAircraft_C":
            current_coordinate = log_entry.get("character").get("location")
            if first_coordinate == None:
                first_coordinate = current_coordinate
    return first_coordinate, current_coordinate