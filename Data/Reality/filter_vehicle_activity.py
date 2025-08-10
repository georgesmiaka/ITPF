import json
import pandas as pd

def filter_vehicle_activities(json_file):
    # Load raw JSON data
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    filtered_records = []
    
    for record in data:
        # Check if it's an activity and if topCandidate.type is "in passenger vehicle"
        if "activity" in record:
            activity = record["activity"]
            top_candidate = activity.get("topCandidate", {})
            if top_candidate.get("type") == "in passenger vehicle":
                # Parse coordinates
                start_coords = activity["start"].replace("geo:", "").split(",")
                end_coords = activity["end"].replace("geo:", "").split(",")

                filtered_records.append({
                    "start_lat": float(start_coords[0]),
                    "start_lon": float(start_coords[1]),
                    "end_lat": float(end_coords[0]),
                    "end_lon": float(end_coords[1]),
                    "startTime": record["startTime"],
                    "endTime": record["endTime"],
                    "distance_m": float(activity.get("distanceMeters", 0)),
                    "probability": float(activity.get("probability", 0))
                })
    
    # Convert to DataFrame for easy viewing/processing
    df = pd.DataFrame(filtered_records)
    return df


# Example usage
if __name__ == "__main__":
    df_vehicle = filter_vehicle_activities("raw_data.json")
    #print(df_vehicle.head())
    # Save to CSV if needed
    df_vehicle.to_csv("vehicle_activities.csv", index=False)
