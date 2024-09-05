import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_earthquake_data(start_date, end_date):
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "geojson",
        "starttime": start_date,
        "endtime": end_date,
        "minmagnitude": 4.0
    }
    response = requests.get(url, params=params)
    data = response.json()
    features = data['features']
    
    earthquake_data = []
    for feature in features:
        properties = feature['properties']
        earthquake_data.append({
            "time": datetime.fromtimestamp(properties['time'] / 1000.0),
            "latitude": feature['geometry']['coordinates'][1],
            "longitude": feature['geometry']['coordinates'][0],
            "depth": feature['geometry']['coordinates'][2],
            "magnitude": properties['mag'],
            "place": properties['place']
        })
    
    df = pd.DataFrame(earthquake_data)
    return df

if __name__ == "__main__":
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    df = fetch_earthquake_data(start_date.isoformat(), end_date.isoformat())
    df.to_csv("data/raw/earthquake_data.csv", index=False)
    print("Earthquake data collected and saved to earthquake_data.csv")