import json
import os
import requests


class WeatherInterface:
    def __init__(self, raw: dict, start_time: int, end_time: int):
        self.raw = raw
        self.start_time = start_time
        self.end_time = end_time

    def time_in_range(self, time: float) -> bool:
        return self.start_time <= time <= self.end_time

    def get_intensity(self, time: float) -> float:
        if not self.time_in_range(time):
            raise ValueError("Time is not in range")

        for day in self.raw["days"]:
            t = int(day["datetimeEpoch"]) + 86400
            if t > time:
                for hour in day["hours"]:
                    if int(hour["datetimeEpoch"]) > time:
                        return float(hour["solarradiation"])

    def save(self):
        try:
            os.makedirs("weather_data")
        except FileExistsError:
            pass
        with open("weather_data/weather.json", "w") as outfile:
            self.raw["beginTime"] = self.start_time
            self.raw["endTime"] = self.end_time
            outfile.write(json.dumps(self.raw))


class WeatherApi:
    def __init__(self, API_KEY: str):
        self.API_KEY = API_KEY

    def get_weather_offline(self):
        with open("weather_data/weather.json", "r") as outfile:
            file = json.loads(outfile.read())
            weather = WeatherInterface(
                file, file["beginTime"], file["endTime"])
            return weather

    def get_weather_online(self, lat: float, lon: float, time: int) -> WeatherInterface:
        end_time = time + 60 * 60 * 24 * 4
        url = 'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/' + str(
            lat) + ',' + str(lon) + f'/{time}/{end_time}' + '?unitGroup=metric&include=hours&key=' + self.API_KEY + '&contentType=json'

        data = json.loads(requests.get(url).text)

        return WeatherInterface(data, time, end_time)


if __name__ == "__main__":
    API_KEY = "UBJES729Z5FC7YCXGZK3CNMYG"
    API_KEY_2 = "9A7GBFDJ89M7SCXAY9LBDW2ZG"
    api = WeatherApi(API_KEY)
