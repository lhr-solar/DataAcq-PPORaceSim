from weather.api import WeatherApi
import logging


class Weather:
    def __init__(self, lat: float, lon: float, time: int):
        API_KEY = "UBJES729Z5FC7YCXGZK3CNMYG"
        API_KEY_2 = "9A7GBFDJ89M7SCXAY9LBDW2ZG"

        self.lat = lat
        self.lon = lon
        self.time = time
        self.api = WeatherApi(API_KEY)
        try:
            logging.info("Loading weather data from file")
            self.weather = self.api.get_weather_offline()
        except:
            logging.warning(
                "Failed to load weather data from file, loading from online API")
            self.weather = self.api.get_weather_online(lat, lon, time)
            self.weather.save()

    def get_intensity(self, time: int) -> float:
        return self.weather.get_intensity(time)


if __name__ == "__main__":
    import datetime

    logging.basicConfig(level=logging.DEBUG)

    start_time = int(datetime.datetime(2023, 1, 10, 9, 0, 0).timestamp())

    w = Weather(40.7128, -74.0060, start_time)
    print(w.weather.get_intensity(start_time + 300000))
