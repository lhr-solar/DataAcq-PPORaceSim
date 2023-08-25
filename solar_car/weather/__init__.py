from solar_car.weather.api import WeatherApi
import logging


class Weather:
    def __init__(self, lat: float, lon: float, time: int):
        API_KEY_1 = "UBJES729Z5FC7YCXGZK3CNMYG"
        API_KEY_2 = "9A7GBFDJ89M7SCXAY9LBDW2ZG"
        API_KEYS = [API_KEY_1, API_KEY_2]

        self.lat = lat
        self.lon = lon
        self.time = time

        # try every api key for convenience, if one fails, try the next one until one works
        for api_key in API_KEYS:
            try:
                self.api = WeatherApi(api_key)
                try:
                    logging.info("Loading weather data from file")
                    self.weather = self.api.get_weather_offline()
                except:
                    logging.warning(
                        "Failed to load weather data from file, loading from online API")
                    self.weather = self.api.get_weather_online(lat, lon, time)
                    self.weather.save()
                logging.info("Weather data loaded successfully")
                break
            except Exception as e:
                logging.warning(
                    f"API key {api_key} failed with error {e}, trying next one")

    def get_intensity(self, time: int) -> float:
        return self.weather.get_intensity(time)


if __name__ == "__main__":
    import datetime

    logging.basicConfig(level=logging.DEBUG)

    start_time = int(datetime.datetime(2023, 1, 10, 9, 0, 0).timestamp())

    w = Weather(40.7128, -74.0060, start_time)
    print(w.weather.get_intensity(start_time + 300000))
