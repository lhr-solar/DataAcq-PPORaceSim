import json, requests, datetime

class Weather:
    # this is the api key for website https://www.visualcrossing.com/weather-api
    # limited to 1000 free / day
    API_KEY = "UBJES729Z5FC7YCXGZK3CNMYG"
    API_KEY_2 = "9A7GBFDJ89M7SCXAY9LBDW2ZG"
    outputfilename = "weather_data/weather.csv"
    csv_data = []
    currWeatherRad = 0

    def __init__(self, latitude, longitude, time):
        self.latitude = latitude
        self.longitude = longitude
        self.time = time

        with open("weather_data/weather.json", "r+") as outfile:
            try:
                file = json.loads(outfile.read())
            except:
                file = None

            if file is None or not (file["startTime"] <= time <= file["endTime"]):
                print("Getting new weather data...")

                url = 'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/' + str(
                    self.latitude) + ',' + str(self.longitude) + f'/{int(time)}/{int(time) + 60*60*24*4}' + '?unitGroup=metric&include=hours&key=' + self.API_KEY_2 + '&contentType=json'
                print(url)
                
                self.data = json.loads(requests.get(url).text)
                self.data["startTime"] = time
                self.data["endTime"] = time + 60*60*24*4

                json.dump(self.data, outfile)
            elif file:
                print("Using cached weather data...")
                self.data = file




    def get_intensity(self):
        # get weather data from API
        # return weather data

        # altitude_deg = pysolar.solar.get_altitude(self.latitude,self.longitude,self.time)
        # return pysolar.radiation.get_radiation_direct(self.time,altitude_deg)

        return self.currWeatherRad

    def pull_weather_data(self, epochTime):
        for day in self.data["days"]:
            t = int(day["datetimeEpoch"]) + 86400 # a day in seconds
            if t > epochTime:
                for hour in day["hours"]:
                    light_intensity = float(hour["solarradiation"])
                    if int(hour["datetimeEpoch"]) > epochTime:
                        print(f'Light Intensity at {day["datetime"]} @ {hour["datetime"]}: {light_intensity}')
                        type(self).currWeatherRad = light_intensity
                        print("LIGHT INTENSITY" + (str)(light_intensity))
                        return light_intensity


if __name__ == "__main__":
    a = Weather(40.7128, -74.0060, "2022-11-12T11:10:00")
    a.pull_weather_data()
