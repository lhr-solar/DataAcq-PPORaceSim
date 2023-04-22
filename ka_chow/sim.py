import datetime
import time
import math
import sys
import logging

from car import Car
from track import Track

logging.basicConfig(level=logging.DEBUG)

trackFile = str(sys.argv[1])

track = Track(trackFile=trackFile)

tickLength = 5  # seconds per tick


class Day:
    def __init__(self, startTime: int, runTime: int) -> None:
        self.startTime = startTime
        self.runTime = runTime


nineHours = 60*60*9
days = [Day(datetime.datetime(2023, 1, 10 + i, 9, 0,
            0).timestamp(), nineHours) for i in range(3)]

totalDist = 0
for i, day in enumerate(days):
    currentTime = day.startTime
    car = Car(track, day.startTime)
    currDist = 0
    print(
        f"\n-------DAY {i+1} STARTING AT {datetime.datetime.fromtimestamp(day.startTime)}-------")
    while currentTime - day.startTime < day.runTime:
        print(f"Time Passed: {currentTime - day.startTime}s")
        print(f"Stoping at: { day.runTime}")
        print(f"Distance Travelled: {currDist}km")
        logging.debug(f"Laps Travelled: {currDist / track.trackLength}")
        slope = track.elevationSlope(track.distanceToT(currDist))/180 * math.pi
        logging.debug(f'Slope: {slope}')
        currDist += car.drive(10, slope, currentTime) / 1000  # to km
        currentTime += tickLength
        time.sleep(1)
        # print()
    totalDist += currDist
print(f'Laps Travelled: {totalDist / track.trackLength}')
