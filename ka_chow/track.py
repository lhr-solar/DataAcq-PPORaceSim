import numpy as np
import splines
import json

from typing import List
from scipy.integrate import quad


class Track:
    # PARAMS:
    #   trackFile: the name of the track file to be used
    def __init__(self, trackFile=None, geojson=None) -> None:
        points = []
        if trackFile:
            with open(trackFile) as f:
                geojson = json.load(f)
                features = geojson["features"]
                for feature in features:
                    geometry = feature["geometry"]
                    coordinates = geometry["coordinates"]
                    properties = feature["properties"]

                    points.append([*coordinates, properties["elevation"]])
                    self.coords = coordinates
        elif geojson:
            features = geojson["features"]
            for feature in features:
                geometry = feature["geometry"]
                coordinates = geometry["coordinates"]
                properties = feature["properties"]

                points.append([*coordinates, properties["elevation"]])
                self.coords = coordinates
        else:
            raise Exception("No track file or geojson provided")

        # Determine aspect ratio for scaling
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        minLat, maxLat, minLon, maxLon = min(ys), max(ys), min(xs), max(xs)
        aspectRatio = (maxLat - minLat)/(maxLon - minLon)

        # Scale points and convert to km
        points = [[(point[0] - minLon)/(maxLon - minLon), (point[1] -
                                                           minLat)/(maxLat - minLat), point[2]] for point in points]
        points = np.array(points)
        points[:, 1] = points[:, 1]*aspectRatio
        points = points*.65  # rough conversion to km from lat/lon
        # ^- conversion not accurate, maybe an issue the points. Should be .4 multiple, but the track would too short then

        self.points = points
        cmr = splines.CatmullRom(points, endconditions="closed")

        self.cmr = cmr

        self.pieceLengths = []
        for i in range(len(cmr.grid) - 1):
            self.pieceLengths += [self.__arcLength(
                cmr, i, i+1)]
        self.trackLength = sum(self.pieceLengths)
        self.tLen = self.trackLength

        minX, maxX, minY, maxY = np.min(points[:, 0]), np.max(
            points[:, 0]), np.min(points[:, 1]), np.max(points[:, 1])
        self.boundingBox = np.array([[minX, minY], [maxX, maxY]])

    # PARAMS
    # y1 = y'(x)
    # y2 = y''(x)
    def __curvature(self, y1: float, y2: float) -> float:
        return (np.abs(y2))/((1+y1**2)**(3/2))

    # Evalulate with distance constant speed across the spline.
    # I.e. ensure the spline is linear in t and C2
    # d is between 0 and the total track length
    def evaluateCS(self, d: float, n=0) -> np.ndarray:
        t = self.distanceToT(d)
        return self.cmr.evaluate(t, n)

    # Returns R at t in the 2d plane
    def curvature(self, t: float) -> float:
        y1 = self.cmr.evaluate(t, 1)[1]
        y2 = self.cmr.evaluate(t, 2)[1]
        return self.__curvature(y1, y2)

    # Returns the slope of the elevation profile at t
    def elevationSlope(self, t: float) -> float:
        return self.cmr.evaluate(np.fmod(t, self.tLen), 1)[2]

    def distanceToT(self, d: float) -> float:
        traveled = 0
        i = 0
        d = np.fmod(d, self.trackLength)
        while d > traveled:
            traveled += self.pieceLengths[i]
            i += 1

        # this assumes the functions have constant f' but that is not true
        # Good enough though
        t = (i) - (traveled - d)/self.pieceLengths[i - 1]

        return t

    def __arcLength(self, spline, t1, t2) -> float:
        def f(x):
            x1, y1, z1 = spline.evaluate(x, 1)
            return np.sqrt(x1**2 + y1**2)
        return quad(f, t1, t2)[0]

    # max speed at T according to friction
    # math.sqrt(self.GRAVITY * self.mass * self.tire_fricts * radius)
    def __maxSpeed(self, t: float, m: float) -> float:
        K = self.curvature(t) * 1  # to m
        F = 3.14
        return np.sqrt(9.8 * m * F * (1/K))

    def maxSpeed(self, d: float, m: float) -> float:
        return self.__maxSpeed(self.distanceToT(d), m)


if __name__ == "__main__":
    t = Track("./track2.json")
    cmr = t.cmr
    print(t.trackLength)
    print(t.evaluateCS(0), cmr.evaluate(0))
    print(t.evaluateCS(t.trackLength - .000001), cmr.evaluate(cmr.grid[-1]))
