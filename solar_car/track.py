import numpy as np
import splines
import json
import plotly.express as px;
from mpl_toolkits.mplot3d import axes3d
import pandas as pd;
import matplotlib.pyplot as plt
from scipy.integrate import quad
import scipy as sp
import scipy.interpolate
from solar_car.helper import plot_spline_3d


class Track:
    """
    The track class defines objects that describe the track the car is on.
    It uses cubic interpolation to describe every point of the track. 
    Additionally, it can calculate the physical maximum speed at any point on the track.

    Parameters
    ----------
    track_file : str, optional
        The path to the track file to be used. The default is None, but either track_file or geojson must be specified.
    geo_json : any, optional
        The geojson object to be used. The default is None, but either track_file or geojson must be specified.
    """

    def __init__(self, track_file: str = None, geo_json: any = None) -> None:
        points = []
        if track_file:
            # with open(track_file) as f:
            #     geo_json = json.load(f)
            #     features = geo_json["features"]
            #     for feature in features:
            #         geometry = feature["geometry"]
            #         coordinates = geometry["coordinates"]
            #         properties = feature["properties"]

            #         points.append([*coordinates, properties["elevation"]])
            #         self.coords = coordinates

            # with open(track_file + ".txt", 'r') as fin, open(track_file + "_commas_out.txt", 'w') as fout:
            #     s = fin.readline()
            #     for line in fin:
            #         s = line.replace('   ', ',').replace("  ", ",").rstrip(",")
            #         fout.write(s)
            df = pd.read_csv(track_file + "_commas_out.txt")
            for i in range(df.shape[0]):
                #slight variation in points for some randomness
                points.append([df.iloc[i, 1] + np.random.uniform(-.001, .001), df.iloc[i, 2] + np.random.uniform(-.001, .001), df.iloc[i, 3] + np.random.uniform(-.001, .001)])
        elif geo_json:
            features = geo_json["features"]
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
        # calc
        first_x = xs[0]
        first_y = ys[0]
        for i in range(len(xs)):
            xs[i] = (xs[i] - first_x) * 111.1 * np.cos(np.pi * ys[i]/180)
            ys[i] = (ys[i] - first_y) * 111.1

        min_lat, max_lat, min_lon, max_lon = min(ys), max(ys), min(xs), max(xs)
        # aspect_ratio = (max_lat - min_lat)/(max_lon - min_lon)

        # Scale points and convert to km
        # points = [[(point[0] - min_lon)/(max_lon - min_lon), (point[1] - min_lat)/(max_lat - min_lat), point[2]] for point in points]
        points = [[xs[i], ys[i], points[i][2]] for i in range(len(points))]
        points = np.array(points)
        # points[:, 1] = points[:, 1]*aspect_ratio # comment out if needed
        # points = points*.65  # rough conversion to km from lat/lon
        # ^- conversion not accurate, maybe an issue the points. Should be .4 multiple, but the track would too short then

        self.points = points
        cmr = splines.CatmullRom(points, endconditions="closed")
        self.piece_lengths = []
        for i in range(len(cmr.grid) - 1):
            self.piece_lengths += [self.__arc_length(cmr, i, i+1)]
        self.track_length = sum(self.piece_lengths)
        self.t_len = self.track_length

        self.cmr = cmr

        min_x, max_x, min_y, max_y = np.min(points[:, 0]), np.max(
            points[:, 0]), np.min(points[:, 1]), np.max(points[:, 1])
        self.bounding_box = np.array([[min_x, min_y], [max_x, max_y]])

        # Determine ground level by taking average.
        self.ground_level = np.mean([point[2] for point in points])

    #def get_standard_length_arr() 
    # PARAMS
    # y1 = y'(x)
    # y2 = y''(x)
    def __curvature(self, y1: float, y2: float) -> float:
        return (np.abs(y2))/((1+y1**2)**(3/2))

    # Evalulate with distance constant speed across the spline.
    # I.e. ensure the spline is linear in t and C2
    # d is between 0 and the total track length
    def evaluate_cs(self, d: float, n=0) -> np.ndarray:
        t = self.distance_to_t(d)
        return self.cmr.evaluate(t, n)

    # Returns R at t in the 2d plane
    def curvature(self, t: float) -> float:
        y1 = self.cmr.evaluate(t, 1)[1]
        y2 = self.cmr.evaluate(t, 2)[1]
        return self.__curvature(y1, y2)

    # # Returns the slope of the elevation profile at t
    def elevation_slope(self, t: float) -> float:
        return self.cmr.evaluate(np.fmod(t, self.t_len), 1)[2]

    def elevation(self, t: float) -> float:
        return self.cmr.evaluate(np.fmod(t, self.t_len))[2]

    def distance_to_t(self, d: float) -> float:
        traveled = 0
        i = 0
        d = np.fmod(d, self.track_length)
        while d > traveled:
            traveled += self.piece_lengths[i]
            i += 1

        # this assumes the functions have constant f' but that is not true
        # Good enough though
        t = (i) - (traveled - d)/self.piece_lengths[i - 1]

        return t

    def __arc_length(self, spline, t1, t2) -> float:
        def f(x):
            x1, y1, z1 = spline.evaluate(x, 1)
            return np.sqrt(x1**2 + y1**2)
        return quad(f, t1, t2)[0]

    # max speed at T according to friction and incline
    # based on http://hyperphysics.phy-astr.gsu.edu/hbase/Mechanics/carbank.html
    def max_speed(self, t: float) -> float:
        r = 1 / self.curvature(t) * 1000
        theta = np.arctan2(self.elevation(t) - self.ground_level, r)
        g = 9.81
        friction_co = 3.14
        return np.sqrt((r*g*(np.sin(theta)+friction_co*np.cos(theta)))/(np.cos(theta)-friction_co*np.sin(theta)))


if __name__ == "__main__":
    t = Track("./track2.json")
    cmr = t.cmr
