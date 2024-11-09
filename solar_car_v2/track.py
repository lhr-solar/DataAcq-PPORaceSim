import pychrono as chrono
from pychrono import vehicle as veh
import numpy as np
import random


def generate_path():
    """
    Generate a path for the solar car environment to follow
    Returns the path
    """
    random.seed(42)
    points = [(x, y, 0) for (x, y) in list(np.random.rand(10, 2) * 150)]

    distances = []
    total_distance = 0
    for i in range(len(points) - 1):
        p1 = np.array(points[i])  # Current point
        p2 = np.array(points[i + 1])  # Next point
        total_distance += np.linalg.norm(p2 - p1)  # Calculate the Euclidean distance
        distances.append(total_distance)


    bounding_box = (
        # min x:
        # np.min(points[0]),
        min(points, key = lambda x: x[0])[0],
        # min y:
        # np.min(points[1]),
        min(points, key = lambda y: y[1])[1],
        # max x:
        # np.max(points[0]),
        max(points, key = lambda x: x[0])[0],
        # max y:
        # np.max(points[1]),
        max(points, key = lambda x: x[1])[1]
    )
    width = bounding_box[2] - bounding_box[0]
    height = bounding_box[3] - bounding_box[1]

    # shift the points to 0, 0
    shift_left = bounding_box[0] + width / 2
    shift_down = bounding_box[1] + height / 2
    points = [(x - shift_left, y - shift_down, 0) for (x, y, z) in points]

    vector_3d_points = [chrono.ChVector3d(x, y, z) for (x, y, z) in points]

    vectot_vector_3d = chrono.vector_ChVector3d(vector_3d_points)
    path = chrono.ChBezierCurve(vectot_vector_3d)
    
    return path, points, distances


def generate_terrain(system, path: chrono.ChBezierCurve):
    """
    Generate terrain for the solar car environment to drive on
    Returns the terrain
    """

    material = chrono.ChContactMaterialSMC()
    material.SetFriction(0.9)
    material.SetRestitution(0.01)
    material.SetYoungModulus(2e7)
    material.SetPoissonRatio(0.3)

    terrain = veh.RigidTerrain(system)

    for i, _point in enumerate(path.GetPoints()[1:]):
        for j in np.linspace(0, 1, 100):
            point: chrono.ChVector3d = path.Eval(i, j)
            deriv: chrono.ChVector3d = path.EvalDer(i, j)

            # normalizedtangent to disregard distance
            x, y = deriv.x, deriv.y
            angle = np.arctan2(y, x)
            normal_angle = angle + np.pi / 2

            normal = chrono.ChVector3d(np.cos(normal_angle), np.sin(normal_angle), 0)

            pos = chrono.ChCoordsysd(point, 0, normal)
            terrain.AddPatch(material, pos, 10, 10, 0)

    return terrain
