import pychrono as chrono
from pychrono import vehicle as veh
import numpy as np


def generate_track():
    """
    Generate a path for the solar car environment to follow
    Returns the path
    """

    points = [(x, y, 0) for (x, y) in list(np.random.rand(10, 2) * 150)]
    vector_3d_points = [chrono.ChVector3d(x, y, z) for (x, y, z) in points]
    vectot_vector_3d = chrono.vector_ChVector3d(vector_3d_points)
    path = chrono.ChBezierCurve(vectot_vector_3d)

    return path


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

    for i, _point in enumerate(path.GetPoints()[:-1]):
        point: chrono.ChVector3d = _point
        prev_point: chrono.ChVector3d = path.GetPoints()[i - 1]

        quat: chrono.ChQuaterniond = chrono.QuatFromVec2Vec(prev_point, point)
        pos = chrono.ChCoordsysd(point, quat)
        len = (point - prev_point).Length()

        terrain.AddPatch(material, pos, len, 10)

    return terrain
