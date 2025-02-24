import numpy as np
import os
import trimesh

from cloth_geometry import ClothGeometry
from pxr import Sdf, Usd, UsdGeom

# ClothGeometry class
# Load USD file to import the full geometry
# The uds file is export form our houdini project
class ClothGeometryMultigrid(ClothGeometry):
    def __init__(self, usd_file):
        super().__init__(usd_file)

        self.layer = self.getPointsAttribute('layer', 0)
        print(self.layer)



