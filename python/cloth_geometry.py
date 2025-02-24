import numpy as np
import os
import trimesh

from pxr import Sdf, Usd, UsdGeom

# ClothGeometry class
# Load USD file to import the full geometry
# The uds file is export form our houdini project
class ClothGeometry:
    def __init__(self, usd_file):
        # check whether the usd file exists
        if not os.path.exists(usd_file):
            print('usd file not found')
            exit()

        # load geometry of the usd file
        stage = Usd.Stage.Open(usd_file)
        self.stage = stage

        # root_layer = stage.GetRootLayer()
        # print(root_layer.ExportToString())

        # get 'usdexport1' prim
        export_prim = stage.GetPrimAtPath('/usdexport1')

        # traverse the export prim and get its children
        for child in export_prim.GetAllChildren():
            if (child.GetTypeName() == 'Mesh'):
                self.mesh = UsdGeom.Mesh(child)

        self.face_counts, self.face_indices, self.triangles = self._loadTriangles()

        # sample other custom attributes named "fixed"
        self.positions = np.array(self.getPoints(0))
        self.positions_init = self.positions
        self.velocities = np.zeros_like(self.positions)
        self.fixed = self.getPointsAttribute('fixed', 0)
        self.masses = self.getPointsAttribute('mass', 0)

        # compute invmasses if masses is zero, set inv_masses to 0
        self.invmasses = np.zeros_like(self.masses)
        self.invmasses[self.masses!= 0] = 1.0 / self.masses[self.masses!= 0]
        self.invmasses[self.fixed!=0] = 0.0

        # compute the edges of the triangles with trimesh
        self.mesh_trimesh = trimesh.Trimesh(vertices=self.positions, faces=self.triangles.reshape(-1, 3))
        self.edges = self.mesh_trimesh.edges_unique
        self.edges_length = np.linalg.norm(self.positions[self.edges[:, 1]] - self.positions[self.edges[:, 0]], axis=1)

    def getPoints(self, simulation_time):
        return self.mesh.GetPointsAttr().Get(simulation_time)
    
    def getTriangles(self):
        return self.triangles
    
    def getFaceCount(self):
        return self.face_counts
    
    def getFaceIndices(self):
        return self.face_indices
    
    def getPointsAttribute(self, attribute_name, simulation_time):
        attr_name = 'primvars:' + attribute_name
        return self.mesh.GetPrim().GetAttribute(attr_name).Get(simulation_time)

    def _loadTriangles(self):
        # the usd file contains animation data, sample the first frame
        face_indices = self.mesh.GetFaceVertexIndicesAttr().Get(0)
        face_counts = self.mesh.GetFaceVertexCountsAttr().Get(0)

        # TODO: check these polygons and make sure they are triangles for cloth simulation
        # if there are quads, triangulate them

        # convert these data to triangles indcies and vertices
        triangles = []
        for i in range(len(face_counts)):
            for j in range(face_counts[i]):
                triangles.append(face_indices[i*3+j])
        triangles = np.array(triangles)
        return face_counts, face_indices, triangles

