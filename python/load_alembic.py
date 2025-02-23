import numpy as np
import os
import trimesh
import warp as wp

from pxr import Sdf, Usd, UsdGeom

# define resource dir and output dir, work dir is the current directory for python script file
work_dir = '.'

resource_dir = '{}/../../resources'.format(work_dir)
output_dir = '{}/../../output'.format(work_dir)


usd_file = '{}/projects/2025_sigasia_rtcloth/scenes/planar_patch/planar_patch.usd'.format(resource_dir)

print(usd_file)

# check whether the usd file exists
if not os.path.exists(usd_file):
    print('usd file not found')
    exit()


# load geometry of the usd file

stage = Usd.Stage.Open(usd_file)
root_layer = stage.GetRootLayer()

# get the mesh prim
print(root_layer.ExportToString())

# get 'usdexport1' prim
export_prim = stage.GetPrimAtPath('/usdexport1')

# print(export_prim.GetPath())

# traverse the export prim and get its children
for child in export_prim.GetAllChildren():
    if (child.GetTypeName() == 'Mesh'):
        # get the mesh data
        mesh = UsdGeom.Mesh(child)

        # the usd file contains animation data, sample the first frame
        points = mesh.GetPointsAttr().Get(0)
        face_indices = mesh.GetFaceVertexIndicesAttr().Get(0)
        face_counts = mesh.GetFaceVertexCountsAttr().Get(0)

        # sample other custom attributes named "fixed"
        fixed = mesh.GetPrim().GetAttribute('primvars:fixed').Get(0)

        # # print the mesh data
        # print(points)
        # print(face_indices)
        # print(face_counts)

# TODO: check these polygons and make sure they are triangles for cloth simulation
# if there are quads, triangulate them

# convert these data to triangles indcies and vertices
triangles = []
vertices = []    
for i in range(len(face_counts)):
    for j in range(face_counts[i]):
        triangles.append(face_indices[i*3+j])
        vertices.append(points[face_indices[i*3+j]])
        

# print the triangles and vertices
vertices = np.array(vertices)
triangles = np.array(triangles)
fixed = np.array(fixed)

# compute edges of the triangles and make sure they are not duplicated
# can use some commonly used geometry library to compute edges
# mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)


################################################################################
# start simulation code here
################################################################################

# convert numpy data to warp data and write computation for each frame

# convert numpy [N, 3] shape to warp array N x vec3
positions = wp.array(vertices, dtype=wp.vec3)
positions_proposed = wp.array(vertices, dtype=wp.vec3)
positions_init = wp.array(vertices, dtype=wp.vec3)
velocities = wp.zeros_like(positions)
fixed = wp.array(fixed, dtype=int)

gravity = wp.vec3(0.0, -9.8, 0.0)

simulation_time = 0.0

# write a warp launch function to compute the gravity integration

def updateGeometry(simulation_time, dt):
    global positions
    # get positions of vertices based on simulation time from mesh data
    vertices = mesh.GetPointsAttr().Get(simulation_time)
    positions = wp.array(np.array(vertices), dtype=wp.vec3)

@wp.kernel
def updatePositions(positions_proposed: wp.array(dtype=wp.vec3),
                    positions: wp.array(dtype=wp.vec3),
                    velocities: wp.array(dtype=wp.vec3),
                    fixed: wp.array(dtype=int),
                    gravity: wp.vec3,
                    dt: float):
    tid = wp.tid()

    if fixed[tid] == 0:
        # compute gravity force
        positions_proposed[tid] = positions[tid] + gravity * dt
    else:
        positions_proposed[tid] = positions[tid]


def update(dt):
    global simulation_time

    simulation_time += dt
    updateGeometry(simulation_time, dt)
    # launch wp kernel
    wp.launch(kernel=updatePositions, dim=len(positions), inputs=[positions_proposed, positions, velocities, fixed, gravity, dt], device=wp.get_preferred_device())

def dump(filepath):
    global positions
    # write data to usd file with triangles and positions
    stage = Usd.Stage.CreateNew(filepath)

    # add meta data for the usd file
    UsdGeom.Xform.Define(stage, '/root')
    # print(dir(stage))

    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(frames_num)
    stage.SetFramesPerSecond(FPS)
    stage.SetTimeCodesPerSecond(FPS)
    # set stage meter unit
    stage.SetMetadata('metersPerUnit', 1.0)
    stage.SetMetadata('upAxis', 'Y')

    # set the kind of root to component
    root_prim = stage.GetPrimAtPath('/root')
    # set the kind of root to component
    root_prim.SetKind("component")
    # root_prim.SetSpecifier(Sdf.SpecifierClass.Component)
    stage.SetDefaultPrim(root_prim)

    # create a root layer and add a mesh to that layer
    mesh_prim = UsdGeom.Mesh.Define(stage, '/root/mesh')

    # set the mesh data
    mesh_prim.GetPointsAttr().Set(positions.numpy())
    mesh_prim.GetFaceVertexIndicesAttr().Set(triangles)
    mesh_prim.GetFaceVertexCountsAttr().Set(np.array([3]*(len(triangles)//3)))

    # # set the fixed attribute
    # fixed_attr = mesh_prim.GetPrim().CreateAttribute('primvars:fixed', Sdf.ValueTypeNames.IntArray)
    # fixed_attr.Set(fixed.numpy())

    # save the stage
    stage.Save()


frames_num = 60
FPS = 60
substeps_num = 100
dt = 1.0/FPS/substeps_num

filepath = '{}/frame_{}.usd'.format(output_dir, 0)
dump(filepath)

for frame_index in range(frames_num):
    for _ in range(substeps_num):
        update(dt)

    filepath = '{}/frame_{}.usd'.format(output_dir, frame_index + 1)
    dump(filepath)
