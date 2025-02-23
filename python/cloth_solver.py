import numpy as np
import warp as wp

from pxr import Usd, UsdGeom, Sdf

@wp.kernel
def updatePositions(positions_proposed: wp.array(dtype=wp.vec3),
                    positions: wp.array(dtype=wp.vec3),
                    positions_animated: wp.array(dtype=wp.vec3),
                    velocities: wp.array(dtype=wp.vec3),
                    fixed: wp.array(dtype=int),
                    gravity: wp.vec3,
                    dt: float):
    tid = wp.tid()

    if fixed[tid] == 0:
        positions_proposed[tid] = positions[tid] + (velocities[tid] + gravity * dt) * dt
    else:
        positions_proposed[tid] = positions_animated[tid]
        positions[tid] = positions_animated[tid]

@wp.kernel
def projectConstraints(positions: wp.array(dtype=wp.vec3),
                       positions_proposed: wp.array(dtype=wp.vec3),
                       invmasses: wp.array(dtype=wp.float32),
                       constraints_indices: wp.array(dtype=wp.vec2i),
                       constraints_restlength: wp.array(dtype=wp.float32),
                       constraints_stiffness: wp.array(dtype=wp.float32),
                       dt: float,
                       delta_positions: wp.array(dtype=wp.vec3)):
    tid = wp.tid()

    # get indices of constraints for this particle
    i0 = constraints_indices[tid].x
    i1 = constraints_indices[tid].y

    k = constraints_stiffness[tid]
    alpha = 1.0 / (k * dt * dt)

    invm0 = invmasses[i0]
    invm1 = invmasses[i1]

    p0 = positions_proposed[i0]
    p1 = positions_proposed[i1]
    p0_last = positions[i0]
    p1_last = positions[i1]

    l0 = constraints_restlength[tid]
    l = wp.length(p0 - p1)
    C = l - l0
    n = (p0 - p1) / l
    dlambda = -0.25 * C / (invm0 + invm1 + alpha)

    wp.atomic_add(delta_positions, i0,  n * invm0 * dlambda)
    wp.atomic_add(delta_positions, i1, -n * invm1 * dlambda)

@wp.kernel
def updateVelocities(delta_positions: wp.array(dtype=wp.vec3),
                     velocities: wp.array(dtype=wp.vec3),
                     positions_proposed: wp.array(dtype=wp.vec3),
                     positions: wp.array(dtype=wp.vec3),
                     fixed: wp.array(dtype=int),
                     dt: float):
    tid = wp.tid()

    if fixed[tid] == 0:
        positions_proposed[tid] = positions_proposed[tid] + delta_positions[tid]
        delta_positions[tid] = wp.vec3(0.0, 0.0, 0.0)
        velocities[tid] = (positions_proposed[tid] - positions[tid]) / dt
        positions[tid] = positions_proposed[tid]

class ClothSolver:

    def __init__(self, geometry, fps):
        self.geometry = geometry    

        # points attributes
        self.positions = wp.array(geometry.positions, dtype=wp.vec3)
        self.positions_proposed = wp.array(self.positions.numpy(), dtype=wp.vec3)
        self.positions_animated = wp.array(self.positions.numpy(), dtype=wp.vec3)
        self.positions_init = wp.array(self.positions.numpy(), dtype=wp.vec3)
        self.velocities = wp.zeros_like(self.positions)
        self.fixed = wp.array(geometry.fixed, dtype=int)
        self.invmasses = wp.array(geometry.invmasses, dtype=wp.float32)

        self.delta_positions = wp.zeros_like(self.positions)

        # primitives attributes
        self.constraints_indices = wp.array(geometry.edges, dtype=wp.vec2i)
        self.constraints_restlength = wp.array(geometry.edges_length, dtype=wp.float32)
        self.constraints_stiffness = wp.array(np.ones(len(geometry.edges))*1e10, dtype=wp.float32)

        self.fps = fps
        self.gravity = wp.vec3(0.0, -9.8065, 0.0)
        self.simulation_time = 0.0

    def updateGeometry(self, simulation_time, dt):
        # get positions of vertices based on simulation time from mesh data
        vertices = self.geometry.mesh.GetPointsAttr().Get(simulation_time * self.fps)
        self.positions_animated = wp.array(np.array(vertices), dtype=wp.vec3)

    def update(self, dt):
        self.simulation_time += dt
        self.updateGeometry(self.simulation_time, dt)
        # launch wp kernel
        wp.launch(kernel=updatePositions, dim=len(self.positions),
                  inputs=[
                      self.positions_proposed,
                      self.positions,
                      self.positions_animated,
                      self.velocities,
                      self.fixed,
                      self.gravity,
                      dt],
                  device=wp.get_preferred_device())
        
        wp.launch(kernel=projectConstraints, dim=len(self.constraints_indices),
                  inputs=[
                      self.positions,
                      self.positions_proposed,
                      self.invmasses,
                      self.constraints_indices,
                      self.constraints_restlength,
                      self.constraints_stiffness,
                      dt,
                      self.delta_positions],
                  device=wp.get_preferred_device())
        
        wp.launch(kernel=updateVelocities, dim=len(self.positions),
                  inputs=[
                      self.delta_positions,
                      self.velocities,
                      self.positions_proposed,
                      self.positions,
                      self.fixed,
                      dt],
                  device=wp.get_preferred_device())
        
    def dump(self, filepath):
        # write data to usd file with triangles and positions
        stage = Usd.Stage.CreateNew(filepath)

        # add meta data for the usd file
        UsdGeom.Xform.Define(stage, '/root')

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
        mesh_prim.GetPointsAttr().Set(self.positions.numpy())
        mesh_prim.GetFaceVertexIndicesAttr().Set(self.geometry.triangles)
        mesh_prim.GetFaceVertexCountsAttr().Set(np.array([3]*(len(self.geometry.triangles)//3)))

        # set the custom attribute for points with name "fixed"
        fixed_attr = mesh_prim.GetPrim().CreateAttribute("primvars:fixed",
                                                         Sdf.ValueTypeNames.IntArray)

        fixed_attr.SetMetadata("interpolation", "vertex")
        fixed_attr.Set(self.fixed.numpy())

        # save the stage
        stage.Save()
