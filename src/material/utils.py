import taichi as ti
import trimesh

def get_rigid_from_mesh(filename):
    mesh = trimesh.load_mesh(filename)
    return mesh

def get_eular_angles(orientation):
    R = orientation
    sy = ti.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = ti.atan2(R[2, 1], R[2, 2])
        y = ti.atan2(-R[2, 0], sy)
        z = ti.atan2(R[1, 0], R[0, 0])
    else:
        x = ti.atan2(-R[1, 2], R[1, 1])
        y = ti.atan2(-R[2, 0], sy)
        z = 0
    # print(np.linalg.det(R.to_numpy()))
    eular_angles = ti.Vector([x, y, z])
    return eular_angles[None][0]

def mesh(vertices, faces):
    mesh = trimesh.Trimesh()
    mesh.vertices = vertices.to_numpy()
    mesh.faces = faces.to_numpy()
    return mesh