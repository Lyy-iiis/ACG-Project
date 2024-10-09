import taichi as ti
import trimesh

@ti.data_oriented
class Mesh:
    def __init__(self, vertices, faces):
        self.vertices = ti.Vector.field(3, dtype=ti.f32, shape=len(vertices))
        self.faces = ti.Vector.field(3, dtype=ti.i32, shape=len(faces))
        
        for i in range(len(vertices)):
            self.vertices[i] = vertices[i]
        
        for i in range(len(faces)):
            self.faces[i] = faces[i]

def get_rigid_from_mesh(filename):
    mesh = trimesh.load_mesh(filename)
    return mesh

def mesh(vertices, faces):
    mesh = trimesh.Trimesh()
    mesh.vertices = vertices.to_numpy()
    mesh.faces = faces.to_numpy()
    return mesh

# @ti.func
# def sample_points(vertices, faces, num_points):
#     volume = []
#     for i in range(len(faces)):
#         volume.append(ti.abs(ti.math.dot(vertices[faces[i][0]], ti.math.cross(vertices[faces[i][1]], vertices[faces[i][2]])) / 6))
#     volume = ti.Vector(volume)
#     points = ti.Vector.field(3, dtype=ti.f32, shape=num_points)
#     face_idx = ti.field(dtype=ti.i32, shape=num_points)
#     cumulative_volume = ti.field(dtype=ti.f32, shape=len(faces))
    
#     cumulative_volume[0] = volume[0]
#     for i in range(1, len(faces)):
#         cumulative_volume[i] = cumulative_volume[i - 1] + volume[i]
    
#     total_volume = cumulative_volume[len(faces) - 1]
    
#     for i in range(num_points):
#         rand_volume = ti.random() * total_volume
#         for j in range(len(faces)):
#             if rand_volume <= cumulative_volume[j]:
#                 face_idx[i] = j
#                 break
#     for i in range(num_points):
#         s, t, u = ti.random(), ti.random(), ti.random()
#         if s + t + u > 1:
#             if t + u > 1:
#                 s, t, u = s, 1 - u, 1 - s - t
#             else:
#                 s, t, u = 1 - t - u, t, s + t + u - 1
#         points[i] = vertices[faces[face_idx[i]][0]] * s + vertices[faces[face_idx[i]][1]] * t + vertices[faces[face_idx[i]][2]] * u
#     return points