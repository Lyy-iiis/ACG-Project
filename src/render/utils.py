import bpy
import bmesh
import taichi as ti

def trimesh_to_blender_object(trimesh_obj, object_name="Bunny"):
    mesh = bpy.data.meshes.new(object_name)
    bm = bmesh.new()

    # Add vertices
    for vertex in trimesh_obj.vertices:
        bm.verts.new(vertex)
    bm.verts.ensure_lookup_table()

    # Add faces
    for face in trimesh_obj.faces:
        bm.faces.new([bm.verts[i] for i in face])
    bm.faces.ensure_lookup_table()

    bm.to_mesh(mesh)
    bm.free()

    # Create object
    obj = bpy.data.objects.new(object_name, mesh)
    bpy.context.collection.objects.link(obj)
        
    mesh_info = {
        "name": obj.name,
        "vertices": len(obj.data.vertices),
        "edges": len(obj.data.edges),
        "faces": len(obj.data.polygons)
    }
    print(f"Mesh Info: {mesh_info}")

    return obj


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