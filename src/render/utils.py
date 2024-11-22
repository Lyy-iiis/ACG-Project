import bpy
import bmesh
import taichi as ti
import numpy as np
import math

def trimesh_to_blender_object(trimesh_obj, object_name="Bunny"):
    mesh = bpy.data.meshes.new(object_name)
    bm = bmesh.new()

    # Add vertices
    for vertex in trimesh_obj.vertices:
        bm.verts.new(vertex)
    bm.verts.ensure_lookup_table()

    # Add faces
    existing_faces = set()
    for face in trimesh_obj.faces:
        face_tuple = tuple(sorted(face))
        if face_tuple in existing_faces:
            # Skip faces that already exist
            continue
        try:
            bm.faces.new([bm.verts[i] for i in face])
            existing_faces.add(face_tuple)
        except ValueError:
            # Skip invalid faces
            continue
    # for face in trimesh_obj.faces:
    #     bm.faces.new([bm.verts[i] for i in face])
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

    
def calculate_camera_rotation(camera_position, look_at):
    """
    Calculate the Euler angles (in degrees) for a camera located at `camera_position`,
    looking at a point specified by `look_at`.

    Parameters:
        camera_position (list or np.array): The position of the camera in 3D space [x, y, z].
        look_at (list or np.array): The target point in 3D space the camera is looking at [x, y, z].

    Returns:
        tuple: Euler angles (yaw, pitch, roll) in degrees.
    """
    # Convert inputs to numpy arrays
    camera_position = np.array(camera_position)
    look_at = np.array(look_at)
    
    # Compute the direction vector
    direction = look_at - camera_position
    direction = direction / np.linalg.norm(direction)  # Normalize the vector

    # Calculate pitch (rotation around x-axis)
    pitch = math.asin(direction[1])  # Positive y-axis is vertical

    # Calculate yaw (rotation around y-axis)
    yaw = math.atan2(direction[0], -direction[2])  # Relative to the negative z-axis

    # Roll (rotation around z-axis) is assumed to be 0
    roll = 0.0

    # Convert radians to degrees
    pitch_deg = math.degrees(pitch)
    yaw_deg = math.degrees(yaw)
    roll_deg = math.degrees(roll)

    return (pitch_deg, -yaw_deg, roll_deg)