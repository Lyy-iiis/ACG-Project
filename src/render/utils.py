import bpy
import bmesh
import taichi as ti
import numpy as np
import math
import openvdb as vdb
import os

def csv_to_openvdb(csv_file, vdb_file, Nx, Ny, Nz):
    """
    Convert a CSV file to an OpenVDB file.

    Parameters:
        csv_file (str): Path to the input CSV file.
        vdb_file (str): Path to the output VDB file.
        Nx (int): Number of grid points in the x-direction.
        Ny (int): Number of grid points in the y-direction.
        Nz (int): Number of grid points in the z-direction.
    """
    # Read CSV file
    density_np = np.loadtxt(csv_file, delimiter=',')
    
    # Reshape to 3D array
    density_np = density_np.reshape((Nz, Ny, Nx))
    
    grid = vdb.FloatGrid()
    accessor = grid.getAccessor()
    grid.name = "density"
    
    for i in range(density_np.shape[0]):
        for j in range(density_np.shape[1]):
            for k in range(density_np.shape[2]):
                accessor.setValueOn((i, j, k), float(density_np[i, j, k]))
    
    # Write to VDB file
    vdb.write(vdb_file, grids=[grid])
    print(f"Converted {csv_file} -> {vdb_file}")

def batch_csv_to_vdb(csv_dir, vdb_dir, Nx, Ny, Nz):
    """
    Batch process all CSV files in a directory and convert them to VDB files.

    Parameters:
        csv_dir (str): Path to the directory containing CSV files.
        vdb_dir (str): Path to the directory to save VDB files.
        Nx (int): Number of grid points in the x-direction.
        Ny (int): Number of grid points in the y-direction.
        Nz (int): Number of grid points in the z-direction.
    """
    # Ensure the output directory exists
    if not os.path.exists(vdb_dir):
        os.makedirs(vdb_dir)
    
    # Process each CSV file in sequence
    csv_files = sorted([f for f in os.listdir(csv_dir) if f.endswith('.csv')])
    
    if not csv_files:
        print("No CSV files found in the directory.")
        return
    
    for csv_file in csv_files:
        csv_path = os.path.join(csv_dir, csv_file)
        vdb_file = os.path.join(vdb_dir, f"{os.path.splitext(csv_file)[0]}.vdb")
        csv_to_openvdb(csv_path, vdb_file, Nx, Ny, Nz)

if __name__ == "__main__":
    # Input and output directories
    csv_dir = "/root/autodl-tmp/Visual-Simulation-of-Smoke/output/csv"  # 输入CSV文件夹路径
    vdb_dir = "/root/autodl-tmp/Visual-Simulation-of-Smoke/output/vdb"  # 输出VDB文件夹路径
    
    # Grid size (modify based on your data)
    Nx, Ny, Nz = 32, 64, 32
    
    batch_csv_to_vdb(csv_dir, vdb_dir, Nx, Ny, Nz)


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