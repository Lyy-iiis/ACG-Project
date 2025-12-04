import bpy
import os
from natsort import natsorted

def render_vdb_to_png(vdb_file, output_file, hdr_path):
    """
    Render a VDB file in Blender and save the result as a PNG.

    Parameters:
        vdb_file (str): Path to the VDB file.
        output_file (str): Path to save the rendered PNG.
        hdr_path (str): Path to the HDR file for the environment.
    """
    # Clear existing objects
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # -----------------------------
    # 1. Import VDB file
    # -----------------------------
    bpy.ops.object.volume_import(filepath=vdb_file)
    vdb_object = bpy.context.selected_objects[0]
    vdb_object.name = 'density'

    # Set VDB object transformation
    vdb_object.location = (-0.94, 0.58, 1.25)
    vdb_object.rotation_mode = 'XYZ'
    vdb_object.rotation_euler = (-1.5708, 0.0, -1.5708)  # -90° 转换为弧度
    vdb_object.scale = (0.076, 0.076, 0.076)

    # -----------------------------
    # 2. Setup Shader for VDB
    # -----------------------------
    # Create a new material
    mat = bpy.data.materials.new(name="DensityMaterial")
    mat.use_nodes = True
    mat.node_tree.nodes.clear()

    # Add Principled Volume shader
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    volume_node = nodes.new(type='ShaderNodeVolumePrincipled')

    # Set properties of Principled Volume
    volume_node.inputs['Color'].default_value = (0.742, 0.742, 0.742, 0.1)  # HSV 转 RGB 近似
    volume_node.inputs['Density'].default_value = 0.3

    # Connect nodes
    links.new(volume_node.outputs['Volume'], output_node.inputs['Volume'])

    # Assign material to VDB object
    if vdb_object.data.materials:
        vdb_object.data.materials[0] = mat
    else:
        vdb_object.data.materials.append(mat)

    # -----------------------------
    # 3. Setup Camera
    # -----------------------------
    cam = bpy.data.cameras.new("Camera")
    cam_obj = bpy.data.objects.new("Camera", cam)
    bpy.context.scene.collection.objects.link(cam_obj)

    cam_obj.location = (13.71, -11.7, 4.958)
    cam_obj.rotation_mode = 'XYZ'
    cam_obj.rotation_euler = (1.243, 0.0, 0.874)  # 71.278°, 0°, 50.12° 转换为弧度
    cam_obj.scale = (1, 1, 1)

    bpy.context.scene.camera = cam_obj

    # -----------------------------
    # 4. Setup Environment HDR
    # -----------------------------
    if bpy.context.scene.world is None:
        bpy.context.scene.world = bpy.data.worlds.new("World")

    bpy.context.scene.world.use_nodes = True
    world_nodes = bpy.context.scene.world.node_tree.nodes
    world_links = bpy.context.scene.world.node_tree.links

    # Clear default nodes
    for node in world_nodes:
        world_nodes.remove(node)

    # Add Environment Texture node
    env_node = world_nodes.new(type='ShaderNodeTexEnvironment')
    env_node.image = bpy.data.images.load(hdr_path)

    bg_node = world_nodes.new(type='ShaderNodeBackground')
    output_node = world_nodes.new(type='ShaderNodeOutputWorld')

    world_links.new(env_node.outputs['Color'], bg_node.inputs['Color'])
    world_links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])

    # -----------------------------
    # 5. Setup Render Settings
    # -----------------------------
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    scene.cycles.samples = 256

    # Output settings
    scene.render.filepath = output_file
    scene.render.image_settings.file_format = 'PNG'

    # -----------------------------
    # 6. Render the Scene
    # -----------------------------
    bpy.ops.render.render(write_still=True)
    print(f"Rendered {vdb_file} -> {output_file}")

if __name__ == "__main__":
    # Input and output directories
    vdb_dir = "/root/autodl-tmp/Visual-Simulation-of-Smoke/output/vdb"  # 输入VDB文件夹路径
    png_dir = "/root/autodl-tmp/Visual-Simulation-of-Smoke/output/png"  # 输出PNG文件夹路径
    hdr_path = "/root/autodl-tmp/background.hdr"  # 环境HDR文件路径

    # Ensure output directory exists
    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    # Process each VDB file in sequence
    vdb_files = natsorted([f for f in os.listdir(vdb_dir) if f.endswith('.vdb')])

    if not vdb_files:
        print("No VDB files found in the directory.")
    else:
        for vdb_file in vdb_files:
            vdb_path = os.path.join(vdb_dir, vdb_file)
            png_file = os.path.join(png_dir, f"{os.path.splitext(vdb_file)[0]}.png")
            render_vdb_to_png(vdb_path, png_file, hdr_path)
            