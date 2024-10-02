import bpy
import bmesh
from src.material import rigid, utils
# import trimesh

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

def init_render():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # Create a new camera, with default location and rotation
    bpy.ops.object.camera_add(location=(0, 0, 0), rotation=(0, 0, 0))
    camera = bpy.context.object
    bpy.context.scene.camera = camera

    # Set background color
    bpy.context.scene.world.use_nodes = True
    bg_node = bpy.context.scene.world.node_tree.nodes['Background']
    bg_node.inputs['Color'].default_value = (0,0,0,1)  # RGBA values for dark gray background

    # Create a new light source
    bpy.ops.object.light_add(type='POINT', location=(5, -5, 5))
    light = bpy.context.object
    light.data.energy = 1000
    

def render_mesh(mesh, output_path):    
    # Set the mesh as the active object
    bpy.context.view_layer.objects.active = mesh
    mesh.select_set(True)
    
    # Set render settings and render the image
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    
    return output_path

def render_rigid_body(mesh, rigid_body: rigid.RigidBody, output_path):
    position = rigid_body.position.to_numpy()
    mesh.location = position
    # print(rigid_body.get_eular_angles())
    mesh.rotation_euler = utils.get_eular_angles(rigid_body.orientation.to_numpy())
    render_mesh(mesh, output_path)