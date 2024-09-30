import bpy
import bmesh
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

    obj = bpy.data.objects.new(object_name, mesh)
    bpy.context.collection.objects.link(obj)

    return obj

def init_render():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # Create a new camera
    bpy.ops.object.camera_add(location=(0, -3, 2), rotation=(1.1, 0, 0))
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
    
    mesh_info = {
        "name": mesh.name,
        "vertices": len(mesh.data.vertices),
        "edges": len(mesh.data.edges),
        "faces": len(mesh.data.polygons)
    }
    print(f"Mesh Info: {mesh_info}")
    bpy.ops.object.select_all(action='DESELECT')
    
    # Set the mesh as the active object
    bpy.context.view_layer.objects.active = mesh
    mesh.select_set(True)

    # Set the origin to geometry and apply transformations
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    print(f"Active object before rendering: {bpy.context.view_layer.objects.active.name}")
    
    # Set render settings and render the image
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    
    return output_path