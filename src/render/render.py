import bpy

from src.material import rigid, utils
# import trimesh

class Render:
    def __init__(self, camera_location=(0,0,0), 
                camera_rotation=(0,0,0),
                bg_color=(0,0,0,1), 
                light_location=(5,-5,5), 
                light_energy=1000):
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        
        # Create a new camera, with default location and rotation
        bpy.ops.object.camera_add(location=camera_location, rotation=camera_rotation)
        camera = bpy.context.object
        bpy.context.scene.camera = camera

        # Set background color
        bpy.context.scene.world.use_nodes = True
        bg_node = bpy.context.scene.world.node_tree.nodes['Background']
        bg_node.inputs['Color'].default_value = bg_color  # RGBA values for dark gray background

        # Create a new light source
        bpy.ops.object.light_add(type='POINT', location=light_location)
        light = bpy.context.object
        light.data.energy = light_energy


    def render_mesh(self, mesh_list:list, output_path):    
        # Set the mesh as the active object
        for mesh in mesh_list:
            bpy.context.view_layer.objects.active = mesh
            mesh.select_set(True)
        
        # Set render settings and render the image
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True)
        
        return output_path

    def render_rigid_body(self, mesh: list, rigid_body: list[rigid.RigidBody], output_path):
        for i in range(len(mesh)):
            position = rigid_body[i].position.to_numpy()
            mesh[i].location = position
            # print(rigid_body.get_eular_angles())
            mesh[i].rotation_euler = utils.get_eular_angles(rigid_body[i].orientation.to_numpy())
        self.render_mesh(mesh, output_path)