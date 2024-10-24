import bpy

from src.material import rigid, fluid, cloth, container
import trimesh
import src.render.utils
from src.render import utils
import src.material.utils
import taichi as ti
import numpy as np

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
        
        self.fluid_mesh = []
        
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
        
    def add_fluid(self, fluid: fluid.Fluid):
        positions = fluid.positions.to_numpy()  
        # print(f"Fluid positions: {positions}")
        # assert False
        for i in range(positions.shape[0]):
            self.fluid_mesh.append(utils.trimesh_to_blender_object(trimesh.creation.icosphere(radius=0.02, center=positions[i]),
                object_name=f"Fluid_{i}"))
        
    def render_fluid(self, fluid: fluid.Fluid, output_path):
        for i in range(len(self.fluid_mesh)):
            # print(f"Fluid position: {fluid.positions[i]}")
            self.fluid_mesh[i].location = fluid.positions[i]
        self.render_mesh(self.fluid_mesh, output_path)
        
    def render_fluid_mesh(self, mesh, output_path):
        bpy.ops.object.select_all(action='DESELECT')
        for obj in bpy.data.objects:
            if obj.type == 'MESH' and obj != mesh:
                obj.select_set(True)
        bpy.ops.object.delete()
        mesh = utils.trimesh_to_blender_object(mesh, object_name="Fluid")
        self.render_mesh([mesh], output_path)
        
        
    def add_container(self, container: container.Container):
        assert False, "Not implemented"
        glass_material = bpy.data.materials.new(name="GlassMaterial")
        glass_material.use_nodes = True
        bsdf = glass_material.node_tree.nodes["Principled BSDF"]
        # bsdf.inputs['Base Color'].default_value = (1, 1, 1, 1)
        bsdf.inputs["Transmission"].default_value = 1
        bsdf.inputs['Roughness'].default_value = 0
        bsdf.inputs['IOR'].default_value = 1
        
        # Create a mesh for the container
        bpy.ops.mesh.primitive_cube_add(size=2, location=container.offset.to_numpy())
        container_mesh = bpy.context.object
        container_mesh.scale = (container.width, container.height, container.depth)
        
        # Assign the glass material to the container mesh
        if container_mesh.data.materials:
            container_mesh.data.materials[0] = glass_material
        else:
            container_mesh.data.materials.append(glass_material)
        
        # Set the container mesh as a rigid body with passive type
        bpy.ops.rigidbody.object_add()
        container_mesh.rigid_body.type = 'PASSIVE'
        container_mesh.rigid_body.collision_shape = 'MESH'

    def render_cloth1(self, mesh: bpy.types.Object, output_path):
        # # Render the current frame with the cloth mesh
        # self.render_mesh(mesh, output_path)
        # # Clear all existing meshes before rendering
        # bpy.ops.object.select_all(action='DESELECT')
        # bpy.ops.object.select_by_type(type='MESH')
        # bpy.ops.object.delete()
        
        # Clear all existing meshes before rendering
        bpy.ops.object.select_all(action='DESELECT')
        for obj in bpy.data.objects:
            if obj.type == 'MESH' and obj != mesh:
                obj.select_set(True)
        bpy.ops.object.delete()

        # Ensure the mesh is the active object in the scene
        bpy.context.view_layer.objects.active = mesh
        
        # Set render parameters (like resolution, camera, etc.)
        bpy.context.scene.render.filepath = output_path  # Set the output path
        bpy.context.scene.render.engine = 'CYCLES'  # Use Cycles engine for rendering

        # Set the output image format
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        
        # Render the current frame to the given file path
        bpy.ops.render.render(write_still=True)