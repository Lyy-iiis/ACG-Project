import bpy, mathutils

from src.material.container import base_container
from src.material.fluid import basefluid
from src.material import rigid, cloth
import trimesh
import src.render.utils
from src.render import utils
import src.material.utils
import taichi as ti
import numpy as np
import math

class Render:
    def __init__(self, camera_location=(0, 0, 0), 
                camera_rotation=(0, 0, 0),
                bg_color=(0, 0, 0, 1), 
                light_location=(0, 5, -6), 
                light_energy=2000):
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        
        # Create a new camera, with default location and rotation
        # direction = mathutils.Vector((0, 0, -6)) - camera.location
        # rot_quat = direction.to_track_quat('Z', 'Y')
        # rotation_euler = rot_quat.to_euler()
        bpy.ops.object.camera_add(align='WORLD', location=camera_location, rotation=camera_rotation)
        camera = bpy.context.object
        # self.look_at(camera, mathutils.Vector((0, 0, -6)))
        bpy.context.scene.camera = camera
        
        # Set background color
        bpy.context.scene.world.use_nodes = True
        world = bpy.context.scene.world
        node_tree = world.node_tree
        nodes = node_tree.nodes
        links = node_tree.links

        # Clear default nodes
        for node in nodes:
            nodes.remove(node)

        # Add Background node
        bg_node = nodes.new(type='ShaderNodeBackground')
        bg_node.location = (200, 0)
        bg_node.inputs['Color'].default_value = bg_color

        # Add Environment Texture node
        env_texture_node = nodes.new(type='ShaderNodeTexEnvironment')
        env_texture_node.location = (-200, 0)
        env_texture_node.image = bpy.data.images.load('assets/background.hdr')

        ## Add Mapping node to rotate the environment texture
        # mapping_node = nodes.new(type='ShaderNodeMapping')
        # mapping_node.location = (-400, 0)
        # mapping_node.inputs['Rotation'].default_value[0] = math.radians(90)  # Rotate 90 degrees around Y axis
        
        # Add Output node
        output_node = nodes.new(type='ShaderNodeOutputWorld')
        output_node.location = (400, 0)
        
        # Link nodes
        # links.new(mapping_node.outputs['Vector'], env_texture_node.inputs['Vector'])
        links.new(env_texture_node.outputs['Color'], bg_node.inputs['Color'])
        links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])
        # bg_node = bpy.context.scene.world.node_tree.nodes['Background']
        # bg_node.inputs['Color'].default_value = bg_color  # RGBA values for dark gray background

        # Create a new light source
        bpy.ops.object.light_add(type='POINT', location=light_location)
        light = bpy.context.object
        light.data.energy = light_energy
        
        self.fluid_mesh = []
        
    def look_at(self, obj_camera, point):
        loc_camera = obj_camera.matrix_world.to_translation()

        direction = point - loc_camera
        # point the cameras '-Z' and use its 'Y' as up
        rot_quat = direction.to_track_quat('-Z', 'Y')

        # assume we're using euler rotation
        obj_camera.rotation_euler = rot_quat.to_euler()
        
    def render_mesh(self, mesh_list: list, output_path):    
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
        
    # def add_fluid(self, fluid: fluid.Fluid):
    #     positions = fluid.positions.to_numpy()  
    #     # print(f"Fluid positions: {positions}")
    #     # assert False
    #     for i in range(positions.shape[0]):
    #         self.fluid_mesh.append(utils.trimesh_to_blender_object(trimesh.creation.icosphere(radius=0.02, center=positions[i]),
    #             object_name=f"Fluid_{i}"))
        
    # def render_fluid(self, fluid: fluid.Fluid, output_path):
    #     for i in range(len(self.fluid_mesh)):
    #         # print(f"Fluid position: {fluid.positions[i]}")
    #         self.fluid_mesh[i].location = fluid.positions[i]
    #     self.render_mesh(self.fluid_mesh, output_path)
        
    def render_fluid(self, mesh, output_path):
        bpy.ops.object.select_all(action='DESELECT')
        for obj in bpy.data.objects:
            if obj.type == 'MESH' and obj != mesh:
                obj.select_set(True)
        bpy.ops.object.delete()
        
        fluid_material = self.get_material("Water", "assets/water.blend")
        
        fluid_mesh = utils.trimesh_to_blender_object(mesh, object_name="Fluid")
        if fluid_mesh.data.materials:
            fluid_mesh.data.materials[0] = fluid_material
        else:
            fluid_mesh.data.materials.append(fluid_material)
        
        self.render_mesh([fluid_mesh], output_path)
        
    def render_coupled_fluid_rigid(self, fluid_mesh, rigid_mesh, container_mesh, output_path):
        bpy.ops.object.select_all(action='DESELECT')
        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                obj.select_set(True)
        bpy.ops.object.delete()

        fluid_material = self.get_material("Water", "assets/water.blend")
        
        fluid_mesh = utils.trimesh_to_blender_object(fluid_mesh, object_name="Fluid")
        if fluid_mesh.data.materials:
            fluid_mesh.data.materials[0] = fluid_material
        else:
            fluid_mesh.data.materials.append(fluid_material)

        rigid_material = self.get_material('Realistic procedural gold', 'assets/rigid.blend')
        rigid_mesh = utils.trimesh_to_blender_object(rigid_mesh, object_name="Rigid")
        if rigid_mesh.data.materials:
            rigid_mesh.data.materials[0] = rigid_material
        else:
            rigid_mesh.data.materials.append(rigid_material)
        
        # container_mesh = self.add_container(container_mesh)
        self.render_mesh([fluid_mesh, rigid_mesh], output_path)
        
    def add_container(self, container_mesh):
        glass_material = self.get_material("Scratched Glass (Procedural)", "assets/glass.blend")
        
        container_mesh = utils.trimesh_to_blender_object(container_mesh, object_name="Container")
        if container_mesh.data.materials:
            container_mesh.data.materials[0] = glass_material
        else:
            container_mesh.data.materials.append(glass_material)
        return container_mesh

    def render_cloth(self, mesh: bpy.types.Object, output_path):
        # Clear all existing meshes before rendering
        bpy.ops.object.select_all(action='DESELECT')
        for obj in bpy.data.objects:
            if obj.type == 'MESH' and obj != mesh:
                obj.select_set(True)
        bpy.ops.object.delete()
        
        # Load the cloth material
        # material = self.get_material("Felt kvadrat 0967", "assets/cloth.blend")
        material = self.get_material("Realistic patterned fabric", "assets/cloth3.blend")
        if mesh.data.materials:
            mesh.data.materials[0] = material
        else:
            mesh.data.materials.append(material)

        # Ensure the mesh is the active object in the scene
        bpy.context.view_layer.objects.active = mesh
        
        # Set render parameters (like resolution, camera, etc.)
        bpy.context.scene.render.filepath = output_path  # Set the output path
        bpy.context.scene.render.engine = 'CYCLES'  # Use Cycles engine for rendering

        # Set the output image format
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        
        # Render the current frame to the given file path
        bpy.ops.render.render(write_still=True)
        
    def render_coupled_cloth_fixed_rigid(self, cloth_mesh, output_path):
        # Clear all existing meshes before rendering
        bpy.ops.object.select_all(action='DESELECT')
        for obj in bpy.data.objects:
            if obj.type == 'MESH' and obj != cloth_mesh:
                obj.select_set(True)
        bpy.ops.object.delete()

        ## Add a rigid body sphere object to the scene
        # Create a new sphere
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=0.1, 
            location=(0, 0, -2)
        )
        sphere = bpy.context.object  # Get the newly added sphere
        
        # Add rigid body physics to the sphere
        bpy.ops.rigidbody.object_add()
        sphere.rigid_body.type = 'PASSIVE'  # Set the sphere as a passive rigid body
        
        # Load the cloth material
        material = self.get_material("Satin Fabric", "assets/cloth6.blend")
        # material = self.get_material("Realistic patterned fabric", "assets/cloth4.blend")
        if cloth_mesh.data.materials:
            cloth_mesh.data.materials[0] = material
        else:
            cloth_mesh.data.materials.append(material)
        
        ## Assign material to the rigid body sphere
        # Load the material
        rigid_material = self.get_material('Realistic procedural gold', 'assets/rigid.blend')
        # Assign the material to the sphere
        if sphere.data.materials:
            sphere.data.materials[0] = rigid_material
        else:
            sphere.data.materials.append(rigid_material)
        
        # # Rotate all meshes around the x-axis by 90 degrees
        # for obj in bpy.data.objects:
        #     if obj.type == 'MESH':
        #         obj.rotation_euler[0] += 1.5708  # 90 degrees in radians

        # Ensure the mesh is the active object in the scene
        bpy.context.view_layer.objects.active = cloth_mesh
        
        # Smooth shading
        bpy.ops.object.shade_smooth()

        # Set render parameters (like resolution, camera, etc.)
        bpy.context.scene.render.filepath = output_path  # Set the output path
        bpy.context.scene.render.engine = 'CYCLES'  # Use Cycles engine for rendering

        # Set the output image format
        bpy.context.scene.render.image_settings.file_format = 'PNG'

        # Render the current frame to the given file path
        bpy.ops.render.render(write_still=True)
        
    def render_coupled_cloth_rigid(self, cloth_mesh, center, radius, output_path):
        # Clear all existing meshes before rendering
        bpy.ops.object.select_all(action='DESELECT')
        for obj in bpy.data.objects:
            if obj.type == 'MESH' and obj != cloth_mesh:
                obj.select_set(True)
        bpy.ops.object.delete()

        ## Add a rigid body sphere object to the scene
        # Create a new sphere
        radius_float = float(radius[None])
        radius_float *= 0.9
        center_tuple = (center[None][0], center[None][1], center[None][2])
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=radius_float, 
            location=center_tuple
        )
        sphere = bpy.context.object  # Get the newly added sphere
        
        # Add rigid body physics to the sphere
        bpy.ops.rigidbody.object_add()
        sphere.rigid_body.type = 'PASSIVE'  # Set the sphere as a passive rigid body

        # Load the cloth material
        material = self.get_material("Satin Fabric", "assets/cloth6.blend")
        # material = self.get_material("Realistic patterned fabric", "assets/cloth4.blend")
        if cloth_mesh.data.materials:
            cloth_mesh.data.materials[0] = material
        else:
            cloth_mesh.data.materials.append(material)
        
        ## Assign material to the rigid body sphere
        # Load the material
        rigid_material = self.get_material('Realistic procedural gold', 'assets/rigid.blend')
        # Assign the material to the sphere
        if sphere.data.materials:
            sphere.data.materials[0] = rigid_material
        else:
            sphere.data.materials.append(rigid_material)
        
        # Ensure the mesh is the active object in the scene
        bpy.context.view_layer.objects.active = cloth_mesh
        
        # Smooth shading
        bpy.ops.object.shade_smooth()

        # Set render parameters (like resolution, camera, etc.)
        bpy.context.scene.render.filepath = output_path  # Set the output path
        bpy.context.scene.render.engine = 'CYCLES'  # Use Cycles engine for rendering

        # Set the output image format
        bpy.context.scene.render.image_settings.file_format = 'PNG'

        # Render the current frame to the given file path
        bpy.ops.render.render(write_still=True)
        
        
        
    def get_material(self, material_name, file_name):
        # Load the .blend file
        blend_file_path = file_name
        with bpy.data.libraries.load(blend_file_path, link=False) as (data_from, data_to):
            print(data_from.materials)
            if material_name in data_from.materials:
                data_to.materials = [material_name]
            else:
                raise ValueError(f"Material {material_name} not found in {blend_file_path}")

        # Return the material
        return data_to.materials[0]