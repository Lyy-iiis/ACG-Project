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

    # Enable double-sided rendering
    if not obj.data.materials:
        mat = bpy.data.materials.new(name="DoubleSidedMaterial")
        obj.data.materials.append(mat)
    else:
        mat = obj.data.materials[0]
    
    # Disable backface culling to enable rendering on both sides
    mat.use_backface_culling = False  
    
    # # Ensure that the object uses nodes for more advanced material control
    # mat.use_nodes = True
    # nodes = mat.node_tree.nodes

    # # Add a Geometry node to control the normal direction
    # if "Geometry" not in nodes:
    #     geometry_node = nodes.new(type="ShaderNodeNewGeometry")
    # else:
    #     geometry_node = nodes["Geometry"]

    # # Use a Mix Shader to mix front and back normals
    # if "Mix Shader" not in nodes:
    #     mix_shader = nodes.new(type="ShaderNodeMixShader")
    # else:
    #     mix_shader = nodes["Mix Shader"]

    # # Link nodes for correct double-sided rendering
    # if "Principled BSDF" in nodes:
    #     bsdf = nodes["Principled BSDF"]
    # else:
    #     bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")

    # # Ensure correct node links for double-sided rendering
    # mat.node_tree.links.new(geometry_node.outputs['Backfacing'], mix_shader.inputs[0])
    # mat.node_tree.links.new(bsdf.outputs['BSDF'], mix_shader.inputs[1])
    # mat.node_tree.links.new(bsdf.outputs['BSDF'], mix_shader.inputs[2])

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