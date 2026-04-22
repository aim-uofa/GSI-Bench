import bpy
import json
import os
import time
import argparse
import sys
import math
from mathutils import Quaternion, Vector, Matrix

# Global constant definitions
PLACEMENT_HEIGHT = 1.5  # Placement area height
RESTRICTED_HEIGHT = 0.05  # Restricted area height

def clear_scene():
    """Clear all objects in the current Blender scene"""
    bpy.ops.object.select_all(action='SELECT')
    # Deselect cameras and lights (if they exist and you don't want to delete them)
    for obj in bpy.context.scene.objects:
        if obj.type in {'CAMERA', 'LIGHT'}:
            obj.select_set(False)
    # Delete selected objects (usually meshes)
    bpy.ops.object.delete()
    # Clean up orphaned data blocks (optional, keeps file clean)
    bpy.ops.outliner.orphans_purge()

def get_object_dimensions(obj):
    """Calculate precise bounding box dimensions of object in local coordinates"""
    if obj.type != 'MESH' or not obj.data.vertices:
        return Vector((0, 0, 0))

    # Ensure transforms are applied (if modifiers need to be considered)
    # depsgraph = bpy.context.evaluated_depsgraph_get()
    # eval_obj = obj.evaluated_get(depsgraph)
    # mesh = eval_obj.to_mesh()

    # Using original mesh data to calculate bounding box is more reliable, especially for objects with unapplied scaling
    mesh = obj.data
    min_coords = Vector((float('inf'), float('inf'), float('inf')))
    max_coords = Vector((float('-inf'), float('-inf'), float('-inf')))

    for vert in mesh.vertices:
        v_co = vert.co
        min_coords.x = min(min_coords.x, v_co.x)
        min_coords.y = min(min_coords.y, v_co.y)
        min_coords.z = min(min_coords.z, v_co.z)
        max_coords.x = max(max_coords.x, v_co.x)
        max_coords.y = max(max_coords.y, v_co.y)
        max_coords.z = max(max_coords.z, v_co.z)

    # Clean up temporary mesh (if to_mesh() was used)
    # eval_obj.to_mesh_clear()

    if float('inf') in min_coords or float('-inf') in max_coords:
        return Vector((0,0,0)) # Empty mesh

    return max_coords - min_coords

def process_one_instance_bpy(i, uid, position_zup, rotation_quat_zup, target_size_zup, model_base_path, instance_id):
    """
    Process a single object instance in Blender.

    Args:
        i (int): Object index.
        uid (str): UID of the model to load.
        position_zup (list): Target position in Z-up coordinate system [x, y, z].
        rotation_quat_zup (list): Target rotation quaternion in Z-up coordinate system [x, y, z, w].
        target_size_zup (list): Target dimensions in Z-up coordinate system [width, depth, height].
        model_base_path (str): Base path for model files.
        instance_id (str): Original instance ID for naming.

    Returns:
        bpy.types.Object or None: Processed Blender object, or None if failed.
    """
    glb_path = os.path.join(model_base_path, f"{uid}.glb")

    if not os.path.exists(glb_path):
        print(f"  File does not exist: {glb_path}")
        return None

    t0 = time.time()

    # --- 0. Import GLB ---
    # Record objects before import
    objects_before = set(bpy.context.scene.objects)
    # Import GLB file
    bpy.ops.import_scene.gltf(filepath=glb_path)
    # Identify newly imported objects
    objects_after = set(bpy.context.scene.objects)
    imported_objects = list(objects_after - objects_before)

    # Filter mesh objects
    imported_meshes = [obj for obj in imported_objects if obj.type == 'MESH']

    if not imported_meshes:
        print(f"  Warning: No mesh objects found in {glb_path}. Skipping instance {instance_id} ({uid}).")
        # Clean up possible imported non-mesh objects (lights, cameras, etc.)
        for obj in imported_objects:
            if obj.name in bpy.data.objects:
                bpy.data.objects.remove(obj, do_unlink=True)
        return None

    # --- 1. Set origin to geometry center ---
    # Deselect all objects, then select all imported meshes
    bpy.ops.object.select_all(action='DESELECT')
    for obj in imported_meshes:
        obj.select_set(True)
    # Set active object (required for join operation)
    bpy.context.view_layer.objects.active = imported_meshes[0]

    # Apply current transforms to objects so bounding box calculation is based on geometry itself
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # Calculate combined bounding box of all selected meshes
    min_coord = Vector((float('inf'), float('inf'), float('inf')))
    max_coord = Vector((float('-inf'), float('-inf'), float('-inf')))

    for obj in imported_meshes:
        # Use object's world coordinate bounding box corners (after applying transforms, local coordinates are world coordinates)
        for corner_local in obj.bound_box:
            corner_world = obj.matrix_world @ Vector(corner_local)
            min_coord.x = min(min_coord.x, corner_world.x)
            min_coord.y = min(min_coord.y, corner_world.y)
            min_coord.z = min(min_coord.z, corner_world.z)
            max_coord.x = max(max_coord.x, corner_world.x)
            max_coord.y = max(max_coord.y, corner_world.y)
            max_coord.z = max(max_coord.z, corner_world.z)

    # Check if bounding box is valid
    if float('inf') in min_coord or float('-inf') in max_coord:
        print(f"  Warning: Cannot calculate valid combined bounding box for object {uid} ({instance_id}).")
        bpy.ops.object.delete() # Delete selected (imported) meshes
        return None

    # Calculate combined bounding box center
    bbox_center = (min_coord + max_coord) / 2.0

    # Move 3D cursor to calculated center point
    bpy.context.scene.cursor.location = bbox_center

    # Set origin of all selected mesh objects to 3D cursor position
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN') # Use 'ORIGIN_CURSOR'

    # Reset 3D cursor position
    bpy.context.scene.cursor.location = (0, 0, 0)

    # Join all selected mesh objects into one
    # Ensure active object is one of the objects to be joined
    if bpy.context.view_layer.objects.active not in imported_meshes:
            bpy.context.view_layer.objects.active = imported_meshes[0]
    bpy.ops.object.join()

    # The joined object is now the active object
    mesh_obj = bpy.context.object

    # --- 2. Calculate and apply scaling ---
    # Get initial dimensions of joined object (origin is already at geometry center)
    initial_dimensions = get_object_dimensions(mesh_obj)

    # Check if initial dimensions are valid
    if initial_dimensions.length < 1e-6 or any(d < 1e-6 for d in initial_dimensions):
        print(f"  Warning: Object {uid} ({instance_id}) has too small or invalid initial dimensions after joining: {initial_dimensions.to_tuple(3)}. Skipping scaling.")
        scale_factors = Vector((1.0, 1.0, 1.0)) # No scaling
    else:
        target_width, target_depth, target_height = target_size_zup
        target_dimensions = Vector((target_width, target_depth, target_height))

        scale_factors = Vector((1.0, 1.0, 1.0))
        valid_scale = True
        for axis in range(3):
            if initial_dimensions[axis] > 1e-6: # Avoid division by zero or very small numbers
                scale_factors[axis] = target_dimensions[axis] / initial_dimensions[axis]
            elif target_dimensions[axis] > 1e-6: # Initial is zero but target is not zero, cannot scale
                print(f"  Warning: Object {uid} ({instance_id}) has initial dimension close to zero on axis {axis}, but target dimension is not zero. Cannot calculate valid scaling.")
                valid_scale = False
                break
            # If both initial and target dimensions are close to zero, scale factor remains 1.0

        if not valid_scale:
            scale_factors = Vector((1.0, 1.0, 1.0)) # If valid scaling cannot be calculated, reset to no scaling

    # Apply scaling (now acts on the single joined object)
    mesh_obj.scale = scale_factors

    # --- 3. Apply rotation ---
    if rotation_quat_zup:
        # Blender uses (w, x, y, z) order
        x, y, z, w = rotation_quat_zup
        quat = Quaternion((w, x, y, z))

        # Check and normalize quaternion
        if abs(quat.magnitude - 1.0) > 1e-3:
                print(f"  Warning: Quaternion magnitude {quat.magnitude:.4f} for object {uid} ({instance_id}) is not close to 1. Normalizing.")
                if quat.magnitude > 1e-6:
                    quat.normalize()
                else:
                    print(f"  Error: Quaternion magnitude too small, cannot normalize. Skipping rotation.")
                    quat = Quaternion((1, 0, 0, 0)) # No rotation

        mesh_obj.rotation_mode = 'QUATERNION'
        mesh_obj.rotation_quaternion = quat
    # (If z_rotation angle support is needed, can add else branch here)
    # else:
    #    z_angle_rad = math.radians(z_rotation_degrees)
    #    mesh_obj.rotation_mode = 'XYZ' # or 'ZXY' etc., depending on your needs
    #    mesh_obj.rotation_euler = (0, 0, z_angle_rad)

    # --- 4. Apply translation ---
    # Since origin is already at geometry center, directly set object position
    mesh_obj.location = Vector(position_zup)

    # --- 5. Name object ---
    # Clean possible illegal characters
    safe_instance_id = "".join(c if c.isalnum() else "_" for c in instance_id)
    safe_uid = "".join(c if c.isalnum() else "_" for c in uid)
    base_name = f"{safe_instance_id}_{safe_uid}"
    mesh_obj.name = base_name
    # Blender will automatically handle duplicates, adding .001, .002 etc.

    print(f"  Object {i} ({uid}, {instance_id}) processed. Target size: {target_size_zup}, Initial size: {initial_dimensions.to_tuple(3)}, Calculated scaling: {scale_factors.to_tuple(3)}")
    return mesh_obj
    

def create_item_placement_zone_bpy(bbox, table_path=None):
    """Create table model representing item_placement_zone in Blender"""
    if table_path is None:
        print(f"  Error: No table model path provided. Skipping creation.")
        return None
        
    x_min, x_max, y_min, y_max = bbox
    width = x_max - x_min
    depth = y_max - y_min # Z-up depth is along Y axis
    height = PLACEMENT_HEIGHT

    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    # In Blender Z-up, object center is at its geometry center
    # We want object top at Z=0, so center at Z = -height / 2
    center_z = -height / 2

    if not os.path.exists(table_path):
        print(f"  Warning: Table model does not exist: {table_path}. Skipping creation.")
        return None

    try:
        objects_before = set(bpy.context.scene.objects)
        bpy.ops.import_scene.gltf(filepath=table_path)
        objects_after = set(bpy.context.scene.objects)
        imported_objects = list(objects_after - objects_before)

        # Filter mesh objects
        imported_meshes = [obj for obj in imported_objects if obj.type == 'MESH']
        
        if not imported_meshes:
            print(f"  Warning: No mesh objects found in {table_path}. Cannot create item_placement_zone.")
            for obj in imported_objects: bpy.data.objects.remove(obj, do_unlink=True)
            return None

        # Deselect all objects, then select all imported meshes
        bpy.ops.object.select_all(action='DESELECT')
        for obj in imported_meshes:
            obj.select_set(True)
        
        # Set active object (required for join operation)
        bpy.context.view_layer.objects.active = imported_meshes[0]
        
        # Apply transforms to all objects
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        
        # Join all mesh objects into one
        if len(imported_meshes) > 1:
            bpy.ops.object.join()
        
        # The joined object is the active object
        table_mesh_obj = bpy.context.active_object
        
        # Set origin to geometry center
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        table_mesh_obj.location = (0, 0, 0)
        table_mesh_obj.rotation_euler = (0, 0, 0) # Clear any imported rotation

        # Calculate scaling
        table_mesh_obj.scale = (1, 1, 1)
        initial_dimensions = get_object_dimensions(table_mesh_obj)
        target_dimensions = Vector((width, depth, height))
        
        scale_factors = Vector((1.0, 1.0, 1.0))
        valid_scale = True
        for axis in range(3):
            if initial_dimensions[axis] > 1e-6:
                scale_factors[axis] = target_dimensions[axis] / initial_dimensions[axis]
            elif target_dimensions[axis] > 1e-6:
                valid_scale = False; break
        
        if not valid_scale:
             print(f"  Warning: Cannot calculate valid scaling for table model {table_path}. Skipping item_placement_zone.")
             bpy.data.objects.remove(table_mesh_obj, do_unlink=True)
             return None

        table_mesh_obj.scale = scale_factors
        
        # Set position
        table_mesh_obj.location = (center_x, center_y, center_z)
        table_mesh_obj.name = "table"

        # 2. Create new material
        mat_name = "PlacementZoneDarkGreyMat"
        dark_grey_material = bpy.data.materials.new(name=mat_name)
        dark_grey_material.use_nodes = True # Ensure using nodes

        print(f"  item_placement_zone created.")
        return table_mesh_obj

    except Exception as e:
        import traceback
        print(f"  Error creating item_placement_zone: {e}")
        print(traceback.format_exc())
        if 'imported_objects' in locals():
             for obj in imported_objects:
                  if obj.name in bpy.data.objects:
                      bpy.data.objects.remove(obj, do_unlink=True)
        return None

def create_restricted_zone_bpy(bbox, sink_path=None):
    """Create object representing restricted_zone in Blender"""
    if sink_path is None:
        print(f"  Error: No sink model path provided. Skipping creation.")
        return None
        
    x_min, x_max, y_min, y_max = bbox
    width = x_max - x_min
    depth = y_max - y_min
    height = RESTRICTED_HEIGHT

    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
     # We want object bottom to touch Z=0, so center at Z = height / 2
    center_z = - height / 2

    if not os.path.exists(sink_path):
        print(f"  Warning: Restricted zone model does not exist: {sink_path}. Skipping creation.")
        return None

    try:
        objects_before = set(bpy.context.scene.objects)
        bpy.ops.import_scene.gltf(filepath=sink_path)
        objects_after = set(bpy.context.scene.objects)
        imported_objects = list(objects_after - objects_before)

        sink_mesh_obj = None
        for obj in imported_objects:
            if obj.type == 'MESH':
                sink_mesh_obj = obj
                break
        
        if not sink_mesh_obj:
            print(f"  Warning: No mesh objects found in {sink_path}. Cannot create restricted_zone.")
            for obj in imported_objects: bpy.data.objects.remove(obj, do_unlink=True)
            return None

        bpy.context.view_layer.objects.active = sink_mesh_obj
        bpy.ops.object.select_all(action='DESELECT')
        sink_mesh_obj.select_set(True)
        
        # Set origin and move to world origin
        bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        sink_mesh_obj.location = (0, 0, 0)
        sink_mesh_obj.rotation_euler = (0, 0, 0) # Clear any imported rotation

        # Calculate scaling
        sink_mesh_obj.scale = (1, 1, 1)
        initial_dimensions = get_object_dimensions(sink_mesh_obj)
        target_dimensions = Vector((width, depth, height))
        
        scale_factors = Vector((1.0, 1.0, 1.0))
        valid_scale = True
        for axis in range(3):
            if initial_dimensions[axis] > 1e-6:
                scale_factors[axis] = target_dimensions[axis] / initial_dimensions[axis]
            elif target_dimensions[axis] > 1e-6:
                valid_scale = False; break
        
        if not valid_scale:
             print(f"  Warning: Cannot calculate valid scaling for sink model {sink_path}. Skipping restricted_zone.")
             bpy.data.objects.remove(sink_mesh_obj, do_unlink=True)
             return None

        sink_mesh_obj.scale = scale_factors

        print(f"  sink created.")
        
        # Set position
        sink_mesh_obj.location = (center_x, center_y, center_z)
        sink_mesh_obj.name = "sink"

        # --- Set grey material using similar method (English API) ---
        # 1. Clear all existing material slots on object (ensure clean state)
        sink_mesh_obj.data.materials.clear()

        # 2. Create a new material
        # bpy.ops.material.new() can also be used, but bpy.data is more direct
        # Give it a name for later lookup, Blender will automatically add .001 etc. if duplicate
        new_mat_name = "BlueEmissionMat"
        emission_material = bpy.data.materials.new(name=new_mat_name)
        emission_material.use_nodes = True

        # 3. Assign new material to object
        sink_mesh_obj.data.materials.append(emission_material)

        # 4. Get material node tree
        mat = bpy.data.materials.get(new_mat_name)
        if mat and mat.use_nodes:
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            
            # Clear all existing nodes
            nodes.clear()
            
            # Create output node
            output_node = nodes.new(type='ShaderNodeOutputMaterial')
            output_node.location = (300, 0)
            
            # Create emission node
            emission_node = nodes.new(type='ShaderNodeEmission')
            emission_node.location = (0, 0)
            
            # Set light blue color
            light_blue = (0.5, 0.5, 0.8, 1.0)  # Light blue RGB values
            emission_node.inputs['Color'].default_value = light_blue
            
            # Set emission strength
            emission_node.inputs['Strength'].default_value = 2.0  # Adjust emission strength
            
            # Connect nodes
            links.new(emission_node.outputs['Emission'], output_node.inputs['Surface'])
            
            print(f"Applied light blue emission material '{new_mat_name}'")
        else:
            print(f"Error: Cannot create or access material '{new_mat_name}'")

        # --- Check vertex colors (keep previous check) ---
        if sink_mesh_obj.data.vertex_colors:
             active_vc_layer = sink_mesh_obj.data.vertex_colors.active
             if active_vc_layer:
                 print(f"  Warning: Object '{sink_mesh_obj.name}' has active vertex color layer ('{active_vc_layer.name}'). This might override material color in some views.")
        # --- End material setup ---

        return sink_mesh_obj

    except Exception as e:
        import traceback
        print(f"  Error creating restricted_zone: {e}")
        print(traceback.format_exc())
        if 'imported_objects' in locals():
             for obj in imported_objects:
                  if obj.name in bpy.data.objects:
                      bpy.data.objects.remove(obj, do_unlink=True)
        return None

def get_scene_bounds(ignore_types={'CAMERA', 'LIGHT'}):
    """
    Calculate combined world coordinate bounding box of the scene according to specific rules:
    - XY boundaries are determined by 'item_placement_zone'.
    - Z minimum boundary is determined by the maximum Z coordinate of 'item_placement_zone'.
    - Z maximum boundary is determined by the highest point of other objects in the scene.
    (Implementation from Version 2)
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    placement_zone_bounds = {
        "min_x": float('inf'), "min_y": float('inf'),
        "max_x": float('-inf'), "max_y": float('-inf'),
        "max_z": float('-inf'), # Need placement zone's max Z as scene's min Z
        "min_z_actual": float('inf')
    }
    max_z_others = float('-inf')
    placement_zone_found = False
    other_objects_found = False

    for obj_orig in bpy.context.scene.objects:
        if obj_orig.type in ignore_types or obj_orig.hide_render: continue
        if obj_orig.type == 'MESH' and not obj_orig.data.vertices: continue
        obj_eval = obj_orig.evaluated_get(depsgraph)
        try: world_bbox_corners = [obj_eval.matrix_world @ Vector(corner) for corner in obj_eval.bound_box]
        except Exception as e: print(f"  Warning: Error getting bounding box for object {obj_orig.name}: {e}. Skipping this object."); continue
        if not world_bbox_corners or len(world_bbox_corners) != 8: print(f"  Warning: Invalid or incomplete bounding box corners for object {obj_orig.name}. Skipping this object."); continue

        obj_min = Vector((float('inf'), float('inf'), float('inf')))
        obj_max = Vector((float('-inf'), float('-inf'), float('-inf')))
        for corner in world_bbox_corners:
            obj_min.x = min(obj_min.x, corner.x); obj_min.y = min(obj_min.y, corner.y); obj_min.z = min(obj_min.z, corner.z)
            obj_max.x = max(obj_max.x, corner.x); obj_max.y = max(obj_max.y, corner.y); obj_max.z = max(obj_max.z, corner.z)

        if obj_orig.name.startswith("item_placement_zone"):
            placement_zone_found = True
            placement_zone_bounds["min_x"] = min(placement_zone_bounds["min_x"], obj_min.x)
            placement_zone_bounds["min_y"] = min(placement_zone_bounds["min_y"], obj_min.y)
            placement_zone_bounds["max_x"] = max(placement_zone_bounds["max_x"], obj_max.x)
            placement_zone_bounds["max_y"] = max(placement_zone_bounds["max_y"], obj_max.y)
            placement_zone_bounds["max_z"] = max(placement_zone_bounds["max_z"], obj_max.z)
            placement_zone_bounds["min_z_actual"] = min(placement_zone_bounds["min_z_actual"], obj_min.z)
            print(f"  Found item_placement_zone: {obj_orig.name}, XY=({obj_min.x:.2f},{obj_min.y:.2f})-({obj_max.x:.2f},{obj_max.y:.2f}), Z=({obj_min.z:.2f}-{obj_max.z:.2f})")
        elif not obj_orig.name.startswith("restricted_zone"):
             other_objects_found = True
             max_z_others = max(max_z_others, obj_max.z)
             print(f"  Considering other object: {obj_orig.name}, Z_max={obj_max.z:.2f}")

    min_coord = Vector((float('inf'), float('inf'), float('inf')))
    max_coord = Vector((float('-inf'), float('-inf'), float('-inf')))

    if placement_zone_found:
        print("  Using item_placement_zone to define XY boundaries and Z minimum.")
        min_coord.x = placement_zone_bounds["min_x"]
        min_coord.y = placement_zone_bounds["min_y"]
        min_coord.z = placement_zone_bounds["max_z"] # Scene Zmin = zone Zmax
        max_coord.x = placement_zone_bounds["max_x"]
        max_coord.y = placement_zone_bounds["max_y"]
        if other_objects_found:
            max_coord.z = max(max_z_others, min_coord.z) # Scene Zmax = highest point of other objects (at least not lower than Zmin)
            print(f"  Highest point of other objects Z={max_z_others:.2f}.")
        else:
            max_coord.z = min_coord.z
            print("  No other objects found, Z maximum set to Z minimum.")
    else:
        print("Warning: No 'item_placement_zone' object found. Will use combined bounding box of all objects.")
        has_objects = False
        for obj_orig in bpy.context.scene.objects:
            if obj_orig.type in ignore_types or obj_orig.hide_render: continue
            if obj_orig.type == 'MESH' and not obj_orig.data.vertices: continue
            obj_eval = obj_orig.evaluated_get(depsgraph)
            try: world_bbox_corners = [obj_eval.matrix_world @ Vector(corner) for corner in obj_eval.bound_box]
            except Exception: continue
            if not world_bbox_corners or len(world_bbox_corners) != 8: continue
            has_objects = True
            for corner in world_bbox_corners:
                min_coord.x = min(min_coord.x, corner.x); min_coord.y = min(min_coord.y, corner.y); min_coord.z = min(min_coord.z, corner.z)
                max_coord.x = max(max_coord.x, corner.x); max_coord.y = max(max_coord.y, corner.y); max_coord.z = max(max_coord.z, corner.z)
        if not has_objects:
             print("Warning: Fallback logic also found no valid objects.")
             min_coord = Vector((0,0,0)); max_coord = Vector((0,0,0))

    if float('inf') in min_coord or float('-inf') in max_coord:
        print("Warning: Cannot calculate valid scene bounding box.")
        return None, None, Vector((0.0, 0.0, 0.0)), Vector((0.0, 0.0, 0.0))

    center = (min_coord + max_coord) / 2.0
    size = max_coord - min_coord
    size.x = max(size.x, 0); size.y = max(size.y, 0); size.z = max(size.z, 0)

    print(f"  Final calculated boundaries: Min={min_coord.to_tuple(3)}, Max={max_coord.to_tuple(3)}")
    print(f"  Final calculated center: {center.to_tuple(3)}")
    print(f"  Final calculated size: {size.to_tuple(3)}")
    return min_coord, max_coord, center, size

def setup_compositor_transparent_background():
    """
    Configure compositor nodes to preserve transparent background in render results.
    """
    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree
    nodes = tree.nodes
    links = tree.links

    # Clear existing nodes
    for node in nodes:
        nodes.remove(node)

    # Create Render Layers node
    render_layers = nodes.new('CompositorNodeRLayers')
    render_layers.location = (-300, 300)

    # Create Composite node
    composite = nodes.new('CompositorNodeComposite')
    composite.location = (0, 300)
    
    # Directly connect render layer to output, preserving transparency
    links.new(render_layers.outputs["Image"], composite.inputs["Image"])

def render_four_views(output_dir, resolution=512, base_distance_factor=1.5, min_distance=0.2):
    """
    Use Cycles GPU to render four standard views of current scene, automatically align to scene center and adjust distance.
    (Incorporates Version 2 details: world light, updated vectors, light energy, min distance)
    """
    print(f"\nStarting four-view rendering to: {output_dir}")
    if not output_dir:
        print("Error: No valid output directory provided.")
        return

    context = bpy.context
    scene = context.scene
    render = scene.render

    # --- 1. Configure renderer (similar to before) ---
    render.engine = 'CYCLES'
    render.resolution_x = resolution
    render.resolution_y = resolution
    render.resolution_percentage = 100
    render.image_settings.file_format = 'PNG'
    render.image_settings.color_mode = 'RGBA'
    render.film_transparent = True

    # --- Add basic world background light (from reconstruct_blender.py) ---
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    try: # Add try-except to prevent errors if node tree is empty
        tree = world.node_tree
        bg_node = tree.nodes.get('Background')
        if not bg_node: # If no background node, try to create
            # Clean up possible existing old nodes
            for node in tree.nodes:
                tree.nodes.remove(node)
            # Create required nodes
            bg_node = tree.nodes.new(type='ShaderNodeBackground')
            output_node = tree.nodes.new(type='ShaderNodeOutputWorld')
            tree.links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])

        if bg_node:
            bg_node.inputs['Color'].default_value = (0.1, 0.1, 0.1, 1.0) # Set weak grey
            bg_node.inputs['Strength'].default_value = 10.0 # Sync strength value
            print("  Basic world background light set.")
        else:
             print("  Warning: Cannot find or create world background node.")
    except Exception as e_world:
        print(f"  Error setting world background: {e_world}")
    # --- End world background light setup ---

    # GPU setup (similar to before)
    scene.cycles.device = 'GPU'
    prefs = context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'CUDA' # or other
    enabled_gpu = False
    try: # Avoid errors when 'cycles' addon is not available
        for device_type in prefs.get_device_types(context):
             prefs.get_devices_for_type(device_type[0])
        for device in prefs.devices:
            if device.type == 'CUDA': # or other
                device.use = True
                enabled_gpu = True
                print(f"Enabled GPU device: {device.name}")
            else:
                device.use = False
    except Exception as e:
         print(f"Error getting GPU devices: {e}, will use CPU.")
         enabled_gpu = False

    if not enabled_gpu:
        print("Warning: No compatible GPU devices found or enabled, will use CPU rendering.")
        scene.cycles.device = 'CPU'

    # Cycles settings (similar to before)
    scene.cycles.samples = 128
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = 'OPENIMAGEDENOISE'
    scene.cycles.diffuse_bounces = 4
    scene.cycles.glossy_bounces = 4
    scene.cycles.transparent_max_bounces = 8
    scene.cycles.transmission_bounces = 8
    scene.render.use_persistent_data = True

    # Configure compositor
    setup_compositor_transparent_background()

    # --- 2. Calculate scene boundaries and center ---
    _, _, scene_center, scene_size = get_scene_bounds()
    print(f"  Calculated scene center: {scene_center.to_tuple(3)}")
    print(f"  Calculated scene size: {scene_size.to_tuple(3)}")

    # Dynamically calculate camera distance
    max_scene_dimension = max(scene_size) if any(d > 1e-6 for d in scene_size) else 1.0
    dynamic_distance = max(max_scene_dimension * base_distance_factor, min_distance)
    print(f"  Dynamically adjusted camera distance: {dynamic_distance:.2f}")

    # --- 3. Setup lighting (sync with reconstruct_blender.py) ---
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='LIGHT')
    bpy.ops.object.delete()

    light_data = bpy.data.lights.new(name="AreaLight", type='AREA')
    light_data.energy = 50 # Use energy value from reconstruct_blender.py
    light_data.size = dynamic_distance # Make light size distance-related (consistent with reconstruct_blender.py)
    light_object = bpy.data.objects.new(name="AreaLight", object_data=light_data)
    scene.collection.objects.link(light_object)
    # Place light diagonally above scene center (consistent with reconstruct_blender.py)
    light_object.location = scene_center + Vector((dynamic_distance * 0.8, -dynamic_distance * 0.8, dynamic_distance * 1.2))
    # Make light point toward scene center (consistent with reconstruct_blender.py)
    light_direction = scene_center - light_object.location
    light_object.rotation_euler = light_direction.to_track_quat('-Z', 'Y').to_euler()

    # --- 4. Setup camera ---
    cam_data = bpy.data.cameras.new("RenderCam")
    cam_data.lens = 50
    cam = bpy.data.objects.new("RenderCam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam

    # --- 5. Define relative view direction vectors and render ---
    # These are direction vectors relative to scene center (viewing center from that direction)
    # (x, y, z)
    view_vectors = {
        "front": Vector((0, -1, 0.6)).normalized(),      # Version 2 Z offset
        # "left": Vector((-1, 0, 0.3)).normalized(),
        "top": Vector((0, -0.3, 1)).normalized(),
        "perspective": Vector((-0.7, -0.7, 0.7)).normalized()
    }

    os.makedirs(output_dir, exist_ok=True)

    for view_name, direction_vec in view_vectors.items():
        print(f"  Rendering view: {view_name}...")
        # Calculate camera position: scene center + direction vector * distance
        cam.location = scene_center + direction_vec * dynamic_distance

        # Calculate camera orientation: from camera position pointing to scene center
        look_at_direction = scene_center - cam.location
        cam.rotation_euler = look_at_direction.to_track_quat('-Z', 'Y').to_euler()

        render.filepath = os.path.join(output_dir, f"{view_name}.png")
        try:
            bpy.ops.render.render(write_still=True)
        except Exception as e:
            print(f"  Error rendering view {view_name}: {e}")
            # Can choose to continue rendering other views or stop here
            continue

    # Cleanup (optional)
    if cam.name in scene.collection.objects: scene.collection.objects.unlink(cam)
    if cam.name in bpy.data.objects: bpy.data.objects.remove(cam)
    if cam_data.name in bpy.data.cameras: bpy.data.cameras.remove(cam_data)
    if light_object.name in scene.collection.objects: scene.collection.objects.unlink(light_object)
    if light_object.name in bpy.data.objects: bpy.data.objects.remove(light_object)
    if light_data.name in bpy.data.lights: bpy.data.lights.remove(light_data)

    print("Four-view rendering completed.")

def reconstruct_scene_from_json_bpy(scene_json_path, results_json_path=None, model_base_path=None, output_path=None, save_glb=True, render_views=True, table_path=None, sink_path=None):
    """
    Use Blender to reconstruct scene from scene JSON and retrieval results JSON.
    """
    # Clean scene
    clear_scene()

    # Read scene JSON file
    if not os.path.exists(scene_json_path):
        print(f"Error: Scene JSON file does not exist: {scene_json_path}")
        return None

    try:
        with open(scene_json_path, 'r') as f:
            scene_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Error parsing scene JSON file: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error reading scene JSON file: {e}")
        return None

    # Check if this is a processed scene with embedded selected_uid
    has_embedded_uids = False
    if scene_data.get("objects") and len(scene_data["objects"]) > 0:
        first_obj = scene_data["objects"][0]
        if "selected_uid" in first_obj:
            has_embedded_uids = True

    # Read results JSON only if needed (traditional format)
    results_data = {}
    if not has_embedded_uids and results_json_path:
        if not os.path.exists(results_json_path):
            print(f"Error: Results JSON file does not exist: {results_json_path}")
            return None
        try:
            with open(results_json_path, 'r') as f:
                results_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: Error parsing JSON files: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error reading JSON files: {e}")
            return None

    # Create scene collection (optional, for organization)
    scene_collection = bpy.context.scene.collection
    # new_collection = bpy.data.collections.new("ReconstructedScene")
    # scene_collection.children.link(new_collection)
    # layer_collection = bpy.context.view_layer.layer_collection.children[new_collection.name]
    # bpy.context.view_layer.active_layer_collection = layer_collection

    # Process item_placement_zone
    if "item_placement_zone" in scene_data:
        print("Creating item_placement_zone...")
        print('######################### Creating item_placement_zone #########################')
        placement_zone_obj = create_item_placement_zone_bpy(scene_data["item_placement_zone"], table_path)
        if placement_zone_obj:
            print("  item_placement_zone created.")
            # link object to collection if using new_collection
            # new_collection.objects.link(placement_zone_obj)
            # scene_collection.objects.unlink(placement_zone_obj) # Unlink from default

    # Process restricted_zone
    if "sink_zone" in scene_data:
        print("Creating restricted_zone...")
        print('######################### Creating sink_zone #########################')
        restricted_zone_obj = create_restricted_zone_bpy(scene_data["sink_zone"], sink_path)
        if restricted_zone_obj:
             print(f"  restricted_zone created (using {os.path.basename(sink_path)}).")
            # link object to collection if using new_collection
            # new_collection.objects.link(restricted_zone_obj)
            # scene_collection.objects.unlink(restricted_zone_obj)

    # Process all objects
    skipped_instances_log = []
    processed_count = 0
    objects_to_process = scene_data.get("objects", [])
    total_objects = len(objects_to_process)

    print(f"\nStarting to process {total_objects} objects...")

    for i, obj_data in enumerate(objects_to_process):
        instance_id = obj_data.get("instance")
        position_zup = obj_data.get("position", [0, 0, 0])
        target_size_zup = obj_data.get("size") # [width, depth, height] in Z-up

        print(f"\nProcessing object: {instance_id} [{i+1}/{total_objects}]")

        # Get selected UID - support both embedded and traditional formats
        selected_uid = None
        
        if has_embedded_uids:
            # Use embedded selected_uid from processed scene
            selected_uid = obj_data.get("selected_uid")
            if selected_uid:
                print(f"  Using embedded UID: {selected_uid}")
        else:
            # get from results JSON
            retrieved_info = results_data.get(instance_id)
            if retrieved_info and retrieved_info.get("retrieved_uids"):
                selected_uid = retrieved_info["retrieved_uids"][0]
                print(f"  Using retrieved UID: {selected_uid}")
        
        if not selected_uid:
            print(f"  Warning: Cannot find valid UID for instance '{instance_id}'. Skipping this object.")
            skipped_instances_log.append(f"{instance_id} (no valid UID)")
            continue

        # Get rotation information (priority: quaternion, then Z rotation angle)
        rotation_quat_zup = obj_data.get("rotation")
        if rotation_quat_zup is None:
            z_angle_deg = obj_data.get("z_rotation", 0.0)
            z_angle_rad = math.radians(z_angle_deg)
            # Blender Z up, rotate around Z axis
            # Quaternion (w, x, y, z)

            rotation_quat_zup = Quaternion((math.cos(z_angle_rad / 2.0), 0, 0, math.sin(z_angle_rad / 2.0)))
            # Original script used [x,y,z,w], here corrected to format accepted by Blender
            rotation_quat_zup = [rotation_quat_zup[1], rotation_quat_zup[2], rotation_quat_zup[3], rotation_quat_zup[0]] # xyzw
            print(f"  Using z_rotation {z_angle_deg}° to generate quaternion (xyzw): {[f'{x:.4f}' for x in rotation_quat_zup]}")
        else:
             # Ensure it's list or tuple
             if isinstance(rotation_quat_zup, (list, tuple)) and len(rotation_quat_zup) == 4:
                  print(f"  Using original quaternion from JSON (xyzw): {[f'{x:.4f}' for x in rotation_quat_zup]}")
             else:
                  print(f"  Warning: Invalid 'rotation' format in original JSON ({rotation_quat_zup}). Will use no rotation.")
                  rotation_quat_zup = [0.0, 0.0, 0.0, 1.0] # xyzw

        # Check required information
        if not selected_uid:
            print(f"  Error: Cannot determine UID for object '{instance_id}'.")
            skipped_instances_log.append(f"{instance_id} (invalid UID)")
            continue
        if not target_size_zup or len(target_size_zup) != 3:
            print(f"  Error: Object '{instance_id}' missing valid target size information.")
            skipped_instances_log.append(f"{instance_id} (invalid size)")
            continue
        if not position_zup or len(position_zup) != 3:
             print(f"  Error: Object '{instance_id}' missing valid position information.")
             skipped_instances_log.append(f"{instance_id} (invalid position)")
             continue

        # Call processing function
        processed_obj = process_one_instance_bpy(
            i, selected_uid, position_zup, rotation_quat_zup, target_size_zup, model_base_path, instance_id
        )

        if processed_obj:
            processed_count += 1
            # link object to collection if using new_collection
            # new_collection.objects.link(processed_obj)
            # try: # Unlink might fail if it was never in the scene collection
            #     scene_collection.objects.unlink(processed_obj)
            # except: pass
        else:
            skipped_instances_log.append(f"{instance_id} ({selected_uid} processing failed)")

    # Export scene to GLB file
    num_objects_in_scene = len([obj for obj in bpy.context.scene.objects if obj.type not in {'CAMERA', 'LIGHT'}]) # Only count non-auxiliary objects
    if num_objects_in_scene > 0:
        print(f"\nPreparing to export/render scene ({num_objects_in_scene} relevant objects)...")
        glb_export_successful = False
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            t0 = time.time()
            bpy.ops.export_scene.gltf(
                filepath=output_path,
                export_format='GLB',
                use_selection=False,
                export_apply=True, # Very important, apply transforms and modifiers
                export_lights=False,
                export_cameras=False
            )
            print(f"\nScene successfully saved to: {output_path}")
            print(f"Processed {processed_count} / {total_objects} objects.")
            print(f"Export time: {time.time() - t0:.2f} seconds")
            glb_export_successful = True # Mark export successful

        except Exception as e:
            import traceback
            print(f"\nError exporting scene to GLB: {e}")
            print(traceback.format_exc())
            output_path = None # Indicate export failed

        # --- Modified rendering call ---
        if glb_export_successful and output_path: # Only render after successful GLB export
            render_output_dir = os.path.join(os.path.dirname(output_path), "rendered_views")
            render_four_views(render_output_dir)
        elif not glb_export_successful:
             print("Skipping rendering due to GLB export failure.")
        else: # output_path is None (possibly due to empty scene or pre-export setting)
            print("Skipping rendering due to invalid output path or empty scene.")

    else:
        print("\nScene is empty, no GLB file saved, no rendering executed.")
        output_path = None

    # Log skipped instances
    if skipped_instances_log:
        print("\nSkipped instances:")
        for log in skipped_instances_log:
            print(f"- {log}")

    return output_path

# --- Blender script entry point ---
def main():
    # Parse arguments after '--' from Blender command line
    argv = sys.argv
    try:
        index = argv.index("--") + 1
        args_list = argv[index:]
    except ValueError:
        args_list = [] # No '--' separator

    parser = argparse.ArgumentParser(description="Use Blender to reconstruct 3D scene from scene JSON and retrieval results JSON")
    parser.add_argument("--scene_json", type=str, required=True, help="Original scene description JSON file path")
    parser.add_argument("--results_json", type=str, help="Retrieval results JSON file path (optional for processed scenes)")
    # --output_glb is no longer required, only needed when --save_glb is set
    parser.add_argument("--output_glb", type=str, help="Output GLB file path (only needed when --save_glb is set)")
    parser.add_argument("--model_base_path", type=str, required=True, help="3D model base path (containing .glb files)")
    # --- Add control flags ---
    # action='store_true' means if this parameter appears in command line, its value is True, otherwise False
    parser.add_argument("--save_glb", action='store_true', help="Whether to save output GLB file")
    parser.add_argument("--render_views", action='store_true', help="Whether to render four views")
    parser.add_argument("--gpu_device_idx", type=str, default="0", help="GPU device index for Blender to use (within visible devices).")
    parser.add_argument("--table_path", type=str, help="Table model GLB file path")
    parser.add_argument("--sink_path", type=str, help="Sink model GLB file path")
    # --------------------

    # Use parse_args to parse arguments from command line
    args = parser.parse_args(args_list)

    # --- Parameter validation ---
    # Parameter validation - results_json is now optional
    if args.save_glb and not args.output_glb:
         parser.error("--output_glb must be provided when --save_glb is set.")
    if args.render_views and not args.output_glb:
         parser.error("--output_glb (for determining render directory) must be provided when --render_views is set, even if not saving GLB.")

    gpu_device_idx = int(args.gpu_device_idx) # Convert to int

    print("="*40)
    print("Starting Blender scene reconstruction")
    print(f"Scene JSON: {args.scene_json}")
    print(f"Results JSON: {args.results_json}")
    print(f"Model path: {args.model_base_path}")
    if args.save_glb: print(f"Output GLB: {args.output_glb}")
    else: print("Output GLB: Disabled")
    if args.render_views: print(f"Render views: Enabled")
    else: print("Render views: Disabled")
    print("="*40)

    t_start = time.time()

    # --- Set Render Device ---
    bpy.context.scene.render.engine = 'CYCLES' # Ensure Cycles is used if needed
    prefs = bpy.context.preferences.addons['cycles'].preferences

    # Enable GPU compute devices. Requires CUDA/OptiX/etc support in Blender build.
    prefs.compute_device_type = 'CUDA' # Or 'OPTIX', 'HIP', 'METAL' depending on GPU/OS
    bpy.context.scene.cycles.device = 'GPU'

    # Unselect all devices first
    for d in prefs.devices:
        d.use = False

    # Attempt to select the specified device index among the available CUDA devices
    cuda_devices = [d for d in prefs.devices if d.type == 'CUDA'] # Or OPTIX etc.
    if cuda_devices:
        if gpu_device_idx < len(cuda_devices):
            cuda_devices[gpu_device_idx].use = True
            print(f"Blender Cycles configured to use GPU device index {gpu_device_idx}: {cuda_devices[gpu_device_idx].name}")
        else:
            print(f"Warning: Requested GPU index {gpu_device_idx} out of range for available CUDA devices ({len(cuda_devices)}). Falling back to default.")
            # Fallback: Enable the first available GPU or let Blender decide
            cuda_devices[0].use = True
    else:
         print("Warning: No CUDA/compatible GPU devices found for Cycles by Blender. Rendering might use CPU.")
    # --- End Render Device Setup ---

    reconstruct_scene_from_json_bpy(
        args.scene_json,
        args.results_json,
        args.model_base_path,
        args.output_glb, # Pass even if not saving, used to determine render path
        save_glb=args.save_glb,             # <--- Pass parameter
        render_views=args.render_views,     # <--- Pass parameter
        table_path=args.table_path,         # <--- New parameter
        sink_path=args.sink_path            # <--- New parameter
    )

    t_end = time.time()
    print(f"\nTotal time: {t_end - t_start:.2f} seconds")
    print("Blender script execution completed.")

if __name__ == "__main__":
    main()
