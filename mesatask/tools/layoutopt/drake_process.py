import torch  
import numpy as np  
import time  
from datetime import timedelta  
import multiprocessing  
from functools import partial  
from typing import List, Tuple, Union, Dict, Optional
import os
from pydrake.solvers import SnoptSolver, IpoptSolver  
from pydrake.multibody.inverse_kinematics import InverseKinematics  
  
from pydrake.all import (
    Context,
    EventStatus,
    InverseKinematics,
    IpoptSolver,
    Simulator,
    SnoptSolver,
    SolverOptions,
    PackageMap,
    DiagramBuilder,
    RgbdSensor,
    RigidTransform,
    RollPitchYaw,
    AddMultibodyPlantSceneGraph,
    Parser,
    SceneGraph
)
from drake_utils import (
    create_plant_and_scene_graph_from_scene_with_cache,
    update_scene_poses_from_plant,
)
from drake_dataclass import (
    PlantSceneGraphCache,
    SceneVecDescription,
)
import re
import tempfile
import trimesh
import matplotlib.pyplot as plt
# from pydrake.geometry import (
#     CameraInfo, ClippingRange, DepthRange,
#     RenderCameraCore, ColorRenderCamera, DepthRenderCamera
# )
import pydrake
import json
from check_scale import check_scale_factors

# Corrected Camera related imports based on diagnostics
from pydrake.geometry import (
    ClippingRange, DepthRange, RenderCameraCore, ColorRenderCamera, DepthRenderCamera, SceneGraph
)
from pydrake.systems.sensors import CameraInfo

import yaml

def load_config(config_path=None):
    """Load configuration file"""
    # Try to find config file
    if not config_path:
        script_dir = os.path.dirname(__file__)
        # 修改配置文件路径
        config_path = os.path.join(script_dir, "..", "config.yaml")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def apply_non_penetration_projection_single_scene(
    scene: torch.Tensor,
    cache: Union[PlantSceneGraphCache, None],
    scene_vec_desc: SceneVecDescription,
    translation_only: bool,
    influence_distance: float = 0.1,
    solver_name: str = "snopt",
    iteration_limit: int = 5000000,
    return_cache: bool = True,
    tol: float = 1e-5,
    min_distance: float = 0.01,
) -> Tuple[torch.Tensor, Union[PlantSceneGraphCache, None], bool]:
    """See `apply_non_penetration_projection` for more details."""
    # Obtain the plant and scene graph.
    cache, _, plant_context = create_plant_and_scene_graph_from_scene_with_cache(
        scene=scene,
        scene_vec_desc=scene_vec_desc,
        cache=cache,
    )
    plant = cache.plant

    # Set up projection NLP.
    ik = InverseKinematics(plant, plant_context)
    q_vars = ik.q()
    prog = ik.prog()

    # Stay close to initial positions.
    q0 = plant.GetPositions(plant_context)
    if len(q0) == 0:
        return scene, cache, False

    for body_idx, model_idx in zip(cache.rigid_body_indices, cache.model_indices):
        if body_idx is None:
            continue

        body = plant.get_body(body_idx)
        if not body.is_floating():
            continue

        # For two quaternion z and z0 with angle θ between their orientation,
        # we know
        # 1-cosθ = 2 - 2*(zᵀz₀)² = 2 - 2zᵀz₀z₀ᵀz
        # So we can add a quadratic cost on the quaternion z.
        q_start_idx = body.floating_positions_start()
        model_quat_vars = q_vars[q_start_idx : q_start_idx + 4]
        z0 = q0[q_start_idx : q_start_idx + 4]
        prog.AddQuadraticCost(
            -4 * np.outer(z0, z0), np.zeros((4,)), 2, model_quat_vars, is_convex=False
        )
        model_pos_vars = q_vars[q_start_idx + 4 : q_start_idx + 7]
        prog.AddQuadraticErrorCost(
            np.eye(3), q0[q_start_idx + 4 : q_start_idx + 7], model_pos_vars
        )

    # Nonpenetration constraint.
    ik.AddMinimumDistanceLowerBoundConstraint(min_distance, influence_distance)

    if translation_only:
        # Add constraints for rotations to stay constant.
        for body_idx, model_idx in zip(cache.rigid_body_indices, cache.model_indices):
            if body_idx is None:
                # Skip empty objects.
                continue

            body = plant.get_body(body_idx)
            if not body.is_floating():
                # Skip non-floating bodies.
                continue

            # Get rotation decision variables.
            q_start_idx = body.floating_positions_start()
            model_quat_vars = q_vars[q_start_idx : q_start_idx + 4]

            # Get initial rotation.
            model_q = plant.GetPositions(plant_context, model_idx)
            model_quat = model_q[:4]

            # Add constraint for rotation to stay constant.
            prog.AddBoundingBoxConstraint(
                model_quat,  # lb
                model_quat,  # ub
                model_quat_vars,  # vars
            )

    # Use the starting positions as the initial guess.
    prog.SetInitialGuess(q_vars, q0)

    # Solve.
    options = SolverOptions()
    if solver_name == "snopt":
        solver = SnoptSolver()
        options.SetOption(solver.id(), "Major feasibility tolerance", 1e-3)
        options.SetOption(solver.id(), "Major optimality tolerance", 1e-3)
        options.SetOption(solver.id(), "Major iterations limit", iteration_limit)
        options.SetOption(solver.id(), "Time limit", 60)
        options.SetOption(solver.id(), "Timing level", 3)
    elif solver_name == "ipopt":
        solver = IpoptSolver()
        options.SetOption(solver.id(), "max_iter", iteration_limit)
        options.SetOption(solver.id(), "tol", tol)
    else:
        raise ValueError(f"Invalid solver: {solver_name}")
    if not solver.available():
        raise ValueError(f"Solver {solver_name} is not available.")

    try:
        result = solver.Solve(prog, None, options)
        success = result.is_success()

        # Update the scene poses.
        plant.SetPositions(plant_context, result.GetSolution(q_vars))
        projected_scene = update_scene_poses_from_plant(
            scene=scene,
            plant=plant,
            plant_context=plant_context,
            model_indices=cache.model_indices,
            scene_vec_desc=scene_vec_desc,
        )
    except Exception as e:
        projected_scene = scene
        success = False

    return projected_scene, (cache if return_cache else None), success

def apply_non_penetration_projection(
    scenes: torch.Tensor,
    scene_vec_desc: SceneVecDescription,
    translation_only: bool,
    influence_distance: float,
    solver_name: str,
    caches: Union[List[PlantSceneGraphCache], List[None]],
    iteration_limit: int = 5000000,
    num_workers: int = 1,
    min_distance: float = 0.01,
) -> Tuple[torch.Tensor, Union[List[PlantSceneGraphCache], List[None]], List[bool]]:
    start_time = time.time()

    scenes = scenes.cpu().detach()

    if num_workers == 1 or len(scenes) == 1:
        projected_scenes, new_caches, successes = [], [], []
        for scene, cache in zip(scenes, caches):
            (
                projected_scene,
                cache,
                success,
            ) = apply_non_penetration_projection_single_scene(
                scene=scene,
                cache=cache,
                scene_vec_desc=scene_vec_desc,
                translation_only=translation_only,
                influence_distance=influence_distance,
                solver_name=solver_name,
                iteration_limit=iteration_limit,
                min_distance=min_distance,
            )
            projected_scenes.append(projected_scene)
            new_caches.append(cache)
            successes.append(success)
    else:
        num_workers = min(num_workers, len(scenes), multiprocessing.cpu_count())
        with multiprocessing.Pool(num_workers) as pool:
            result = pool.starmap(
                partial(
                    apply_non_penetration_projection_single_scene,
                    scene_vec_desc=scene_vec_desc,
                    translation_only=translation_only,
                    solver_name=solver_name,
                    iteration_limit=iteration_limit,
                    return_cache=False,  # Can't return non-pickeable objects.
                    min_distance=min_distance,
                ),
                zip(scenes, caches),
            )
            projected_scenes, _, successes = zip(*result)
            successes = list(successes)

        # Caches stay the same as the objects in the scene aren't changed by the
        # projection.
        new_caches = caches


    return torch.stack(projected_scenes), new_caches, successes

def generate_scaled_urdf(original_urdf_path, scale, tag=None, uid=None):
    """
    Generate a new temporary URDF file with specified scale and mesh path.
    - Modifies <robot name="..."> to ensure uniqueness
    - Replaces mesh filename with obj_path/{uid}.obj
    Returns the path to the new file.
    """
    if isinstance(scale, (tuple, list)):
        scale_str = " ".join(str(s) for s in scale)
    else:
        scale_str = str(scale) # Should already be a string if single

    with open(original_urdf_path, "r") as f:
        urdf_content = f.read()

    # --- NEW: Make robot name unique within the URDF content ---
    unique_robot_name = f"scaled_mesh_robot_{tag}" # e.g., scaled_mesh_robot_0
    urdf_content_renamed = re.sub(
        r'<robot\s+name="[^"]*"', # Matches <robot name="anything_here"
        f'<robot name="{unique_robot_name}"', # Replaces with unique name
        urdf_content,
        count=1 # Only replace the first occurrence
    )

    # --- NEW: Replace mesh filename ---
    if uid is not None:
        cfg = load_config()
        obj_dir = cfg.get("drake", {}).get("model_package_path", "./MesaTask-10K/Assets_library_obj")
        new_mesh_path = os.path.join(obj_dir, f'{uid}.obj')
        urdf_content_renamed = re.sub(
            r'<mesh filename="[^"]*"', 
            f'<mesh filename="{new_mesh_path}"', 
            urdf_content_renamed
        )

    urdf_new_scale = re.sub(r'scale="[^"]*"', f'scale="{scale_str}"', urdf_content_renamed)

    base = os.path.basename(original_urdf_path).replace('.urdf', '')
    filename_tag = str(tag).replace(' ', '_') if tag is not None else scale_str.replace(' ', '_')
    tmp_urdf_path = os.path.join(tempfile.gettempdir(), f"{base}_scale_{filename_tag}.urdf")

    with open(tmp_urdf_path, "w") as f:
        f.write(urdf_new_scale)
    add_origin_rpy_to_urdf(tmp_urdf_path, "0 0 0")
    if not os.path.exists(new_mesh_path):
        print(f"Mesh file not found: {new_mesh_path}")
    return tmp_urdf_path

def add_origin_rpy_to_urdf(urdf_path, rpy="-1.5708 0 0"):
    with open(urdf_path, "r") as f:
        urdf = f.read()
    # Insert <origin rpy="..."/> after <visual> and <collision> tags
    urdf = re.sub(r'(<visual>\s*)', r'\1<origin rpy="{}" xyz="0 0 0"/>'.format(rpy), urdf)
    urdf = re.sub(r'(<collision>\s*)', r'\1<origin rpy="{}" xyz="0 0 0"/>'.format(rpy), urdf)
    with open(urdf_path, "w") as f:
        f.write(urdf)

def save_drake_topdown_image(urdf_paths, positions, rotations, image_path="drake_topdown.png"):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)

    renderer_name_for_cameras = "DefaultRenderer" # Default fallback
    try:
        # Try to explicitly register a VTK renderer and use its name.
        # These imports should be valid based on your dir(pydrake.geometry)
        from pydrake.geometry import RenderEngineVtkParams, MakeRenderEngineVtk, Rgba
        
        vtk_params = RenderEngineVtkParams()
        # You can customize parameters if needed, e.g.:
        # vtk_params.default_clear_color = Rgba(0.8, 0.8, 0.8, 1.0) # Gray background
        
        # Attempt to register the renderer
        # The method name might be AddRenderer or RegisterRenderer.
        # We'll try RegisterRenderer first as it's common in newer Drake.
        if hasattr(scene_graph, "RegisterRenderer"):
            scene_graph.RegisterRenderer("my_vtk_renderer", MakeRenderEngineVtk(vtk_params))
            renderer_name_for_cameras = "my_vtk_renderer"
            print("Successfully registered 'my_vtk_renderer' with RegisterRenderer.")
        elif hasattr(scene_graph, "AddRenderer"): # Some older versions might use AddRenderer
            scene_graph.AddRenderer("my_vtk_renderer", MakeRenderEngineVtk(vtk_params))
            renderer_name_for_cameras = "my_vtk_renderer"
            print("Successfully registered 'my_vtk_renderer' with AddRenderer.")
        else:
            print("Warning: SceneGraph instance does not have 'RegisterRenderer' or 'AddRenderer'. Using fallback renderer name.")
            # If explicit registration is not available/fails, rely on AddMultibodyPlantSceneGraph
            # having set up a default renderer, and guess its name.
            # "DefaultRenderer" or "vtk" are common. Let's stick with "DefaultRenderer" as a primary guess.
            # renderer_name_for_cameras = "vtk" # Alternative guess

    except ImportError as ie:
        print(f"Warning: Could not import VTK renderer components ({ie}). Using fallback renderer name.")
    except Exception as e_reg:
        print(f"Warning: Failed to explicitly register a VTK renderer ({e_reg}). Using fallback renderer name.")
    
    print(f"Using renderer name for cameras: '{renderer_name_for_cameras}'")

    parser = Parser(plant)

    model_counter = 0

    for urdf, pos, quat in zip(urdf_paths, positions, rotations):
        model_name = f"scaled_mesh_{model_counter}"
        model_counter += 1
        try:
            model_instances = parser.AddModels(file_name=urdf, model_name=model_name)
            if not model_instances:
                print(f"Warning: AddModels(file_name={urdf}, model_name={model_name}) returned no model instances.")
                continue
            model_instance = model_instances[0]
            body = plant.GetBodyByName("body", model_instance)
        except Exception as e:
            try:
                print(f"Note: AddModels with explicit model_name failed ('{e}'), trying without...")
                model_instances = parser.AddModels(file_name=urdf)
                if not model_instances:
                     print(f"Warning: AddModels(file_name={urdf}) on fallback returned no model instances.")
                     continue
                model_instance = model_instances[0]
                body = plant.GetBodyByName("body", model_instance)
            except Exception as e_fallback:
                print(f"Error loading model {urdf}: With model_name: {e}, Fallback without: {e_fallback}")
                continue

        X_WB = RigidTransform(
            RollPitchYaw(0, 0, 0),
            pos
        )
        plant.WeldFrames(plant.world_frame(), body.body_frame(), X_WB)

    plant.Finalize()

    width = 800
    height = 800
    fov_y_radians = np.pi / 4.0
    clipping_near = 0.1
    clipping_far = 100.0


    cam_intrinsics = CameraInfo(width, height, fov_y_radians)
    cam_clipping = ClippingRange(clipping_near, clipping_far)
    X_BC_pose = RigidTransform() # Camera pose in body frame (identity if sensor is the body)

    color_camera_core = RenderCameraCore(
        renderer_name_for_cameras, cam_intrinsics, cam_clipping, X_BC_pose)
    color_camera = ColorRenderCamera(color_camera_core, show_window=False)

    # It's good practice to define the depth camera even if not immediately used for saving
    depth_camera_core = RenderCameraCore(
        renderer_name_for_cameras, cam_intrinsics, cam_clipping, X_BC_pose)
    depth_camera_range = DepthRange(clipping_near, clipping_far) # Use the same clipping for depth
    depth_camera = DepthRenderCamera(depth_camera_core, depth_camera_range)

    scene_center = np.mean(positions, axis=0) if positions and len(positions) > 0 else np.array([0.,0.,0.])

    # Position the sensor for a top-down view
    # The sensor body is at sensor_pose_in_world. The camera itself is identity wrt sensor body.
    sensor_pose_in_world = RigidTransform(
        RollPitchYaw(np.pi/2, 0, np.pi), # X-axis up, Y-axis left, Z-axis forward (camera looks along +Z)
                                         # For top-down: Roll=pi/2 (to point Z down), Yaw can be adjusted
        [scene_center[0], scene_center[1], scene_center[2] + 50.0] # Positioned above the scene center
    )

    sensor = builder.AddSystem(
        RgbdSensor(
            parent_id=scene_graph.world_frame_id(), # Attach sensor to world frame
            X_PB=sensor_pose_in_world,             # Pose of sensor in parent (world)
            color_camera=color_camera,
            depth_camera=depth_camera
        )
    )

    builder.Connect(
        scene_graph.get_query_output_port(),
        sensor.query_object_input_port()
    )

    diagram = builder.Build()
    
    # Create a new context for this diagram (do not reuse from projection)
    diagram_context = diagram.CreateDefaultContext()
    plant_context_render = plant.GetMyContextFromRoot(diagram_context) # If needed for plant specific settings
    # scene_graph_context_render = scene_graph.GetMyContextFromRoot(diagram_context) # If needed

    # Use diagram_context directly since no simulation steps are needed

    # Get sensor context from diagram context to avoid RuntimeError
    sensor_subsystem_context = diagram.GetSubsystemContext(sensor, diagram_context)

    # Eval the output port using the sensor's specific context
    color_image = sensor.color_image_output_port().Eval(sensor_subsystem_context)

    print("Image shape:", color_image.data.shape)
    print("Image dtype:", color_image.data.dtype)

    image_data = color_image.data
    if image_data.ndim == 3 and image_data.shape[2] == 4:
        image_data = image_data[:, :, :3]

    plt.imsave(image_path, image_data)
    print(f"Drake top-down view saved to: {image_path}")

def get_positions_and_scales_from_json(json_path, glb_dir):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get object list (adapt to new format)
    objects = data.get('objects', [])
    
    pos_list = []
    scale_factors_list = []
    uid_list = []
    
    for obj in objects:
        # Convert position from centimeters to meters
        pos_cm = obj['position']
        pos_m = [p / 100.0 for p in pos_cm]  # centimeters to meters
        pos_list.append(pos_m)
        
        size_cm = obj['size']
        # Check selected_uid instead of retrieved_uids
        if not obj.get('selected_uid'):
            print(f"Object {obj.get('instance', '')} has no selected_uid, skip.")
            scale_factors_list.append([1.0, 1.0, 1.0])
            continue
            
        uid = obj['selected_uid']
        # Process uid (if needed)
        base_uid = re.split(r'_x_[-\d\.]+_y_[-\d\.]+', uid)[0]
        base_uid = base_uid.split('_x_')[0].split('_y_')[0]
        uid_list.append(base_uid)
        
        glb_path = os.path.join(glb_dir, f"{base_uid}.glb")
        if not os.path.exists(glb_path):
            print(f"GLB not found: {glb_path}, use scale 1.0")
            scale_factors_list.append([1.0, 1.0, 1.0])
            continue
            
        # Convert target size from centimeters to meters
        size_m = [s / 100.0 for s in size_cm]  # centimeters to meters
        scale_factors, _ = check_scale_factors(glb_path, size_m)
        
        if scale_factors is not None:
            scale_factors_list.append(scale_factors.tolist())
        else:
            scale_factors_list.append([1.0, 1.0, 1.0])
    
    return pos_list, scale_factors_list, uid_list

def batch_optimize_by_position(scene_tensor, scene_vec_desc, batch_size=10):
    N = scene_tensor.shape[0]
    projected_scene = scene_tensor.clone()
    
    # Sort by x coordinate
    x_positions = scene_tensor[:, 0]  # Assuming position is in the 4th column of the tensor
    sorted_indices = torch.argsort(x_positions)
    
    # Batch optimization
    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        batch_indices = sorted_indices[i:end]
        
        # Extract current batch objects
        batch_scene = projected_scene[batch_indices]
        
        # Apply projection to current batch
        batch_projected, cache, success = apply_non_penetration_projection_single_scene(
            scene=batch_scene,
            cache=None,
            scene_vec_desc=scene_vec_desc,
            translation_only=True,
            influence_distance=1,
            solver_name="ipopt",
            iteration_limit=100000,
            tol=1e-6,
            min_distance=0.01,
        )
        
        if success:
            # Update results
            projected_scene[batch_indices] = batch_projected
        else:
            print(f"Batch {i//batch_size + 1} failed, using original positions")
    
    return projected_scene, success

def save_updated_json(original_json_path, updated_positions, output_suffix="_optimized"):
    """
    Save updated JSON file with optimized positions.
    Note: updated_positions are in meters, need to convert back to centimeters.
    Maintains original scene_processed_scene.json format.
    """
    # Read original JSON file
    with open(original_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Update positions (convert meters to centimeters)
    for obj, new_pos_m in zip(data.get('objects', []), updated_positions):
        new_pos_cm = [p * 100.0 for p in new_pos_m]  # meters to centimeters
        obj['position'] = new_pos_cm
    
    # Generate new filename
    base_path = os.path.splitext(original_json_path)[0]
    output_path = f"{base_path}{output_suffix}.json"
    
    # Save updated JSON, maintaining original format
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Optimized scene saved to: {output_path}")
    return output_path

def check_collisions(scene_tensor, scene_vec_desc):
    """Check collisions in the scene"""
    cache, diagram_context, plant_context = create_plant_and_scene_graph_from_scene_with_cache(
        scene=scene_tensor,
        scene_vec_desc=scene_vec_desc,
        cache=None,
    )
    plant = cache.plant
    
    try:
        scene_graph_context = cache.scene_graph.GetMyContextFromRoot(diagram_context)
        query_object = cache.scene_graph.get_query_output_port().Eval(scene_graph_context)
        
        collision_pairs = []
        for i, body_idx1 in enumerate(cache.rigid_body_indices):
            if body_idx1 is None:
                continue
            for j, body_idx2 in enumerate(cache.rigid_body_indices[i+1:], i+1):
                if body_idx2 is None:
                    continue
                
                try:
                    # Get geometry ids for each body
                    body1 = plant.get_body(body_idx1)
                    body2 = plant.get_body(body_idx2)
                    
                    # Get all geometry ids for each body
                    geometry_ids1 = plant.GetCollisionGeometriesForBody(body1)
                    geometry_ids2 = plant.GetCollisionGeometriesForBody(body2)
                    
                    # Check distance between each pair of geometries
                    min_distance = float('inf')
                    for geom_id1 in geometry_ids1:
                        for geom_id2 in geometry_ids2:
                            try:
                                distance_pair = query_object.ComputeSignedDistancePairClosestPoints(
                                    geom_id1, geom_id2
                                )
                                distance = distance_pair.distance
                                min_distance = min(min_distance, distance)
                            except Exception as e:
                                print(f"Error computing distance between geometries {geom_id1} and {geom_id2}: {e}")
                    
                    if min_distance < 0.05:  # 5cm collision detection threshold
                        collision_pairs.append((i, j, min_distance))
                        
                except Exception as e:
                    print(f"Error processing objects {i} and {j}: {e}")
        
        return collision_pairs
        
    except Exception as e:
        print(f"Error in collision detection: {e}")
        return []

def smart_batch_optimize(scene_tensor, scene_vec_desc, batch_size=3):
    """基于碰撞关系的智能批次划分"""
    N = scene_tensor.shape[0]
    projected_scene = scene_tensor.clone()
    
    # 检查初始碰撞关系
    initial_collisions = check_collisions(scene_tensor, scene_vec_desc)
    
    # 构建碰撞图
    collision_graph = {}
    for i in range(N):
        collision_graph[i] = []
    
    for obj1, obj2, _ in initial_collisions:
        collision_graph[obj1].append(obj2)
        collision_graph[obj2].append(obj1)
    
    # 基于碰撞关系分组
    batches = []
    used = set()
    
    for i in range(N):
        if i in used:
            continue
        
        # 找到与物体i有碰撞关系的所有物体
        batch = [i]
        used.add(i)
        
        # 添加与当前批次中物体有碰撞的物体
        for obj in batch:
            for neighbor in collision_graph[obj]:
                if neighbor not in used and len(batch) < batch_size:
                    batch.append(neighbor)
                    used.add(neighbor)
        
        batches.append(batch)
    
    # 按批次优化
    for batch_idx, batch in enumerate(batches):
        print(f"Optimizing batch {batch_idx + 1}: {batch}")
        
        batch_scene = projected_scene[batch]
        batch_projected, cache, success = apply_non_penetration_projection_single_scene(
            scene=batch_scene,
            cache=None,
            scene_vec_desc=scene_vec_desc,
            translation_only=True,
            influence_distance=0.3,
            solver_name="ipopt",
            iteration_limit=50000
        )
        
        if success:
            projected_scene[batch] = batch_projected
    
    return projected_scene, success

def apply_non_penetration_projection_single_scene_with_constraints(
    scene: torch.Tensor,
    cache: Union[PlantSceneGraphCache, None],
    scene_vec_desc: SceneVecDescription,
    translation_only: bool,
    influence_distance: float = 0.5,
    solver_name: str = "snopt",
    iteration_limit: int = 100000,
    return_cache: bool = True,
    tol: float = 1e-8,
    movement_constraints: Dict[int, Dict] = None,
    min_distance: float = 0.01,
) -> Tuple[torch.Tensor, Union[PlantSceneGraphCache, None], bool]:
    cache, _, plant_context = create_plant_and_scene_graph_from_scene_with_cache(
        scene=scene,
        scene_vec_desc=scene_vec_desc,
        cache=cache,
    )
    plant = cache.plant

    # Set up projection NLP.
    ik = InverseKinematics(plant, plant_context)
    q_vars = ik.q()
    prog = ik.prog()

    # Stay close to initial positions.
    q0 = plant.GetPositions(plant_context)
    if len(q0) == 0:
        return scene, cache, False

    for body_idx, model_idx in zip(cache.rigid_body_indices, cache.model_indices):
        if body_idx is None:
            continue

        body = plant.get_body(body_idx)
        if not body.is_floating():
            continue

        # For two quaternion z and z0 with angle θ between their orientation,
        q_start_idx = body.floating_positions_start()
        model_quat_vars = q_vars[q_start_idx : q_start_idx + 4]
        z0 = q0[q_start_idx : q_start_idx + 4]
        prog.AddQuadraticCost(
            -4 * np.outer(z0, z0), np.zeros((4,)), 2, model_quat_vars, is_convex=False
        )
        model_pos_vars = q_vars[q_start_idx + 4 : q_start_idx + 7]
        prog.AddQuadraticErrorCost(
            np.eye(3), q0[q_start_idx + 4 : q_start_idx + 7], model_pos_vars
        )

    # Nonpenetration constraint.
    ik.AddMinimumDistanceLowerBoundConstraint(min_distance, influence_distance)  # 5cm minimum distance

    if translation_only:
        # Add constraints for rotations to stay constant.
        for body_idx, model_idx in zip(cache.rigid_body_indices, cache.model_indices):
            if body_idx is None:
                continue

            body = plant.get_body(body_idx)
            if not body.is_floating():
                continue

            q_start_idx = body.floating_positions_start()
            model_quat_vars = q_vars[q_start_idx : q_start_idx + 4]
            model_q = plant.GetPositions(plant_context, model_idx)
            model_quat = model_q[:4]

            prog.AddBoundingBoxConstraint(
                model_quat, model_quat, model_quat_vars,
            )

    # Add movement direction constraints and boundaries
    if movement_constraints is not None:
        for obj_idx, constraints in movement_constraints.items():
            if obj_idx < len(cache.rigid_body_indices):
                body_idx = cache.rigid_body_indices[obj_idx]
                if body_idx is not None:
                    body = plant.get_body(body_idx)
                    if body.is_floating():
                        q_start_idx = body.floating_positions_start()
                        initial_pos = q0[q_start_idx + 4:q_start_idx + 7] 
                        
                        # Get boundaries for this object
                        boundaries = constraints.get("boundaries", {})
                        
                        # x direction constraint
                        if "x" in constraints:
                            x_pos_var = q_vars[q_start_idx + 4]
                            if constraints["x"] == "negative":
                                x_max = boundaries.get("x_max", initial_pos[0])
                                prog.AddBoundingBoxConstraint(-np.inf, x_max, x_pos_var)
                            elif constraints["x"] == "positive":
                                x_min = boundaries.get("x_min", initial_pos[0])
                                prog.AddBoundingBoxConstraint(x_min, np.inf, x_pos_var)
                            elif constraints["x"] == "fixed":
                                prog.AddBoundingBoxConstraint(initial_pos[0], initial_pos[0], x_pos_var)
                            else:  # "free"
                                x_min = boundaries.get("x_min", -np.inf)
                                x_max = boundaries.get("x_max", np.inf)
                                prog.AddBoundingBoxConstraint(x_min, x_max, x_pos_var)
                        
                        # y direction constraint
                        if "y" in constraints:
                            y_pos_var = q_vars[q_start_idx + 5]
                            if constraints["y"] == "negative":
                                y_max = boundaries.get("y_max", initial_pos[1])
                                prog.AddBoundingBoxConstraint(-np.inf, y_max, y_pos_var)
                            elif constraints["y"] == "positive":
                                y_min = boundaries.get("y_min", initial_pos[1])
                                prog.AddBoundingBoxConstraint(y_min, np.inf, y_pos_var)
                            elif constraints["y"] == "fixed":
                                prog.AddBoundingBoxConstraint(initial_pos[1], initial_pos[1], y_pos_var)
                            else:  # "free"
                                y_min = boundaries.get("y_min", -np.inf)
                                y_max = boundaries.get("y_max", np.inf)
                                prog.AddBoundingBoxConstraint(y_min, y_max, y_pos_var)
                        
                        # z direction constraint
                        if "z" in constraints:
                            z_pos_var = q_vars[q_start_idx + 6]
                            if constraints["z"] == "negative":
                                z_max = boundaries.get("z_max", initial_pos[2])
                                if z_max is not None:
                                    prog.AddBoundingBoxConstraint(-np.inf, z_max, z_pos_var)
                                else:
                                    prog.AddBoundingBoxConstraint(-np.inf, initial_pos[2], z_pos_var)
                            elif constraints["z"] == "positive":
                                z_min = boundaries.get("z_min", initial_pos[2])
                                prog.AddBoundingBoxConstraint(z_min, np.inf, z_pos_var)
                            elif constraints["z"] == "fixed":
                                prog.AddBoundingBoxConstraint(initial_pos[2], initial_pos[2], z_pos_var)
                            else:  # "free"
                                z_min = boundaries.get("z_min", -np.inf)
                                z_max = boundaries.get("z_max", np.inf)
                                if z_max is not None:
                                    prog.AddBoundingBoxConstraint(z_min, z_max, z_pos_var)
                                else:
                                    max_height = initial_pos[2] + 0.25  
                                    prog.AddBoundingBoxConstraint(initial_pos[2], max_height, z_pos_var)
                                    print(f"Object {obj_idx}: z-axis constrained to max height {max_height:.2f}m (initial: {initial_pos[2]:.2f}m)")

    # Use the starting positions as the initial guess.
    prog.SetInitialGuess(q_vars, q0)

    # Solve.
    options = SolverOptions()
    if solver_name == "snopt":
        solver = SnoptSolver()
        options.SetOption(solver.id(), "Major feasibility tolerance", 1e-3)
        options.SetOption(solver.id(), "Major optimality tolerance", 1e-3)
        options.SetOption(solver.id(), "Major iterations limit", iteration_limit)
        options.SetOption(solver.id(), "Time limit", 60)
        options.SetOption(solver.id(), "Timing level", 3)
    elif solver_name == "ipopt":
        solver = IpoptSolver()
        options.SetOption(solver.id(), "max_iter", iteration_limit)
        options.SetOption(solver.id(), "tol", tol)
    else:
        raise ValueError(f"Invalid solver: {solver_name}")
    if not solver.available():
        raise ValueError(f"Solver {solver_name} is not available.")

    try:
        result = solver.Solve(prog, None, options)
        success = result.is_success()

        plant.SetPositions(plant_context, result.GetSolution(q_vars))
        projected_scene = update_scene_poses_from_plant(
            scene=scene,
            plant=plant,
            plant_context=plant_context,
            model_indices=cache.model_indices,
            scene_vec_desc=scene_vec_desc,
        )
    except Exception as e:
        projected_scene = scene
        success = False

    return projected_scene, (cache if return_cache else None), success

def batch_optimize_by_position_with_constraints(
    scene_tensor, 
    scene_vec_desc, 
    batch_size=3,
    movement_constraints=None,  # Use complete constraints dictionary
    min_distance: float = 0.01
):
    N = scene_tensor.shape[0]
    projected_scene = scene_tensor.clone()
    
    # Sort by x coordinate
    x_positions = scene_tensor[:, 0]
    sorted_indices = torch.argsort(x_positions)
    
    # Batch optimization
    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        batch_indices = sorted_indices[i:end]
        
        # Extract current batch objects
        batch_scene = projected_scene[batch_indices]
        
        # Determine which objects in current batch need constraints
        batch_constraints = {}
        if movement_constraints is not None:
            for global_obj_idx, constraints in movement_constraints.items():
                if global_obj_idx in batch_indices:
                    # Convert global index to batch-local index
                    batch_local_idx = list(batch_indices).index(global_obj_idx)
                    # Use complete constraint configuration
                    batch_constraints[batch_local_idx] = constraints
        
        # Apply projection to current batch
        batch_projected, cache, success = apply_non_penetration_projection_single_scene_with_constraints(
            scene=batch_scene,
            cache=None,
            scene_vec_desc=scene_vec_desc,
            translation_only=True,
            influence_distance=0.2,
            solver_name="ipopt",
            iteration_limit=50000,
            tol=1e-6,
            movement_constraints=batch_constraints,
            min_distance=min_distance
        )
        
        if success:
            projected_scene[batch_indices] = batch_projected
            print(f"Batch {i//batch_size + 1} successful")
        else:
            print(f"Batch {i//batch_size + 1} failed, using original positions")
    
    return projected_scene, success

def sliding_window_optimize(
    scene_tensor, 
    scene_vec_desc, 
    window_size=3,
    step_size=1,  # Slide step
    movement_constraints=None,
    min_distance: float = 0.01
):
    """
    Optimize using sliding window, wrapping around to form a complete cycle
    
    Args:
        scene_tensor: Scene tensor
        scene_vec_desc: Scene vector description
        window_size: Window size (number of objects to optimize at a time)
        step_size: Slide step (how many objects to move at a time)
        movement_constraints: Movement constraints dictionary
    """
    N = scene_tensor.shape[0]
    projected_scene = scene_tensor.clone()
    
    # Sort by x coordinate
    x_positions = scene_tensor[:, 0]
    sorted_indices = torch.argsort(x_positions)
    
    print(f"Total objects: {N}, Window size: {window_size}, Step size: {step_size}")
    
    # Calculate how many windows are needed to cover all objects (including wrap-around)
    # To ensure each object can be optimized with its neighbors, we need more windows
    total_windows = N + window_size - 1  # Ensure all possible neighbor combinations are covered
    
    window_count = 0
    for window_idx in range(total_windows):
        # Calculate the start index of the current window (allowing wrap-around)
        start_idx = window_idx % N
        
        # Build the index list for the current window (including wrap-around)
        window_indices = []
        for i in range(window_size):
            idx = (start_idx + i) % N
            window_indices.append(sorted_indices[idx])
        
        window_count += 1
        
        print(f"Window {window_count}: optimizing objects {window_indices}")
        
        # Extract objects for the current window
        window_scene = projected_scene[torch.tensor(window_indices)]
        
        # Determine which objects in the current window need constraints
        window_constraints = {}
        if movement_constraints is not None:
            for global_obj_idx, constraints in movement_constraints.items():
                if global_obj_idx in window_indices:
                    # Convert global index to window-local index
                    window_local_idx = list(window_indices).index(global_obj_idx)
                    window_constraints[window_local_idx] = constraints
        
        # Apply projection to the current window
        window_projected, cache, success = apply_non_penetration_projection_single_scene_with_constraints(
            scene=window_scene,
            cache=None,
            scene_vec_desc=scene_vec_desc,
            translation_only=True,
            influence_distance=0.1,
            solver_name="ipopt",
            iteration_limit=50000,
            tol=1e-6,
            movement_constraints=window_constraints,
            min_distance=min_distance
        )
        
        if success:
            projected_scene[torch.tensor(window_indices)] = window_projected
            print(f"Window {window_count} successful")
        else:
            print(f"Window {window_count} failed, using original positions")
    
    return projected_scene, True

def adaptive_sliding_window_optimize(
    scene_tensor, 
    scene_vec_desc, 
    min_window_size=2,
    max_window_size=5,
    movement_constraints=None,
    max_iterations=3,
    min_distance: float = 0.01
):
    """
    Adaptive sliding window optimization - Adjusts window size based on collisions, supports wrap-around
    
    Args:
        scene_tensor: Scene tensor
        scene_vec_desc: Scene vector description
        min_window_size: Minimum window size
        max_window_size: Maximum window size
        movement_constraints: Movement constraints dictionary
        max_iterations: Maximum number of iterations
    """
    N = scene_tensor.shape[0]
    projected_scene = scene_tensor.clone()
    
    # Sort by x coordinate
    x_positions = scene_tensor[:, 0]
    sorted_indices = torch.argsort(x_positions)
    
    print(f"Adaptive sliding window optimization: {N} objects")
    
    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")
        
        # Check current collisions
        current_collisions = check_collisions(projected_scene, scene_vec_desc)
        print(f"Current collisions: {len(current_collisions)}")
        
        if len(current_collisions) == 0:
            print("No collisions remaining, optimization complete!")
            break
        
        # Build collision graph
        collision_graph = {}
        for i in range(N):
            collision_graph[i] = []
        
        for obj1, obj2, _ in current_collisions:
            collision_graph[obj1].append(obj2)
            collision_graph[obj2].append(obj1)
        
        # Use sliding window optimization
        window_size = min(max_window_size, max(min_window_size, len(current_collisions) // 2 + 1))
        step_size = max(1, window_size // 2)  # Step is half the window size to ensure overlap
        
        print(f"Using window_size={window_size}, step_size={step_size}")
        
        # Calculate how many windows are needed to cover all objects (including wrap-around)
        total_windows = N + window_size - 1
        
        # Sliding window optimization (supports wrap-around)
        for window_idx in range(total_windows):
            # Calculate the start index of the current window (allowing wrap-around)
            start_idx = window_idx % N
            
            # Build the index list for the current window (including wrap-around)
            window_indices = []
            for i in range(window_size):
                idx = (start_idx + i) % N
                window_indices.append(sorted_indices[idx])
            
            # Check if the current window has collisions
            window_has_collision = False
            for obj1, obj2, _ in current_collisions:
                if obj1 in window_indices and obj2 in window_indices:
                    window_has_collision = True
                    break
            
            if not window_has_collision:
                continue  # Skip windows with no collisions
            
            print(f"Optimizing window: {window_indices}")
            
            # Extract objects for the current window
            window_scene = projected_scene[torch.tensor(window_indices)]
            
            # Determine which objects in the current window need constraints
            window_constraints = {}
            if movement_constraints is not None:
                for global_obj_idx, constraints in movement_constraints.items():
                    if global_obj_idx in window_indices:
                        window_local_idx = list(window_indices).index(global_obj_idx)
                        window_constraints[window_local_idx] = constraints
            
            # Apply projection to the current window
            window_projected, cache, success = apply_non_penetration_projection_single_scene_with_constraints(
                scene=window_scene,
                cache=None,
                scene_vec_desc=scene_vec_desc,
                translation_only=True,
                influence_distance=0.2,
                solver_name="ipopt",
                iteration_limit=50000,
                tol=1e-6,
                movement_constraints=window_constraints,
                min_distance=min_distance
            )
            
            if success:
                projected_scene[torch.tensor(window_indices)] = window_projected
                print(f"Window optimization successful")
            else:
                print(f"Window optimization failed")
        
        # Check optimization effect
        new_collisions = check_collisions(projected_scene, scene_vec_desc)
        improvement = len(current_collisions) - len(new_collisions)
        print(f"Collision reduction: {improvement}")
        
        if improvement == 0:
            print("No improvement in this iteration, stopping")
            break
    
    return projected_scene, True

def circular_sliding_window_optimize(
    scene_tensor, 
    scene_vec_desc, 
    window_size=3,
    movement_constraints=None,
    max_rounds=5,
    convergence_threshold=0,
    min_distance: float = 0.01
):
    """
    Circular sliding window optimization - Multiple rounds until convergence
    
    Args:
        scene_tensor: Scene tensor
        scene_vec_desc: Scene description
        window_size: Window size (number of objects to optimize at a time)
        movement_constraints: Movement constraints dictionary
        max_rounds: Maximum optimization rounds
        convergence_threshold: Convergence threshold (stop when collision count no longer decreases)
    """
    N = scene_tensor.shape[0]
    projected_scene = scene_tensor.clone()
    
    # Sort by x coordinate
    x_positions = scene_tensor[:, 0]
    sorted_indices = torch.argsort(x_positions)
    
    print(f"Circular sliding window optimization: {N} objects, window_size={window_size}")
    
    # Check initial collisions
    initial_collisions = check_collisions(projected_scene, scene_vec_desc)
    print(f"Initial collisions: {len(initial_collisions)}")
    
    previous_collision_count = len(initial_collisions)
    
    for round_num in range(max_rounds):
        print(f"\n=== Round {round_num + 1} ===")
        
        # Calculate how many windows are needed to cover all objects (including wrap-around)
        total_windows = N + window_size - 1
        
        # Circular sliding window optimization
        for window_idx in range(total_windows):
            # Calculate the start index of the current window (allowing wrap-around)
            start_idx = window_idx % N
            
            # Build the index list for the current window (including wrap-around)
            window_indices = []
            for i in range(window_size):
                idx = (start_idx + i) % N
                window_indices.append(sorted_indices[idx])
            
            # Extract objects for the current window
            window_scene = projected_scene[torch.tensor(window_indices)]
            
            # Determine which objects in the current window need constraints (apply global constraints to each window)
            window_constraints = {}
            if movement_constraints is not None:
                for global_obj_idx, constraints in movement_constraints.items():
                    if global_obj_idx in window_indices:
                        window_local_idx = list(window_indices).index(global_obj_idx)
                        window_constraints[window_local_idx] = constraints
                        print(f"Window {window_idx}: Added movement constraints for object {global_obj_idx} (local index {window_local_idx}): {constraints}")
            
            # Debug information: show constraints for the current window
            if window_constraints:
                print(f"Window {window_idx} constraints: {window_constraints}")
                print(f"Window {window_idx} indices: {window_indices}")
            
            # Apply projection to the current window
            window_projected, cache, success = apply_non_penetration_projection_single_scene_with_constraints(
                scene=window_scene,
                cache=None,
                scene_vec_desc=scene_vec_desc,
                translation_only=True,
                influence_distance=0.2,
                solver_name="ipopt",
                iteration_limit=50000,
                tol=1e-6,
                movement_constraints=window_constraints,
                min_distance=min_distance
            )
            
            if success:
                projected_scene[torch.tensor(window_indices)] = window_projected
        
        # Check optimization effect for this round
        current_collisions = check_collisions(projected_scene, scene_vec_desc)
        current_collision_count = len(current_collisions)
        improvement = previous_collision_count - current_collision_count
        
        print(f"Round {round_num + 1} complete: {current_collision_count} collisions (improvement: {improvement})")
        
        # Check for convergence
        if current_collision_count <= convergence_threshold:
            print(f"Converged! Collisions reduced to {current_collision_count}")
            break
        
        # if improvement <= 0:
        #     print(f"No improvement in round {round_num + 1}, stopping")
        #     break
        
        previous_collision_count = current_collision_count
    
    final_collisions = check_collisions(projected_scene, scene_vec_desc)
    print(f"\nFinal result: {len(final_collisions)} collisions remaining")
    
    return projected_scene, len(final_collisions) == 0

def verify_units(scene_tensor, pos_list):
    """Verify unit conversion is correct"""
    print("=== Unit Verification ===")
    print("Original positions (cm):")
    for i, pos_cm in enumerate(pos_list):
        print(f"  Object {i}: {pos_cm}")
    
    print("\nScene tensor positions (should be meters):")
    for i in range(scene_tensor.shape[0]):
        pos_m = scene_tensor[i][:3].tolist()
        print(f"  Object {i}: {pos_m}")
    
    print("\nDistance checks (should be reasonable meters):")
    for i in range(scene_tensor.shape[0]):
        for j in range(i+1, scene_tensor.shape[0]):
            pos1 = scene_tensor[i][:3]
            pos2 = scene_tensor[j][:3]
            dist = torch.norm(pos1 - pos2).item()
            print(f"  Objects {i}-{j}: {dist:.3f}m ({dist*100:.1f}cm)")
    
    print("=== End Verification ===\n")

def set_constraints(scene_layout_txt_path: str, json_path: str = None) -> Dict[int, Dict[str, str]]:
    import re
    import json
    
    with open(scene_layout_txt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    scene_graph_match = re.search(r'Scene Graph:(.*?)(?=Therefore, the scene layout is:|$)', content, re.DOTALL)
    if not scene_graph_match:
        print("Warning: Could not find Scene Graph section")
        return {}
    
    scene_graph_text = scene_graph_match.group(1)

    relations = []
    lines = scene_graph_text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('(') and line.endswith(')'):

            parts = line[1:-1].split(',')
            if len(parts) == 3:
                obj1 = parts[0].strip()
                relation = parts[1].strip()
                obj2 = parts[2].strip()
                relations.append((obj1, relation, obj2))
    
    instances = []
    placement_zone = None
    
    if json_path and os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        objects = json_data.get('objects', [])
        instances = [obj['instance'] for obj in objects]
        
        placement_zone = json_data.get('item_placement_zone', [0, 60, 0, 40])
        print(f"Placement zone: {placement_zone}")
    else:
        print("Warning: JSON file not found, using default instances")
        instances = ["books_0", "stack_0", "reading_lamp_0", "plant_0", "glasses_0"]
        placement_zone = [0, 60, 0, 40]  
    
    instance_to_index = {}
    for i, instance in enumerate(instances):
        instance_to_index[instance] = i
    
    num_objects = len(instances)
    movement_constraints = {}  

    if len(placement_zone) >= 8:
        x_min = (placement_zone[0] / 100.0) + 0.02
        x_max = (placement_zone[1] / 100.0) - 0.02
        y_min = (placement_zone[2] / 100.0) + 0.02
        y_max = (placement_zone[3] / 100.0) - 0.02
    else:
        x_min, x_max, y_min, y_max = 0.0, 1, 0.0, 1  
    

    z_min = 0.0

    for i in range(num_objects):
        movement_constraints[i] = {
            "x": "free",
            "y": "free", 
            "z": "fixed", 
            "boundaries": {
                "x_min": x_min, "x_max": x_max,
                "y_min": y_min, "y_max": y_max,
                "z_min": z_min, "z_max": None  
            }
        }
    
    z_free_objects = set()
    xy_fixed_objects = set()
    
    for obj1, relation, obj2 in relations:
        if relation in ["above", "below", "in"]:
            if relation in ["above", "below"]:
                if obj1 in instance_to_index:
                    z_free_objects.add(obj1)
                    xy_fixed_objects.add(obj1)
                if obj2 in instance_to_index:
                    z_free_objects.add(obj2)
                    xy_fixed_objects.add(obj2)
            elif relation == "in":
                if obj1 in instance_to_index:
                    z_free_objects.add(obj1)
                    print(f"Object {obj1} is 'in' {obj2}, z-axis constraint released")
    
    for instance in z_free_objects:
        if instance in instance_to_index:
            obj_index = instance_to_index[instance]
            movement_constraints[obj_index]["z"] = "free"
            print(f"Object {instance} (index {obj_index}) has special relation, z-axis constraint released")
    
    for instance in xy_fixed_objects:
        if instance in instance_to_index:
            obj_index = instance_to_index[instance]
            movement_constraints[obj_index]["x"] = "fixed"
            movement_constraints[obj_index]["y"] = "fixed"
            print(f"Object {instance} (index {obj_index}) has above/below relation, x,y-axis constraints fixed")
    
    return movement_constraints


def optimize_scene_from_json(
    floder_path: str,
    config_path: str = "../../config.yaml", 
    movement_constraints: Optional[Dict] = None,
) -> Tuple[bool, str]:

    try:
        # Load configuration
        config = load_config(config_path)
        drake_config = config.get('drake', {})
        
        # Get parameters from config
        glb_dir = drake_config.get('glb_dir')
        original_urdf = drake_config.get('original_urdf')
        model_package_path = drake_config.get('model_package_path')
        window_size = drake_config.get('window_size', 6)
        max_rounds = drake_config.get('max_rounds', 3)
        convergence_threshold = drake_config.get('convergence_threshold', 0)
        output_suffix = drake_config.get('output_suffix', "_optimized")
        min_distance = drake_config.get('min_distance', 0.01)
        json_path = os.path.join(floder_path, "scene_processed_scene.json")
        sg_path = os.path.join(floder_path, "scene_layout.txt")

        movement_constraints = set_constraints(sg_path, json_path)

        # Create Drake package mapping
        drake_package_map = PackageMap()
        drake_package_map.Add("models", model_package_path)

        # Get positions, scales and UIDs from JSON
        pos_list, scale_factors_list, uid_list = get_positions_and_scales_from_json(json_path, glb_dir)

        # Generate scaled URDF and build scene data
        json_objects = []
        for i, (scale, uid) in enumerate(zip(scale_factors_list, uid_list)):
            urdf_path = generate_scaled_urdf(original_urdf, scale, tag=f"{i}", uid=uid)
            json_objects.append({
                "model_path": urdf_path,
                "position": pos_list[i], 
                "rotation": [1.0, 0.0, 0.0, 0.0],
                "is_welded": False
            })
        json_data = {"objects": json_objects}

        # Create scene description and convert to tensor
        scene_desc = SceneVecDescription.from_json(json_data, drake_package_map)
        scene_tensor = scene_desc.json_to_scene_tensor(json_data)

        # Check initial collisions
        initial_collisions = check_collisions(scene_tensor, scene_desc)
        print(f"Initial collisions: {len(initial_collisions)}")

        # Use circular sliding window optimization
        projected_scene, success = circular_sliding_window_optimize(
            scene_tensor, 
            scene_desc, 
            window_size=window_size,
            movement_constraints=movement_constraints or {},
            max_rounds=max_rounds,
            convergence_threshold=convergence_threshold,
            min_distance=min_distance
        )

        # Check final collisions
        final_collisions = check_collisions(projected_scene, scene_desc)
        print(f"Collisions after optimization: {len(final_collisions)}")

        # Save optimization results
        if success or len(final_collisions) < len(initial_collisions):
            result_json = scene_desc.scene_tensor_to_json(projected_scene)
            updated_positions = [obj['position'] for obj in result_json["objects"]]
            optimized_json_path = save_updated_json(json_path, updated_positions, output_suffix)
            return (len(final_collisions) == 0, optimized_json_path)
        else:
            return (False, "")

    except Exception as e:
        print(f"Optimization error: {str(e)}")
        import traceback
        traceback.print_exc()
        return (False, "")


# Interface call example
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--floder_path", type=str, required=True,
                       help="Path to scene JSON file")
    parser.add_argument("--config", type=str, default="../../config.yaml",
                       help="Path to config.yaml file (default: ../../config.yaml)")
    args = parser.parse_args()
    
    success, optimized_path = optimize_scene_from_json(
        floder_path=args.floder_path,
        config_path=args.config
    )
    
    if success and optimized_path:
        print(f"Optimization successful! Results saved to: {optimized_path}")
    else:
        print("Optimization failed or no better solution found")


    