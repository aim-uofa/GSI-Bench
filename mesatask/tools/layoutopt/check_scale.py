import trimesh
import numpy as np

def get_mesh_dimensions(mesh):
    bounds = mesh.bounds  # shape: (2, 3)
    min_corner, max_corner = bounds
    size = max_corner - min_corner
    return size

def check_scale_factors(mesh_path, target_size):
    """
    Calculate actual mesh dimensions and scale factors
    Args:
        mesh_path (str): Path to mesh file
        target_size (list/tuple): Target dimensions [x, y, z]
    Returns:
        (scale_factors, initial_dimensions)
    """
    try:
        mesh = trimesh.load(mesh_path, force='mesh')
    except Exception as e:
        print(f"Failed to load {mesh_path}: {e}")
        return None, None

    initial_dimensions = get_mesh_dimensions(mesh)
    # Check dimension validity
    if np.any(initial_dimensions < 1e-6):
        print(f"Warning: Model {mesh_path} has invalid or too small initial dimensions: {initial_dimensions}")
        return None, initial_dimensions

    target_dimensions = np.array(target_size)
    scale_factors = target_dimensions / initial_dimensions
    return scale_factors, initial_dimensions

def check_multiple_models(model_paths_and_sizes):
    results = {}
    for mesh_path, target_size in model_paths_and_sizes:
        scale_factors, initial_dims = check_scale_factors(mesh_path, target_size)
        results[mesh_path] = {
            'target_size': target_size,
            'initial_dimensions': initial_dims.tolist() if initial_dims is not None else None,
            'scale_factors': scale_factors.tolist() if scale_factors is not None else None,
            'success': scale_factors is not None
        }
        print(f"Model path: {mesh_path}")
        print(f"Scale factors: {scale_factors}")
    return results
        

