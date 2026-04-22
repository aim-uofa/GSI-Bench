import os
import trimesh
import numpy as np
from pathlib import Path
import re
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm

# Add thread lock for printing
print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    """Thread-safe print function"""
    with print_lock:
        print(*args, **kwargs)

def glb_to_obj(glb_path: str, output_dir: str = None) -> str:
    """
    Convert a GLB file to OBJ format, attempting to ensure normals are present.
    """
    try:
        # Load the GLB file
        loaded_asset = trimesh.load(glb_path, force='scene' if glb_path.lower().endswith('.glb') else None)
        
        # Get output directory
        if output_dir is None:
            output_dir = os.path.dirname(glb_path)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output path with .obj extension
        base_name = Path(glb_path).stem
        obj_path = os.path.join(output_dir, f"{base_name}.obj")

        export_mesh = None
        if isinstance(loaded_asset, trimesh.Scene):
            if len(loaded_asset.geometry) > 0:
                export_mesh = trimesh.util.concatenate(
                    [geom for geom in loaded_asset.geometry.values() if isinstance(geom, trimesh.Trimesh)]
                )
            else:
                safe_print(f"Warning: Scene from '{glb_path}' contains no mesh geometry to export.")
                return None
        elif isinstance(loaded_asset, trimesh.Trimesh):
            export_mesh = loaded_asset
        else:
            safe_print(f"Warning: Loaded '{glb_path}' as an unsupported type: {type(loaded_asset)}")
            return None

        if export_mesh:
            # Check and fix normals
            if len(export_mesh.vertex_normals) != len(export_mesh.vertices) or np.all(export_mesh.vertex_normals == 0):
                export_mesh.fix_normals()
            
            # Export as OBJ
            export_mesh.apply_translation(-export_mesh.centroid)
            export_mesh.export(obj_path)
            safe_print(f"✓ Successfully exported {os.path.basename(glb_path)} -> {os.path.basename(obj_path)}")
            return obj_path
        else:
            safe_print(f"No valid mesh to export from {glb_path}")
            return None
            
    except Exception as e:
        safe_print(f"✗ Error converting {glb_path}: {e}")
        return None

def batch_convert_glb_to_obj_parallel(glb_dir: str, output_dir: str = None, max_workers: int = 8) -> list:
    """
    Convert GLB files to OBJ format in parallel using multiple threads
    
    Args:
        glb_dir (str): Directory containing GLB files
        output_dir (str, optional): Directory to save OBJ files
        max_workers (int): Maximum number of threads
        
    Returns:
        list: List of generated OBJ file paths
    """
    if output_dir is None:
        output_dir = glb_dir
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all GLB files for conversion
    glb_files = []
    for file in os.listdir(glb_dir):
        if file.lower().endswith('.glb'):
            glb_path = os.path.join(glb_dir, file)
            base_name = Path(file).stem
            obj_path = os.path.join(output_dir, f"{base_name}.obj")
            
            # Check if OBJ file already exists
            if os.path.exists(obj_path):
                safe_print(f"Skipping existing file: {os.path.basename(obj_path)}")
                continue
                
            glb_files.append((glb_path, output_dir))
    
    if not glb_files:
        safe_print("No files to convert")
        return []
    
    safe_print(f"Starting conversion of {len(glb_files)} files using {max_workers} threads...")
    
    converted_files = []
    failed_files = []
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(glb_to_obj, glb_path, output_dir): glb_path 
                         for glb_path, output_dir in glb_files}
        
        # Show progress with tqdm
        with tqdm(total=len(glb_files), desc="Conversion progress") as pbar:
            for future in as_completed(future_to_file):
                glb_path = future_to_file[future]
                try:
                    obj_path = future.result()
                    if obj_path:
                        converted_files.append(obj_path)
                    else:
                        failed_files.append(glb_path)
                except Exception as e:
                    safe_print(f"✗ Conversion failed for {os.path.basename(glb_path)}: {e}")
                    failed_files.append(glb_path)
                
                pbar.update(1)
    
    safe_print(f"\nConversion completed!")
    safe_print(f"Successfully converted: {len(converted_files)} files")
    safe_print(f"Failed conversions: {len(failed_files)} files")
    
    if failed_files:
        safe_print("Failed files:")
        for file in failed_files:
            safe_print(f"  {os.path.basename(file)}")
    
    return converted_files

def convert_json_assets_to_obj_parallel(json_path, glb_dir, obj_dir, max_workers: int = 8):
    """
    Convert GLB files listed in JSON file to OBJ format in parallel
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        objects = json.load(f)
    
    os.makedirs(obj_dir, exist_ok=True)
    
    # Collect files for conversion
    conversion_tasks = []
    for obj in objects:
        if not obj.get("retrieved_uids"):
            continue
        uid = obj["retrieved_uids"][0]
        # Remove _x_... and _y_... suffixes
        base_uid = re.split(r'_x_[-\d\.]+_y_[-\d\.]+', uid)[0]
        base_uid = base_uid.split('_x_')[0].split('_y_')[0]
        glb_path = os.path.join(glb_dir, f"{base_uid}.glb")
        
        if not os.path.exists(glb_path):
            safe_print(f"GLB not found: {glb_path}")
            continue
            
        obj_path = os.path.join(obj_dir, f"{base_uid}.obj")
        if os.path.exists(obj_path):
            safe_print(f"Skipping existing file: {os.path.basename(obj_path)}")
            continue
            
        conversion_tasks.append((glb_path, obj_dir))
    
    if not conversion_tasks:
        safe_print("No files to convert")
        return []
    
    safe_print(f"Starting conversion of {len(conversion_tasks)} files using {max_workers} threads...")
    
    converted_files = []
    failed_files = []
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(glb_to_obj, glb_path, obj_dir): glb_path 
                         for glb_path, obj_dir in conversion_tasks}
        
        with tqdm(total=len(conversion_tasks), desc="Conversion progress") as pbar:
            for future in as_completed(future_to_file):
                glb_path = future_to_file[future]
                try:
                    obj_path = future.result()
                    if obj_path:
                        converted_files.append(obj_path)
                    else:
                        failed_files.append(glb_path)
                except Exception as e:
                    safe_print(f"✗ Conversion failed for {os.path.basename(glb_path)}: {e}")
                    failed_files.append(glb_path)
                
                pbar.update(1)
    
    safe_print(f"\nConversion completed!")
    safe_print(f"Successfully converted: {len(converted_files)} files")
    safe_print(f"Failed conversions: {len(failed_files)} files")
    
    return converted_files

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-threaded GLB to OBJ conversion tool')
    parser.add_argument('--glb_dir', type=str, 
                       default="path/to/Assets_library",
                       help='Directory containing GLB files')
    parser.add_argument('--obj_dir', type=str, 
                       default="path/to/Assets_library_obj",
                       help='Output directory for OBJ files')
    parser.add_argument('--json_path', type=str, default=None,
                       help='JSON file path (if specified, read files to convert from JSON)')
    parser.add_argument('--max_workers', type=int, default=16,
                       help='Maximum number of threads')
    
    args = parser.parse_args()
    
    if args.json_path:
        # Convert from JSON file
        converted_files = convert_json_assets_to_obj_parallel(
            args.json_path, args.glb_dir, args.obj_dir, args.max_workers
        )
    else:
        # Convert all GLB files in directory
        converted_files = batch_convert_glb_to_obj_parallel(
            args.glb_dir, args.obj_dir, args.max_workers
        )
    
    print(f"\nConversion completed! Converted {len(converted_files)} files")

