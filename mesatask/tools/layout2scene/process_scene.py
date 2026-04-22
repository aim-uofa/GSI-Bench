import sys
import os
import time
import argparse
import json
import traceback
import yaml

from .json2img import initialize_pipeline, process_scene_data


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def validate_config_paths(config: dict) -> bool:
    """Validate that all required paths exist"""
    required_paths = [
        'annotations_path',
        'sbert_features_path',
        'model_base_path',
        'blender_executable'
    ]
    
    all_valid = True
    for path_key in required_paths:
        path = config.get(path_key)
        if path and not os.path.exists(path):
            print(f"Warning: Path not found - {path_key}: {path}")
            all_valid = False
    
    return all_valid


def extract_json_from_response(response_str: str) -> dict:
    """Extract and parse JSON from MesaTask response string"""
    # Find the final answer marker
    layout_start_marker = "\nTherefore, the scene layout is:\n**Final Answer**\n"
    layout_index = response_str.find(layout_start_marker)
    
    if layout_index == -1:
        print("Error: Could not find layout start marker.")
        return None
        
    layout_json_str = response_str[layout_index + len(layout_start_marker):].strip()
    
    # Remove trailing markers
    if layout_json_str.endswith("<|im_end|>"):
        layout_json_str = layout_json_str[:-len("<|im_end|>")].strip()

    # Clean up JSON string
    if layout_json_str.endswith("```"):
        layout_json_str = layout_json_str[:-3].strip()
    if layout_json_str.startswith("```json"):
       layout_json_str = layout_json_str[len("```json"):].strip()
    elif layout_json_str.startswith("```"):
       layout_json_str = layout_json_str[3:].strip()

    # Validate and parse JSON
    if not (layout_json_str.startswith('{') and layout_json_str.endswith('}')):
        print(f"Warning: Content doesn't look like JSON: '{layout_json_str[:50]}...'")
        return None
    
    try:
        return json.loads(layout_json_str)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse scene layout JSON: {e}")
        return None


def process_single_scene_layout(scene_layout_string: str, output_base: str, config: dict) -> bool:
    """
    Process a single scene layout string to generate 3D scene
    """
    print("=== Processing Single Scene Layout ===")
    
    # Initialize pipeline
    print("Initializing pipeline components...")
    t_init_start = time.time()
    retriever, device, annotations_data = initialize_pipeline(config)
    t_init_end = time.time()

    if retriever is None or device is None or annotations_data is None:
        print("Pipeline initialization failed.")
        return False
    print(f"Initialization took {t_init_end - t_init_start:.2f} seconds.")

    # Extract JSON from response
    print("\nExtracting JSON from response...")
    scene_dict = extract_json_from_response(scene_layout_string)
    
    if scene_dict is None:
        print("Failed to extract valid JSON from response.")
        return False

    print("JSON extracted successfully.")

    # Process scene data
    print("\nProcessing scene data...")
    t_process_start = time.time()
    
    try:
        success = process_scene_data(
            scene_data_dict=scene_dict,
            output_base=output_base,
            retriever=retriever,
            device=device,
            annotations_data=annotations_data,
            config=config
        )
    except Exception as e:
        print(f"Error during scene processing: {e}")
        traceback.print_exc()
        success = False

    t_process_end = time.time()
    print(f"Processing took {t_process_end - t_process_start:.2f} seconds.")

    # Check outputs
    if success:
        print("\n=== Checking Outputs ===")
        output_dir = os.path.dirname(output_base)
        render_dir_name = "rendered_views"
        expected_render_dir = os.path.join(output_dir, render_dir_name)
        expected_glb = f"{output_base}_reconstructed_bpy.glb"

        render_ok = os.path.isdir(expected_render_dir) and len(os.listdir(expected_render_dir)) > 0
        glb_ok = os.path.exists(expected_glb)
        
        print(f"Render directory check: {'Success' if render_ok else 'Failed'}")
        print(f"GLB file check: {'Success' if glb_ok else 'Failed'}")
        
        if render_ok and glb_ok:
            print("✅ Scene processing completed successfully!")
        else:
            print("⚠️ Scene processing completed but some outputs may be missing.")
    else:
        print("❌ Scene processing failed.")

    return success


def main():
    """Main function for command-line execution"""
    parser = argparse.ArgumentParser(description="Process single scene layout string to generate 3D scene")
    
    # Input arguments
    parser.add_argument("--scene_layout_file", type=str, required=True,
                       help="Path to file containing scene layout string")
    parser.add_argument("--output_base", type=str, required=True,
                       help="Base path for output files")
    # 修改默认配置文件路径
    parser.add_argument("--config", type=str, default="../../config.yaml",
                       help="Path to config.yaml file (default: ../../config.yaml)")
    
    # Optional overrides
    parser.add_argument("--model_path", type=str, default=None,
                       help="Override model base path from config")
    parser.add_argument("--blender_exec", type=str, default=None,
                       help="Override Blender executable from config")
    parser.add_argument("--annotations_path", type=str, default=None,
                       help="Override annotations path from config")

    args = parser.parse_args()

    print("=" * 60)
    print("Single Scene Layout Processing")
    print("=" * 60)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments if provided
    if args.model_path:
        config['model_base_path'] = args.model_path
    if args.blender_exec:
        config['blender_executable'] = args.blender_exec
    if args.annotations_path:
        config['annotations_path'] = args.annotations_path
    
    # Validate paths
    print("\nValidating configuration paths...")
    if not validate_config_paths(config):
        print("Warning: Some configured paths are invalid. Processing may fail.")
    else:
        print("All configuration paths are valid.")

    # Validate input file
    if not os.path.exists(args.scene_layout_file):
        print(f"Error: Scene layout file not found: {args.scene_layout_file}")
        sys.exit(1)

    # Load scene layout string
    try:
        with open(args.scene_layout_file, 'r', encoding='utf-8') as f:
            scene_layout_string = f.read().strip()
        print(f"\nLoaded scene layout string ({len(scene_layout_string)} characters)")
    except Exception as e:
        print(f"Error: Failed to load scene layout file: {e}")
        sys.exit(1)

    # Process scene layout
    success = process_single_scene_layout(
        scene_layout_string=scene_layout_string,
        output_base=args.output_base,
        config=config
    )

    if success:
        print("\n Processing completed successfully!")
    else:
        print("\n Processing failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()