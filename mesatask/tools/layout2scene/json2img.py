import subprocess
import os
import json
import torch
import open_clip
from sentence_transformers import SentenceTransformer
import traceback

from .retrival_object import ObjathorRetriever


def perform_retrieval_processing(scene_data, retriever, device, annotations_data):
    """
    Process scene data using initialized retriever and annotations_data, return retrieval results dictionary
    """
    results_output = {}
    objects_to_process = scene_data.get('objects', [])

    if not annotations_data:
        print("Error: annotations_data not provided to perform_retrieval_processing.")
        return None

    if not objects_to_process:
        print("No 'objects' list found in scene data or list is empty")
        return {}

    queries, target_sizes, instance_ids = [], [], []
    for obj in objects_to_process:
        instance_id = obj.get('instance')
        description = obj.get('description')
        size = obj.get('size')
        if not instance_id or not description or not size or len(size) != 3:
            print(f"Warning: Object missing name, description or valid size: {obj}. Skipping.")
            continue
        queries.append(description)
        target_sizes.append({'x': size[0], 'y': size[1], 'z': size[2]})
        instance_ids.append(instance_id)

    if not queries:
        print("No valid objects for retrieval.")
        return {}

    print(f"Starting batch retrieval for {len(queries)} objects...")
    try:
        if not retriever or not hasattr(retriever, 'retrieve_text_size'):
            print("Error: Retriever object is not valid or missing 'retrieve_text_size' method.")
            return None
        batch_results = retriever.retrieve_text_size(queries, target_sizes, device=device)
    except Exception as e:
        print(f"Error calling retriever.retrieve_text_size: {e}")
        traceback.print_exc()
        return None

    original_objects_dict = {obj.get('instance'): obj for obj in objects_to_process if obj.get('instance')}

    for i, instance_id in enumerate(instance_ids):
        if batch_results is None or i >= len(batch_results):
            print(f"Warning: Insufficient retrieval results or None. Skipping instance_id {instance_id}.")
            results_output[instance_id] = {
                'description': original_objects_dict.get(instance_id, {}).get('description', ""),
                'original_size': original_objects_dict.get(instance_id, {}).get('size'),
                'retrieved_uids': [],
                'similarity_scores': [],
                'status': 'Retrieval result missing or index out of bounds'
            }
            continue

        retrieved_uids, similarity_scores = batch_results[i]
        original_obj = original_objects_dict.get(instance_id)
        original_size = original_obj.get('size') if original_obj else None
        original_description = original_obj.get('description') if original_obj else ""

        # Filter retrieval results
        if retrieved_uids:
            filtered_uids = []
            filtered_scores = []
            uid_score_map = dict(zip(retrieved_uids, similarity_scores)) if isinstance(similarity_scores, list) else {uid: 0.0 for uid in retrieved_uids}

            for uid in retrieved_uids:
                if not isinstance(annotations_data, dict):
                    print("Error: annotations_data is not a valid dictionary.")
                    continue

                annotation_info = annotations_data.get(uid)
                if (annotation_info and 
                    annotation_info.get("onTable", True) and 
                    not annotation_info.get("mark_deletion", False) and 
                    not annotation_info.get("is_composite", False)):
                    filtered_uids.append(uid)
                    filtered_scores.append(uid_score_map.get(uid, 0.0))

            retrieved_uids = filtered_uids
            similarity_scores = filtered_scores

        if not isinstance(similarity_scores, list):
            similarity_scores = []

        if retrieved_uids:
            results_output[instance_id] = {
                'description': original_description,
                'original_size': original_size,
                'retrieved_uids': retrieved_uids,
                'similarity_scores': similarity_scores
            }
        else:
            results_output[instance_id] = {
                'description': original_description,
                'original_size': original_size,
                'retrieved_uids': [],
                'similarity_scores': [],
                'status': 'Retrieval failed or all candidates filtered out'
            }

    return results_output


def initialize_pipeline(config: dict):
    """
    Initialize object retriever, related models, and load annotation data
    Returns retriever, device, annotations_data
    """
    print("Initializing object retriever and loading annotation data...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Load annotation data from config
        annotations_path = config['annotations_path']
        print(f"Loading annotation file: {annotations_path}")
        if not os.path.exists(annotations_path):
            raise FileNotFoundError(f"Annotation JSON file not found: {annotations_path}")
        with open(annotations_path, 'r', encoding='utf-8') as f:
            annotations_data = json.load(f)
        print(f"Annotation data loaded ({len(annotations_data)} entries).")

        sbert_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device="cpu")

        # Initialize Retriever (simplified version, only needs SBERT)
        retriever = ObjathorRetriever(
            sbert_model=sbert_model,
            config=config
        )
        
        retriever.sbert_model.to(device)

        print(f"Retriever initialization completed, using device: {device}")
        return retriever, device, annotations_data

    except Exception as e:
        print(f"Error during initialization: {e}")
        traceback.print_exc()
        return None, None, None


def run_reconstruction_and_render(scene_json_path, output_base_name, config: dict):
    """
    Call reconstruct_blender.py to reconstruct scene and render images
    """
    print(f"\n--- Step 2: Starting scene reconstruction and rendering ---")

    output_glb_path = f"{output_base_name}_reconstructed_bpy.glb"
    render_output_dir = os.path.join(os.path.dirname(output_glb_path), "rendered_views")

    model_base_path = config['model_base_path']
    blender_executable = config['blender_executable']
    table_path = config.get('table_model_path')  
    sink_path = config.get('sink_model_path')   

    script_dir = os.path.dirname(os.path.abspath(__file__))
    reconstruct_script = os.path.join(script_dir, "reconstruct_blender.py")

    paths_to_check = [
        (reconstruct_script, "Reconstruction script not found"),
        (blender_executable, "Blender executable not found"),
        (model_base_path, "Model base path not found"),
        (scene_json_path, "Input scene JSON file does not exist")
    ]
    
    if table_path:
        paths_to_check.append((table_path, "Table model file not found"))
    if sink_path:
        paths_to_check.append((sink_path, "Sink model file not found"))
    
    for path, error_msg in paths_to_check:
        if not os.path.exists(path):
            print(f"Error: {error_msg}: {path}")
            return None

    os.makedirs(os.path.dirname(output_glb_path), exist_ok=True)

    # Modified command - no longer need results_json when using processed scene
    command = [
        blender_executable,
        "--background", "--python", reconstruct_script, "--",
        "--scene_json", scene_json_path,
        "--output_glb", output_glb_path, 
        "--model_base_path", model_base_path
    ]
    
    if table_path:
        command.extend(["--table_path", table_path])
    if sink_path:
        command.extend(["--sink_path", sink_path])

    print(command)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        print(f"\nError: Blender script execution failed, return code: {process.returncode}")
        print("Blender script stderr:"); print(stderr)
        return None
    elif stderr:
        print("\nBlender script stderr (may contain warnings):"); print(stderr)

    # Check render directory
    if os.path.exists(render_output_dir) and os.path.isdir(render_output_dir):
        try:
            png_files = [f for f in os.listdir(render_output_dir) if f.lower().endswith('.png')]
            if png_files: 
                print(f"\nReconstruction and rendering successful. Found {len(png_files)} PNG files.")
                return render_output_dir
            else: 
                print("\nWarning: Render directory is empty but created.")
                return render_output_dir
        except OSError as e: 
            print(f"\nError: Unable to read render directory contents: {e}")
            return None
    else:
        print(f"\nError: Expected render directory not generated: {render_output_dir}")
        return None


def create_processed_scene_with_selected_uids(scene_data, retrieval_results_dict, output_path):
    """
    Create processed scene JSON with selected UIDs already embedded
    """
    processed_scene = scene_data.copy()
    processed_objects = []
    
    objects_to_process = scene_data.get('objects', [])
    
    for obj_data in objects_to_process:
        instance_id = obj_data.get('instance')
        
        # Get retrieval results and select UID (same logic as reconstruct_blender.py line 814)
        retrieved_info = retrieval_results_dict.get(instance_id)
        selected_uid = None
        if retrieved_info and retrieved_info.get("retrieved_uids"):
            selected_uid = retrieved_info["retrieved_uids"][0]
        
        # Add selected_uid to object data
        processed_obj = obj_data.copy()
        processed_obj["selected_uid"] = selected_uid
        processed_objects.append(processed_obj)
    
    processed_scene["objects"] = processed_objects
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_scene, f, indent=2, ensure_ascii=False)
    
    print(f"Processed scene with selected UIDs saved to: {output_path}")
    return output_path



def process_scene_data(scene_data_dict, output_base, retriever, device, annotations_data, config: dict):
    """
    Process scene data dictionary, save it to temporary file, then call file-based processing pipeline
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_base)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Perform retrieval processing
    retrieval_results_dict = perform_retrieval_processing(scene_data_dict, retriever, device, annotations_data)
    
    if not retrieval_results_dict:
        print("Error: Retrieval processing failed or returned empty results.")
        return False

    # Save original retrieval results (with all 5 UIDs)
    retrieval_output_path = f"{output_base}_retrieval_results.json"
    with open(retrieval_output_path, 'w', encoding='utf-8') as f:
        json.dump(retrieval_results_dict, f, indent=2, ensure_ascii=False)
    print(f"Full retrieval results saved to: {retrieval_output_path}")

    # Create processed scene with selected UIDs (for Blender)
    processed_scene_path = f"{output_base}_processed_scene.json"
    create_processed_scene_with_selected_uids(scene_data_dict, retrieval_results_dict, processed_scene_path)

    # Scene reconstruction and rendering (using processed scene)
    render_dir = run_reconstruction_and_render(
        processed_scene_path,  # Use processed scene instead of original
        output_base,
        config
    )

    success = render_dir is not None
    if success:
        print(f"\nScene processing successful. Rendered images at: {os.path.abspath(render_dir)}")
    else:
        print("\nScene reconstruction or rendering step failed.")

    return success
