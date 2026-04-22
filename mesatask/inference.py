"""
Integrated Scene Generation Pipeline
Combines task info loading, scene layout generation, and 3D scene rendering
"""

import json
import argparse
import torch
import random
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from tools.prompt import LAYOUT_GENERATION_PROMPT_CONCISE
from tools.layout2scene.process_scene import process_single_scene_layout, load_config, validate_config_paths
from tools.layout2scene.json2img import run_reconstruction_and_render
from tools.layoutopt.drake_process import optimize_scene_from_json


class MesaTaskModel:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        
    def load_model(self):
        """Load the tokenizer and model"""
        if self.model is None:
            print(f"Loading MesaTask model from: {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype="auto",
                device_map="auto"
            )
    
    def generate_layout(self, task_info: dict) -> str:
        self.load_model()
        
        # Prepare the prompt
        task_info_json = json.dumps(task_info, ensure_ascii=False, indent=2)
        prompt = LAYOUT_GENERATION_PROMPT_CONCISE + "\n" + task_info_json
        
        # Prepare model input
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True  
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        # Generate response
        print("Generating scene layout with MesaTask...")
        seed = 42
        torch.manual_seed(seed)
        random.seed(seed)
        generated_ids = self.model.generate(**model_inputs,
                                             max_new_tokens=32768,
                                             temperature=0.7,
                                             top_p=0.9,
                                             top_k=50)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        
        # Decode the output
        content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        return content


def find_next_scene_folder(task_dir: str) -> str:
    """Find the next available scene folder name within a task directory"""
    scene_counter = 1
    while True:
        scene_folder_name = f"scene_{scene_counter:03d}"
        scene_folder_path = os.path.join(task_dir, scene_folder_name)
        if not os.path.exists(scene_folder_path):
            return scene_folder_path
        scene_counter += 1

def find_task_folder_from_input(input_file: str) -> str:
    """Extract task folder from input file path"""
    input_dir = os.path.dirname(input_file)
    # Check if input_file is in a task_xxx folder
    if os.path.basename(input_dir).startswith('task_'):
        return input_dir
    else:
        # If not, assume it's in the current directory and create a default task folder
        return None

def integrated_scene_generation(
    input_file: str,
    mesatask_model_path: str,
    output_dir: str = "output",
    config_file: str = "tools/layout2scene/config.yaml",
    enable_drake_optimization: bool = True,
    enable_rendering: bool = True
):
    """
    Integrated pipeline: task info -> scene layout -> 3D scene
    """
    
    # Determine task folder from input file
    task_folder = find_task_folder_from_input(input_file)
    
    if task_folder is None:
        # If input file is not in a task folder, create a fallback structure
        os.makedirs(output_dir, exist_ok=True)
        task_folder = os.path.join(output_dir, "task_default")
        os.makedirs(task_folder, exist_ok=True)
        print(f"Warning: Input file not in task folder structure. Using: {task_folder}")
    
    # Find next available scene folder within the task folder
    scene_folder = find_next_scene_folder(task_folder)
    os.makedirs(scene_folder, exist_ok=True)
    
    # Set output base for the scene
    output_base = os.path.join(scene_folder, "scene")
    
    print(f"Using task folder: {os.path.basename(task_folder)}")
    print(f"Using scene folder: {os.path.basename(scene_folder)}")
    
    # Step 1: Load task information
    print("=" * 60)
    print("Step 1: Loading task information")
    print("=" * 60)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        task_info = json.load(f)
    print(f"Task info loaded from: {input_file}")
    
    # Step 2: Generate scene layout using MesaTask
    print("\n" + "=" * 60)
    print("Step 2: Generating scene layout with MesaTask")
    print("=" * 60)
    
    mesatask = MesaTaskModel(mesatask_model_path)
    scene_layout = mesatask.generate_layout(task_info)
    print("Scene layout generated successfully")
    
    # Save scene layout to scene folder
    scene_layout_file = os.path.join(scene_folder, "scene_layout.txt")
    with open(scene_layout_file, 'w', encoding='utf-8') as f:
        f.write(scene_layout)
    print(f"Scene layout saved to: {scene_layout_file}")
    
    # Step 3: Load configuration for 3D scene generation
    print("\n" + "=" * 60)
    print("Step 3: Loading 3D scene generation configuration")
    print("=" * 60)
    
    config = load_config(config_file)
    if not validate_config_paths(config):
        print("Warning: Some configured paths are invalid. Processing may fail.")
    else:
        print("All configuration paths are valid.")
    
    # Step 4: Place 3D scene
    print("\n" + "=" * 60)
    print("Step 4: Generating 3D scene")
    print("=" * 60)
    
    success = process_single_scene_layout(
        scene_layout_string=scene_layout,
        output_base=output_base,
        config=config
    )
    
    # Step 5: Drake Optimization (optional)
    drake_success = False
    optimized_scene_path = None
    
    if success and enable_drake_optimization:
        print("\n" + "=" * 60)
        print("Step 5: Drake Optimization")
        print("=" * 60)
        
        # Find the processed scene file
        processed_scene_file = os.path.join(scene_folder, "scene_processed_scene.json")
        if os.path.exists(processed_scene_file):
            print(f"Starting Drake optimization on: {processed_scene_file}")
            try:
                drake_success, optimized_scene_path = optimize_scene_from_json(
                    json_path=processed_scene_file,
                    config_path=config_file
                )
                if drake_success and optimized_scene_path:
                    print(f"Drake optimization successful! Optimized scene saved to: {optimized_scene_path}")
                else:
                    print("Drake optimization failed or no improvement found.")
            except Exception as e:
                print(f"Drake optimization error: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"Warning: Processed scene file not found: {processed_scene_file}")
    
    # Step 6: Render Optimized Scene (optional)
    if drake_success and optimized_scene_path and enable_rendering:
        print("\n" + "=" * 60)
        print("Step 6: Rendering Optimized Scene")
        print("=" * 60)
        
        try:
            # Create optimized_scene subfolder
            optimized_scene_folder = os.path.join(scene_folder, "optimized_scene")
            os.makedirs(optimized_scene_folder, exist_ok=True)
            
            # Copy optimized JSON to optimized_scene folder
            optimized_json_filename = os.path.basename(optimized_scene_path)
            optimized_json_dest = os.path.join(optimized_scene_folder, optimized_json_filename)
            import shutil
            shutil.copy2(optimized_scene_path, optimized_json_dest)
            print(f"Optimized JSON copied to: {optimized_json_dest}")
            
            # Set output base for optimized scene
            optimized_output_base = os.path.join(optimized_scene_folder, "scene_optimized")
            render_dir = run_reconstruction_and_render(
                scene_json_path=optimized_json_dest,  # Use the copied file
                output_base_name=optimized_output_base,
                config=config
            )
            
            if render_dir:
                print(f"Optimized scene rendering successful! Images saved to: {render_dir}")
            else:
                print("Optimized scene rendering failed.")
        except Exception as e:
            print(f"Rendering error: {e}")
            import traceback
            traceback.print_exc()

    # Report results
    if success:
        print("\n" + "=" * 60)
        print("Pipeline Results")
        print("=" * 60)
        print("✅ Basic 3D scene generation: SUCCESS")
        
        if enable_drake_optimization:
            if drake_success:
                print("✅ Drake optimization: SUCCESS")
            else:
                print("❌ Drake optimization: FAILED")
        
        if enable_rendering and drake_success:
            print("✅ Optimized scene rendering: SUCCESS")
        
        print(f"\nScene folder: {scene_folder}")
        
        # List generated files
        print("\nGenerated files:")
        for root, dirs, files in os.walk(scene_folder):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, scene_folder)
                print(f"  - {rel_path}")
        
        return True
    else:
        print("\n❌ 3D scene generation failed!")
        return False


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="Integrated scene generation pipeline")
    parser.add_argument("--input_file", type=str, required=True,
                       help="Input task info JSON file path (should be in task_xxx folder)")
    parser.add_argument("--output_dir", type=str, default="output",
                       help="Base output directory (only used for fallback if input not in task folder)")
    parser.add_argument("--mesatask_model_path", type=str, 
                       default="./MesaTask-10K/MesaTask_model",
                       help="Path to MesaTask model")
    parser.add_argument("--config_file", type=str, 
                       default="./config.yaml",
                       help="Configuration file for 3D scene generation")
    parser.add_argument("--physical_optimization", action="store_true",
                       help="Enable physical optimization using Drake")
    parser.add_argument("--rendering", action="store_true",
                       help="Enable scene rendering")
    
    args = parser.parse_args()
    
    # Simple flags
    enable_drake_optimization = args.physical_optimization
    enable_rendering = args.rendering

    print("🚀 Starting Integrated Scene Generation Pipeline")
    print(f"📥 Input: {args.input_file}")
    print(f"📤 Output: {args.output_dir}")
    print(f"🤖 Model: {args.mesatask_model_path}")
    print(f"⚙️ Config: {args.config_file}")
    print(f"🔧 Physical Optimization: {'Yes' if enable_drake_optimization else 'No'}")
    print(f"🎨 Scene Rendering: {'Yes' if enable_rendering else 'No'}")
    
    success = integrated_scene_generation(
        input_file=args.input_file,
        mesatask_model_path=args.mesatask_model_path,
        output_dir=args.output_dir,
        config_file=args.config_file,
        enable_drake_optimization=enable_drake_optimization,
        enable_rendering=enable_rendering
    )
    
    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
