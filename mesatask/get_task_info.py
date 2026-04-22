"""
Task Information Generation Interface
Input table_type and task_name, call GPT to generate task info and save as JSON file
"""

import json
import argparse
import os
from tools.taskinfo_gen import generate_and_validate_task_info, save_task_info_to_file


def get_task_info(
    task_name: str, 
    table_type: str, 
    api_key: str,
    model: str,
    base_url: str,
    validate: bool = True
):
    """
    Generate task information using GPT
    
    Args:
        task_name (str): Task name
        table_type (str): Table type  
        api_key (str): GPT API key
        model (str): GPT model to use
        base_url (str): API base URL
        validate (bool): Whether to validate results
        
    Returns:
        dict or None: Generated task information, None if failed
    """
    return generate_and_validate_task_info(
        task_name=task_name,
        table_type=table_type,
        api_key=api_key,
        model=model,
        base_url=base_url,
        validate=validate
    )


def find_next_task_folder(base_output_dir: str) -> str:
    """Find the next available task folder name"""
    task_counter = 1
    while True:
        task_folder_name = f"task_{task_counter:03d}"
        task_folder_path = os.path.join(base_output_dir, task_folder_name)
        if not os.path.exists(task_folder_path):
            return task_folder_path
        task_counter += 1


def main():
    """
    Command line interface main function
    """
    parser = argparse.ArgumentParser(description="Generate task information using GPT")
    parser.add_argument("--task_name", type=str, required=True, 
                       help="Task name, e.g.: 'Organize books and magazines'")
    parser.add_argument("--table_type", type=str, required=True,
                       help="Table type, e.g.: 'Nightstand', 'TV stand', 'side table'")
    parser.add_argument("--api_key", type=str, required=True,
                       help="GPT API key")
    parser.add_argument("--model", type=str, default="gpt-4o",
                       help="GPT model to use (default: gpt-4o)")
    parser.add_argument("--base_url", type=str, default="https://api.openai.com/v1",
                       help="API base URL")
    parser.add_argument("--output_dir", type=str, default="output",
                       help="Base output directory (default: output)")
    parser.add_argument("--no_validate", action="store_true",
                       help="Skip result validation")
    
    args = parser.parse_args()
    
    # Create task folder
    os.makedirs(args.output_dir, exist_ok=True)
    task_folder = find_next_task_folder(args.output_dir)
    os.makedirs(task_folder, exist_ok=True)
    
    # Set output file path
    output_file = os.path.join(task_folder, "task_info.json")
    
    print("=" * 60)
    print("Task Information Generation")
    print("=" * 60)
    print(f"Task: {args.task_name}")
    print(f"Table Type: {args.table_type}")
    print(f"GPT Model: {args.model}")
    print(f"Base URL: {args.base_url}")
    print(f"Task Folder: {task_folder}")
    print(f"Output File: {output_file}")
    
    # Generate task information
    task_info = get_task_info(
        task_name=args.task_name,
        table_type=args.table_type,
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url,
        validate=not args.no_validate
    )
    
    if task_info is not None:
        print("\n✅ Task information generated successfully!")
        
        # Print JSON format
        print("\n📄 Generated Task Information:")
        print("-" * 40)
        print(json.dumps(task_info, ensure_ascii=False, indent=2))
        print("-" * 40)
        
        # Save to file
        if save_task_info_to_file(task_info, output_file):
            print(f"\n💾 Task info saved to: {output_file}")
            print(f"📁 Task folder: {task_folder}")
        else:
            print(f"\n❌ Failed to save to: {output_file}")
        
        return task_info, task_folder
        
    else:
        print("\n❌ Task information generation failed!")
        return None, None


if __name__ == "__main__":
    main()
