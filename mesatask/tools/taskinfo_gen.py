"""
Task Information Generator - Single Task Version
Convert task instructions to detailed task information
"""

import json
from typing import Dict, Any, Optional
from .gpt import GPT


def generate_single_task_info(
    task_name: str, 
    table_type: str, 
    api_key: str, 
    model: str,
    base_url: str
) -> Optional[Dict[str, Any]]:
    """
    Generate detailed task information for a single task
    
    Args:
        task_name (str): Task name (e.g. "Organize books and magazines")
        table_type (str): Table type (e.g. "Nightstand", "TV stand", "side table")
        api_key (str): GPT API key
        model (str): GPT model to use
        base_url (str): API base URL
        
    Returns:
        Dict[str, Any] or None: Generated task information, None if failed
    """
    try:
        gpt = GPT(
            api_key=api_key,
            model=model,
            base_url=base_url
        )
        
        return gpt.generate_task_info(task_name, table_type)
        
    except Exception:
        return None


def validate_task_info(task_info: Dict[str, Any]) -> bool:
    """
    Validate task information completeness
    
    Args:
        task_info (Dict): Task information to validate
        
    Returns:
        bool: Whether validation passed
    """
    required_fields = [
        "Environment",
        "item_placement_zone", 
        "Task",
        "Goal",
        "Action Sequence",
        "Objects cluster"
    ]
    
    for field in required_fields:
        if field not in task_info:
            return False
    
    # Validate item_placement_zone format
    placement_zone = task_info.get("item_placement_zone")
    if not isinstance(placement_zone, list) or len(placement_zone) != 4:
        return False
    
    # Validate lists
    if not all(isinstance(task_info.get(field), list) for field in ["Goal", "Action Sequence", "Objects cluster"]):
        return False
    
    return True


def generate_and_validate_task_info(
    task_name: str, 
    table_type: str, 
    api_key: str, 
    model: str,
    base_url: str,
    validate: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Generate and validate task information
    
    Args:
        task_name (str): Task name
        table_type (str): Table type
        api_key (str): GPT API key
        model (str): GPT model to use
        base_url (str): API base URL
        validate (bool): Whether to validate results
        
    Returns:
        Dict[str, Any] or None: Validated task information, None if failed
    """
    task_info = generate_single_task_info(
        task_name=task_name,
        table_type=table_type,
        api_key=api_key,
        model=model,
        base_url=base_url
    )
    
    if task_info is None:
        return None
    
    if validate and not validate_task_info(task_info):
        return None
    
    return task_info


def save_task_info_to_file(task_info: Dict[str, Any], output_path: str) -> bool:
    """
    Save task information to JSON file
    
    Args:
        task_info (Dict): Task information
        output_path (str): Output file path
        
    Returns:
        bool: Whether save succeeded
    """
    import os
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(task_info, f, ensure_ascii=False, indent=2)
    
    return True