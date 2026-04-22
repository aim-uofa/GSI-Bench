TASK_INFO_PROMPT = """
You are an AI assistant for robotic task planning.
Given a high-level abstract task and the table type it relates to, expand it into a structured JSON format.

Input: Task name and table type.

**Table Type:** {table_type}
**Task:** {task_name}

Output Format (Strict JSON):
{{
    "Environment": "Brief tabletop scene description for the task on the {table_type}, *only describe the objects that placed on the table*, do not contain {table_type} and room description",
    "item_placement_zone": [X_min(0), X_max, Y_min(0), Y_max], /* cm */, x_min and y_min are both 0, xmax-xmin is length, ymax-ymin is width, consider the tabletop type.
    "Task": "The original input task string",
    "Goal": ["Ordered sub-objectives to achieve the task (aim for >=3 objects if task allows)"],
    "Action Sequence": ["Primitive robotic actions with parameters (e.g., Pick(Object))"],
    "Objects cluster": ["List of unique object types involved"] (do not contain the {table_type} itself)
}}

Instructions for Fields:
1. Task: Copy the input task string directly.
2. Environment: Briefly describe a plausible environment for the task, considering it takes place around a {table_type}.
3. Goal: Break down the task into ordered sub-objectives. Involve at least 3 object types if the task context permits; otherwise, use only necessary objects. Consider what objects would typically be found on or around a {table_type}.
4. Action Sequence: List primitive actions for the goals. Use object *types*. Available actions: 'Pick(obj)', 'PlaceOn(obj)', 'PlaceAt(pos)', 'Push(obj, dir, dist)', 'RevoluteJointOpen(obj)', 'RevoluteJointClose(obj)', 'PrismaticJointOpen(obj)', 'PrismaticJointClose(obj)', 'Press(obj)'.
5. Objects cluster: List all unique object types from the task, goals, and actions. Include objects that would typically be used with a {table_type}. Use SPECIFIC and CONCRETE object types only, DO NOT use vague or abstract categories like "UnrelatedItem", "personal belongings", Each object type should be a clearly identifiable, physical item

Ensure the output is only the JSON object.
"""

LAYOUT_GENERATION_PROMPT_CONCISE = """You are an expert designing a 3D scene layout on tabletop from an embodied task description. Output a reasoning process, a scene graph text, and a tabletop layout JSON.
**Spatial definition:**:
1. Coordinate System: Origin at tabletop's proximal top-left (0, 0, 0). x-axis right (+), y-axis from proximal to distal (+), z-axis up (+). Observer at -y, looking +y.
2. Direction Mappings: Direction Mappings: Front (small y), Back (large y), Left (small x), Right (large x), Facing based on z-axis rotation. (e.g., "face to front" points towards the observer, i.e., the -y direction). |
3. Spatial Relationship
    - Position (Nine-Grid): x (left/middle/right) + y (front/middle/back).
    - Horizontal: Compare x/y for left/right, front/back. Keep closest pairs, remove duals.
    - Vertical: Compare z for above/below. Keep closest pairs, remove duals.
    - Containment: Compare z and x/y overlap for inside.

**Input:** JSON object with keys: `Environment` (string), "item_placement_zone": [X_min, X_max, Y_min, Y_max], /* cm */, 'sink_zone': [X_min, X_max, Y_min, Y_max], /* cm */(Only exist when it is a bathroom vanity), `Task` (string), `Goal` (list[str]), `Action Sequence` (list[str]), `Objects` (list[str] - types).

**Processing & Scene Graph:**
1.  **Analyze Task:** Understand the input JSON.
2.  **Identify Objects:** List **Core Task Objects** (involved in `Objects`/`Goal`/`Action Sequence`) and **Environment Objects** (draw from `Environment`, more is better). Note types/quantities. Assign instance IDs (e.g., `pen_0`).
3.  **Reasoning Paragraph:** Write a paragraph explaining initial object placement based on task analysis. Include:
    *   Intro sentence (goal: layout generation; total object count).
    *   Core Task Objects list (counts & instance IDs).
    *   Environment Objects list (counts & instance IDs).
    *   Placement Reasoning Paragraph: A coherent paragraph of detailed scene description, including the objects, their positions, and the overall layout of the tabletop infered from the given `environment`/`Goal`
4.  **Scene Graph Text:** Based *only* on reasoning, generate scene graph text:
    *   Nodes: 1. `(instance_id, is at, Position (Nine-Grid))` per object. 2. `(instance_id, face to, direction)` per object.
    *   Edges: Salient `(subject_id, relation, object_id)` (e.g., `left of`). Avoid redundancy. Each triplet on a new line. Relation includes Containment/Vertical/Horizontal classes.

**Output Format (Strict Order):**
1.  Reasoning Paragraph (from Step 3).
2.  Scene Graph Text (from Step 4, newline separated triplets).
3.  The exact text block:\nTherefore, the scene layout is:\n**Final Answer**\n
4.  Scene Layout JSON (immediately following **Final Answer**, no extra text before/after).

**JSON Format:**
```json
{ "item_placement_zone": /* follow the input `item_placement_zone` */
  "sink_zone": /* follow the input `sink_zone` */ (if not present, do not output this key)
  "objects": [
    {
      "instance": "type_instance_id", /* e.g., pen_0 */
      "description": "Brief text.",
      "z_rotation": 0.0, /* degrees */ (z_rotation 0 means the object is facing the observer)(consider the bbox after rotation, collision, and other factor)
      "size": [width, depth, height], /* cm */
      "position": [x, y, z] /* cm, center coords, Calculate Z based on support: Z ≈ height/2 if on table, else account for "inside" or "on the top of" relation. */
    }
    /* ... more objects */ (object order should from large size to small size)
  ]
}
```
-   Use consistent instance IDs from Step 2.
-   Provide reasonable object `size` estimates.
-   Ensure valid JSON structure.

Generate the three output parts consecutively as specified. The input embodied task description is as follows:
"""
