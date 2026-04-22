#!/usr/bin/env python3
"""
指令模板库 - 支持中英文多样化表达
"""

TEMPLATES = {
    "move_right": {
        "zh": [
            "请将{obj}向右移动{value}厘米。",
            "请把{obj}往右边挪动{value}厘米。",
            "将{obj}向右平移{value}厘米。",
            "把{obj}朝右移{value}厘米。",
        ],
        "en": [
            "Move the {obj} {value} centimeters to the right.",
            "Shift the {obj} {value} cm to the right.",
            "Please move the {obj} rightward by {value} centimeters.",
            "Relocate the {obj} {value} cm to the right.",
        ]
    },
    "move_left": {
        "zh": [
            "请将{obj}向左移动{value}厘米。",
            "请把{obj}往左边挪动{value}厘米。",
            "将{obj}向左平移{value}厘米。",
            "把{obj}朝左移{value}厘米。",
        ],
        "en": [
            "Move the {obj} {value} centimeters to the left.",
            "Shift the {obj} {value} cm to the left.",
            "Please move the {obj} leftward by {value} centimeters.",
            "Relocate the {obj} {value} cm to the left.",
        ]
    },
    "move_forward": {
        "zh": [
            "请将{obj}向远离视角的方向移动{value}厘米。",
            "请把{obj}往桌面远端挪动{value}厘米。",
            "将{obj}向前推{value}厘米。",
            "把{obj}朝远处移{value}厘米。",
        ],
        "en": [
            "Move the {obj} {value} centimeters away from the viewpoint.",
            "Push the {obj} {value} cm toward the far end of the table.",
            "Move the {obj} forward by {value} centimeters.",
            "Shift the {obj} {value} cm away from you.",
        ]
    },
    "move_backward": {
        "zh": [
            "请将{obj}向靠近视角的方向移动{value}厘米。",
            "请把{obj}往视角方向挪动{value}厘米。",
            "将{obj}向后拉{value}厘米。",
            "把{obj}朝近处移{value}厘米。",
        ],
        "en": [
            "Move the {obj} {value} centimeters toward the viewpoint.",
            "Pull the {obj} {value} cm toward you.",
            "Move the {obj} backward by {value} centimeters.",
            "Shift the {obj} {value} cm closer to you.",
        ]
    },
    "rotate_clockwise": {
        "zh": [
            "请将{obj}顺时针旋转{value}度。",
            "请把{obj}向右转动{value}度。",
            "将{obj}顺时针转{value}度。",
            "把{obj}右旋{value}度。",
        ],
        "en": [
            "Rotate the {obj} {value} degrees clockwise.",
            "Turn the {obj} {value} degrees to the right.",
            "Please rotate the {obj} clockwise by {value} degrees.",
            "Spin the {obj} {value} degrees clockwise.",
        ]
    },
    "rotate_counterclockwise": {
        "zh": [
            "请将{obj}逆时针旋转{value}度。",
            "请把{obj}向左转动{value}度。",
            "将{obj}逆时针转{value}度。",
            "把{obj}左旋{value}度。",
        ],
        "en": [
            "Rotate the {obj} {value} degrees counterclockwise.",
            "Turn the {obj} {value} degrees to the left.",
            "Please rotate the {obj} counterclockwise by {value} degrees.",
            "Spin the {obj} {value} degrees counterclockwise.",
        ]
    },
    "scale_up": {
        "zh": [
            "请将{obj}放大{value}%。",
            "请把{obj}缩放为原来的{scale}倍大小。",
            "将{obj}扩大{value}%。",
            "把{obj}放大到{scale}倍。",
        ],
        "en": [
            "Scale up the {obj} by {value}%.",
            "Enlarge the {obj} by {value} percent.",
            "Make the {obj} {scale} times larger.",
            "Increase the size of the {obj} by {value}%.",
        ]
    },
    "scale_down": {
        "zh": [
            "请将{obj}缩小{value}%。",
            "请把{obj}缩放为原来的{scale}倍大小。",
            "将{obj}缩小{value}%。",
            "把{obj}缩小到{scale}倍。",
        ],
        "en": [
            "Scale down the {obj} by {value}%.",
            "Shrink the {obj} by {value} percent.",
            "Make the {obj} {scale} times smaller.",
            "Reduce the size of the {obj} by {value}%.",
        ]
    },
}
