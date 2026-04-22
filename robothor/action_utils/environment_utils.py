"""
Environment-related utility functions for RoboTHOR environment
"""


def restore_view(controller, view_info):
    """
    恢复到指定视角

    Args:
        controller: AI2-THOR controller
        view_info: 视角信息字典，包含 pos, rot, horizon 等字段
    """
    if view_info is not None:
        controller.step(
            action="TeleportFull",
            position=view_info["pos"],
            rotation={"x": 0, "y": int(view_info["rot"]), "z": 0},
            horizon=int(view_info.get("horizon", 0)),
            standing=True,
            forceAction=True
        )


def apply_physics_settings(controller, disable_physics=False):
    """
    应用物理设置

    注意：AI2-THOR 默认启用物理模拟，reset() 后也会恢复到启用状态。
    因此只需要在 disable_physics=True 时主动暂停物理，
    disable_physics=False 时无需任何操作（已经是启用状态）。

    Args:
        controller: AI2-THOR controller
        disable_physics: 是否禁用物理模拟
    """
    if disable_physics:
        # 使用 PausePhysicsAutoSim 暂停自动物理模拟
        event = controller.step(action="PausePhysicsAutoSim")
        if event.metadata.get("lastActionSuccess", False):
            print("[Info] Physics auto-simulation paused for better performance")
        else:
            print("[Warning] Failed to pause physics auto-simulation")
    # else: 不需要任何操作，默认就是启用状态
