# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import EventCfg

from .loco_manip_env_cfg import DigitLocoManipEnvCfg, DigitLocoManipObservations
from .loco_manip_env_cfg_ovx import applied_body_force, apply_random_external_force_torque


@configclass
class DigitLocoManipForceObservations(DigitLocoManipObservations):
    """Policy observations plus critic-only applied end-effector force observations."""

    @configclass
    class CriticCfg(ObsGroup):
        left_hand_applied_force = ObsTerm(
            func=applied_body_force,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="left_arm_wrist_yaw")},
        )
        right_hand_applied_force = ObsTerm(
            func=applied_body_force,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="right_arm_wrist_yaw")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    critic = CriticCfg()


@configclass
class DigitLocoManipForceEvents(EventCfg):
    """Events that apply random-direction end-effector forces every 5-10 seconds."""

    left_hand_force = EventTermCfg(
        func=apply_random_external_force_torque,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="left_arm_wrist_yaw"),
            "force_magnitude_range": (0.0, 15.0),
            "torque": (0.0, 0.0, 0.0),
            "is_global": True,
        },
    )

    right_hand_force = EventTermCfg(
        func=apply_random_external_force_torque,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="right_arm_wrist_yaw"),
            "force_magnitude_range": (0.0, 15.0),
            "torque": (0.0, 0.0, 0.0),
            "is_global": True,
        },
    )


@configclass
class DigitLocoManipForceEnvCfg(DigitLocoManipEnvCfg):
    """Flat Digit loco-manipulation training with end-effector external force events enabled."""

    events: DigitLocoManipForceEvents = DigitLocoManipForceEvents()
    observations: DigitLocoManipForceObservations = DigitLocoManipForceObservations()


class DigitLocoManipForceEnvCfg_PLAY(DigitLocoManipForceEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        # Make a smaller scene for play.
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # Disable randomization for play.
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.left_hand_force = None
        self.events.right_hand_force = None
