# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch

from isaaclab.managers import CurriculumTermCfg, EventTermCfg, SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab_tasks.manager_based.manipulation.reach.mdp as manipulation_mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.digit.rough_env_cfg import (
    DigitRewards,
    DigitRoughEnvCfg,
    TerminationsCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import EventCfg

from isaaclab_assets.robots.agility import ARM_JOINT_NAMES, LEG_JOINT_NAMES


def apply_constant_external_force_torque(
    env,
    env_ids: torch.Tensor,
    force: tuple[float, float, float],
    torque: tuple[float, float, float] = (0.0, 0.0, 0.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    is_global: bool = True,
):
    """Apply a constant external wrench to selected bodies."""
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies
    forces = torch.tensor(force, dtype=torch.float32, device=asset.device).view(1, 1, 3).repeat(len(env_ids), num_bodies, 1)
    torques = torch.tensor(torque, dtype=torch.float32, device=asset.device).view(1, 1, 3).repeat(
        len(env_ids), num_bodies, 1
    )
    asset.permanent_wrench_composer.set_forces_and_torques(
        forces=forces,
        torques=torques,
        body_ids=asset_cfg.body_ids,
        env_ids=env_ids,
        is_global=is_global,
    )


def apply_random_external_force_torque(
    env,
    env_ids: torch.Tensor,
    force_magnitude_range: tuple[float, float],
    torque: tuple[float, float, float] = (0.0, 0.0, 0.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    is_global: bool = True,
):
    """Apply a random constant force with random direction and bounded magnitude to selected bodies."""
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies

    directions = torch.randn((len(env_ids), num_bodies, 3), device=asset.device)
    directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)

    min_force, max_force = force_magnitude_range
    magnitudes = torch.empty((len(env_ids), num_bodies, 1), device=asset.device).uniform_(min_force, max_force)
    forces = directions * magnitudes
    torques = torch.tensor(torque, dtype=torch.float32, device=asset.device).view(1, 1, 3).repeat(
        len(env_ids), num_bodies, 1
    )
    asset.permanent_wrench_composer.set_forces_and_torques(
        forces=forces,
        torques=torques,
        body_ids=asset_cfg.body_ids,
        env_ids=env_ids,
        is_global=is_global,
    )


def terrain_levels_vel_locomanip(
    env,
    env_ids,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Terrain curriculum for rough locomanip with a relaxed promotion threshold."""
    asset = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    move_up = distance > 3.0
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def applied_body_force(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Return the currently applied permanent external force on selected bodies in world frame."""
    asset = env.scene[asset_cfg.name]
    return asset.permanent_wrench_composer.composed_force_as_torch[:, asset_cfg.body_ids].view(env.num_envs, -1)


@configclass
class DigitLocoManipRewards(DigitRewards):
    joint_deviation_arms = None
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_leg_toe_roll"),
            "threshold": 0.5,
            "command_name": "base_velocity",
        },
    )

    joint_vel_hip_yaw = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_leg_hip_yaw"])},
    )

    left_ee_pos_tracking = RewTerm(
        func=manipulation_mdp.position_command_error,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="left_arm_wrist_yaw"),
            "command_name": "left_ee_pose",
        },
    )

    left_ee_pos_tracking_fine_grained = RewTerm(
        func=manipulation_mdp.position_command_error_tanh,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="left_arm_wrist_yaw"),
            "std": 0.05,
            "command_name": "left_ee_pose",
        },
    )

    left_end_effector_orientation_tracking = RewTerm(
        func=manipulation_mdp.orientation_command_error,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="left_arm_wrist_yaw"),
            "command_name": "left_ee_pose",
        },
    )

    right_ee_pos_tracking = RewTerm(
        func=manipulation_mdp.position_command_error,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="right_arm_wrist_yaw"),
            "command_name": "right_ee_pose",
        },
    )

    right_ee_pos_tracking_fine_grained = RewTerm(
        func=manipulation_mdp.position_command_error_tanh,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="right_arm_wrist_yaw"),
            "std": 0.05,
            "command_name": "right_ee_pose",
        },
    )

    right_end_effector_orientation_tracking = RewTerm(
        func=manipulation_mdp.orientation_command_error,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="right_arm_wrist_yaw"),
            "command_name": "right_ee_pose",
        },
    )


@configclass
class DigitLocoManipTerminations(TerminationsCfg):
    hip_or_arm_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_leg_hip_.*", ".*_arm_.*"]),
            "threshold": 1.0,
        },
    )


@configclass
class DigitLocoManipObservations:
    """Configuration for the Digit Locomanipulation environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        left_ee_pose_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "left_ee_pose"},
        )
        right_ee_pose_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "right_ee_pose"},
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES + ARM_JOINT_NAMES)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES + ARM_JOINT_NAMES)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy = PolicyCfg()


@configclass
class DigitLocoManipCommands:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.05,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1.2),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(0.0, 0.0),
        ),
    )

    left_ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="left_arm_wrist_yaw",
        resampling_time_range=(1.0, 3.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.10, 0.50),
            pos_y=(0.05, 0.50),
            pos_z=(-0.20, 0.20),
            roll=(-0.1, 0.1),
            pitch=(-0.1, 0.1),
            yaw=(math.pi / 2.0 - 0.1, math.pi / 2.0 + 0.1),
        ),
    )

    right_ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="right_arm_wrist_yaw",
        resampling_time_range=(1.0, 3.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.10, 0.50),
            pos_y=(-0.50, -0.05),
            pos_z=(-0.20, 0.20),
            roll=(-0.1, 0.1),
            pitch=(-0.1, 0.1),
            yaw=(-math.pi / 2.0 - 0.1, -math.pi / 2.0 + 0.1),
        ),
    )


@configclass
class DigitEvents(EventCfg):
    # Add an external force to simulate a payload being carried.
    left_hand_force = EventTermCfg(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="left_arm_wrist_yaw"),
            "force_range": (-10.0, 10.0),
            "torque_range": (-1.0, 1.0),
        },
    )

    right_hand_force = EventTermCfg(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="right_arm_wrist_yaw"),
            "force_range": (-10.0, 10.0),
            "torque_range": (-1.0, 1.0),
        },
    )


@configclass
class DigitLocoManipEnvCfg(DigitRoughEnvCfg):
    rewards: DigitLocoManipRewards = DigitLocoManipRewards()
    observations: DigitLocoManipObservations = DigitLocoManipObservations()
    commands: DigitLocoManipCommands = DigitLocoManipCommands()
    terminations: DigitLocoManipTerminations = DigitLocoManipTerminations()

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 14.0

        # Rewards:
        self.rewards.flat_orientation_l2.weight = -10.5
        self.rewards.termination_penalty.weight = -100.0

        # Change terrain to flat.
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # Remove height scanner.
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # Remove terrain curriculum.
        self.curriculum.terrain_levels = None


class DigitLocoManipEnvCfg_PLAY(DigitLocoManipEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        # Make a smaller scene for play.
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # Disable randomization for play.
        self.observations.policy.enable_corruption = False
        # Remove random pushing.
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class DigitRoughLocoManipObservations:
    """Configuration for the rough-terrain Digit locomanipulation environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        left_ee_pose_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "left_ee_pose"},
        )
        right_ee_pose_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "right_ee_pose"},
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES + ARM_JOINT_NAMES)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES + ARM_JOINT_NAMES)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy = PolicyCfg()

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
class DigitRoughLocoManipEvents(EventCfg):
    """Events for the rough-terrain Digit locomanipulation environment."""

    left_hand_force = EventTermCfg(
        func=apply_random_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="left_arm_wrist_yaw"),
            "force_magnitude_range": (0.0, 15.0),
            "torque": (0.0, 0.0, 0.0),
            "is_global": True,
        },
    )

    right_hand_force = EventTermCfg(
        func=apply_random_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="right_arm_wrist_yaw"),
            "force_magnitude_range": (0.0, 15.0),
            "torque": (0.0, 0.0, 0.0),
            "is_global": True,
        },
    )


@configclass
class DigitRoughLocoManipCurriculum:
    """Curriculum for the rough-terrain Digit locomanipulation environment."""

    terrain_levels = CurriculumTermCfg(func=terrain_levels_vel_locomanip)


@configclass
class DigitRoughLocoManipEnvCfg(DigitRoughEnvCfg):
    rewards: DigitLocoManipRewards = DigitLocoManipRewards()
    observations: DigitRoughLocoManipObservations = DigitRoughLocoManipObservations()
    commands: DigitLocoManipCommands = DigitLocoManipCommands()
    terminations: DigitLocoManipTerminations = DigitLocoManipTerminations()
    events: DigitRoughLocoManipEvents = DigitRoughLocoManipEvents()
    curriculum: DigitRoughLocoManipCurriculum = DigitRoughLocoManipCurriculum()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 50.0


class DigitRoughLocoManipEnvCfg_PLAY(DigitRoughLocoManipEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        # Make a smaller scene for play.
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # Spawn the robot randomly in the grid (instead of their terrain levels).
        self.scene.terrain.max_init_terrain_level = None
        # Reduce the number of terrains to save memory.
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # Disable randomization for play.
        self.observations.policy.enable_corruption = False
        # Remove random pushing and training-time arm load events.
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.left_hand_force = None
        self.events.right_hand_force = None
