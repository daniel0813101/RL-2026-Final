# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab_tasks.manager_based.manipulation.reach.mdp as manipulation_mdp
from isaaclab_assets.robots.agility import ARM_JOINT_NAMES, LEG_JOINT_NAMES

from .loco_manip_env_cfg_ovx import (
    DigitLocoManipEnvCfg,
    DigitLocoManipObservations,
    DigitLocoManipRewards,
    DigitRoughLocoManipEnvCfg,
)


def _pbrs_potential(
    env: ManagerBasedRLEnv,
    potential_func: Callable,
    asset_cfg: SceneEntityCfg | None = None,
    sensor_cfg: SceneEntityCfg | None = None,
    command_name: str | None = None,
    threshold: float | None = None,
    std: float | None = None,
    command_threshold: float | None = None,
    yaw_threshold: float | None = None,
    target_width: float | None = None,
    min_width: float | None = None,
    max_width: float | None = None,
    center_weight: float | None = None,
) -> torch.Tensor:
    """Evaluate the original reward function used as the PBRS potential."""
    params = {}
    if asset_cfg is not None:
        params["asset_cfg"] = asset_cfg
    if sensor_cfg is not None:
        params["sensor_cfg"] = sensor_cfg
    if command_name is not None:
        params["command_name"] = command_name
    if threshold is not None:
        params["threshold"] = threshold
    if std is not None:
        params["std"] = std
    if command_threshold is not None:
        params["command_threshold"] = command_threshold
    if yaw_threshold is not None:
        params["yaw_threshold"] = yaw_threshold
    if target_width is not None:
        params["target_width"] = target_width
    if min_width is not None:
        params["min_width"] = min_width
    if max_width is not None:
        params["max_width"] = max_width
    if center_weight is not None:
        params["center_weight"] = center_weight
    return potential_func(env, **params)


def pbrs_delta(
    env: ManagerBasedRLEnv,
    cache_name: str,
    potential_func: Callable,
    asset_cfg: SceneEntityCfg | None = None,
    sensor_cfg: SceneEntityCfg | None = None,
    command_name: str | None = None,
    threshold: float | None = None,
    std: float | None = None,
    command_threshold: float | None = None,
    yaw_threshold: float | None = None,
    target_width: float | None = None,
    min_width: float | None = None,
    max_width: float | None = None,
    center_weight: float | None = None,
) -> torch.Tensor:
    """Return (phi(s') - phi(s)) / step_dt, guarded on terminated/reset envs."""
    current_phi = _pbrs_potential(
        env,
        potential_func,
        asset_cfg,
        sensor_cfg,
        command_name,
        threshold,
        std,
        command_threshold,
        yaw_threshold,
        target_width,
        min_width,
        max_width,
        center_weight,
    )
    previous_phi = getattr(env, "_pbrs_phi_prev", {}).get(cache_name)
    if previous_phi is None:
        previous_phi = current_phi
    delta_phi = (~env.reset_buf).to(current_phi.dtype) * (current_phi - previous_phi)
    return delta_phi / env.step_dt


def feet_air_time_positive_biped_yaw_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    yaw_threshold: float = 0.1,
) -> torch.Tensor:
    """Reward biped stepping when either linear or yaw command asks the robot to move."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)

    command = env.command_manager.get_command(command_name)
    moving = (torch.norm(command[:, :2], dim=1) > 0.1) | (torch.abs(command[:, 2]) > yaw_threshold)
    return reward * moving


def stand_still_joint_deviation_l1_yaw_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.06,
    yaw_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint deviation only when both linear and yaw commands are near zero."""
    command = env.command_manager.get_command(command_name)
    standing = (torch.norm(command[:, :2], dim=1) < command_threshold) & (torch.abs(command[:, 2]) < yaw_threshold)
    return mdp.joint_deviation_l1(env, asset_cfg) * standing


def leg_mirror_symmetry_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES),
    command_name: str | None = None,
    threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize instantaneous left/right leg joint asymmetry while commanded to walk."""
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos
    device = joint_pos.device

    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, slice):
        joint_ids = list(range(asset.num_joints))[joint_ids]
    joint_names = [asset.joint_names[joint_id] for joint_id in joint_ids]
    joint_name_to_id = dict(zip(joint_names, joint_ids, strict=False))

    same_left_ids: list[int] = []
    same_right_ids: list[int] = []
    opposite_left_ids: list[int] = []
    opposite_right_ids: list[int] = []
    same_sign_tokens = ("hip_pitch", "knee", "toe_a", "toe_b")

    for joint_name, joint_id in zip(joint_names, joint_ids, strict=False):
        if "left" not in joint_name:
            continue
        counterpart_id = joint_name_to_id.get(joint_name.replace("left", "right", 1))
        if counterpart_id is None:
            continue
        if any(token in joint_name for token in same_sign_tokens):
            same_left_ids.append(joint_id)
            same_right_ids.append(counterpart_id)
        else:
            opposite_left_ids.append(joint_id)
            opposite_right_ids.append(counterpart_id)

    penalty = torch.zeros(env.num_envs, device=device)
    if same_left_ids:
        penalty += torch.sum(torch.abs(joint_pos[:, same_left_ids] - joint_pos[:, same_right_ids]), dim=1)
    if opposite_left_ids:
        penalty += torch.sum(torch.abs(joint_pos[:, opposite_left_ids] + joint_pos[:, opposite_right_ids]), dim=1)

    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        penalty *= (torch.norm(command[:, :2], dim=1) > threshold).to(penalty.dtype)
    return penalty


def feet_lateral_width_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    min_width: float = 0.38,
    max_width: float = 0.55,
    center_weight: float = 1.0,
) -> torch.Tensor:
    """Penalize lateral foot width outside a target range and lateral center offset in the base frame."""
    robot = env.scene[asset_cfg.name]

    foot_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :]
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w

    num_envs, num_feet, _ = foot_pos_w.shape
    rel_pos_w = foot_pos_w - root_pos_w[:, None, :]

    root_quat_w_flat = root_quat_w[:, None, :].expand(-1, num_feet, -1).reshape(-1, 4)
    foot_pos_b_flat = math_utils.quat_rotate_inverse(root_quat_w_flat, rel_pos_w.reshape(-1, 3))
    foot_pos_b = foot_pos_b_flat.reshape(num_envs, num_feet, 3)

    y_left = foot_pos_b[:, 0, 1]
    y_right = foot_pos_b[:, 1, 1]

    width = torch.abs(y_left - y_right)
    width_error = torch.clamp(min_width - width, min=0.0) + torch.clamp(width - max_width, min=0.0)
    center_error = 0.5 * (y_left + y_right)
    return width_error.square() + center_weight * center_error.square()


class PbrsManagerBasedRLEnv(ManagerBasedRLEnv):
    """Manager-based RL env with PBRS potentials cached immediately before physics."""

    def _cache_pbrs_potentials(self):
        if not hasattr(self, "_pbrs_phi_prev"):
            self._pbrs_phi_prev = {}

        for term_cfg in self.reward_manager._term_cfgs:
            if term_cfg.func is not pbrs_delta:
                continue
            cache_name = term_cfg.params["cache_name"]
            self._pbrs_phi_prev[cache_name] = _pbrs_potential(
                self,
                term_cfg.params["potential_func"],
                term_cfg.params.get("asset_cfg"),
                term_cfg.params.get("sensor_cfg"),
                term_cfg.params.get("command_name"),
                term_cfg.params.get("threshold"),
                term_cfg.params.get("std"),
                term_cfg.params.get("command_threshold"),
                term_cfg.params.get("yaw_threshold"),
                term_cfg.params.get("target_width"),
                term_cfg.params.get("min_width"),
                term_cfg.params.get("max_width"),
                term_cfg.params.get("center_weight"),
            ).detach().clone()

    def step(self, action: torch.Tensor):
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()
        self._cache_pbrs_potentials()

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()
            self.recorder_manager.record_post_reset(reset_env_ids)

        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.obs_buf = self.observation_manager.compute(update_history=True)
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras


@configclass
class DigitPbrsLocoManipRewards(DigitLocoManipRewards):
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=2.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-8.0,
    )
    feet_air_time = RewTerm(
        func=feet_air_time_positive_biped_yaw_command,
        weight=1.4,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_leg_toe_roll"),
            "threshold": 0.8,
            "command_name": "base_velocity",
            "yaw_threshold": 0.1,
        },
    )
    stand_still = RewTerm(
        func=stand_still_joint_deviation_l1_yaw_command,
        weight=-0.4,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.06,
            "yaw_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES),
        },
    )
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=0.0)
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES + ARM_JOINT_NAMES)},
    )
    joint_deviation_hip_roll = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_leg_hip_roll")},
    )
    joint_deviation_hip_yaw = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_leg_hip_yaw")},
    )
    joint_deviation_knee = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_tarsus")},
    )
    joint_deviation_feet = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_toe_a", ".*_toe_b"])},
    )
    leg_mirror_symmetry = RewTerm(
        func=leg_mirror_symmetry_l1,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES),
            "command_name": "base_velocity",
            "threshold": 0.1,
        },
    )
    feet_lateral_width = RewTerm(
        func=feet_lateral_width_penalty,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_leg_toe_roll", "right_leg_toe_roll"],
                preserve_order=True,
            ),
            "min_width": 0.38,
            "max_width": 0.55,
            "center_weight": 0.8,
        },
    )
    left_ee_pos_tracking_fine_grained = RewTerm(
        func=manipulation_mdp.position_command_error_tanh,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="left_arm_wrist_yaw"),
            "std": 0.05,
            "command_name": "left_ee_pose",
        },
    )
    right_ee_pos_tracking_fine_grained = RewTerm(
        func=manipulation_mdp.position_command_error_tanh,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="right_arm_wrist_yaw"),
            "std": 0.05,
            "command_name": "right_ee_pose",
        },
    )

    feet_air_time_pb = RewTerm(
        func=pbrs_delta,
        weight=0.0,
        params={
            "cache_name": "feet_air_time",
            "potential_func": feet_air_time_positive_biped_yaw_command,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_leg_toe_roll"),
            "threshold": 0.8,
            "command_name": "base_velocity",
            "yaw_threshold": 0.1,
        },
    )
    dof_torques_l2_pb = RewTerm(
        func=pbrs_delta,
        weight=-1.0e-6,
        params={"cache_name": "dof_torques_l2", "potential_func": mdp.joint_torques_l2},
    )
    dof_acc_l2_pb = RewTerm(
        func=pbrs_delta,
        weight=-2.0e-7,
        params={
            "cache_name": "dof_acc_l2",
            "potential_func": mdp.joint_acc_l2,
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES + ARM_JOINT_NAMES),
        },
    )
    joint_deviation_hip_roll_pb = RewTerm(
        func=pbrs_delta,
        weight=-0.1,
        params={
            "cache_name": "joint_deviation_hip_roll",
            "potential_func": mdp.joint_deviation_l1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_leg_hip_roll"),
        },
    )
    joint_deviation_hip_yaw_pb = RewTerm(
        func=pbrs_delta,
        weight=-0.05,
        params={
            "cache_name": "joint_deviation_hip_yaw",
            "potential_func": mdp.joint_deviation_l1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_leg_hip_yaw"),
        },
    )
    joint_deviation_knee_pb = RewTerm(
        func=pbrs_delta,
        weight=-0.2,
        params={
            "cache_name": "joint_deviation_knee",
            "potential_func": mdp.joint_deviation_l1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_tarsus"),
        },
    )
    joint_deviation_feet_pb = RewTerm(
        func=pbrs_delta,
        weight=-0.1,
        params={
            "cache_name": "joint_deviation_feet",
            "potential_func": mdp.joint_deviation_l1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_toe_a", ".*_toe_b"]),
        },
    )
    leg_mirror_symmetry_pb = RewTerm(
        func=pbrs_delta,
        weight=0.0,
        params={
            "cache_name": "leg_mirror_symmetry",
            "potential_func": leg_mirror_symmetry_l1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES),
            "command_name": "base_velocity",
            "threshold": 0.1,
        },
    )
    feet_lateral_width_pb = RewTerm(
        func=pbrs_delta,
        weight=0.0,
        params={
            "cache_name": "feet_lateral_width",
            "potential_func": feet_lateral_width_penalty,
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_leg_toe_roll", "right_leg_toe_roll"],
                preserve_order=True,
            ),
            "min_width": 0.38,
            "max_width": 0.55,
            "center_weight": 0.8,
        },
    )
    left_ee_pos_tracking_fine_grained_pb = RewTerm(
        func=pbrs_delta,
        weight=1.0,
        params={
            "cache_name": "left_ee_pos_tracking_fine_grained",
            "potential_func": manipulation_mdp.position_command_error_tanh,
            "asset_cfg": SceneEntityCfg("robot", body_names="left_arm_wrist_yaw"),
            "std": 0.05,
            "command_name": "left_ee_pose",
        },
    )
    right_ee_pos_tracking_fine_grained_pb = RewTerm(
        func=pbrs_delta,
        weight=1.0,
        params={
            "cache_name": "right_ee_pos_tracking_fine_grained",
            "potential_func": manipulation_mdp.position_command_error_tanh,
            "asset_cfg": SceneEntityCfg("robot", body_names="right_arm_wrist_yaw"),
            "std": 0.05,
            "command_name": "right_ee_pose",
        },
    )


@configclass
class DigitPbrsLocoManipEnvCfg(DigitLocoManipEnvCfg):
    rewards: DigitPbrsLocoManipRewards = DigitPbrsLocoManipRewards()

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_standing_envs = 0.05
        self.commands.base_velocity.rel_heading_envs = 0.0


class DigitPbrsLocoManipEnvCfg_PLAY(DigitPbrsLocoManipEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class DigitPbrsRoughLocoManipEnvCfg(DigitRoughLocoManipEnvCfg):
    rewards: DigitPbrsLocoManipRewards = DigitPbrsLocoManipRewards()

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_standing_envs = 0.05
        self.commands.base_velocity.rel_heading_envs = 0.0


class DigitPbrsRoughLocoManipEnvCfg_PLAY(DigitPbrsRoughLocoManipEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.left_hand_force = None
        self.events.right_hand_force = None


@configclass
class DigitPbrsFlatPolicyRoughLocoManipEnvCfg_PLAY(DigitPbrsRoughLocoManipEnvCfg_PLAY):
    """Rough-terrain play env with flat-terrain policy observations for flat-trained checkpoints."""

    observations: DigitLocoManipObservations = DigitLocoManipObservations()
