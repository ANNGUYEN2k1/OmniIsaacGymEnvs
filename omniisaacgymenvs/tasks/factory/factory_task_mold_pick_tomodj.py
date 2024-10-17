# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 'AS IS'
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''Factory: Class for mold pick task.

Inherits mold environment class and abstract task class (not enforced). Can be executed with
ISAACPYTHON4 scripts/rlgames_train.py task=FactoryTaskMoldPick_TomoDJ
'''

import asyncio
import math
from typing import Dict, Tuple

import hydra
import omegaconf
import omni.isaac.core.utils.torch as torch_utils
import omni.kit
import torch
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.torch.transformations import tf_combine, tf_vector
from pxr import Gf, UsdGeom

import omniisaacgymenvs.tasks.factory.factory_control as fc
from omniisaacgymenvs.tasks.factory.factory_env_mold_tomodj import FactoryEnvMold
from omniisaacgymenvs.tasks.factory.factory_schema_class_task import FactoryABCTask
from omniisaacgymenvs.tasks.factory.factory_schema_config_task import FactorySchemaConfigTask


class FactoryTaskMoldPick_TomoDJ(FactoryEnvMold, FactoryABCTask):
    '''[summary]'''

    def __init__(self, name, sim_config, env, offset=None) -> None:
        '''Initialize environment superclass. Initialize instance variables.'''

        super().__init__(name, sim_config, env, offset)

        # Get task params
        self.hand = 'gripper'
        self._get_task_yaml_params()

        # Mold pose tensors
        self.mold_grasp_pos_local = torch.tensor([], dtype=torch.float32, device=self._device)
        self.mold_grasp_quat_local = torch.tensor([], dtype=torch.float32, device=self._device)
        self.mold_grasp_quat = torch.tensor([], dtype=torch.float32, device=self._device)
        self.mold_grasp_pos = torch.tensor([], dtype=torch.float32, device=self._device)

        # Vectors to align grasp pose
        self.hand_side_axis = torch.tensor([0, 1, 0], device=self._device, dtype=torch.float32).repeat(self._num_envs, 1)
        self.mold_side_axis = torch.tensor([0, -1, 0], device=self._device, dtype=torch.float32).repeat(self._num_envs, 1)

        # Keypoint tensors
        self.keypoint_hand_offsets = (
            self._get_keypoint_offsets(self.cfg_task.rl.num_keypoints) * self.cfg_task.rl.keypoint_scale
        )
        self.keypoint_mold_offsets = (
            self._get_keypoint_offsets(self.cfg_task.rl.num_keypoints) * self.cfg_task.rl.keypoint_scale
        )
        self.keypoints_hand = torch.zeros(
            (self._num_envs, self.cfg_task.rl.num_keypoints, 3),
            dtype=torch.float32,
            device=self._device
        )
        self.keypoints_mold = torch.zeros_like(self.keypoints_hand, device=self._device)

        # Action tensor
        self.actions = torch.zeros((self._num_envs, self._num_actions), device=self._device)

    def _get_task_yaml_params(self) -> None:
        '''Initialize instance variables from YAML files.'''

        # Set factory data
        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name='factory_schema_config_task', node=FactorySchemaConfigTask)

        # Task params
        self.cfg_task = omegaconf.OmegaConf.create(self._task_cfg)
        self.max_episode_length = self.cfg_task.rl.max_episode_length  # Required instance var for VecTask

        # Model params
        ppo_path = 'train/FactoryTaskMoldPick_TomoDJPPO.yaml'  # Relative to Gym's Hydra search path (cfg dir)
        self.cfg_ppo = hydra.compose(config_name=ppo_path)
        self.cfg_ppo = self.cfg_ppo['train']  # Strip superfluous nesting

    def post_reset(self) -> None:
        '''Reset the world. Called only once, before simulation begins.'''

        # Disable gravity if needed
        if self.cfg_task.sim.disable_gravity:
            self.disable_gravity()

        # Acquire new tensors
        self.acquire_base_tensors()
        self._acquire_task_tensors()

        # Refresh tensors
        self.refresh_base_tensors()
        self.refresh_env_tensors()
        self._refresh_task_tensors()

        # Reset all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        asyncio.ensure_future(self.reset_idx_async(indices, randomize_hand_pose=False))

    def _acquire_task_tensors(self) -> None:
        '''Acquire task tensors.'''

        # Mold local pose tensors
        mold_grasp_heights = self.mold_heights * 0.2
        mold_grasp_widths = self.mold_widths * 0.5
        mold_grasp_lengths = self.mold_lengths * 0.4
        self.mold_grasp_pos_local = torch.cat(
            (mold_grasp_lengths, mold_grasp_widths, mold_grasp_heights),
            dim=-1
        )
        self.mold_grasp_quat_local = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device).repeat(self._num_envs, 1)

    def pre_physics_step(self, actions) -> None:
        '''Reset environments. Apply actions from policy. Simulation step called after this method.'''

        # Check if the environments need to be reset
        if not self.world.is_playing():
            return
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids, randomize_hand_pose=True)

        # Apply actions
        self.actions = actions.clone().to(self._device)  # Shape = (num_envs, num_actions); values = [-1, 1]
        self._apply_actions_as_ctrl_targets(
            actions=self.actions,
            ctrl_target_hand_dof_pos=self.asset_info_robot_table.hand.gripper.hand_width_max,
            do_scale=True
        )

    async def pre_physics_step_async(self, actions) -> None:
        '''Reset environments. Apply actions from policy. Simulation step called after this method.'''

        # Check if the environments need to be reset
        if not self.world.is_playing():
            return
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            await self.reset_idx_async(env_ids, randomize_hand_pose=True)

        # Apply actions
        self.actions = actions.clone().to(self._device)  # Shape = (num_envs, num_actions); values = [-1, 1]
        self._apply_actions_as_ctrl_targets(
            actions=self.actions,
            ctrl_target_hand_dof_pos=self.asset_info_robot_table.hand.gripper.hand_width_max,
            do_scale=True
        )

    def reset_idx(self, env_ids, randomize_hand_pose) -> None:
        '''Reset specified environments.'''

        # Reset scene
        self._reset_robot(env_ids)
        self._reset_object(env_ids)
        if randomize_hand_pose:
            self._randomize_hand_pose(env_ids, sim_steps=self.cfg_task.env.num_hand_move_sim_steps)

        # Reset buffers
        self._reset_buffers(env_ids)

    async def reset_idx_async(self, env_ids, randomize_hand_pose) -> None:
        '''Reset specified environments.'''

        # Reset scene
        self._reset_robot(env_ids)
        self._reset_object(env_ids)
        if randomize_hand_pose:
            await self._randomize_hand_pose_async(env_ids, sim_steps=self.cfg_task.env.num_hand_move_sim_steps)

        # Reset buffers
        self._reset_buffers(env_ids)

    def _reset_robot(self, env_ids) -> None:
        '''Reset DOF states and DOF targets of TomoDJ.'''

        # TomoDJ's dof positions
        self.dof_pos[env_ids] = torch.cat(
            (
                torch.tensor(self.cfg_task.randomize.tomodj_arm_initial_dof_pos, dtype=torch.float32, device=self._device),
                torch.tensor(
                    [self.asset_info_robot_table.hand.gripper.hand_width_max] * 2,
                    dtype=torch.float32,
                    device=self._device
                )
            ),
            dim=-1
        )  # Shape = (num_envs, num_dofs)

        # TomoDJ's dof velocities
        self.dof_vel[env_ids] = 0.0  # Shape = (num_envs, num_dofs)

        # Set TomoDJ dof's target positions
        self.ctrl_target_dof_pos[env_ids] = self.dof_pos[env_ids]

        # Reset TomoDJ's dofs
        indices = env_ids.to(dtype=torch.int32)
        self.tomodjs.set_joint_positions(self.dof_pos[env_ids], indices=indices)
        self.tomodjs.set_joint_velocities(self.dof_vel[env_ids], indices=indices)

    def _reset_object(self, env_ids) -> None:
        '''Reset root states of mold.'''

        # Randomize root state of mold
        mold_noise_xy = 2 * (torch.rand((self._num_envs, 2), dtype=torch.float32, device=self._device) - 0.5)  # [-1, 1]
        mold_noise_xy = mold_noise_xy @ torch.diag(torch.tensor(self.cfg_task.randomize.mold_pos_xy_noise, device=self._device))

        # Mold's positions
        self.mold_pos[env_ids, 0] = self.cfg_task.randomize.mold_pos_xy_initial[0] + mold_noise_xy[env_ids, 0]
        self.mold_pos[env_ids, 1] = self.cfg_task.randomize.mold_pos_xy_initial[1] + mold_noise_xy[env_ids, 1]
        self.mold_pos[env_ids, 2] = self.cfg_base.env.table_height

        # Mold's orientations
        mold_z_euler = (-35 * (torch.rand((self._num_envs), dtype=torch.float32, device=self._device)) + 185) * math.pi / 180
        quat_tensor = torch.stack(
            [
                torch.cos(mold_z_euler / 2),
                torch.zeros_like(mold_z_euler),
                torch.zeros_like(mold_z_euler),
                torch.sin(mold_z_euler / 2)
            ],
            dim=-1
        )
        self.mold_quat[env_ids, :] = quat_tensor.to(dtype=torch.float32, device=self._device)

        # Mold's velocities
        self.mold_linvel[env_ids, :] = 0.0
        self.mold_angvel[env_ids, :] = 0.0

        # Reset molds
        indices = env_ids.to(dtype=torch.int32)
        self.molds.set_world_poses(self.mold_pos[env_ids] + self._env_pos[env_ids], self.mold_quat[env_ids], indices=indices)
        self.molds.set_velocities(torch.cat((self.mold_linvel[env_ids], self.mold_angvel[env_ids]), dim=1), indices=indices)

    def _reset_buffers(self, env_ids) -> None:
        '''Reset buffers.'''

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def _apply_actions_as_ctrl_targets(self, actions, ctrl_target_hand_dof_pos, do_scale) -> None:
        '''Apply actions from policy as position/rotation/force/torque targets.'''

        # Interpret actions as target pos displacements and set pos target
        pos_actions = actions[:, 0:3]
        if do_scale:
            pos_actions = pos_actions @ torch.diag(torch.tensor(self.cfg_task.rl.pos_action_scale, device=self._device))
        self.ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = actions[:, 3:6]
        if do_scale:
            rot_actions = rot_actions @ torch.diag(torch.tensor(self.cfg_task.rl.rot_action_scale, device=self._device))

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)
        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        if self.cfg_task.rl.clamp_rot:
            rot_actions_quat = torch.where(
                angle.unsqueeze(-1).repeat(1, 4) > self.cfg_task.rl.clamp_rot_thresh,
                rot_actions_quat,
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device).repeat(self._num_envs, 1)
            )
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

        # Interpret actions as target forces and target torques
        if self.cfg_ctrl['do_force_ctrl']:
            force_actions = actions[:, 6:9]
            if do_scale:
                force_actions = (
                    force_actions @ torch.diag(torch.tensor(self.cfg_task.rl.force_action_scale, device=self._device))
                )
            torque_actions = actions[:, 9:12]
            if do_scale:
                torque_actions = (
                    torque_actions @ torch.diag(torch.tensor(self.cfg_task.rl.torque_action_scale, device=self._device))
                )
            self.ctrl_target_fingertip_contact_wrench = torch.cat((force_actions, torque_actions), dim=-1)

        # Set hand dof's target positions
        self.ctrl_target_hand_dof_pos = ctrl_target_hand_dof_pos

        # Calculate control's signals and control TomoDJ
        self.generate_ctrl_signals()

    def post_physics_step(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        '''Step buffers. Refresh tensors. Compute observations and reward. Reset environments.'''

        # Update states
        self.progress_buf[:] += 1
        if self.world.is_playing():

            # In this policy, episode length is constant
            is_last_step = self.progress_buf[0] == self.max_episode_length - 1
            if is_last_step:

                # At this point, robot has executed RL policy. Now close hand and lift (open-loop)
                if self.cfg_task.env.close_and_lift:
                    self._close_hand(sim_steps=self.cfg_task.env.num_hand_close_sim_steps)
                    self._lift_hand(
                        tomodj_hand_width=0.0,
                        lift_distance=0.3,
                        sim_steps=self.cfg_task.env.num_hand_lift_sim_steps
                    )
            self.refresh_base_tensors()
            self.refresh_env_tensors()
            self._refresh_task_tensors()
            self.get_observations()
            self.get_states()
            self.calculate_metrics()
            self.get_extras()
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    async def post_physics_step_async(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        '''Step buffers. Refresh tensors. Compute observations and reward. Reset environments.'''

        # Update states
        self.progress_buf[:] += 1
        if self.world.is_playing():

            # In this policy, episode length is constant
            is_last_step = self.progress_buf[0] == self.max_episode_length - 1
            if self.cfg_task.env.close_and_lift:

                # At this point, robot has executed RL policy. Now close hand and lift (open-loop)
                if is_last_step:
                    await self._close_hand_async(sim_steps=self.cfg_task.env.num_hand_close_sim_steps)
                    await self._lift_hand_async(sim_steps=self.cfg_task.env.num_hand_lift_sim_steps)
            self.refresh_base_tensors()
            self.refresh_env_tensors()
            self._refresh_task_tensors()
            self.get_observations()
            self.get_states()
            self.calculate_metrics()
            self.get_extras()
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _refresh_task_tensors(self):
        '''Refresh tensors.'''

        # Compute mold's grasp pose
        self.mold_grasp_quat, self.mold_grasp_pos = tf_combine(
            self.mold_quat,
            self.mold_pos,
            self.mold_grasp_quat_local,
            self.mold_grasp_pos_local
        )

        # Compute pos of keypoints on hand and mold in world frame
        for idx, keypoint_offset in enumerate(self.keypoint_hand_offsets):
            self.keypoints_hand[:, idx] = tf_combine(
                self.fingertip_midpoint_quat,
                self.fingertip_midpoint_pos,
                self.identity_quat,
                keypoint_offset.repeat(self._num_envs, 1)
            )[1]
        for idx, keypoint_offset in enumerate(self.keypoint_mold_offsets):
            self.keypoints_mold[:, idx] = tf_combine(
                self.mold_grasp_quat,
                self.mold_grasp_pos,
                self.identity_quat,
                keypoint_offset.repeat(self._num_envs, 1)
            )[1]

    def get_observations(self) -> dict:
        '''Compute observations.'''

        # # Mold and hand's keypoints poses
        # for i in range(self._num_envs):
        #     for j in range(self.cfg_task.rl.num_keypoints):

        #         # Hand
        #         hand_pose_cube = UsdGeom.Xformable(self._stage.GetPrimAtPath(f'/World/envs/env_{i}/hand_{j}'))
        #         hand_pose_cube_ops = hand_pose_cube.GetOrderedXformOps()
        #         hand_pose_cube_translate_op = None
        #         hand_pose_cube_orient_op = None
        #         for op in hand_pose_cube_ops:
        #             if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
        #                 hand_pose_cube_translate_op = op
        #             elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
        #                 hand_pose_cube_orient_op = op
        #         position = self.keypoints_hand[i, j, :].cpu().tolist()
        #         orientation = self.fingertip_midpoint_quat[i].cpu().tolist()
        #         hand_pose_cube_translate_op.Set(Gf.Vec3f(position[0], position[1], position[2]))
        #         hand_pose_cube_orient_op.Set(Gf.Quatd(orientation[0], orientation[1], orientation[2], orientation[3]))

        #         # Mold
        #         mold_pose_cube = UsdGeom.Xformable(self._stage.GetPrimAtPath(f'/World/envs/env_{i}/mold_{j}'))
        #         mold_pose_cube_ops = mold_pose_cube.GetOrderedXformOps()
        #         mold_pose_cube_translate_op = None
        #         for op in mold_pose_cube_ops:
        #             if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
        #                 mold_pose_cube_translate_op = op
        #         position = self.keypoints_mold[i, j, :].cpu().tolist()
        #         mold_pose_cube_translate_op.Set(Gf.Vec3f(position[0], position[1], position[2]))

        # Shallow copies of tensors
        obs_tensors = [
            self.fingertip_midpoint_pos,
            self.fingertip_midpoint_quat,
            self.fingertip_midpoint_linvel,
            self.fingertip_midpoint_angvel,
            self.mold_grasp_pos,
            self.mold_grasp_quat
        ]
        self.obs_buf = torch.cat(obs_tensors, dim=-1)  # Shape = (num_envs, num_observations)
        observations = {self.tomodjs.name: {'obs_buf': self.obs_buf}}
        return observations

    def calculate_metrics(self) -> None:
        '''Update reward and reset buffers.'''

        self._update_reset_buf()
        self._update_rew_buf()

    def _update_reset_buf(self) -> None:
        '''Assign environments for reset if successful or failed.'''

        # If max episode length has been reached
        self.reset_buf[:] = torch.where(
            self.progress_buf[:] >= self.max_episode_length - 1,
            torch.ones_like(self.reset_buf),
            self.reset_buf
        )

    def _update_rew_buf(self) -> None:
        '''Compute reward at current timestep.'''

        # Action
        action_penalty = torch.norm(self.actions, p=2, dim=-1)  # * self.cfg_task.rl.action_penalty_scale

        # Around handle
        around_handle_reward = torch.zeros_like(action_penalty)
        dz_fingers = torch.abs(self.fingertip_attributes['left_finger'][0][:, 2] - self.fingertip_attributes['right_finger'][0][:, 2])
        dz_left_finger_vs_mold_grasp = torch.abs(self.fingertip_attributes['left_finger'][0][:, 2] - self.mold_grasp_pos[:, 2])
        dz_right_finger_vs_mold_grasp = torch.abs(self.fingertip_attributes['right_finger'][0][:, 2] - self.mold_grasp_pos[:, 2])
        around_handle_reward = torch.where(
            (
                (dz_fingers <= self.cfg_task.rl.fingertip_threshold)
                & (dz_left_finger_vs_mold_grasp <= self.cfg_task.rl.fingertip_threshold)
                & (dz_right_finger_vs_mold_grasp <= self.cfg_task.rl.fingertip_threshold)
            ),
            around_handle_reward + 1 - self.cfg_task.rl.fingertip_threshold * 100,
            around_handle_reward
        )
        
        # Collision
        contact_penalty = self.get_collision_contacts(interested_objects={'mold': 1})

        # Fingertips pos and rot
        fingertip_pos_reward = -torch.abs(
            torch.norm(self.fingertip_attributes['left_finger'][0] - self.mold_grasp_pos, p=2, dim=-1)
            + torch.norm(self.fingertip_attributes['right_finger'][0] - self.mold_grasp_pos, p=2, dim=-1)
            - self.asset_info_robot_table.hand.gripper.hand_width_max
        )
        fingertip_rot_reward = -calculate_angles(
            self.fingertip_attributes['left_finger'][0] - self.mold_grasp_pos,
            self.mold_grasp_pos - self.fingertip_attributes['right_finger'][0]
        )
        
        # Grasp pos and rot
        grasp_pos_reward = -torch.norm(self.fingertip_midpoint_pos - self.mold_grasp_pos, p=2, dim=-1)
        axis1 = tf_vector(self.fingertip_midpoint_quat, self.hand_side_axis)
        axis2 = tf_vector(self.mold_grasp_quat, self.mold_side_axis)
        dot1 = (
            torch.bmm(axis1.view(self._num_envs, 1, 3), axis2.view(self._num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        )  # alignment of side axis for hand
        # Reward for matching the side orientation of the hand to the mold (fingers wrapped)
        grasp_rot_reward = torch.sign(dot1) * dot1**2
        grasp_rot_reward = torch.where(grasp_rot_reward < 0.8, 0, grasp_rot_reward)
        grasp_rot_reward = torch.where(grasp_rot_reward >= 0.95, 0.5 + grasp_rot_reward, grasp_rot_reward) - 1

        # # Keypoints_distance
        # keypoint_reward = -self._get_keypoint_dist()

        # Reward buffer
        self.rew_buf = (
            -action_penalty * self.cfg_task.rl.action_penalty_scale
            + around_handle_reward * self.cfg_task.rl.around_handle_reward_scale
            - contact_penalty * self.cfg_task.rl.contact_penalty_scale
            + fingertip_pos_reward * self.cfg_task.rl.fingertip_pos_reward_scale
            + fingertip_rot_reward * self.cfg_task.rl.fingertip_rot_reward_scale
            + grasp_pos_reward * self.cfg_task.rl.grasp_pos_reward_scale
            + grasp_rot_reward * self.cfg_task.rl.grasp_rot_reward_scale
            # + keypoint_reward * self.cfg_task.rl.keypoint_reward_scale
        )

        # In this policy, episode length is constant across all envs
        is_last_step = self.progress_buf[0] == self.max_episode_length - 1
        if is_last_step:

            # Check if mold is picked up and above table
            lift_success = self._check_lift_success(height_multiple=3.0)
            self.rew_buf[:] += lift_success * self.cfg_task.rl.success_bonus
            self.extras['successes'] = torch.mean(lift_success.float())

        # print(
        #     self.progress_buf[0],
        #     -action_penalty.mean() * self.cfg_task.rl.action_penalty_scale,
        #     around_handle_reward.mean() * self.cfg_task.rl.around_handle_reward_scale,
        #     -contact_penalty.mean() * self.cfg_task.rl.contact_penalty_scale,
        #     fingertip_pos_reward.mean() * self.cfg_task.rl.fingertip_pos_reward_scale,
        #     fingertip_rot_reward.mean() * self.cfg_task.rl.fingertip_rot_reward_scale,
        #     grasp_pos_reward.mean() * self.cfg_task.rl.grasp_pos_reward_scale,
        #     grasp_rot_reward.mean() * self.cfg_task.rl.grasp_rot_reward_scale,
        #     # keypoint_reward.mean() * self.cfg_task.rl.keypoint_reward_scale
        # )

    def _get_keypoint_offsets(self, num_keypoints, axis='z_1') -> torch.Tensor:
        '''Get uniformly-spaced keypoints along a line of unit length, centered at 0.'''

        # Create keypoint's offsets
        keypoint_offsets = torch.zeros((num_keypoints, 3), device=self._device)
        offsets = torch.linspace(0.0, 1.0, num_keypoints, device=self._device) - 0.5
        axis_split = axis.split('_')
        if 'x' in axis:
            keypoint_offsets[:, 0] = float(axis_split[axis_split.index('x') + 1]) * offsets
        if 'y' in axis:
            keypoint_offsets[:, 1] = float(axis_split[axis_split.index('y') + 1]) * offsets
        if 'z' in axis:
            keypoint_offsets[:, -1] = float(axis_split[axis_split.index('z') + 1]) * offsets

        # Normalize
        keypoint_offsets /= torch.norm(keypoint_offsets, p=2, dim=-1)[0]
        return keypoint_offsets

    def _get_keypoint_dist(self) -> torch.Tensor:
        '''Get keypoint distance.'''

        keypoint_dist = torch.sum(
            torch.norm(self.keypoints_mold - self.keypoints_hand, p=2, dim=-1), dim=-1
        )
        return keypoint_dist

    def _close_hand(self, sim_steps=20) -> None:
        '''Fully close hand using controller. Called outside RL loop (i.e., after last step of episode).'''

        self._move_hand_to_dof_pos(hand_dof_pos=0.0, sim_steps=sim_steps)

    def _move_hand_to_dof_pos(self, hand_dof_pos, sim_steps=20) -> None:
        '''Move hand fingers to specified DOF position using controller.'''

        # Set target hand dofs
        delta_hand_pose = torch.zeros((self._num_envs, 6), device=self._device)  # No hand motion
        self._apply_actions_as_ctrl_targets(delta_hand_pose, hand_dof_pos, do_scale=False)

        # Step sim
        for _ in range(sim_steps):
            SimulationContext.step(self.world, render=True)

    def _lift_hand(self, tomodj_hand_width: torch.Tensor | float = 0.0, lift_distance=0.3, sim_steps=20) -> None:
        '''Lift hand by specified distance. Called outside RL loop (i.e., after last step of episode).'''

        # Set hand motion
        delta_hand_pose = torch.zeros([self._num_envs, 6], device=self._device)
        delta_hand_pose[:, 2] = lift_distance

        # Step sim
        for _ in range(sim_steps):
            self._apply_actions_as_ctrl_targets(delta_hand_pose, tomodj_hand_width, do_scale=False)
            SimulationContext.step(self.world, render=True)

    async def _close_hand_async(self, sim_steps=20) -> None:
        '''Fully close hand using controller. Called outside RL loop (i.e., after last step of episode).'''

        await self._move_hand_to_dof_pos_async(hand_dof_pos=0.0, sim_steps=sim_steps)

    async def _move_hand_to_dof_pos_async(self, hand_dof_pos, sim_steps=20) -> None:
        '''Move hand fingers to specified DOF position using controller.'''

        # Set target hand dofs
        delta_hand_pose = torch.zeros((self._num_envs, 6), device=self._device)  # No hand motion
        self._apply_actions_as_ctrl_targets(delta_hand_pose, hand_dof_pos, do_scale=False)

        # Step sim
        for _ in range(sim_steps):
            await omni.kit.app.get_app().next_update_async()

    async def _lift_hand_async(self, tomodj_hand_width: torch.Tensor | float = 0.0, lift_distance=0.3, sim_steps=20) -> None:
        '''Lift hand by specified distance. Called outside RL loop (i.e., after last step of episode).'''

        # Set hand motion
        delta_hand_pose = torch.zeros([self._num_envs, 6], device=self._device)
        delta_hand_pose[:, 2] = lift_distance

        # Step sim
        for _ in range(sim_steps):
            self._apply_actions_as_ctrl_targets(delta_hand_pose, tomodj_hand_width, do_scale=False)
            await omni.kit.app.get_app().next_update_async()

    def _check_lift_success(self, height_multiple) -> torch.Tensor:
        '''Check if mold is above table by more than specified multiple times height of mold.'''

        lift_success = torch.where(
            self.mold_pos[:, 2] > self.cfg_base.env.table_height + self.mold_heights.squeeze(-1) * height_multiple,
            torch.ones((self._num_envs), device=self._device),
            torch.zeros((self._num_envs), device=self._device)
        )
        return lift_success

    def _randomize_hand_pose(self, env_ids, sim_steps) -> None:
        '''Move hand to random pose.'''

        # Step once to update physx with the newly set joint positions from reset_robot()
        SimulationContext.step(self.world, render=True)

        # Set target pos above table
        self.ctrl_target_fingertip_midpoint_pos = (
            torch.tensor([0.0, 0.0, self.cfg_base.env.table_height], device=self._device)
            + torch.tensor(self.cfg_task.randomize.fingertip_midpoint_pos_initial, device=self._device)
        )
        self.ctrl_target_fingertip_midpoint_pos = (
            self.ctrl_target_fingertip_midpoint_pos.unsqueeze(0).repeat(self._num_envs, 1)
        )
        fingertip_midpoint_pos_noise = (
            2 * (torch.rand((self._num_envs, 3), dtype=torch.float32, device=self._device) - 0.5)
        )  # [-1, 1]
        fingertip_midpoint_pos_noise = (
            fingertip_midpoint_pos_noise
            @ torch.diag(torch.tensor(self.cfg_task.randomize.fingertip_midpoint_pos_noise, device=self._device))
        )
        self.ctrl_target_fingertip_midpoint_pos += fingertip_midpoint_pos_noise

        # Set target rot
        ctrl_target_fingertip_midpoint_euler = (
            torch.tensor(self.cfg_task.randomize.fingertip_midpoint_rot_initial, device=self._device)
        ).unsqueeze(0).repeat(self._num_envs, 1)
        fingertip_midpoint_rot_noise = (
            2 * (torch.rand((self._num_envs, 3), dtype=torch.float32, device=self._device) - 0.5)
        )  # [-1, 1]
        fingertip_midpoint_rot_noise = (
            fingertip_midpoint_rot_noise
            @ torch.diag(torch.tensor(self.cfg_task.randomize.fingertip_midpoint_rot_noise, device=self._device))
        )
        ctrl_target_fingertip_midpoint_euler += fingertip_midpoint_rot_noise
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            ctrl_target_fingertip_midpoint_euler[:, 0],
            ctrl_target_fingertip_midpoint_euler[:, 1],
            ctrl_target_fingertip_midpoint_euler[:, 2]
        )

        # Step sim and render
        for _ in range(sim_steps):
            if not self.world.is_playing():
                return

            # Refresh tensors
            self.refresh_base_tensors()
            self.refresh_env_tensors()
            self._refresh_task_tensors()

            # Compute actions
            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos,
                fingertip_midpoint_quat=self.fingertip_midpoint_quat,
                ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
                ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
                jacobian_type=self.cfg_ctrl['jacobian_type'],
                rot_error_type='axis_angle'
            )
            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
            actions = torch.zeros((self._num_envs, self.cfg_task.env.numActions), device=self._device)
            actions[:, :6] = delta_hand_pose

            # Apply actions
            self._apply_actions_as_ctrl_targets(
                actions=actions,
                ctrl_target_hand_dof_pos=self.asset_info_robot_table.hand.gripper.hand_width_max,
                do_scale=False
            )
            SimulationContext.step(self.world, render=True)

        # Reset TomoDJ's velocities
        self.dof_vel[env_ids, :] = torch.zeros_like(self.dof_vel[env_ids])
        indices = env_ids.to(dtype=torch.int32)
        self.tomodjs.set_joint_velocities(self.dof_vel[env_ids], indices=indices)

        # Step once to update physx with the newly set joint velocities
        SimulationContext.step(self.world, render=True)

    async def _randomize_hand_pose_async(self, env_ids, sim_steps) -> None:
        '''Move hand to random pose.'''

        # Step once to update physx with the newly set joint positions from reset_robot()
        await omni.kit.app.get_app().next_update_async()

        # Set target pos above table
        self.ctrl_target_fingertip_midpoint_pos = (
            torch.tensor([0.0, 0.0, self.cfg_base.env.table_height], device=self._device)
            + torch.tensor(self.cfg_task.randomize.fingertip_midpoint_pos_initial, device=self._device)
        )
        self.ctrl_target_fingertip_midpoint_pos = (
            self.ctrl_target_fingertip_midpoint_pos.unsqueeze(0).repeat(self._num_envs, 1)
        )
        fingertip_midpoint_pos_noise = (
            2 * (torch.rand((self._num_envs, 3), dtype=torch.float32, device=self._device) - 0.5)
        )  # [-1, 1]
        fingertip_midpoint_pos_noise = (
            fingertip_midpoint_pos_noise
            @ torch.diag(torch.tensor(self.cfg_task.randomize.fingertip_midpoint_pos_noise, device=self._device))
        )
        self.ctrl_target_fingertip_midpoint_pos += fingertip_midpoint_pos_noise

        # Set target rot
        ctrl_target_fingertip_midpoint_euler = (
            torch.tensor(self.cfg_task.randomize.fingertip_midpoint_rot_initial, device=self._device)
        ).unsqueeze(0).repeat(self._num_envs, 1)
        fingertip_midpoint_rot_noise = (
            2 * (torch.rand((self._num_envs, 3), dtype=torch.float32, device=self._device) - 0.5)
        )  # [-1, 1]
        fingertip_midpoint_rot_noise = (
            fingertip_midpoint_rot_noise
            @ torch.diag(torch.tensor(self.cfg_task.randomize.fingertip_midpoint_rot_noise, device=self._device))
        )
        ctrl_target_fingertip_midpoint_euler += fingertip_midpoint_rot_noise
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            ctrl_target_fingertip_midpoint_euler[:, 0],
            ctrl_target_fingertip_midpoint_euler[:, 1],
            ctrl_target_fingertip_midpoint_euler[:, 2]
        )

        # Step sim and render
        for _ in range(sim_steps):

            # Refresh tensors
            self.refresh_base_tensors()
            self.refresh_env_tensors()
            self._refresh_task_tensors()

            # Compute actions
            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos,
                fingertip_midpoint_quat=self.fingertip_midpoint_quat,
                ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
                ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
                jacobian_type=self.cfg_ctrl['jacobian_type'],
                rot_error_type='axis_angle'
            )
            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
            actions = torch.zeros((self._num_envs, self.cfg_task.env.numActions), device=self._device)
            actions[:, :6] = delta_hand_pose

            # Apply actions
            self._apply_actions_as_ctrl_targets(
                actions=actions,
                ctrl_target_hand_dof_pos=self.asset_info_robot_table.hand.gripper.hand_width_max,
                do_scale=False
            )
            await omni.kit.app.get_app().next_update_async()

        # Reset TomoDJ's velocities
        self.dof_vel[env_ids, :] = torch.zeros_like(self.dof_vel[env_ids])
        indices = env_ids.to(dtype=torch.int32)
        self.tomodjs.set_joint_velocities(self.dof_vel[env_ids], indices=indices)

        # Step once to update physx with the newly set joint velocities
        await omni.kit.app.get_app().next_update_async()


def calculate_angles(vector1: torch.Tensor, vector2: torch.Tensor):
    '''Calculate the angle between two 3D vectors (shape: (N, 3)).'''

    # Compute dot products
    dot_products = torch.einsum('ij,ij->i', vector1, vector2)

    # Compute magnitudes (norms)
    magnitudes1 = torch.norm(vector1, dim=1)
    magnitudes2 = torch.norm(vector2, dim=1)

    # Compute cosine of angles
    cos_theta = dot_products / (magnitudes1 * magnitudes2)

    # Compute angles in radians
    angles_rad = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))  # Clamp values to avoid numerical issues
    return angles_rad
