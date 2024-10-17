# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

'''Required python modules'''
import math
import os
import numpy as np
import torch

from omni.isaac.cloner import Cloner
from omni.isaac.core.objects import DynamicCuboid, VisualCuboid
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.rotations import tensor_clamp
from omni.isaac.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from omni.physx import get_physx_simulation_interface
from pxr import Gf, PhysicsSchemaTools, UsdGeom  # type: ignore

from omniisaacgymenvs.utils.usd_utils.create_instanceable_assets import FixedCuboid
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.cabinet import Cabinet
from omniisaacgymenvs.robots.articulations.tomodj import TomoDJ
from omniisaacgymenvs.robots.articulations.views.cabinet_view import CabinetView
from omniisaacgymenvs.robots.articulations.views.tomodj_view import TomoDJView

current_file_path = os.path.abspath(__file__)
data_folder_path = os.path.dirname(os.path.dirname(current_file_path)) + '/data'


class TomoDJCabinetTask(RLTask):
    '''[summary]'''

    def __init__(self, name, sim_config, env, offset=None) -> None:
        self.update_config(sim_config)

        self.distX_offset = 0.04
        self.dt = 1 / 60.0

        # dof_pos_scaled: 9, tomodj_dof_vel: 9, to_target: 3, cabinet_dof_pos[:, 3]: 1, cabinet_dof_vel[:, 3]: 1
        self._num_observations = 23
        self._num_actions = 9

        super().__init__(name, env, offset)

        self.tomodj = None

        self._tomodjs = None
        self._cabinets = None
        self._props = None

        self._stage = get_current_stage()

        self.num_cabinet_dofs = 0

        self.tomodj_local_grasp_pos = torch.tensor([], device=self._device, dtype=torch.float32)
        self.tomodj_local_grasp_rot = torch.tensor([], device=self._device, dtype=torch.float32)

        self.drawer_local_grasp_pos = torch.tensor([], device=self._device, dtype=torch.float32)
        self.drawer_local_grasp_rot = torch.tensor([], device=self._device, dtype=torch.float32)

        self.cabinet_dof_pos = torch.tensor([], device=self._device, dtype=torch.float32)
        self.cabinet_dof_vel = torch.tensor([], device=self._device, dtype=torch.float32)
        self.tomodj_dof_pos = torch.tensor([], device=self._device, dtype=torch.float32)

        self.tomodj_grasp_rot = torch.tensor([], device=self._device, dtype=torch.float32)
        self.tomodj_grasp_pos = torch.tensor([], device=self._device, dtype=torch.float32)
        self.drawer_grasp_rot = torch.tensor([], device=self._device, dtype=torch.float32)
        self.drawer_grasp_pos = torch.tensor([], device=self._device, dtype=torch.float32)

        self.tomodj_lfinger_pos = torch.tensor([], device=self._device, dtype=torch.float32)
        self.tomodj_lfinger_rot = torch.tensor([], device=self._device, dtype=torch.float32)
        self.tomodj_rfinger_pos = torch.tensor([], device=self._device, dtype=torch.float32)
        self.tomodj_rfinger_rot = torch.tensor([], device=self._device, dtype=torch.float32)

        self.num_tomodj_dofs = 0
        self.tomodj_dof_lower_limits = torch.tensor([], device=self._device, dtype=torch.float32)
        self.tomodj_dof_upper_limits = torch.tensor([], device=self._device, dtype=torch.float32)
        self.tomodj_dof_speed_scales = torch.tensor([], device=self._device, dtype=torch.float32)
        self.tomodj_dof_targets = torch.tensor([], device=self._device, dtype=torch.float32)

        self.default_prop_pos = torch.tensor([], device=self._device, dtype=torch.float32)
        self.default_prop_rot = torch.tensor([], device=self._device, dtype=torch.float32)
        self.prop_indices = torch.tensor([], device=self._device, dtype=torch.float32)

        # Physics Simulation
        self.physx_interface = None

        drawer_local_grasp_pose = torch.tensor([0.3, 0.01, 0.0, 1.0, 0.0, 0.0, 0.0], device=self._device)
        self.drawer_local_grasp_pos = drawer_local_grasp_pose[0:3].repeat((self._num_envs, 1))
        self.drawer_local_grasp_rot = drawer_local_grasp_pose[3:7].repeat((self._num_envs, 1))

        self.gripper_forward_axis = (
            torch.tensor([0, 0, 1], device=self._device, dtype=torch.float32).repeat((self._num_envs, 1))
        )
        self.drawer_inward_axis = (
            torch.tensor([-1, 0, 0], device=self._device, dtype=torch.float32).repeat((self._num_envs, 1))
        )
        self.gripper_up_axis = (
            torch.tensor([0, 1, 0], device=self._device, dtype=torch.float32).repeat((self._num_envs, 1))
        )
        self.drawer_up_axis = (
            torch.tensor([0, 0, 1], device=self._device, dtype=torch.float32).repeat((self._num_envs, 1))
        )

        self.tomodj_default_dof_pos = torch.tensor(
            [0.52, 0.52, 0.26, -0.26, -0.25, 0.17, -0.52, 0.035, 0.035], device=self._device
        )

        self.actions = torch.zeros((self._num_envs, self._num_actions), device=self._device)

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg['env']['numEnvs']
        self._env_spacing = self._task_cfg['env']['envSpacing']

        self._max_episode_length = self._task_cfg['env']['episodeLength']

        self.action_scale = self._task_cfg['env']['actionScale']
        self.start_position_noise = self._task_cfg['env']['startPositionNoise']
        self.start_rotation_noise = self._task_cfg['env']['startRotationNoise']
        self.num_props = self._task_cfg['env']['numProps']

        self.dof_vel_scale = self._task_cfg['env']['dofVelocityScale']
        self.dist_reward_scale = self._task_cfg['env']['distRewardScale']
        self.rot_reward_scale = self._task_cfg['env']['rotRewardScale']
        self.around_handle_reward_scale = self._task_cfg['env']['aroundHandleRewardScale']
        self.open_reward_scale = self._task_cfg['env']['openRewardScale']
        self.finger_dist_reward_scale = self._task_cfg['env']['fingerDistRewardScale']
        self.action_penalty_scale = self._task_cfg['env']['actionPenaltyScale']
        self.finger_close_reward_scale = self._task_cfg['env']['fingerCloseRewardScale']

    def set_up_scene(
        self,
        scene,
        replicate_physics=True,
        collision_filter_global_paths=None,
        filter_collisions=False,
        copy_from_source=False
    ) -> None:
        '''[summary]'''

        self.get_tomodj()
        self.get_cabinet()
        if self.num_props > 0:
            self.get_props()

        VisualCuboid(
            prim_path=self.default_zero_env_path + '/hand_grasp',
            name='hand_grasp',
            scale=[0.01, 0.01, 0.01],
            color=np.array([1.0, 0.0, 0.0])
        )
        VisualCuboid(
            prim_path=self.default_zero_env_path + '/drawer_grasp',
            name='drawer_grasp',
            scale=[0.01, 0.01, 0.01],
            color=np.array([0.0, 0.0, 1.0])
        )

        # Set up scene
        collision_filter_global_paths = [] if collision_filter_global_paths is None else collision_filter_global_paths
        RLTask.set_up_scene(
            self,
            scene,
            replicate_physics=replicate_physics,
            collision_filter_global_paths=collision_filter_global_paths,
            filter_collisions=filter_collisions,
            copy_from_source=copy_from_source)
        self._tomodjs = TomoDJView(prim_paths_expr='/World/envs/.*/tomodj', name='tomodj_view')
        self._cabinets = CabinetView(prim_paths_expr='/World/envs/.*/cabinet', name='cabinet_view')

        scene.add(self._tomodjs)
        scene.add(self._tomodjs.hands)
        for fingertip_view in self._tomodjs.fingertip_views.values():
            scene.add(fingertip_view)
        scene.add(self._cabinets)
        scene.add(self._cabinets.drawers)

        if self.num_props > 0:
            self._props = RigidPrimView(
                prim_paths_expr='/World/envs/.*/prop/.*', name='prop_view', reset_xform_properties=False
            )
            scene.add(self._props)

        self.init_data()
        return

    def initialize_views(self, scene):
        super().initialize_views(scene)
        if scene.object_exists('tomodj_view'):
            scene.remove_object('tomodj_view', registry_only=True)
        if scene.object_exists('hands_view'):
            scene.remove_object('hands_view', registry_only=True)
        for fingertip_view_name in self._tomodjs.fingertip_views:
            if scene.object_exists(fingertip_view_name):
                scene.remove_object(fingertip_view_name, registry_only=True)
        if scene.object_exists('cabinet_view'):
            scene.remove_object('cabinet_view', registry_only=True)
        if scene.object_exists('drawers_view'):
            scene.remove_object('drawers_view', registry_only=True)
        if scene.object_exists('prop_view'):
            scene.remove_object('prop_view', registry_only=True)
        self._tomodjs = TomoDJView(prim_paths_expr='/World/envs/.*/tomodj', name='tomodj_view')
        self._cabinets = CabinetView(prim_paths_expr='/World/envs/.*/cabinet', name='cabinet_view')

        scene.add(self._tomodjs)
        scene.add(self._tomodjs.hands)
        for fingertip_view in self._tomodjs.fingertip_views.values():
            scene.add(fingertip_view)
        scene.add(self._cabinets)
        scene.add(self._cabinets.drawers)

        if self.num_props > 0:
            self._props = RigidPrimView(
                prim_paths_expr='/World/envs/.*/prop/.*', name='prop_view', reset_xform_properties=False
            )
            scene.add(self._props)

        self.init_data()

    def get_tomodj(self):
        '''[summary]'''

        tomodj = TomoDJ(
            prim_path=self.default_zero_env_path + '/tomodj',
            name='tomodj',
            usd_path=(data_folder_path + '/TomoDJ/tomodj_instanceable.usd')
        )
        self._sim_config.apply_articulation_settings(
            'tomodj',
            get_prim_at_path(tomodj.prim_path),
            self._sim_config.parse_actor_config('tomodj')
        )
        self.tomodj = tomodj
        FixedCuboid(
            prim_path=self.default_zero_env_path + '/pillar',
            name='pillar',
            translation=[tomodj.position[0], tomodj.position[1], tomodj.position[2] / 2],
            scale=[0.3, 0.7, tomodj.position[2]],
            contact_offset=0.005
        )

    def get_cabinet(self):
        '''[summary]'''

        cabinet = Cabinet(
            prim_path=self.default_zero_env_path + '/cabinet',
            name='cabinet',
            usd_path=(data_folder_path + '/Cabinet/sektion_cabinet_instanceable_adjusted.usd')
        )
        self._sim_config.apply_articulation_settings(
            'cabinet',
            get_prim_at_path(cabinet.prim_path),
            self._sim_config.parse_actor_config('cabinet')
        )

    def get_props(self):
        '''[summary]'''

        prop_cloner = Cloner()
        drawer_pos = [0.0515, 0.0, 0.7172]
        prop_color = np.array([0.2, 0.4, 0.6])

        props_per_row = int(math.ceil(math.sqrt(self.num_props)))
        prop_size = 0.08
        prop_spacing = 0.09
        xmin = -0.5 * prop_spacing * (props_per_row - 1)
        zmin = -0.5 * prop_spacing * (props_per_row - 1)
        prop_count = 0

        prop_pos = []
        for j in range(props_per_row):
            prop_up = zmin + j * prop_spacing
            for k in range(props_per_row):
                if prop_count >= self.num_props:
                    break
                propx = xmin + k * prop_spacing
                prop_pos.append([propx, prop_up, 0.0])
                prop_count += 1

        prop = DynamicCuboid(
            prim_path=self.default_zero_env_path + '/prop/prop_0',
            name='prop',
            color=prop_color,
            size=prop_size,
            density=100.0
        )
        self._sim_config.apply_articulation_settings(
            'prop',
            get_prim_at_path(prop.prim_path),
            self._sim_config.parse_actor_config('prop')
        )

        prop_paths = [f'{self.default_zero_env_path}/prop/prop_{j}' for j in range(self.num_props)]
        prop_cloner.clone(
            source_prim_path=self.default_zero_env_path + '/prop/prop_0',
            prim_paths=prop_paths,
            positions=(np.array(prop_pos) + np.array(drawer_pos)).tolist(),
            replicate_physics=False
        )

    def init_data(self) -> None:
        '''[summary]'''

        def get_env_local_pose(env_pos, xformable, device):
            '''Compute pose in env-local coordinates'''
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device, dtype=torch.float)

        hand_pose = get_env_local_pose(
            self._env_pos[0],
            UsdGeom.Xformable(self._stage.GetPrimAtPath('/World/envs/env_0/tomodj/left_arm_link7')),
            self._device
        )
        lfinger_pose = get_env_local_pose(
            self._env_pos[0],
            UsdGeom.Xformable(self._stage.GetPrimAtPath('/World/envs/env_0/tomodj/left_finger')),
            self._device
        )
        rfinger_pose = get_env_local_pose(
            self._env_pos[0],
            UsdGeom.Xformable(self._stage.GetPrimAtPath('/World/envs/env_0/tomodj/right_finger')),
            self._device
        )

        finger_pose = torch.zeros(7, device=self._device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        finger_pose[3:7] = lfinger_pose[3:7]
        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])

        tomodj_local_grasp_pose_rot, tomodj_local_pose_pos = (
            tf_combine(hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3])
        )
        tomodj_local_pose_pos += torch.tensor([0.04, 0.0, 0.0], device=self._device)
        self.tomodj_local_grasp_pos = tomodj_local_pose_pos.repeat((self._num_envs, 1))
        self.tomodj_local_grasp_rot = tomodj_local_grasp_pose_rot.repeat((self._num_envs, 1))

        # Physics Simulation
        self.physx_interface = get_physx_simulation_interface()

    def get_collision_contact(self):
        '''[summary]'''

        contact_headers = self.physx_interface.get_contact_report()[0]
        for contact_header in contact_headers:
            collider_1 = str(PhysicsSchemaTools.intToSdfPath(contact_header.actor0))  # type: ignore
            collider_2 = str(PhysicsSchemaTools.intToSdfPath(contact_header.actor1))  # type: ignore

            contacts = [collider_1, collider_2]
            print(contacts)

    def get_observations(self) -> dict:
        hand_pos, hand_rot = self._tomodjs.hands.get_world_poses(clone=False)
        drawer_pos, drawer_rot = self._cabinets.drawers.get_world_poses(clone=False)
        tomodj_dof_pos = self._tomodjs.get_joint_positions(clone=False)
        tomodj_dof_vel = self._tomodjs.get_joint_velocities(clone=False)
        self.cabinet_dof_pos = self.ensure_tensor(self._cabinets.get_joint_positions(clone=False))
        self.cabinet_dof_vel = self.ensure_tensor(self._cabinets.get_joint_velocities(clone=False))
        self.tomodj_dof_pos = self.ensure_tensor(tomodj_dof_pos)

        (
            self.tomodj_grasp_rot,
            self.tomodj_grasp_pos,
            self.drawer_grasp_rot,
            self.drawer_grasp_pos
        ) = compute_grasp_transforms(
            hand_rot,
            hand_pos,
            self.tomodj_local_grasp_rot,
            self.tomodj_local_grasp_pos,
            drawer_rot,
            drawer_pos,
            self.drawer_local_grasp_rot,
            self.drawer_local_grasp_pos
        )

        for i in range(self._num_envs):
            grasp_pose_cube = UsdGeom.Xformable(self._stage.GetPrimAtPath(f'/World/envs/env_{i}/grasp_pose'))
            drawer_pose_cube = UsdGeom.Xformable(self._stage.GetPrimAtPath(f'/World/envs/env_{i}/drawer_pose'))
            grasp_pose_cube_ops = grasp_pose_cube.GetOrderedXformOps()
            drawer_pose_cube_ops = drawer_pose_cube.GetOrderedXformOps()
            grasp_pose_cube_translate_op = None
            grasp_pose_cube_orient_op = None

            for op in grasp_pose_cube_ops:
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    grasp_pose_cube_translate_op = op
                elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                    grasp_pose_cube_orient_op = op

            drawer_pose_cube_op = None
            for op in drawer_pose_cube_ops:
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    drawer_pose_cube_op = op
                    break

            position = (self.tomodj_grasp_pos[i] - self._env_pos[i]).cpu().tolist()
            orientation = self.tomodj_grasp_rot[i].cpu().tolist()
            grasp_pose_cube_translate_op.Set(Gf.Vec3f(position[0], position[1], position[2]))
            grasp_pose_cube_orient_op.Set(Gf.Quatd(orientation[0], orientation[1], orientation[2], orientation[3]))
            position = (self.drawer_grasp_pos[i] - self._env_pos[i]).cpu().tolist()
            drawer_pose_cube_op.Set(Gf.Vec3f(position[0], position[1], position[2]))

        tomodj_lfinger_pos, self.tomodj_lfinger_rot = (
            self._tomodjs.fingertip_views['left_fingers_view'].get_world_poses(clone=False)
        )
        tomodj_rfinger_pos, self.tomodj_rfinger_rot = (
            self._tomodjs.fingertip_views['right_fingers_view'].get_world_poses(clone=False)
        )
        self.tomodj_lfinger_pos = self.ensure_tensor(tomodj_lfinger_pos)
        self.tomodj_rfinger_pos = self.ensure_tensor(tomodj_rfinger_pos)

        dof_pos_scaled = (
            2.0 * (tomodj_dof_pos - self.tomodj_dof_lower_limits)
            / (self.tomodj_dof_upper_limits - self.tomodj_dof_lower_limits)
            - 1.0
        )
        to_target = self.drawer_grasp_pos - self.tomodj_grasp_pos
        self.obs_buf = torch.cat(
            (
                self.ensure_tensor(dof_pos_scaled),
                self.ensure_tensor(tomodj_dof_vel * self.dof_vel_scale),
                self.ensure_tensor(to_target),
                self.cabinet_dof_pos[:, 3].unsqueeze(-1),
                self.cabinet_dof_vel[:, 3].unsqueeze(-1)
            ),
            dim=-1
        )

        observations = {self._tomodjs.name: {'obs_buf': self.obs_buf}}
        # self.get_collision_contact()
        return observations

    def pre_physics_step(self, actions) -> None:
        if not self.world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        self.actions = actions.clone().to(self._device)
        targets = self.tomodj_dof_targets + self.tomodj_dof_speed_scales * self.dt * self.actions * self.action_scale
        self.tomodj_dof_targets[:] = tensor_clamp(targets, self.tomodj_dof_lower_limits, self.tomodj_dof_upper_limits)
        env_ids_int32 = torch.arange(self._tomodjs.count, dtype=torch.int32, device=self._device)

        self._tomodjs.set_joint_position_targets(self.tomodj_dof_targets, indices=env_ids_int32)

    def reset_idx(self, env_ids):
        '''[summary]'''

        indices = env_ids.to(dtype=torch.int32)
        num_indices = len(indices)

        # reset tomodj
        pos = tensor_clamp(
            (
                self.tomodj_default_dof_pos.unsqueeze(0)
                + 0.25 * (torch.rand((len(env_ids), self.num_tomodj_dofs), device=self._device) - 0.5)
            ),
            self.tomodj_dof_lower_limits,
            self.tomodj_dof_upper_limits
        )
        dof_pos = torch.zeros((num_indices, self._tomodjs.num_dof), device=self._device)
        dof_vel = torch.zeros((num_indices, self._tomodjs.num_dof), device=self._device)
        dof_pos[:, :] = pos
        self.tomodj_dof_targets[env_ids, :] = pos
        self.tomodj_dof_pos[env_ids, :] = pos

        # reset cabinet
        self._cabinets.set_joint_positions(
            torch.zeros((len(env_ids), self.num_cabinet_dofs), dtype=torch.float32, device=self._device), indices=indices
        )
        self._cabinets.set_joint_velocities(
            torch.zeros((len(env_ids), self.num_cabinet_dofs), dtype=torch.float32, device=self._device), indices=indices
        )

        # reset props
        if self.num_props > 0:
            self._props.set_world_poses(
                self.default_prop_pos[self.prop_indices[env_ids].flatten()],
                self.default_prop_rot[self.prop_indices[env_ids].flatten()],
                self.prop_indices[env_ids].flatten().to(torch.int32),
            )

        self._tomodjs.set_joint_position_targets(self.tomodj_dof_targets[env_ids], indices=indices)
        self._tomodjs.set_joint_positions(dof_pos, indices=indices)
        self._tomodjs.set_joint_velocities(dof_vel, indices=indices)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def post_reset(self):
        self.num_tomodj_dofs = self._tomodjs.num_dof
        self.tomodj_dof_pos = torch.zeros((self._num_envs, self.num_tomodj_dofs), device=self._device)
        dof_limits = self.ensure_tensor(self._tomodjs.get_dof_limits())
        self.tomodj_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.tomodj_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)
        self.tomodj_dof_speed_scales = torch.ones_like(self.tomodj_dof_lower_limits)
        self.tomodj_dof_speed_scales[self._tomodjs.finger_indices] = 0.1
        self.tomodj_dof_targets = torch.zeros((self._num_envs, self.num_tomodj_dofs), dtype=torch.float, device=self._device)

        self.num_cabinet_dofs = self._cabinets.num_dof

        if self.num_props > 0:
            default_prop_pos, default_prop_rot = self._props.get_world_poses()
            self.default_prop_pos = self.ensure_tensor(default_prop_pos)
            self.default_prop_rot = self.ensure_tensor(default_prop_rot)
            self.prop_indices = (
                torch.arange(self._num_envs * self.num_props, device=self._device).view(self._num_envs, self.num_props)
            )

        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        self.rew_buf[:] = self.compute_tomodj_reward(
            self.actions,
            self.cabinet_dof_pos,
            self.tomodj_grasp_pos,
            self.drawer_grasp_pos,
            self.tomodj_grasp_rot,
            self.drawer_grasp_rot,
            self.tomodj_lfinger_pos,
            self.tomodj_rfinger_pos,
            self.tomodj_dof_pos,
            self.gripper_forward_axis,
            self.drawer_inward_axis,
            self.gripper_up_axis,
            self.drawer_up_axis,
            self._num_envs,
            self.dist_reward_scale,
            self.rot_reward_scale,
            self.around_handle_reward_scale,
            self.open_reward_scale,
            self.finger_dist_reward_scale,
            self.action_penalty_scale,
            self.finger_close_reward_scale
        )

    def is_done(self) -> None:
        '''[summary]'''

        # reset if drawer is open or max length reached
        self.reset_buf = torch.where(
            self.cabinet_dof_pos[:, 3] > 0.19,
            torch.ones_like(self.reset_buf),
            self.reset_buf
        )
        self.reset_buf = torch.where(
            self.progress_buf >= self._max_episode_length - 1,
            torch.ones_like(self.reset_buf),
            self.reset_buf
        )

    def compute_tomodj_reward(
        self,
        actions: torch.Tensor,
        cabinet_dof_pos: torch.Tensor,
        tomodj_grasp_pos: torch.Tensor,
        drawer_grasp_pos: torch.Tensor,
        tomodj_grasp_rot: torch.Tensor,
        drawer_grasp_rot: torch.Tensor,
        tomodj_lfinger_pos: torch.Tensor,
        tomodj_rfinger_pos: torch.Tensor,
        joint_positions: torch.Tensor,
        gripper_forward_axis: torch.Tensor,
        drawer_inward_axis: torch.Tensor,
        gripper_up_axis: torch.Tensor,
        drawer_up_axis: torch.Tensor,
        num_envs: int,
        dist_reward_scale: float,
        rot_reward_scale: float,
        around_handle_reward_scale: float,
        open_reward_scale: float,
        finger_dist_reward_scale: float,
        action_penalty_scale: float,
        finger_close_reward_scale: float
    ) -> torch.Tensor:
        '''[summary]'''

        # distance from hand to the drawer
        d = torch.norm(tomodj_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
        dist_reward = -d

        axis1 = tf_vector(tomodj_grasp_rot, gripper_forward_axis)
        axis2 = tf_vector(drawer_grasp_rot, drawer_inward_axis)
        axis3 = tf_vector(tomodj_grasp_rot, gripper_up_axis)
        axis4 = tf_vector(drawer_grasp_rot, drawer_up_axis)

        dot1 = (
            torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        )  # alignment of forward axis for gripper
        dot2 = (
            torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        )  # alignment of up axis for gripper
        # reward for matching the orientation of the hand to the drawer (fingers wrapped)
        rot_reward = 0.5 * (torch.sign(dot1) * dot1**2 + torch.sign(dot2) * dot2**2)
        rot_reward = torch.where(rot_reward[:] < 0.6, 0, rot_reward)
        rot_reward = torch.where(rot_reward[:] >= 0.8, 2 * rot_reward, rot_reward)

        # bonus if left finger is above the drawer handle and right below
        around_handle_reward = torch.zeros_like(rot_reward)
        around_handle_reward = torch.where(
            tomodj_lfinger_pos[:, 2] <= drawer_grasp_pos[:, 2],
            torch.where(
                tomodj_rfinger_pos[:, 2] >= drawer_grasp_pos[:, 2],
                around_handle_reward + 0.5,
                around_handle_reward
            ),
            around_handle_reward
        )
        # reward for distance of each finger from the drawer
        finger_dist_reward = torch.zeros_like(rot_reward)
        lfinger_dist = torch.abs(tomodj_lfinger_pos[:, 2] - drawer_grasp_pos[:, 2])
        rfinger_dist = torch.abs(tomodj_rfinger_pos[:, 2] - drawer_grasp_pos[:, 2])
        finger_dist_reward = torch.where(
            tomodj_lfinger_pos[:, 2] <= drawer_grasp_pos[:, 2],
            torch.where(
                tomodj_rfinger_pos[:, 2] >= drawer_grasp_pos[:, 2],
                (0.04 - lfinger_dist) + (0.04 - rfinger_dist),
                finger_dist_reward
            ),
            finger_dist_reward
        )

        finger_close_reward = torch.zeros_like(rot_reward)
        finger_close_reward = torch.where(
            d <= 0.03,
            (0.04 - joint_positions[:, 7]) + (0.04 - joint_positions[:, 8]),
            finger_close_reward
        )

        # regularization on the actions (summed for each environment)
        action_penalty = torch.sum(actions**2, dim=-1)

        # how far the cabinet has been opened out
        open_reward = cabinet_dof_pos[:, 3] * around_handle_reward + cabinet_dof_pos[:, 3]  # drawer_top_joint

        rewards = (
            dist_reward_scale * dist_reward
            + rot_reward_scale * rot_reward
            + around_handle_reward_scale * around_handle_reward
            + open_reward_scale * open_reward
            + finger_dist_reward_scale * finger_dist_reward
            - action_penalty_scale * action_penalty
            + finger_close_reward_scale * finger_close_reward
        )

        # bonus for opening drawer properly
        rewards = torch.where(cabinet_dof_pos[:, 3] > 0.01, rewards + 0.5, rewards)
        rewards = torch.where(cabinet_dof_pos[:, 3] > 0.1, rewards + around_handle_reward, rewards)
        rewards = torch.where(cabinet_dof_pos[:, 3] > 0.19, rewards + (2.0 * around_handle_reward), rewards)

        # # prevent bad style in opening drawer
        # rewards = torch.where(tomodj_lfinger_pos[:, 0] < drawer_grasp_pos[:, 0] - distX_offset,
        #                       torch.ones_like(rewards) * -1, rewards)
        # rewards = torch.where(tomodj_rfinger_pos[:, 0] < drawer_grasp_pos[:, 0] - distX_offset,
        #                       torch.ones_like(rewards) * -1, rewards)

        return rewards


def compute_grasp_transforms(
    hand_rot,
    hand_pos,
    tomodj_local_grasp_rot,
    tomodj_local_grasp_pos,
    drawer_rot,
    drawer_pos,
    drawer_local_grasp_rot,
    drawer_local_grasp_pos
):
    '''[summary]'''

    global_tomodj_rot, global_tomodj_pos = tf_combine(
        hand_rot,
        hand_pos,
        tomodj_local_grasp_rot,
        tomodj_local_grasp_pos
    )
    global_drawer_rot, global_drawer_pos = tf_combine(
        drawer_rot,
        drawer_pos,
        drawer_local_grasp_rot,
        drawer_local_grasp_pos
    )

    return global_tomodj_rot, global_tomodj_pos, global_drawer_rot, global_drawer_pos
