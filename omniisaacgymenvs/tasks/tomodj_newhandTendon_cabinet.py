# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import math

import numpy as np
import torch

from omni.isaac.cloner import Cloner
from omni.isaac.core.objects import DynamicCuboid, VisualCuboid
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.utils.torch.transformations import *
from omniisaacgymenvs.utils.usd_utils.create_instanceable_assets import FixedCuboid
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.cabinet import Cabinet
from omniisaacgymenvs.robots.articulations.tomodj_newhandTendon import TomoDJ
from omniisaacgymenvs.robots.articulations.views.cabinet_view import CabinetView
from omniisaacgymenvs.robots.articulations.views.tomodj_newhandTendon_view import TomoDJView
from pxr import UsdGeom, UsdPhysics, Sdf
from omni.physx import get_physx_simulation_interface
from omni.physx.scripts.physicsUtils import *

current_file_path = os.path.abspath(__file__)
data_folder_path = os.path.dirname(os.path.dirname(current_file_path)) + '/data'

class TomoDJnewhandTendonCabinetTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self.update_config(sim_config)

        self.distX_offset = 0.04
        self.dt = 1 / 60.0

        self._num_observations = 57 # (dof_pos_scaled: 26, tomodj_dof_vel: 26, to_target: 3, cabinet_dof_pos[:, 3]: 1, cabinet_dof_vel[:, 3]: 1)
        self._num_actions = 22 # 7 + 3 + 4 * 3

        RLTask.__init__(self, name, env)
        return

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]

        self._max_episode_length = self._task_cfg["env"]["episodeLength"]

        self.action_scale = self._task_cfg["env"]["actionScale"]
        self.start_position_noise = self._task_cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self._task_cfg["env"]["startRotationNoise"]
        self.num_props = self._task_cfg["env"]["numProps"]

        self.dof_vel_scale = self._task_cfg["env"]["dofVelocityScale"]
        self.dist_reward_scale = self._task_cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self._task_cfg["env"]["rotRewardScale"]
        self.around_handle_reward_scale = self._task_cfg["env"]["aroundHandleRewardScale"]
        self.open_reward_scale = self._task_cfg["env"]["openRewardScale"]
        self.finger_dist_reward_scale = self._task_cfg["env"]["fingerDistRewardScale"]
        self.action_penalty_scale = self._task_cfg["env"]["actionPenaltyScale"]
        self.finger_close_reward_scale = self._task_cfg["env"]["fingerCloseRewardScale"]
            
    def set_up_scene(self, scene) -> None:
        self._stage = get_current_stage()
        
        self.get_tomodj()
        self.get_cabinet()
        if self.num_props > 0:
            self.get_props()
        
        # VisualCuboid(
        #     prim_path=self.default_zero_env_path + f"/hand",
        #     name="hand",
        #     scale=np.array([0.01, 0.01, 0.01]),
        #     color=np.array([0.0, 1.0, 0.0])
        # )
        # for i in range(5):
        #     VisualCuboid(
        #         prim_path=self.default_zero_env_path + f"/finger_{i}",
        #         name="finger_{i}",
        #         scale=np.array([0.01, 0.01, 0.01]),
        #         color=np.array([1.0, 0.0, 0.0])
        #     )
        #     VisualCuboid(
        #         prim_path=self.default_zero_env_path + f"/drawer_{i}",
        #         name="drawer_{i}",
        #         scale=np.array([0.01, 0.01, 0.01]),
        #         color=np.array([0.0, 0.0, 1.0])
        #     )
        
        super().set_up_scene(scene, filter_collisions=False)
        self._tomodjs = TomoDJView(prim_paths_expr="/World/envs/.*/tomodj", name="tomodj_view")
        self._cabinets = CabinetView(prim_paths_expr="/World/envs/.*/cabinet", name="cabinet_view")

        scene.add(self._tomodjs)
        scene.add(self._tomodjs._hands)
        scene.add(self._tomodjs._thumbs)
        scene.add(self._tomodjs._forefingers)
        scene.add(self._tomodjs._middle_fingers)
        scene.add(self._tomodjs._ring_fingers)
        scene.add(self._tomodjs._little_fingers)
        scene.add(self._cabinets)
        scene.add(self._cabinets._drawers)

        if self.num_props > 0:
            self._props = RigidPrimView(
                prim_paths_expr="/World/envs/.*/prop/.*", name="prop_view", reset_xform_properties=False
            )
            scene.add(self._props)

        self.init_data()
        return

    def initialize_views(self, scene):
        super().initialize_views(scene)
        if scene.object_exists("tomodj_view"):
            scene.remove_object("tomodj_view", registry_only=True)
        if scene.object_exists("hands_view"):
            scene.remove_object("hands_view", registry_only=True)
        if scene.object_exists("thumbs_view"):
            scene.remove_object("thumbs_view", registry_only=True)
        if scene.object_exists("forefingers_view"):
            scene.remove_object("forefingers_view", registry_only=True)
        if scene.object_exists("middle_fingers_view"):
            scene.remove_object("middle_fingers_view", registry_only=True)
        if scene.object_exists("ring_fingers_view"):
            scene.remove_object("ring_fingers_view", registry_only=True)
        if scene.object_exists("little_fingers_view"):
            scene.remove_object("little_fingers_view", registry_only=True)
        if scene.object_exists("cabinet_view"):
            scene.remove_object("cabinet_view", registry_only=True)
        if scene.object_exists("drawers_view"):
            scene.remove_object("drawers_view", registry_only=True)
        if scene.object_exists("prop_view"):
            scene.remove_object("prop_view", registry_only=True)
        self._tomodjs = TomoDJView(prim_paths_expr="/World/envs/.*/tomodj", name="tomodj_view")
        self._cabinets = CabinetView(prim_paths_expr="/World/envs/.*/cabinet", name="cabinet_view")

        scene.add(self._tomodjs)
        scene.add(self._tomodjs._hands)
        scene.add(self._tomodjs._thumbs)
        scene.add(self._tomodjs._forefingers)
        scene.add(self._tomodjs._middle_fingers)
        scene.add(self._tomodjs._ring_fingers)
        scene.add(self._tomodjs._little_fingers)
        scene.add(self._cabinets)
        scene.add(self._cabinets._drawers)

        if self.num_props > 0:
            self._props = RigidPrimView(
                prim_paths_expr="/World/envs/.*/prop/.*", name="prop_view", reset_xform_properties=False
            )
            scene.add(self._props)

        self.init_data()

    def get_tomodj(self):
        tomodj = TomoDJ(prim_path=self.default_zero_env_path + "/tomodj", 
                          name="tomodj", 
                          usd_path=data_folder_path + '/TomoDJ/newhand/tendon_joints/tomodj_instanceable.usd')
        self._sim_config.apply_articulation_settings(
            "tomodj", get_prim_at_path(tomodj.prim_path), self._sim_config.parse_actor_config("tomodj")
        )
        tomodj.set_tomodj_properties(stage=self._stage, prim=tomodj.prim)
        self.tomodj = tomodj
        pillar = FixedCuboid(
            prim_path=self.default_zero_env_path + "/pillar",
            name="pillar",
            translation=np.array([tomodj._position[0], tomodj._position[1], tomodj._position[2]/2]),
            scale=np.array([0.3, 0.7, tomodj._position[2]]),
            contact_offset=0.005
        )

    def get_cabinet(self):
        cabinet = Cabinet(self.default_zero_env_path + "/cabinet", 
                          name="cabinet",
                          usd_path=data_folder_path + '/Cabinet/bigger_upper_drawer/sektion_cabinet_instanceable.usd')
        self._sim_config.apply_articulation_settings(
            "cabinet", get_prim_at_path(cabinet.prim_path), self._sim_config.parse_actor_config("cabinet")
        )

    def get_props(self):
        prop_cloner = Cloner()
        drawer_pos = torch.tensor([0.0515, 0.0, 0.7172])
        prop_color = torch.tensor([0.2, 0.4, 0.6])

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
            prim_path=self.default_zero_env_path + "/prop/prop_0",
            name="prop",
            color=prop_color,
            size=prop_size,
            density=100.0,
        )
        self._sim_config.apply_articulation_settings(
            "prop", get_prim_at_path(prop.prim_path), self._sim_config.parse_actor_config("prop")
        )

        prop_paths = [f"{self.default_zero_env_path}/prop/prop_{j}" for j in range(self.num_props)]
        prop_cloner.clone(
            source_prim_path=self.default_zero_env_path + "/prop/prop_0",
            prim_paths=prop_paths,
            positions=np.array(prop_pos) + drawer_pos.numpy(),
            replicate_physics=False,
        )
        

    def init_data(self) -> None:
        tomodj_local_thumb_pose_pos = torch.tensor([0.038, 0.008, 0.0], device=self._device)
        tomodj_local_another_finger_pose_pos = torch.tensor([0.024, -0.002, 0.0], device=self._device)
        self.tomodj_local_thumb_pose_pos = tomodj_local_thumb_pose_pos.repeat((self._num_envs, 1))
        self.tomodj_local_another_finger_pose_pos = tomodj_local_another_finger_pose_pos.repeat((self._num_envs, 1))

        self.tomodj_default_dof_pos = torch.tensor(
            [0.52, 0.52, 0.26, -0.26, -0.25, 0.17, -0.52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device=self._device
        ) # 7 arm dof + 3 thumb dof + 4 * 4 another fingers dof

        self.left_hand_forward_axis = torch.tensor([1, 0, 0], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1)
        )
        self.drawer_inward_axis = torch.tensor([-1, 0, 0], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1)
        )
        self.left_hand_up_axis = torch.tensor([0, 0, 1], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1)
        )
        self.drawer_up_axis = torch.tensor([0, 0, 1], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1)
        )
        
        self.actions = torch.zeros((self._num_envs, self.num_actions), device=self._device)
        
        # Subscribe to contact reporting
        # stage = get_current_stage()
        # contactReportAPI = PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(self.tomodj.prim_path + '/LF1_3'))
        # contactReportAPI.CreateThresholdAttr().Set(1)
        # self.physx_interface = get_physx_simulation_interface()
    
    def _on_contact_report_event(self, contact_headers, contact_data):
        for contact_header in contact_headers:         
            collider_1 = str(PhysicsSchemaTools.intToSdfPath(contact_header.actor0))
            collider_2 = str(PhysicsSchemaTools.intToSdfPath(contact_header.actor1))

            contacts = [collider_1, collider_2]
            print(contacts)
            
            
    def get_observations(self) -> dict:
        left_hand_pos, self.left_hand_rot = self._tomodjs._hands.get_world_poses(clone=False)
        thumb_pos, thumb_rot = self._tomodjs._thumbs.get_world_poses(clone=False)
        forefinger_pos, forefinger_rot = self._tomodjs._forefingers.get_world_poses(clone=False)
        middle_finger_pos, middle_finger_rot = self._tomodjs._middle_fingers.get_world_poses(clone=False)
        ring_finger_pos, ring_finger_rot = self._tomodjs._ring_fingers.get_world_poses(clone=False)
        little_finger_pos, little_finger_rot = self._tomodjs._little_fingers.get_world_poses(clone=False)
        self.drawer_pos, self.drawer_rot = self._cabinets._drawers.get_world_poses(clone=False)
        tomodj_dof_pos = self._tomodjs.get_joint_positions(clone=False)
        tomodj_dof_vel = self._tomodjs.get_joint_velocities(clone=False)
        self.cabinet_dof_pos = self._cabinets.get_joint_positions(clone=False)
        self.cabinet_dof_vel = self._cabinets.get_joint_velocities(clone=False)
        self.tomodj_dof_pos = tomodj_dof_pos

        for idx, keypoint_offset in enumerate(self.drawer_local_keypoint_offsets):
            self.keypoints_drawer[:, idx] = tf_combine(
                self.drawer_rot,
                self.drawer_pos,
                self.identity_quat,
                keypoint_offset.repeat(self.num_envs, 1),
            )[1]
        
        fingers_pos = [thumb_pos, forefinger_pos, middle_finger_pos, ring_finger_pos, little_finger_pos]
        self.fingers_rot = [thumb_rot, forefinger_rot, middle_finger_rot, ring_finger_rot, little_finger_rot]
        
        self.keypoints_finger[:, 0] = tf_combine(
            self.fingers_rot[0],
            fingers_pos[0],
            self.identity_quat,
            self.tomodj_local_thumb_pose_pos
        )[1]
        
        for i in range(4):
            self.keypoints_finger[:, i + 1] = tf_combine(
                self.fingers_rot[i + 1],
                fingers_pos[i + 1],
                self.identity_quat,
                self.tomodj_local_another_finger_pose_pos
            )[1]
        
        # stage = get_current_stage()
        # for i in range(self.num_envs):
        #     hand_pose_cube = UsdGeom.Xformable(stage.GetPrimAtPath(f"/World/envs/env_{i}/hand"))
        #     hand_pose_cube_ops = hand_pose_cube.GetOrderedXformOps()
        #     hand_pose_cube_translate_op = None
        #     hand_pose_cube_orient_op = None
        #     for op in hand_pose_cube_ops:
        #         if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
        #             hand_pose_cube_translate_op = op
        #         elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
        #             hand_pose_cube_orient_op = op
        #     position = (left_hand_pos[i, :] - self._env_pos[i] + torch.tensor([0.0, 0.0, 0.08], device=self.device)).cpu().tolist()
        #     orientation = (self.left_hand_rot[i]).cpu().tolist()
        #     hand_pose_cube_translate_op.Set(Gf.Vec3f(position[0], position[1], position[2]))
        #     hand_pose_cube_orient_op.Set(Gf.Quatd(orientation[0], orientation[1], orientation[2], orientation[3]))
        #     for j in range(5):
        #         finger_pose_cube = UsdGeom.Xformable(stage.GetPrimAtPath(f"/World/envs/env_{i}/finger_{j}"))
        #         finger_pose_cube_ops = finger_pose_cube.GetOrderedXformOps()
        #         finger_pose_cube_translate_op = None
        #         finger_pose_cube_orient_op = None
        #         for op in finger_pose_cube_ops:
        #             if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
        #                 finger_pose_cube_translate_op = op
        #             elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
        #                 finger_pose_cube_orient_op = op
        #         position = (self.keypoints_finger[i, j, :] - self._env_pos[i]).cpu().tolist()
        #         orientation = (self.fingers_rot[j][i]).cpu().tolist()
        #         finger_pose_cube_translate_op.Set(Gf.Vec3f(position[0], position[1], position[2]))
        #         finger_pose_cube_orient_op.Set(Gf.Quatd(orientation[0], orientation[1], orientation[2], orientation[3]))
                
        #         drawer_pose_cube = UsdGeom.Xformable(stage.GetPrimAtPath(f"/World/envs/env_{i}/drawer_{j}"))    
        #         drawer_pose_cube_ops = drawer_pose_cube.GetOrderedXformOps()      
        #         drawer_pose_cube_translate_op = None
        #         for op in drawer_pose_cube_ops:
        #             if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
        #                 drawer_pose_cube_translate_op = op
        #         position = (self.keypoints_drawer[i, j, :] - self._env_pos[i]).cpu().tolist()
        #         drawer_pose_cube_translate_op.Set(Gf.Vec3f(position[0], position[1], position[2]))
        
        self.dof_pos_scaled = (
            2.0
            * (tomodj_dof_pos - self.tomodj_dof_lower_limits)
            / (self.tomodj_dof_upper_limits - self.tomodj_dof_lower_limits)
            - 1.0
        )
        to_target = torch.sum(self.keypoints_drawer - self.keypoints_finger, dim = 1)
        self.obs_buf = torch.cat(
            (
                self.dof_pos_scaled,
                tomodj_dof_vel * self.dof_vel_scale,
                to_target,
                self.cabinet_dof_pos[:, 3].unsqueeze(-1),
                self.cabinet_dof_vel[:, 3].unsqueeze(-1),
            ),
            dim=-1,
        )
        
        observations = {self._tomodjs.name: {"obs_buf": self.obs_buf}}
        
        # collision_report = self.physx_interface.get_contact_report()
        # self._on_contact_report_event(collision_report[0], collision_report[1])
        
        return observations

    def pre_physics_step(self, actions) -> None:
        if not self.world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        self.actions = actions.clone().to(self._device)
        targets = self.tomodj_dof_targets[:, self.actuated_dof_indices] + self.tomodj_dof_speed_scales[self.actuated_dof_indices] * self.dt * self.actions * self.action_scale
        self.tomodj_dof_targets[:, self.actuated_dof_indices] = tensor_clamp(targets, self.tomodj_dof_lower_limits[self.actuated_dof_indices], self.tomodj_dof_upper_limits[self.actuated_dof_indices])
        env_ids_int32 = torch.arange(self._tomodjs.count, dtype=torch.int32, device=self._device)

        self._tomodjs.set_joint_position_targets(self.tomodj_dof_targets[:, self.actuated_dof_indices], indices=env_ids_int32, joint_indices=self.actuated_dof_indices)

    def reset_idx(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)
        num_indices = len(indices)

        # reset tomodj
        pos = tensor_clamp(
            self.tomodj_default_dof_pos.unsqueeze(0)
            + 0.25 * (torch.rand((len(env_ids), self.num_tomodj_dofs), device=self._device) - 0.5),
            self.tomodj_dof_lower_limits,
            self.tomodj_dof_upper_limits,
        )
        dof_pos = torch.zeros((num_indices, self._tomodjs.num_dof), device=self._device)
        dof_vel = torch.zeros((num_indices, self._tomodjs.num_dof), device=self._device)
        dof_pos[:, :] = pos
        self.tomodj_dof_targets[env_ids, :] = pos
        self.tomodj_dof_pos[env_ids, :] = pos

        # reset cabinet
        self._cabinets.set_joint_positions(
            torch.zeros_like(self._cabinets.get_joint_positions(clone=False)[env_ids]), indices=indices
        )
        self._cabinets.set_joint_velocities(
            torch.zeros_like(self._cabinets.get_joint_velocities(clone=False)[env_ids]), indices=indices
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
        self.actuated_dof_indices = self._tomodjs.actuated_dof_indices
        self.tomodj_dof_pos = torch.zeros((self.num_envs, self.num_tomodj_dofs), device=self._device)
        dof_limits = self._tomodjs.get_dof_limits()
        self.tomodj_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.tomodj_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)
        self.tomodj_dof_speed_scales = torch.ones_like(self.tomodj_dof_lower_limits)
        self.tomodj_dof_speed_scales[self._tomodjs._finger_indices] = 5
        self.tomodj_dof_targets = torch.zeros(
            (self._num_envs, self.num_tomodj_dofs), dtype=torch.float, device=self._device
        )
        self.keypoints_finger = torch.zeros(
            (self.num_envs, 5, 3), dtype=torch.float32, device=self.device
        )
        
        self.keypoints_drawer = torch.zeros_like(
            self.keypoints_finger, device=self.device
        )
        
        self.identity_quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )
        
        self.drawer_local_keypoint_offsets = torch.zeros((5, 3), device=self.device)
        self.drawer_local_keypoint_offsets[1:, 1] = (torch.linspace(0.0, 1.0, 4, device=self.device) - 0.5) * 0.1 - 0.01
        self.drawer_local_keypoint_offsets[0, 1] = self.drawer_local_keypoint_offsets[1, 1]
        for i in range(5):
            self.drawer_local_keypoint_offsets[i, 0] = 0.305
            if i == 0:
                self.drawer_local_keypoint_offsets[i, 2] = 0.0165
            else:
                self.drawer_local_keypoint_offsets[i, 2] = 0.0135
            
        if self.num_props > 0:
            self.default_prop_pos, self.default_prop_rot = self._props.get_world_poses()
            self.prop_indices = torch.arange(self._num_envs * self.num_props, device=self._device).view(
                self._num_envs, self.num_props
            )

        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)
        self.terminate_buf = torch.zeros_like(self.reset_buf)

    def calculate_metrics(self) -> None:
        self.rew_buf[:] = self.compute_tomodj_reward(
            self.reset_buf,
            self.terminate_buf,
            self.progress_buf,
            self.actions,
            self.cabinet_dof_pos,
            self.keypoints_drawer,
            self.keypoints_finger,
            self.fingers_rot,
            self.left_hand_rot,
            self.drawer_pos,
            self.drawer_rot,
            self.left_hand_forward_axis,
            self.drawer_inward_axis,
            self.left_hand_up_axis,
            self.drawer_up_axis,
            self._num_envs,
            self.dist_reward_scale,
            self.rot_reward_scale,
            self.around_handle_reward_scale,
            self.open_reward_scale,
            self.finger_dist_reward_scale,
            self.action_penalty_scale,
            self.distX_offset,
            self._max_episode_length,
            self.tomodj_dof_pos,
            self.finger_close_reward_scale,
        )

    def is_done(self) -> None:
        # reset if drawer is open or max length reached
        self.reset_buf = torch.where(self.cabinet_dof_pos[:, 3] > 0.19, torch.ones_like(self.reset_buf), self.reset_buf)
        self.reset_buf = torch.where(self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)

        self.terminate_buf = torch.zeros_like(self.reset_buf)
        self.terminate_buf = torch.where(torch.any(self.dof_pos_scaled < -1.05, dim=1), torch.ones_like(self.terminate_buf), self.terminate_buf)
        self.terminate_buf = torch.where(torch.any(self.dof_pos_scaled > 1.05, dim=1), torch.ones_like(self.terminate_buf), self.terminate_buf)
        self.reset_buf = torch.where(self.terminate_buf == 1, torch.ones_like(self.reset_buf), self.reset_buf)

    def compute_tomodj_reward(
        self,
        reset_buf,
        terminate_buf,
        progress_buf,
        actions,
        cabinet_dof_pos,
        keypoints_drawer,
        keypoints_finger,
        fingers_rot,
        left_hand_rot,
        drawer_pos,
        drawer_rot,
        left_hand_forward_axis,
        drawer_inward_axis,
        left_hand_up_axis,
        drawer_up_axis,
        num_envs,
        dist_reward_scale,
        rot_reward_scale,
        around_handle_reward_scale,
        open_reward_scale,
        finger_dist_reward_scale,
        action_penalty_scale,
        distX_offset,
        max_episode_length,
        joint_positions,
        finger_close_reward_scale,
    ):
        ## type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float, float, float, float, Tensor) -> Tuple[Tensor, Tensor]

        # distance from hand to the drawer
        d_keypoints = torch.norm(self.keypoints_drawer - self.keypoints_finger, p=2, dim=-1)
        d = torch.sum(d_keypoints[:, 1:], dim=-1)
        dist_reward = -(d + d_keypoints[:, 0])/2
        
        axis1 = tf_vector(left_hand_rot, left_hand_forward_axis)
        axis2 = tf_vector(drawer_rot, drawer_inward_axis)
        axis3 = tf_vector(left_hand_rot, left_hand_up_axis)
        axis4 = tf_vector(drawer_rot, drawer_up_axis)

        dot1 = (
            torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        )  # alignment of forward axis for left hand
        dot2 = (
            torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        )  # alignment of up axis for left hand
        # reward for matching the orientation of the hand to the drawer (fingers wrapped)
        rot_reward = 0.5 * (torch.sign(dot1) * dot1**2 + torch.sign(dot2) * dot2**2)
                  
        # bonus if fingers are setup correctly
        around_handle_reward = torch.zeros_like(dist_reward)

        # check if all fingers in the gap between cabinet and handle
        keypoints_finger_drawer_distance = []
        for i in range(5):
            keypoints_finger_drawer_distance.append(keypoints_finger[:, i, 0] - drawer_pos[:, 0])
        
        around_handle_reward = torch.where(
            (keypoints_finger_drawer_distance[0][:] > 0.2732) & (keypoints_finger_drawer_distance[0][:] < 0.3149) & \
            (keypoints_finger_drawer_distance[1][:] > 0.2732) & (keypoints_finger_drawer_distance[1][:] < 0.3149) & \
            (keypoints_finger_drawer_distance[2][:] > 0.2732) & (keypoints_finger_drawer_distance[2][:] < 0.3149) & \
            (keypoints_finger_drawer_distance[3][:] > 0.2732) & (keypoints_finger_drawer_distance[3][:] < 0.3149) & \
            (keypoints_finger_drawer_distance[4][:] > 0.2732) & (keypoints_finger_drawer_distance[4][:] < 0.3149),
            around_handle_reward + 0.5,
            around_handle_reward
        )

        # # reward for distance of each finger from the drawer
        # finger_dist_reward = torch.zeros_like(dist_reward)
        # lfinger_dist = torch.abs(tomodj_lfinger_pos[:, 2] - drawer_grasp_pos[:, 2])
        # rfinger_dist = torch.abs(tomodj_rfinger_pos[:, 2] - drawer_grasp_pos[:, 2])
        # finger_dist_reward = torch.where(
        #     tomodj_lfinger_pos[:, 2] <= drawer_grasp_pos[:, 2],
        #     torch.where(
        #         tomodj_rfinger_pos[:, 2] >= drawer_grasp_pos[:, 2],
        #         (0.04 - lfinger_dist) + (0.04 - rfinger_dist),
        #         finger_dist_reward,
        #     ),
        #     finger_dist_reward,
        # )
        
        # finger_close_reward = torch.zeros_like(dist_reward)
        # finger_close_reward = torch.where(
        #     d <= 0.03, (0.04 - joint_positions[:, 6]) + (0.04 - joint_positions[:, 7]), finger_close_reward
        # )

        # regularization on the actions (summed for each environment)
        action_penalty = torch.sum(actions**2, dim=-1)

        # how far the cabinet has been opened out
        open_reward = cabinet_dof_pos[:, 3] * around_handle_reward + cabinet_dof_pos[:, 3]  # drawer_top_joint

        rewards = (
            dist_reward_scale * dist_reward
            + rot_reward_scale * rot_reward
            + around_handle_reward_scale * around_handle_reward
            + open_reward_scale * open_reward
            # + finger_dist_reward_scale * finger_dist_reward
            - action_penalty_scale * action_penalty
            # + finger_close_reward_scale * finger_close_reward
        )
        
        # print('a', dot1[0], dot2[0], rot_reward[0])
        # print('b', dist_reward[0], around_handle_reward[0], open_reward[0])
        
        # bonus for opening drawer properly
        rewards = torch.where(cabinet_dof_pos[:, 3] > 0.01, rewards + 0.5, rewards)
        rewards = torch.where(cabinet_dof_pos[:, 3] > 0.1, rewards + around_handle_reward, rewards)
        rewards = torch.where(cabinet_dof_pos[:, 3] > 0.19, rewards + (2.0 * around_handle_reward), rewards)

        # # prevent bad style in opening drawer
        # rewards = torch.where(tomodj_lfinger_pos[:, 0] < drawer_grasp_pos[:, 0] - distX_offset,
        #                       torch.ones_like(rewards) * -1, rewards)
        # rewards = torch.where(tomodj_rfinger_pos[:, 0] < drawer_grasp_pos[:, 0] - distX_offset,
        #                       torch.ones_like(rewards) * -1, rewards)

        rewards = torch.where(terminate_buf == 1, -100, rewards)
        
        return rewards
