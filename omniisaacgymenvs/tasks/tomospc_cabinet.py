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
from omniisaacgymenvs.robots.articulations.tomospc import TomoSPC
from omniisaacgymenvs.robots.articulations.views.cabinet_view import CabinetView
from omniisaacgymenvs.robots.articulations.views.tomospc_view import TomoSPCView
from pxr import UsdGeom, UsdPhysics, Sdf
from omni.physx import get_physx_simulation_interface
from omni.physx.scripts.physicsUtils import *

current_file_path = os.path.abspath(__file__)
data_folder_path = os.path.dirname(os.path.dirname(current_file_path)) + '/data'

class TomoSPCCabinetTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self.update_config(sim_config)

        self.distX_offset = 0.04
        self.dt = 1 / 60.0

        self._num_observations = 21 # (dof_pos_scaled: 8, tomospc_dof_vel: 8, to_target: 3, cabinet_dof_pos[:, 3]: 1, cabinet_dof_vel[:, 3]: 1)
        self._num_actions = 8

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
        self.get_tomospc()
        self.get_cabinet()
        if self.num_props > 0:
            self.get_props()
            
        # VisualCuboid(
        #     prim_path=self.default_zero_env_path + "/grasp_pose",
        #     name="grasp_pose",
        #     scale=np.array([0.01, 0.01, 0.01]),
        #     color=np.array([1.0, 0.0, 0.0])
        # )
        # VisualCuboid(
        #     prim_path=self.default_zero_env_path + "/drawer_pose",
        #     name="drawer_pose",
        #     scale=np.array([0.01, 0.01, 0.01]),
        #     color=np.array([0.0, 0.0, 1.0])
        # )
        
        super().set_up_scene(scene, filter_collisions=False)
        self._tomospcs = TomoSPCView(prim_paths_expr="/World/envs/.*/tomospc", name="tomospc_view")
        self._cabinets = CabinetView(prim_paths_expr="/World/envs/.*/cabinet", name="cabinet_view")

        scene.add(self._tomospcs)
        scene.add(self._tomospcs._camera)
        scene.add(self._tomospcs._hands)
        scene.add(self._tomospcs._lfingers)
        scene.add(self._tomospcs._rfingers)
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
        if scene.object_exists("tomospc_view"):
            scene.remove_object("tomospc_view", registry_only=True)
        if scene.object_exists("camera_view"):
            scene.remove_object("camera_view", registry_only=True)
        if scene.object_exists("hands_view"):
            scene.remove_object("hands_view", registry_only=True)
        if scene.object_exists("lfingers_view"):
            scene.remove_object("lfingers_view", registry_only=True)
        if scene.object_exists("rfingers_view"):
            scene.remove_object("rfingers_view", registry_only=True)
        if scene.object_exists("cabinet_view"):
            scene.remove_object("cabinet_view", registry_only=True)
        if scene.object_exists("drawers_view"):
            scene.remove_object("drawers_view", registry_only=True)
        if scene.object_exists("prop_view"):
            scene.remove_object("prop_view", registry_only=True)
        self._tomospcs = TomoSPCView(prim_paths_expr="/World/envs/.*/tomospc", name="tomospc_view")
        self._cabinets = CabinetView(prim_paths_expr="/World/envs/.*/cabinet", name="cabinet_view")

        scene.add(self._tomospcs)
        scene.add(self._tomospcs._camera)
        scene.add(self._tomospcs._hands)
        scene.add(self._tomospcs._lfingers)
        scene.add(self._tomospcs._rfingers)
        scene.add(self._cabinets)
        scene.add(self._cabinets._drawers)

        if self.num_props > 0:
            self._props = RigidPrimView(
                prim_paths_expr="/World/envs/.*/prop/.*", name="prop_view", reset_xform_properties=False
            )
            scene.add(self._props)

        self.init_data()

    def get_tomospc(self):
        tomospc = TomoSPC(prim_path=self.default_zero_env_path + "/tomospc", 
                          name="tomospc", 
                          usd_path=data_folder_path + '/TomoSPC/tomospc_instanceable.usd')
        self._sim_config.apply_articulation_settings(
            "tomospc", get_prim_at_path(tomospc.prim_path), self._sim_config.parse_actor_config("tomospc")
        )
        self.tomospc = tomospc
        pillar = FixedCuboid(
            prim_path=self.default_zero_env_path + "/pillar",
            name="pillar",
            translation=np.array([tomospc._position[0], tomospc._position[1], tomospc._position[2]/2]),
            scale=np.array([0.3, 0.7, tomospc._position[2]]),
            contact_offset=0.005
        )

    def get_cabinet(self):
        cabinet = Cabinet(self.default_zero_env_path + "/cabinet", 
                          name="cabinet",
                          usd_path=data_folder_path + '/Cabinet/sektion_cabinet_instanceable_adjusted.usd')
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
        
        # barier = FixedCuboid(
        #     prim_path=self.default_zero_env_path + "/cube",
        #     name="cube",
        #     color=prop_color,
        #     translation=np.array([0.85, -0.165, 0.55]),
        #     scale=np.array([0.125, 0.001, 0.3])
        # )
        
        # stage = get_current_stage()
        # barier_ = UsdGeom.Xform(stage.GetPrimAtPath(barier.prim_path))
        # # Hide render barier
        # barier_.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)

    def init_data(self) -> None:
        def get_env_local_pose(env_pos, xformable, device):
            """Compute pose in env-local coordinates"""
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

        stage = get_current_stage()
        hand_pose = get_env_local_pose(
            self._env_pos[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/tomospc/left_arm_link6")),
            self._device,
        )
        lfinger_pose = get_env_local_pose(
            self._env_pos[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/tomospc/left_finger")),
            self._device,
        )
        rfinger_pose = get_env_local_pose(
            self._env_pos[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/tomospc/right_finger")),
            self._device,
        )

        finger_pose = torch.zeros(7, device=self._device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        finger_pose[3:7] = lfinger_pose[3:7]
        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])

        grasp_pose_axis = 1
        tomospc_local_grasp_pose_rot, tomospc_local_pose_pos = tf_combine(
            hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3]
        )
        tomospc_local_pose_pos += torch.tensor([0, 0.04, 0], device=self._device)
        self.tomospc_local_grasp_pos = tomospc_local_pose_pos.repeat((self._num_envs, 1))
        self.tomospc_local_grasp_rot = tomospc_local_grasp_pose_rot.repeat((self._num_envs, 1))

        drawer_local_grasp_pose = torch.tensor([0.3, 0.01, 0.0, 1.0, 0.0, 0.0, 0.0], device=self._device)
        self.drawer_local_grasp_pos = drawer_local_grasp_pose[0:3].repeat((self._num_envs, 1))
        self.drawer_local_grasp_rot = drawer_local_grasp_pose[3:7].repeat((self._num_envs, 1))

        self.gripper_forward_axis = torch.tensor([0, 0, 1], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1)
        )
        self.drawer_inward_axis = torch.tensor([-1, 0, 0], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1)
        )
        self.gripper_up_axis = torch.tensor([0, -1, 0], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1)
        )
        self.drawer_up_axis = torch.tensor([0, 0, 1], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1)
        )

        self.tomospc_default_dof_pos = torch.tensor(
            [-0.52, 0.26, 0.79, 0.35, 1.05, 0, 0.035, 0.035], device=self._device
        )

        self.actions = torch.zeros((self._num_envs, self.num_actions), device=self._device)
        
        # # Subscribe to contact reporting
        # contactReportAPI = PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(self.tomospc.prim_path + '/left_arm_link3'))
        # contactReportAPI.CreateThresholdAttr().Set(1)
        # self.physx_interface = get_physx_simulation_interface()
    
    def _on_contact_report_event(self, contact_headers, contact_data):
        for contact_header in contact_headers:         
            collider_1 = str(PhysicsSchemaTools.intToSdfPath(contact_header.actor0))
            collider_2 = str(PhysicsSchemaTools.intToSdfPath(contact_header.actor1))

            contacts = [collider_1, collider_2]
            print(contacts)
            
            
    def get_observations(self) -> dict:
        hand_pos, hand_rot = self._tomospcs._hands.get_world_poses(clone=False)
        drawer_pos, drawer_rot = self._cabinets._drawers.get_world_poses(clone=False)
        tomospc_dof_pos = self._tomospcs.get_joint_positions(clone=False)
        tomospc_dof_vel = self._tomospcs.get_joint_velocities(clone=False)
        self.cabinet_dof_pos = self._cabinets.get_joint_positions(clone=False)
        self.cabinet_dof_vel = self._cabinets.get_joint_velocities(clone=False)
        self.tomospc_dof_pos = tomospc_dof_pos

        (
            self.tomospc_grasp_rot,
            self.tomospc_grasp_pos,
            self.drawer_grasp_rot,
            self.drawer_grasp_pos,
        ) = self.compute_grasp_transforms(
            hand_rot,
            hand_pos,
            self.tomospc_local_grasp_rot,
            self.tomospc_local_grasp_pos,
            drawer_rot,
            drawer_pos,
            self.drawer_local_grasp_rot,
            self.drawer_local_grasp_pos,
        )

        # stage = get_current_stage()
        # for i in range(self.num_envs):
        #     grasp_pose_cube = UsdGeom.Xformable(stage.GetPrimAtPath(f"/World/envs/env_{i}/grasp_pose"))
        #     drawer_pose_cube = UsdGeom.Xformable(stage.GetPrimAtPath(f"/World/envs/env_{i}/drawer_pose"))
        #     grasp_pose_cube_ops = grasp_pose_cube.GetOrderedXformOps()
        #     drawer_pose_cube_ops = drawer_pose_cube.GetOrderedXformOps()
        #     grasp_pose_cube_translate_op = None
        #     grasp_pose_cube_orient_op = None

        #     for op in grasp_pose_cube_ops:
        #         if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
        #             grasp_pose_cube_translate_op = op
        #         elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
        #             grasp_pose_cube_orient_op = op
                
        #     drawer_pose_cube_op = None
        #     for op in drawer_pose_cube_ops:
        #         if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
        #             drawer_pose_cube_op = op
        #             break
                
        #     position = (self.tomospc_grasp_pos[i] - self._env_pos[i]).cpu().tolist()
        #     orientation = self.tomospc_grasp_rot[i].cpu().tolist()
        #     grasp_pose_cube_translate_op.Set(Gf.Vec3f(position[0], position[1], position[2]))
        #     grasp_pose_cube_orient_op.Set(Gf.Quatd(orientation[0], orientation[1], orientation[2], orientation[3]))
        #     position = (self.drawer_grasp_pos[i] - self._env_pos[i]).cpu().tolist()
        #     drawer_pose_cube_op.Set(Gf.Vec3f(position[0], position[1], position[2]))

        self.tomospc_lfinger_pos, self.tomospc_lfinger_rot = self._tomospcs._lfingers.get_world_poses(clone=False)
        self.tomospc_rfinger_pos, self.tomospc_rfinger_rot = self._tomospcs._rfingers.get_world_poses(clone=False)

        dof_pos_scaled = (
            2.0
            * (tomospc_dof_pos - self.tomospc_dof_lower_limits)
            / (self.tomospc_dof_upper_limits - self.tomospc_dof_lower_limits)
            - 1.0
        )
        to_target = self.drawer_grasp_pos - self.tomospc_grasp_pos
        self.obs_buf = torch.cat(
            (
                dof_pos_scaled,
                tomospc_dof_vel * self.dof_vel_scale,
                to_target,
                self.cabinet_dof_pos[:, 3].unsqueeze(-1),
                self.cabinet_dof_vel[:, 3].unsqueeze(-1),
            ),
            dim=-1,
        )
        
        observations = {self._tomospcs.name: {"obs_buf": self.obs_buf}}
        
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
        targets = self.tomospc_dof_targets + self.tomospc_dof_speed_scales * self.dt * self.actions * self.action_scale
        self.tomospc_dof_targets[:] = tensor_clamp(targets, self.tomospc_dof_lower_limits, self.tomospc_dof_upper_limits)
        env_ids_int32 = torch.arange(self._tomospcs.count, dtype=torch.int32, device=self._device)

        self._tomospcs.set_joint_position_targets(self.tomospc_dof_targets, indices=env_ids_int32)

    def reset_idx(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)
        num_indices = len(indices)

        # reset tomospc
        pos = tensor_clamp(
            self.tomospc_default_dof_pos.unsqueeze(0)
            + 0.25 * (torch.rand((len(env_ids), self.num_tomospc_dofs), device=self._device) - 0.5),
            self.tomospc_dof_lower_limits,
            self.tomospc_dof_upper_limits,
        )
        dof_pos = torch.zeros((num_indices, self._tomospcs.num_dof), device=self._device)
        dof_vel = torch.zeros((num_indices, self._tomospcs.num_dof), device=self._device)
        dof_pos[:, :] = pos
        self.tomospc_dof_targets[env_ids, :] = pos
        self.tomospc_dof_pos[env_ids, :] = pos

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

        self._tomospcs.set_joint_position_targets(self.tomospc_dof_targets[env_ids], indices=indices)
        self._tomospcs.set_joint_positions(dof_pos, indices=indices)
        self._tomospcs.set_joint_velocities(dof_vel, indices=indices)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def post_reset(self):

        self.num_tomospc_dofs = self._tomospcs.num_dof
        self.tomospc_dof_pos = torch.zeros((self.num_envs, self.num_tomospc_dofs), device=self._device)
        dof_limits = self._tomospcs.get_dof_limits()
        self.tomospc_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.tomospc_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)
        self.tomospc_dof_speed_scales = torch.ones_like(self.tomospc_dof_lower_limits)
        self.tomospc_dof_speed_scales[self._tomospcs.gripper_indices] = 0.1
        self.tomospc_dof_targets = torch.zeros(
            (self._num_envs, self.num_tomospc_dofs), dtype=torch.float, device=self._device
        )

        if self.num_props > 0:
            self.default_prop_pos, self.default_prop_rot = self._props.get_world_poses()
            self.prop_indices = torch.arange(self._num_envs * self.num_props, device=self._device).view(
                self._num_envs, self.num_props
            )

        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        self.rew_buf[:] = self.compute_tomospc_reward(
            self.reset_buf,
            self.progress_buf,
            self.actions,
            self.cabinet_dof_pos,
            self.tomospc_grasp_pos,
            self.drawer_grasp_pos,
            self.tomospc_grasp_rot,
            self.drawer_grasp_rot,
            self.tomospc_lfinger_pos,
            self.tomospc_rfinger_pos,
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
            self.distX_offset,
            self._max_episode_length,
            self.tomospc_dof_pos,
            self.finger_close_reward_scale,
        )

    def is_done(self) -> None:
        # reset if drawer is open or max length reached
        self.reset_buf = torch.where(self.cabinet_dof_pos[:, 3] > 0.19, torch.ones_like(self.reset_buf), self.reset_buf)
        self.reset_buf = torch.where(
            self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf
        )

    def compute_grasp_transforms(
        self,
        hand_rot,
        hand_pos,
        tomospc_local_grasp_rot,
        tomospc_local_grasp_pos,
        drawer_rot,
        drawer_pos,
        drawer_local_grasp_rot,
        drawer_local_grasp_pos,
    ):

        global_tomospc_rot, global_tomospc_pos = tf_combine(
            hand_rot, hand_pos, tomospc_local_grasp_rot, tomospc_local_grasp_pos
        )
        global_drawer_rot, global_drawer_pos = tf_combine(
            drawer_rot, drawer_pos, drawer_local_grasp_rot, drawer_local_grasp_pos
        )

        return global_tomospc_rot, global_tomospc_pos, global_drawer_rot, global_drawer_pos

    def compute_tomospc_reward(
        self,
        reset_buf,
        progress_buf,
        actions,
        cabinet_dof_pos,
        tomospc_grasp_pos,
        drawer_grasp_pos,
        tomospc_grasp_rot,
        drawer_grasp_rot,
        tomospc_lfinger_pos,
        tomospc_rfinger_pos,
        gripper_forward_axis,
        drawer_inward_axis,
        gripper_up_axis,
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
        d = torch.norm(tomospc_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
        dist_reward = -d

        axis1 = tf_vector(tomospc_grasp_rot, gripper_forward_axis)
        axis2 = tf_vector(drawer_grasp_rot, drawer_inward_axis)
        axis3 = tf_vector(tomospc_grasp_rot, gripper_up_axis)
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
            tomospc_lfinger_pos[:, 2] <= drawer_grasp_pos[:, 2],
            torch.where(
                tomospc_rfinger_pos[:, 2] >= drawer_grasp_pos[:, 2], around_handle_reward + 0.5, around_handle_reward
            ),
            around_handle_reward,
        )
        # reward for distance of each finger from the drawer
        finger_dist_reward = torch.zeros_like(rot_reward)
        lfinger_dist = torch.abs(tomospc_lfinger_pos[:, 2] - drawer_grasp_pos[:, 2])
        rfinger_dist = torch.abs(tomospc_rfinger_pos[:, 2] - drawer_grasp_pos[:, 2])
        finger_dist_reward = torch.where(
            tomospc_lfinger_pos[:, 2] <= drawer_grasp_pos[:, 2],
            torch.where(
                tomospc_rfinger_pos[:, 2] >= drawer_grasp_pos[:, 2],
                (0.04 - lfinger_dist) + (0.04 - rfinger_dist),
                finger_dist_reward,
            ),
            finger_dist_reward,
        )
        
        finger_close_reward = torch.zeros_like(rot_reward)
        finger_close_reward = torch.where(
            d <= 0.03, (0.04 - joint_positions[:, 6]) + (0.04 - joint_positions[:, 7]), finger_close_reward
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
        # print('a', dot1[0], dot2[0], rot_reward[0], d[0])
        # print('b', tomospc_lfinger_pos[0], tomospc_rfinger_pos[0], drawer_grasp_pos[0])
        # print('c', dist_reward[0], finger_dist_reward[0], around_handle_reward[0], finger_close_reward[0])
        
        # bonus for opening drawer properly
        rewards = torch.where(cabinet_dof_pos[:, 3] > 0.01, rewards + 0.5, rewards)
        rewards = torch.where(cabinet_dof_pos[:, 3] > 0.1, rewards + around_handle_reward, rewards)
        rewards = torch.where(cabinet_dof_pos[:, 3] > 0.19, rewards + (2.0 * around_handle_reward), rewards)

        # # prevent bad style in opening drawer
        # rewards = torch.where(tomospc_lfinger_pos[:, 0] < drawer_grasp_pos[:, 0] - distX_offset,
        #                       torch.ones_like(rewards) * -1, rewards)
        # rewards = torch.where(tomospc_rfinger_pos[:, 0] < drawer_grasp_pos[:, 0] - distX_offset,
        #                       torch.ones_like(rewards) * -1, rewards)

        return rewards
