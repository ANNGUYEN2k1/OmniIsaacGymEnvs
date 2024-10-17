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

'''Factory: base class.

Inherits Gym's RLTask class and abstract base class. Inherited by environment classes. Not directly executed.

Configuration defined in FactoryBase.yaml. Asset info defined in factory_asset_info_robot_table.yaml.
'''

import math
from typing import Dict, Optional, Sequence

import carb
import hydra
import numpy as np
import torch
from omni.isaac.core.utils.stage import get_current_stage
from pxr import PhysicsSchemaTools, PhysxSchema, UsdPhysics

import omniisaacgymenvs.tasks.factory.factory_control as fc
from omniisaacgymenvs.robots.articulations.factory_franka import FactoryFranka
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.tasks.factory.factory_schema_class_base import FactoryABCBase
from omniisaacgymenvs.tasks.factory.factory_schema_config_base import FactorySchemaConfigBase
from omniisaacgymenvs.utils.usd_utils.create_instanceable_assets import FixedCuboid


class FactoryBase(RLTask, FactoryABCBase):
    '''[summary]'''

    def __init__(self, name, sim_config, env, offset=None) -> None:
        '''Initialize instance variables. Initialize RLTask superclass.'''

        # Set instance variables from base YAML
        self._get_base_yaml_params()
        self._env_spacing = self.cfg_base.env.env_spacing

        # Set instance variables from task and train YAMLs
        self._sim_config = sim_config
        self._cfg = sim_config.config  # CL args, task config, and train config
        self._task_cfg = sim_config.task_config  # Just task config
        self._num_envs = sim_config.task_config['env']['numEnvs']
        self._num_observations = sim_config.task_config['env']['numObservations']
        self._num_actions = sim_config.task_config['env']['numActions']

        super().__init__(name, env, offset)

        # Stage
        self._stage = get_current_stage()

        # Franka's articulation
        self.franka = None

        # Dof tensors
        self.dof_pos = torch.tensor([], dtype=torch.float32, device=self._device)
        self.dof_vel = torch.tensor([], dtype=torch.float32, device=self._device)
        self.dof_torque = torch.tensor([], dtype=torch.float32, device=self._device)

        # Arm tensors
        self.arm_dof_pos = torch.tensor([], dtype=torch.float32, device=self._device)
        self.arm_mass_matrix = torch.tensor([], dtype=torch.float32, device=self._device)

        # Hand tensors
        self.hand_quat = torch.tensor([], dtype=torch.float32, device=self._device)
        self.hand_force = torch.tensor([], dtype=torch.float32, device=self._device)

        # Fingertip tensors
        self.finger_midpoint_pos = torch.tensor([], dtype=torch.float32, device=self._device)
        self.fingertip_midpoint_pos = torch.tensor([], dtype=torch.float32, device=self._device)
        self.fingertip_midpoint_quat = torch.tensor([], dtype=torch.float32, device=self._device)
        self.fingertip_midpoint_linvel = torch.tensor([], dtype=torch.float32, device=self._device)
        self.fingertip_midpoint_angvel = torch.tensor([], dtype=torch.float32, device=self._device)
        self.fingertip_midpoint_jacobian = torch.tensor([], dtype=torch.float32, device=self._device)
        self.fingertip_midpoint_jacobian_tf = torch.tensor([], dtype=torch.float32, device=self._device)
        self.fingertip_attributes = {}

        # Properties
        self.num_arm_dofs = 0
        self.num_hand_dofs = 0
        self.num_dofs = 0

        # Control target tensor
        self.ctrl_target_fingertip_midpoint_pos = torch.tensor([], dtype=torch.float32, device=self._device)
        self.ctrl_target_fingertip_midpoint_quat = torch.tensor([], dtype=torch.float32, device=self._device)
        self.ctrl_target_dof_pos = torch.tensor([], dtype=torch.float32, device=self._device)
        self.ctrl_target_hand_dof_pos = torch.tensor([], dtype=torch.float32, device=self._device)
        self.ctrl_target_fingertip_contact_wrench = torch.tensor([], dtype=torch.float32, device=self._device)

        # Action tensor
        self.prev_actions = torch.tensor([], dtype=torch.float32, device=self._device)

        # Control config
        self.cfg_ctrl_keys = {
            'num_envs',
            'jacobian_type',
            'hand_prop_gains',
            'hand_deriv_gains',
            'motor_ctrl_mode',
            'gain_space',
            'ik_method',
            'joint_prop_gains',
            'joint_deriv_gains',
            'do_motion_ctrl',
            'task_prop_gains',
            'task_deriv_gains',
            'do_inertial_comp',
            'motion_ctrl_axes',
            'do_force_ctrl',
            'force_ctrl_method',
            'wrench_prop_gains',
            'force_ctrl_axes'
        }
        self.cfg_ctrl = {}
        if not hasattr(self, 'cfg_task'):
            self.cfg_task = self._task_cfg

        # Keypoint tensors
        self.identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device).repeat(self._num_envs, 1)

    def _get_base_yaml_params(self):
        '''Initialize instance variables from YAML files.'''

        # Set factory base config
        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name='factory_schema_config_base', node=FactorySchemaConfigBase)

        # Base params
        config_path = 'task/FactoryBase.yaml'  # Relative to Gym's Hydra search path (cfg dir)
        self.cfg_base = hydra.compose(config_name=config_path)
        self.cfg_base = self.cfg_base['task']  # Strip superfluous nesting

        # Asset params
        asset_info_path = '../tasks/factory/yaml/factory_asset_info_robot_table.yaml'
        # Relative to Gym's Hydra search path (cfg dir)
        self.asset_info_robot_table = hydra.compose(config_name=asset_info_path)
        self.asset_info_robot_table = (
            self.asset_info_robot_table['']['']['']['tasks']['factory']['yaml']
        )  # Strip superfluous nesting

    def import_robot_assets(self, add_to_stage=True):
        '''Set Franka and table asset options. Import assets.'''

        # Add Franka and Table to stage
        if add_to_stage:

            # Franka
            franka_translation = [self.cfg_base.env.robot_depth, 0.0, 0.0]
            franka_orientation = [0.0, 0.0, 0.0, 1.0]
            franka = FactoryFranka(
                prim_path=self.default_zero_env_path + '/franka',
                name='franka',
                translation=franka_translation,
                orientation=franka_orientation
            )
            self._sim_config.apply_articulation_settings(
                'franka',
                self._stage.GetPrimAtPath(franka.prim_path),
                self._sim_config.parse_actor_config('franka')
            )
            for link_prim in franka.prim.GetChildren():
                if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                    rb = PhysxSchema.PhysxRigidBodyAPI.Get(self._stage, link_prim.GetPrimPath())
                    rb.GetDisableGravityAttr().Set(True)
                    rb.GetRetainAccelerationsAttr().Set(False)
                    if self.cfg_base.sim.add_damping:
                        rb.GetLinearDampingAttr().Set(1.0)  # default = 0.0; increased to improve stability
                        rb.GetMaxLinearVelocityAttr().Set(1.0)  # default = 1000.0; reduced to prevent CUDA errors
                        rb.GetAngularDampingAttr().Set(5.0)  # default = 0.5; increased to improve stability
                        rb.GetMaxAngularVelocityAttr().Set(2 / math.pi * 180)  # default = 64.0; reduced to prevent CUDA errors
                    else:
                        rb.GetLinearDampingAttr().Set(0.0)
                        rb.GetMaxLinearVelocityAttr().Set(1000.0)
                        rb.GetAngularDampingAttr().Set(0.5)
                        rb.GetMaxAngularVelocityAttr().Set(64 / math.pi * 180)
            self.franka = franka

            # Table
            table_translation = [0.0, 0.0, self.cfg_base.env.table_height * 0.5]
            table_orientation = [1.0, 0.0, 0.0, 0.0]
            table_scale = [
                self.asset_info_robot_table.table.depth,
                self.asset_info_robot_table.table.width,
                self.cfg_base.env.table_height
            ]
            FixedCuboid(
                prim_path=self.default_zero_env_path + '/table',
                name='table',
                translation=table_translation,
                orientation=table_orientation,
                scale=table_scale,
                size=1.0,
                color=np.array([0, 0, 0])
            )

            # Number of Franka's dofs
            self.num_arm_dofs = len(franka.local_arm_joint_paths)
            self.num_hand_dofs = len(franka.local_finger_joint_paths)
            self.num_dofs = self.num_arm_dofs + self.num_hand_dofs

        # Set controller params
        self.parse_controller_spec(add_to_stage=add_to_stage)

    def get_collision_contacts(self, interested_objects: Optional[Dict[str, float]] = None):
        '''[summay]'''

        # Contact report
        contact_headers = self.physx_interface.get_contact_report()[0]

        # Check Franka contact
        contact_penalty = torch.zeros((self._num_envs), dtype=torch.float32, device=self._device)
        for contact_header in contact_headers:
            actor0 = str(PhysicsSchemaTools.intToSdfPath(contact_header.actor0))
            actor1 = str(PhysicsSchemaTools.intToSdfPath(contact_header.actor1))
            if 'franka' in actor0 or 'franka' in actor1:
                num_envs = int(actor0.split('/')[3].split('_')[-1])

                # Check fingertips contact
                for fingertip_link_name, attribute in self.fingertip_attributes.items():
                    if (
                        (not torch.equal(attribute[-1][num_envs], torch.tensor([0.0, 0.0, 0.0], device=self._device)))
                        and (fingertip_link_name in actor0 or fingertip_link_name in actor1)
                    ):
                        contact_penalty[num_envs] += 1

                # Check hand contact
                if (
                    (not torch.equal(self.hand_force[num_envs], torch.tensor([0.0, 0.0, 0.0], device=self._device)))
                    and ('hand' in actor0 or 'hand' in actor1)
                ):
                    contact_penalty[num_envs] += 2

                # Check object contact
                if interested_objects is not None:
                    contact_penalty[num_envs] += sum(
                        weight for obj, weight in interested_objects.items() if obj in actor0 or obj in actor1
                    )
        return contact_penalty

    def check_grasp_ready(self, interested_object: str, num_contacts: int):
        '''[summay]'''

        # Contact report
        contact_headers = self.physx_interface.get_contact_report()[0]

        # Check Franka contact
        contacts = torch.zeros((self._num_envs), dtype=torch.float32, device=self._device)
        for contact_header in contact_headers:
            actor0 = str(PhysicsSchemaTools.intToSdfPath(contact_header.actor0))
            actor1 = str(PhysicsSchemaTools.intToSdfPath(contact_header.actor1))
            for contact_link in self.franka_contact_links:
                if (
                    (contact_link in actor0 or contact_link in actor1)
                    and (interested_object in actor0 or interested_object in actor1)
                ):
                    num_envs = int(actor0.split('/')[3].split('_')[-1])

                    if contact_link == 'hand':
                        contact_force = self.frankas.hands.get_net_contact_forces(indices=[num_envs], clone=False)[0]
                    else:
                        contact_force = self.frankas.fingertip_views[contact_link + 's_view'].get_net_contact_forces(
                            indices=[num_envs],
                            clone=False
                        )[0]
                    if not torch.equal(contact_force, torch.tensor([0.0, 0.0, 0.0], device=self._device)):
                        contacts[num_envs] += 1

        contacts = torch.where(
            contacts >= num_contacts,
            torch.ones((self._num_envs), device=self._device),
            torch.zeros((self._num_envs), device=self._device)
        )
        return contacts

    def acquire_base_tensors(self):
        '''Acquire base tensors.'''

        # Dof tensors
        self.dof_pos = torch.zeros((self._num_envs, self.num_dofs), device=self._device)
        self.dof_vel = torch.zeros((self._num_envs, self.num_dofs), device=self._device)
        self.dof_torque = torch.zeros((self._num_envs, self.num_dofs), device=self._device)

        # Control target tensors
        self.ctrl_target_fingertip_midpoint_pos = torch.zeros((self._num_envs, 3), device=self._device)
        self.ctrl_target_fingertip_midpoint_quat = torch.zeros((self._num_envs, 4), device=self._device)
        self.ctrl_target_dof_pos = torch.zeros((self._num_envs, self.num_dofs), device=self._device)
        self.ctrl_target_hand_dof_pos = torch.zeros((self._num_envs, self.num_hand_dofs), device=self._device)
        self.ctrl_target_fingertip_contact_wrench = torch.zeros((self._num_envs, 6), device=self._device)

        # Action tensor
        self.prev_actions = torch.zeros((self._num_envs, self._num_actions), device=self._device)

    def refresh_base_tensors(self):
        '''Refresh base tensors.'''

        # Check if sim is playing
        if not self.world.is_playing():
            return

        # Dof state
        self.dof_pos = self.frankas.get_joint_positions(clone=False)
        self.dof_vel = self.frankas.get_joint_velocities(clone=False)

        # Franka state
        franka_jacobian = self.frankas.get_jacobians()  # [num_envs, num_bodies - 1, 6, num_dofs] (root has no Jacobian)
        franka_mass_matrix = self.frankas.get_mass_matrices(clone=False)  # [num_envs, num_dofs, num_dofs]

        # Arm state
        self.arm_dof_pos = self.dof_pos[:, 0:self.num_arm_dofs]
        self.arm_mass_matrix = franka_mass_matrix[:, 0:self.num_arm_dofs, 0:self.num_arm_dofs]  # for Franka arm (not hand)

        # Hand state
        self.hand_quat = self.ensure_tensor(self.frankas.hands.get_world_poses(clone=False)[1])
        hand_forces = self.frankas.hands.get_net_contact_forces(clone=False)
        self.hand_force = hand_forces[:, 0:3]

        # Fingertip centered state
        fingertip_centered_pos, fingertip_centered_quat = self.frankas.fingertip_centereds.get_world_poses(clone=False)
        fingertip_centered_pos -= self._env_pos
        fingertip_centered_velocities = self.frankas.fingertip_centereds.get_velocities(clone=False)
        fingertip_centered_linvel = fingertip_centered_velocities[:, 0:3]
        fingertip_centered_angvel = fingertip_centered_velocities[:, 3:6]

        # Left finger state
        left_finger_pos, left_finger_quat = self.frankas.fingertip_views['panda_leftfingers_view'].get_world_poses(clone=False)
        left_finger_pos -= self._env_pos
        left_finger_jacobian = franka_jacobian[:, self.num_arm_dofs + 1, 0:6, 0:self.num_arm_dofs]
        left_finger_forces = self.frankas.fingertip_views['panda_leftfingers_view'].get_net_contact_forces(clone=False)
        left_finger_force = left_finger_forces[:, 0:3]
        left_fingertip_pos = fc.translate_along_local_z(
            pos=left_finger_pos,
            quat=self.hand_quat,
            offset=self.asset_info_robot_table.hand.gripper.finger_length,
            device=self._device
        )  # End point of finger

        # Right finger state
        right_finger_pos, right_finger_quat = (
            self.frankas.fingertip_views['panda_rightfingers_view'].get_world_poses(clone=False)
        )
        right_finger_pos -= self._env_pos
        right_finger_jacobian = franka_jacobian[:, self.num_arm_dofs + 2, 0:6, 0:self.num_arm_dofs]
        right_finger_forces = self.frankas.fingertip_views['panda_rightfingers_view'].get_net_contact_forces(clone=False)
        right_finger_force = right_finger_forces[:, 0:3]
        right_fingertip_pos = fc.translate_along_local_z(
            pos=right_finger_pos,
            quat=self.hand_quat,
            offset=self.asset_info_robot_table.hand.gripper.finger_length,
            device=self._device
        )  # End point of finger

        # Update fingertip attributes
        self.fingertip_attributes['left_finger'] = [
            left_fingertip_pos,
            left_finger_quat,
            left_finger_force
        ]
        self.fingertip_attributes['right_finger'] = [
            right_fingertip_pos,
            right_finger_quat,
            right_finger_force
        ]

        # Fingertip midpoint state
        self.finger_midpoint_pos = (left_finger_pos + right_finger_pos) / 2
        self.fingertip_midpoint_pos = fc.translate_along_local_z(
            pos=self.finger_midpoint_pos,
            quat=self.hand_quat,
            offset=self.asset_info_robot_table.hand.gripper.finger_length,
            device=self._device
        )
        self.fingertip_midpoint_jacobian = 0.5 * (left_finger_jacobian + right_finger_jacobian)
        self.fingertip_midpoint_quat = fingertip_centered_quat  # always equal
        self.fingertip_midpoint_linvel = (
            fingertip_centered_linvel
            + torch.cross(fingertip_centered_angvel, self.fingertip_midpoint_pos - fingertip_centered_pos, dim=1)
        )  # The issue was added to the bug tracker: Add relative velocity term
        # (see https://dynamicsmotioncontrol487379916.files.wordpress.com/2020/11/21-me258pointmovingrigidbody.pdf)
        self.fingertip_midpoint_angvel = fingertip_centered_angvel  # always equal
        # From sum of angular velocities
        # (https://physics.stackexchange.com/questions/547698/understanding-addition-of-angular-velocity),
        # angular velocity of midpoint w.r.t. world is equal to sum of
        # angular velocity of midpoint w.r.t. hand and angular velocity of hand w.r.t. world.
        # Midpoint is in sliding contact (i.e., linear relative motion) with hand;
        # angular velocity of midpoint w.r.t. hand is zero.
        # Thus, angular velocity of midpoint w.r.t. world is equal to angular velocity of hand w.r.t. world.

    def parse_controller_spec(self, add_to_stage):
        '''Parse controller specification into lower-level controller configuration.'''

        # Update controller config
        self.cfg_ctrl['num_envs'] = self._num_envs
        self.cfg_ctrl['jacobian_type'] = self.cfg_task.ctrl.all.jacobian_type
        self.cfg_ctrl['hand_prop_gains'] = torch.tensor(
            self.cfg_task.ctrl.all.hand_prop_gains,
            device=self._device
        ).repeat((self._num_envs, 1))
        self.cfg_ctrl['hand_deriv_gains'] = torch.tensor(
            self.cfg_task.ctrl.all.hand_deriv_gains,
            device=self._device
        ).repeat((self._num_envs, 1))
        ctrl_type = self.cfg_task.ctrl.ctrl_type
        if ctrl_type == 'gym_default':
            self.cfg_ctrl['motor_ctrl_mode'] = 'gym'
            self.cfg_ctrl['gain_space'] = 'joint'
            self.cfg_ctrl['ik_method'] = self.cfg_task.ctrl.gym_default.ik_method
            self.cfg_ctrl['joint_prop_gains'] = torch.tensor(
                self.cfg_task.ctrl.gym_default.joint_prop_gains,
                device=self._device
            ).repeat((self._num_envs, 1))
            self.cfg_ctrl['joint_deriv_gains'] = torch.tensor(
                self.cfg_task.ctrl.gym_default.joint_deriv_gains,
                device=self._device
            ).repeat((self._num_envs, 1))
            self.cfg_ctrl['hand_prop_gains'] = torch.tensor(
                self.cfg_task.ctrl.gym_default.hand_prop_gains,
                device=self._device
            ).repeat((self._num_envs, 1))
            self.cfg_ctrl['hand_deriv_gains'] = torch.tensor(
                self.cfg_task.ctrl.gym_default.hand_deriv_gains,
                device=self._device
            ).repeat((self._num_envs, 1))
        elif ctrl_type == 'joint_space_ik':
            self.cfg_ctrl['motor_ctrl_mode'] = 'manual'
            self.cfg_ctrl['gain_space'] = 'joint'
            self.cfg_ctrl['ik_method'] = self.cfg_task.ctrl.joint_space_ik.ik_method
            self.cfg_ctrl['joint_prop_gains'] = torch.tensor(
                self.cfg_task.ctrl.joint_space_ik.joint_prop_gains,
                device=self._device
            ).repeat((self._num_envs, 1))
            self.cfg_ctrl['joint_deriv_gains'] = torch.tensor(
                self.cfg_task.ctrl.joint_space_ik.joint_deriv_gains,
                device=self._device
            ).repeat((self._num_envs, 1))
            self.cfg_ctrl['do_inertial_comp'] = False
        elif ctrl_type == 'joint_space_id':
            self.cfg_ctrl['motor_ctrl_mode'] = 'manual'
            self.cfg_ctrl['gain_space'] = 'joint'
            self.cfg_ctrl['ik_method'] = self.cfg_task.ctrl.joint_space_id.ik_method
            self.cfg_ctrl['joint_prop_gains'] = torch.tensor(
                self.cfg_task.ctrl.joint_space_id.joint_prop_gains,
                device=self._device
            ).repeat((self._num_envs, 1))
            self.cfg_ctrl['joint_deriv_gains'] = torch.tensor(
                self.cfg_task.ctrl.joint_space_id.joint_deriv_gains,
                device=self._device
            ).repeat((self._num_envs, 1))
            self.cfg_ctrl['do_inertial_comp'] = True
        elif ctrl_type == 'task_space_impedance':
            self.cfg_ctrl['motor_ctrl_mode'] = 'manual'
            self.cfg_ctrl['gain_space'] = 'task'
            self.cfg_ctrl['do_motion_ctrl'] = True
            self.cfg_ctrl['task_prop_gains'] = torch.tensor(
                self.cfg_task.ctrl.task_space_impedance.task_prop_gains,
                device=self._device
            ).repeat((self._num_envs, 1))
            self.cfg_ctrl['task_deriv_gains'] = torch.tensor(
                self.cfg_task.ctrl.task_space_impedance.task_deriv_gains,
                device=self._device
            ).repeat((self._num_envs, 1))
            self.cfg_ctrl['do_inertial_comp'] = False
            self.cfg_ctrl['motion_ctrl_axes'] = torch.tensor(
                self.cfg_task.ctrl.task_space_impedance.motion_ctrl_axes,
                device=self._device
            ).repeat((self._num_envs, 1))
            self.cfg_ctrl['do_force_ctrl'] = False
        elif ctrl_type == 'operational_space_motion':
            self.cfg_ctrl['motor_ctrl_mode'] = 'manual'
            self.cfg_ctrl['gain_space'] = 'task'
            self.cfg_ctrl['do_motion_ctrl'] = True
            self.cfg_ctrl['task_prop_gains'] = torch.tensor(
                self.cfg_task.ctrl.operational_space_motion.task_prop_gains,
                device=self._device
            ).repeat((self._num_envs, 1))
            self.cfg_ctrl['task_deriv_gains'] = torch.tensor(
                self.cfg_task.ctrl.operational_space_motion.task_deriv_gains,
                device=self._device
            ).repeat((self._num_envs, 1))
            self.cfg_ctrl['do_inertial_comp'] = True
            self.cfg_ctrl['motion_ctrl_axes'] = torch.tensor(
                self.cfg_task.ctrl.operational_space_motion.motion_ctrl_axes,
                device=self._device
            ).repeat((self._num_envs, 1))
            self.cfg_ctrl['do_force_ctrl'] = False
        elif ctrl_type == 'open_loop_force':
            self.cfg_ctrl['motor_ctrl_mode'] = 'manual'
            self.cfg_ctrl['gain_space'] = 'task'
            self.cfg_ctrl['do_motion_ctrl'] = False
            self.cfg_ctrl['do_force_ctrl'] = True
            self.cfg_ctrl['force_ctrl_method'] = 'open'
            self.cfg_ctrl['force_ctrl_axes'] = torch.tensor(
                self.cfg_task.ctrl.open_loop_force.force_ctrl_axes,
                device=self._device
            ).repeat((self._num_envs, 1))
        elif ctrl_type == 'closed_loop_force':
            self.cfg_ctrl['motor_ctrl_mode'] = 'manual'
            self.cfg_ctrl['gain_space'] = 'task'
            self.cfg_ctrl['do_motion_ctrl'] = False
            self.cfg_ctrl['do_force_ctrl'] = True
            self.cfg_ctrl['force_ctrl_method'] = 'closed'
            self.cfg_ctrl['wrench_prop_gains'] = torch.tensor(
                self.cfg_task.ctrl.closed_loop_force.wrench_prop_gains,
                device=self._device
            ).repeat((self._num_envs, 1))
            self.cfg_ctrl['force_ctrl_axes'] = torch.tensor(
                self.cfg_task.ctrl.closed_loop_force.force_ctrl_axes,
                device=self._device
            ).repeat((self._num_envs, 1))
        elif ctrl_type == 'hybrid_force_motion':
            self.cfg_ctrl['motor_ctrl_mode'] = 'manual'
            self.cfg_ctrl['gain_space'] = 'task'
            self.cfg_ctrl['do_motion_ctrl'] = True
            self.cfg_ctrl['task_prop_gains'] = torch.tensor(
                self.cfg_task.ctrl.hybrid_force_motion.task_prop_gains,
                device=self._device
            ).repeat((self._num_envs, 1))
            self.cfg_ctrl['task_deriv_gains'] = torch.tensor(
                self.cfg_task.ctrl.hybrid_force_motion.task_deriv_gains,
                device=self._device
            ).repeat((self._num_envs, 1))
            self.cfg_ctrl['do_inertial_comp'] = True
            self.cfg_ctrl['motion_ctrl_axes'] = torch.tensor(
                self.cfg_task.ctrl.hybrid_force_motion.motion_ctrl_axes,
                device=self._device
            ).repeat((self._num_envs, 1))
            self.cfg_ctrl['do_force_ctrl'] = True
            self.cfg_ctrl['force_ctrl_method'] = 'closed'
            self.cfg_ctrl['wrench_prop_gains'] = torch.tensor(
                self.cfg_task.ctrl.hybrid_force_motion.wrench_prop_gains,
                device=self._device
            ).repeat((self._num_envs, 1))
            self.cfg_ctrl['force_ctrl_axes'] = torch.tensor(
                self.cfg_task.ctrl.hybrid_force_motion.force_ctrl_axes,
                device=self._device
            ).repeat((self._num_envs, 1))
        cfg_ctrl_keys = list(self.cfg_ctrl.keys())
        cfg_ctrl_missing_keys = {cfg_ctrl_key: None for cfg_ctrl_key in self.cfg_ctrl_keys if cfg_ctrl_key not in cfg_ctrl_keys}
        self.cfg_ctrl.update(cfg_ctrl_missing_keys)

        # Local joint paths
        local_arm_joint_paths = self.franka.local_arm_joint_paths
        local_finger_joint_paths = self.franka.local_finger_joint_paths

        # Edit the drive params
        if add_to_stage:
            if self.cfg_ctrl['motor_ctrl_mode'] == 'gym':

                # Arm drives
                for i, local_arm_joint_path in enumerate(local_arm_joint_paths):
                    joint_prim = self._stage.GetPrimAtPath(self.default_zero_env_path + '/franka/' + local_arm_joint_path)
                    drive = UsdPhysics.DriveAPI.Apply(joint_prim, 'angular')
                    drive.GetStiffnessAttr().Set(self.cfg_ctrl['joint_prop_gains'][0, i].item() * np.pi / 180)
                    drive.GetDampingAttr().Set(self.cfg_ctrl['joint_deriv_gains'][0, i].item() * np.pi / 180)

                # Hand drives
                for i, local_finger_joint_path in enumerate(local_finger_joint_paths):
                    joint_prim = self._stage.GetPrimAtPath(
                        self.default_zero_env_path + '/franka/' + local_finger_joint_path
                    )
                    drive = UsdPhysics.DriveAPI.Apply(joint_prim, 'linear')
                    drive.GetStiffnessAttr().Set(self.cfg_ctrl['hand_deriv_gains'][0, i].item())
                    drive.GetDampingAttr().Set(self.cfg_ctrl['hand_deriv_gains'][0, i].item())
            elif self.cfg_ctrl['motor_ctrl_mode'] == 'manual':

                # Arm drives
                for local_arm_joint_path in local_arm_joint_paths:
                    joint_prim = self._stage.GetPrimAtPath(self.default_zero_env_path + '/franka/' + local_arm_joint_path)
                    joint_prim.RemoveAPI(UsdPhysics.DriveAPI, 'angular')
                    drive = UsdPhysics.DriveAPI.Apply(joint_prim, 'None')
                    drive.GetStiffnessAttr().Set(0.0)
                    drive.GetDampingAttr().Set(0.0)

                # Hand drives
                for local_finger_joint_path in local_finger_joint_paths:
                    joint_prim = self._stage.GetPrimAtPath(
                        self.default_zero_env_path + '/franka/' + local_finger_joint_path
                    )
                    joint_prim.RemoveAPI(UsdPhysics.DriveAPI, 'linear')
                    drive = UsdPhysics.DriveAPI.Apply(joint_prim, 'None')
                    drive.GetStiffnessAttr().Set(0.0)
                    drive.GetDampingAttr().Set(0.0)

    def generate_ctrl_signals(self):
        '''Get Jacobian. Set Franka DOF position targets or DOF torques.'''

        # Get desired Jacobian
        if self.cfg_ctrl['jacobian_type'] == 'geometric':
            self.fingertip_midpoint_jacobian_tf = self.fingertip_midpoint_jacobian
        elif self.cfg_ctrl['jacobian_type'] == 'analytic':
            self.fingertip_midpoint_jacobian_tf = fc.get_analytic_jacobian(
                fingertip_quat=self.fingertip_midpoint_quat,
                fingertip_jacobian=self.fingertip_midpoint_jacobian,
                num_envs=self._num_envs,
                device=self._device
            )

        # Set PD joint pos target or joint torque
        if not isinstance(self.ctrl_target_hand_dof_pos, torch.Tensor):
            self.ctrl_target_hand_dof_pos *= torch.ones((self._num_envs, self.num_hand_dofs), device=self._device)
        if self.cfg_ctrl['motor_ctrl_mode'] == 'gym':
            self._set_dof_pos_target()
        elif self.cfg_ctrl['motor_ctrl_mode'] == 'manual':
            self._set_dof_torque()

    def _set_dof_pos_target(self, joint_indices: Optional[Sequence[int]] = None):
        '''Set Franka DOF position target to move fingertips towards target pose.'''

        if joint_indices is None:
            joint_indices = list(range(self.num_dofs))
        self.ctrl_target_dof_pos = fc.compute_dof_pos_target(
            cfg_ctrl=self.cfg_ctrl,
            arm_dof_pos=self.arm_dof_pos,
            fingertip_midpoint_pos=self.fingertip_midpoint_pos,
            fingertip_midpoint_quat=self.fingertip_midpoint_quat,
            jacobian=self.fingertip_midpoint_jacobian_tf,
            ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
            ctrl_target_hand_dof_pos=self.ctrl_target_hand_dof_pos,
            device=self._device
        )[:, joint_indices]
        self.frankas.set_joint_position_targets(positions=self.ctrl_target_dof_pos, joint_indices=joint_indices)

    def _set_dof_torque(self, joint_indices: Optional[Sequence[int]] = None):
        '''Set Franka DOF torque to move fingertips towards target pose.'''

        if joint_indices is None:
            joint_indices = list(range(self.num_dofs))
        self.dof_torque = fc.compute_dof_torque(
            cfg_ctrl=self.cfg_ctrl,
            dof_pos=self.dof_pos,
            dof_vel=self.dof_vel,
            fingertip_midpoint_pos=self.fingertip_midpoint_pos,
            fingertip_midpoint_quat=self.fingertip_midpoint_quat,
            fingertip_midpoint_linvel=self.fingertip_midpoint_linvel,
            fingertip_midpoint_angvel=self.fingertip_midpoint_angvel,
            finger_forces=[attributes[-1] for attributes in self.fingertip_attributes.values()],
            jacobian=self.fingertip_midpoint_jacobian_tf,
            arm_mass_matrix=self.arm_mass_matrix,
            ctrl_target_hand_dof_pos=self.ctrl_target_hand_dof_pos,
            ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
            ctrl_target_fingertip_contact_wrench=self.ctrl_target_fingertip_contact_wrench,
            device=self._device
        )[:, joint_indices]
        self.frankas.set_joint_efforts(efforts=self.dof_torque, joint_indices=joint_indices)

    def enable_gravity(self, gravity_mag):
        '''Enable gravity.'''

        gravity = [0.0, 0.0, -gravity_mag]
        self.world._physics_sim_view.set_gravity(carb.Float3(gravity[0], gravity[1], gravity[2]))

    def disable_gravity(self):
        '''Disable gravity.'''

        gravity = [0.0, 0.0, 0.0]
        self.world._physics_sim_view.set_gravity(carb.Float3(gravity[0], gravity[1], gravity[2]))
