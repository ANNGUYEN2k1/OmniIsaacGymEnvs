# Copyright (c) 2018-2022, NVIDIA Corporation
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

'''Required python modules.'''
import torch
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView


class TomoDJView(ArticulationView):
    '''[summary]'''

    def __init__(self, prim_paths_expr: str, name: str = 'TomoDJView') -> None:
        '''Initialize articulation view.'''

        super().__init__(prim_paths_expr=prim_paths_expr, name=name, reset_xform_properties=False)

        self._hands = RigidPrimView(
            prim_paths_expr='/World/envs/.*/tomodj/left_arm_link7',
            name='hands_view',
            reset_xform_properties=False
        )
        self._thumbs = RigidPrimView(
            prim_paths_expr='/World/envs/.*/tomodj/LF1_3',
            name='LF1_3s_view',
            reset_xform_properties=False
        )
        self._forefingers = RigidPrimView(
            prim_paths_expr='/World/envs/.*/tomodj/LF2_4',
            name='LF2_4s_view',
            reset_xform_properties=False
        )
        self._middle_fingers = RigidPrimView(
            prim_paths_expr='/World/envs/.*/tomodj/LF3_4',
            name='LF3_4s_view',
            reset_xform_properties=False
        )
        self._ring_fingers = RigidPrimView(
            prim_paths_expr='/World/envs/.*/tomodj/LF4_4',
            name='LF4_4s_view',
            reset_xform_properties=False
        )
        self._little_fingers = RigidPrimView(
            prim_paths_expr='/World/envs/.*/tomodj/LF5_4',
            name='LF5_4s_view',
            reset_xform_properties=False
        )

        # Properties
        self._fingertip_views = {
            'LF1_3s_view': self._thumbs,
            'LF2_4s_view': self._forefingers,
            'LF3_4s_view': self._middle_fingers,
            'LF4_4s_view': self._ring_fingers,
            'LF5_4s_view': self._little_fingers,
        }
        self._arm_joint_names = [
            'left_arm_joint1',
            'left_arm_joint2',
            'left_arm_joint3',
            'left_arm_joint4',
            'left_arm_joint5',
            'left_arm_joint6',
            'left_arm_joint7'
        ]
        self._finger_joint_names = [
            'LF1_1_joint',
            'LF1_2_joint',
            'LF1_3_joint',
            'LF2_1_joint',
            'LF2_2_joint',
            'LF2_3_joint',
            'LF3_1_joint',
            'LF3_2_joint',
            'LF3_3_joint',
            'LF4_1_joint',
            'LF4_2_joint',
            'LF4_3_joint',
            'LF5_1_joint',
            'LF5_2_joint',
            'LF5_3_joint'
        ]
        self._arm_indices = []
        self._finger_indices = []
        self._actuated_dof_indices = []

    def initialize(self, physics_sim_view=None):
        '''Initialize physics simulation view.'''

        super().initialize(physics_sim_view)

        print('Bodies:', self.body_names)
        print('Dofs:', self.dof_names)

        # Set tendon joints's params
        limit_stiffness = torch.tensor([30.0] * self.num_fixed_tendons, device=self._device)
        damping = torch.tensor([0.1] * self.num_fixed_tendons, device=self._device)
        self.set_fixed_tendon_properties(dampings=damping, limit_stiffnesses=limit_stiffness)

        # Get actuator joint_indices
        self._actuated_dof_indices = self._arm_indices + self._finger_indices

    @property
    def arm_indices(self):
        '''Retrieves the list indices of arm joints.

        Returns:
            arm_indices (List[int]): Indices of arm joints.
        '''

        self._arm_indices = [self.get_dof_index(joint_name) for joint_name in self._arm_joint_names]
        return self._arm_indices

    @property
    def finger_indices(self):
        '''Retrieves the list indices of finger joints.

        Returns:
            finger_indices (List[int]): Indices of finger joints.
        '''

        self._finger_indices = [self.get_dof_index(joint_name) for joint_name in self._finger_joint_names]
        return self._finger_indices

    @property
    def actuated_dof_indices(self):
        '''Retrieves the list indices of actuator joints (not Tendon or Mimic joints).

        Returns:
            finger_indices (List[int]): Indices of actuator joints.
        '''

        self._actuated_dof_indices = self._arm_indices + self._finger_indices
        return self._actuated_dof_indices

    @property
    def fingertip_views(self):
        '''Retrieves the dict of fingertips's views.

        Returns:
            fingertip_views (Dict{str: RigidPrimView}): ArticulationViews of fingertips.
        '''

        return self._fingertip_views

    @property
    def hands(self):
        '''Retrieves the hands's view.

        Returns:
            hands (RigidPrimView): ArticulationView of hands.
        '''

        return self._hands
