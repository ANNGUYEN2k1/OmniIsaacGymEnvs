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

'''Required python modules'''
import math
from typing import Optional, Sequence

import numpy as np
import torch
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from pxr import PhysxSchema

from omniisaacgymenvs.tasks.utils.usd_utils import set_drive


class TomoSPC(Robot):
    '''[summary]'''

    def __init__(
        self,
        prim_path: str,
        name: str = 'tomospc',
        usd_path: Optional[str] = None,
        translation: Optional[Sequence[float]] = None,
        orientation: Optional[Sequence[float]] = None
    ) -> None:
        '''[summary]'''

        # Porperties
        self._usd_path = usd_path
        self._name = name
        self._position = [0.95, 0.0, 0.4] if translation is None else translation
        self._orientation = [0.0, 0.0, 0.0, 1.0] if orientation is None else orientation
        if self._usd_path is not None:
            add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=self._position,
            orientation=self._orientation,
            articulation_controller=None
        )

        # Set arm drives's params
        self._dof_paths = [
            'body_base/left_arm_joint1',
            'left_arm_link1/left_arm_joint2',
            'left_arm_link2/left_arm_joint3',
            'left_arm_link3/left_arm_joint4',
            'left_arm_link4/left_arm_joint5',
            'left_arm_link5/left_arm_joint6'
        ]
        drive_type = ['angular'] * 6
        default_dof_pos = [math.degrees(x) for x in [-0.52, 0.26, 0.79, 0.35, 1.05, 0]]
        stiffness = [400 * np.pi / 180] * 6
        damping = [80 * np.pi / 180] * 6
        max_force = [87, 87, 87, 87, 87, 87]
        max_velocity = [math.degrees(x) for x in [2.175, 2.175, 2.175, 2.175, 2.175, 2.175]]
        for i, dof in enumerate(self._dof_paths):
            set_drive(
                prim_path=f'{prim_path}/{dof}',
                drive_type=drive_type[i],
                target_type='position',
                target_value=default_dof_pos[i],
                stiffness=stiffness[i],
                damping=damping[i],
                max_force=max_force[i]
            )
            PhysxSchema.PhysxJointAPI(get_prim_at_path(f'{prim_path}/{dof}')).CreateMaxJointVelocityAttr().Set(  # type: ignore
                max_velocity[i]
            )

        # Set finger drives's params
        self._finger_joints_config = {
            'left_hand/LF1_1_joint': {'stiffness': 1, 'damping': 0.1, 'max_force': 2.3722},
            'left_hand/LF2_1_joint': {'stiffness': 1, 'damping': 0.1, 'max_force': 2.3722},
            'left_hand/LF3_1_joint': {'stiffness': 1, 'damping': 0.1, 'max_force': 2.3722},
            'left_hand/LF4_1_joint': {'stiffness': 1, 'damping': 0.1, 'max_force': 2.3722},
            'left_hand/LF5_1_joint': {'stiffness': 1, 'damping': 0.1, 'max_force': 2.3722},
            'LF1_1/LF1_2_joint': {'stiffness': 1, 'damping': 0.1, 'max_force': 1.45},
            'LF1_2/LF1_3_joint': {'stiffness': 1, 'damping': 0.1, 'max_force': 0.99},
            'LF2_1/LF2_2_joint': {'stiffness': 1, 'damping': 0.1, 'max_force': 0.9},
            'LF2_2/LF2_3_joint': {'stiffness': 1, 'damping': 0.1, 'max_force': 0.9},
            'LF2_3/LF2_4_joint': {'stiffness': 1, 'damping': 0.1, 'max_force': 0.7245},
            'LF3_1/LF3_2_joint': {'stiffness': 1, 'damping': 0.1, 'max_force': 0.9},
            'LF3_2/LF3_3_joint': {'stiffness': 1, 'damping': 0.1, 'max_force': 0.9},
            'LF3_3/LF3_4_joint': {'stiffness': 1, 'damping': 0.1, 'max_force': 0.7245},
            'LF4_1/LF4_2_joint': {'stiffness': 1, 'damping': 0.1, 'max_force': 0.9},
            'LF4_2/LF4_3_joint': {'stiffness': 1, 'damping': 0.1, 'max_force': 0.9},
            'LF4_3/LF4_4_joint': {'stiffness': 1, 'damping': 0.1, 'max_force': 0.7245},
            'LF5_1/LF5_2_joint': {'stiffness': 1, 'damping': 0.1, 'max_force': 0.9},
            'LF5_2/LF5_3_joint': {'stiffness': 1, 'damping': 0.1, 'max_force': 0.9},
            'LF5_3/LF5_4_joint': {'stiffness': 1, 'damping': 0.1, 'max_force': 0.7245}
        }
        for finger_joint_name, config in self._finger_joints_config.items():
            set_drive(
                f'{self.prim_path}/{finger_joint_name}',
                'angular',
                'position',
                0.0,
                config['stiffness'] * np.pi / 180,
                config['damping'] * np.pi / 180,
                config['max_force']
            )

    def set_tomospc_properties(self, stage):
        '''Disable gravity for all child rigid bodies of the prim.'''

        for link_prim in self.prim.GetChildren():
            if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                rb = PhysxSchema.PhysxRigidBodyAPI.Get(stage, link_prim.GetPrimPath())
                rb.GetDisableGravityAttr().Set(True)  # type: ignore

    @property
    def position(self) -> torch.tensor:
        '''Position of robot

        Returns:
            position (torch.Tensor[3]): position of the robot

        Example:

        .. code-block:: python

            >>> tomodj.position
            [0 0 0]
        '''

        return torch.tensor(self._position)

    @property
    def local_finger_joint_paths(self) -> list:
        '''Position of robot

        Returns:
            local_finger_joint_paths (List[str]): list local path of the robot's finger joints

        Example:

        .. code-block:: python

            >>> tomodj.local_finger_joint_paths
            ['left_hand/LF1_1_joint' 'LF1_1/LF1_2_joint' ... ]
        '''

        return list(self._finger_joints_config.keys())

    @property
    def local_arm_joint_paths(self) -> list:
        '''Position of robot

        Returns:
            local_finger_joint_paths (List[str]): list local path of the robot's finger joints

        Example:

        .. code-block:: python

            >>> tomodj.local_arm_joint_paths
            ['body_base/left_arm_joint1' 'left_arm_link1/left_arm_joint2', ...]
        '''

        return self._dof_paths
