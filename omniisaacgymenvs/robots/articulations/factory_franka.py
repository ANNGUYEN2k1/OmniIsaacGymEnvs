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

import carb
import numpy as np
import torch
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.nucleus import get_assets_root_path
from pxr import PhysxSchema

from omniisaacgymenvs.tasks.utils.usd_utils import set_drive


class FactoryFranka(Robot):
    '''[summary]'''

    def __init__(
        self,
        prim_path: str,
        name: str = 'franka',
        usd_path: Optional[str] = None,
        translation: Optional[Sequence[float]] = None,
        orientation: Optional[Sequence[float]] = None
    ) -> None:
        '''[summary]'''

        # Porperties
        self._usd_path = usd_path
        self._name = name
        self._position = [1.0, 0.0, 0.0] if translation is None else translation
        self._orientation = [0.0, 0.0, 0.0, 1.0] if orientation is None else orientation
        if self._usd_path is None:
            assets_root_path = get_assets_root_path()
            if assets_root_path is None:
                carb.log_error('Could not find Isaac Sim assets folder')
            else:
                self._usd_path = assets_root_path + '/Isaac/Robots/FactoryFranka/factory_franka.usd'
        if self._usd_path is not None:
            add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=self._position,
            orientation=self._orientation,
            articulation_controller=None
        )

        # Set franka drives's params
        self._dof_paths = [
            'panda_link0/panda_joint1',
            'panda_link1/panda_joint2',
            'panda_link2/panda_joint3',
            'panda_link3/panda_joint4',
            'panda_link4/panda_joint5',
            'panda_link5/panda_joint6',
            'panda_link6/panda_joint7',
            'panda_hand/panda_finger_joint1',
            'panda_hand/panda_finger_joint2'
        ]
        drive_type = ['angular'] * 7 + ['linear'] * 2
        default_dof_pos = [math.degrees(x) for x in [0.0, -1.0, 0.0, -2.2, 0.0, 2.4, 0.8]] + [0.02, 0.02]
        stiffness = [40 * np.pi / 180] * 7 + [500] * 2
        damping = [80 * np.pi / 180] * 7 + [20] * 2
        max_force = [87, 87, 87, 87, 12, 12, 12, 200, 200]
        max_velocity = [math.degrees(x) for x in [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61]] + [0.2, 0.2]
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

    def set_franka_properties(self, stage):
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
            ['left_hand/finger_joint1' 'left_hand/finger_joint2']
        '''

        return self._dof_paths[-2:]

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

        return self._dof_paths[:-2]
