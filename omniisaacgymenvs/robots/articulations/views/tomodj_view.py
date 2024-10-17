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
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView


class TomoDJView(ArticulationView):
    '''[summary]'''

    def __init__(self, prim_paths_expr: str, name: str = 'TomoDJView') -> None:
        '''Initialize articulation view.'''

        super().__init__(prim_paths_expr=prim_paths_expr, name=name, reset_xform_properties=False)

        # Create views
        self._hands = RigidPrimView(
            prim_paths_expr='/World/envs/.*/tomodj/left_arm_link7',
            name='hands_view',
            reset_xform_properties=False
        )
        self._lfingers = RigidPrimView(
            prim_paths_expr='/World/envs/.*/tomodj/left_finger',
            name='left_fingers_view',
            reset_xform_properties=False
        )
        self._rfingers = RigidPrimView(
            prim_paths_expr='/World/envs/.*/tomodj/right_finger',
            name='right_fingers_view',
            reset_xform_properties=False
        )

        # Properties
        self._fingertip_views = {'left_fingers_view': self._lfingers, 'right_fingers_view': self._rfingers}
        self._finger_indices = []

    def initialize(self, physics_sim_view=None):
        '''Initialize physics simulation view.'''

        super().initialize(physics_sim_view)

        print('Bodies:', self.body_names)
        print('Dofs:', self.dof_names)

    @property
    def finger_indices(self):
        '''Retrieves the list indices of finger joints.

        Returns:
            finger_indices (List[int]): Indices of finger joints.
        '''

        self._finger_indices = [self.get_dof_index('finger_joint1'), self.get_dof_index('finger_joint2')]
        return self._finger_indices

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
