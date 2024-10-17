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

'''Factory: class for mold env.

Inherits base class and abstract environment class. Inherited by mold task classes. Not directly executed.

Configuration defined in FactoryEnvMold_TomoDJ.yaml.
'''
from typing import List

import hydra
import numpy as np  # noqa: F401
import torch
from omni.isaac.core.objects import VisualCuboid  # noqa: F401
from omni.isaac.core.prims import RigidPrimView, XFormPrim
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.physx import get_physx_simulation_interface
from omni.physx.scripts import physicsUtils, utils

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.tasks.factory.factory_base_tomodj import FactoryBase
from omniisaacgymenvs.tasks.factory.factory_schema_class_env import FactoryABCEnv
from omniisaacgymenvs.tasks.factory.factory_schema_config_env import FactorySchemaConfigEnv


class FactoryEnvMold(FactoryBase, FactoryABCEnv):
    '''[summary]'''

    def __init__(self, name, sim_config, env, offset=None) -> None:
        '''Initialize base superclass. Initialize instance variables.'''

        super().__init__(name, sim_config, env, offset)

        if not hasattr(self, 'hand'):
            self.hand = 'gripper'

        # Get env params
        self._get_env_yaml_params()

        # Views
        self.molds = None

        # Mold properties
        self.moldPhysicsMaterialPath = '/World/Physics_Materials/MoldMaterial'
        self.mold_heights = torch.tensor([], dtype=torch.float32, device=self._device)
        self.mold_widths = torch.tensor([], dtype=torch.float32, device=self._device)
        self.mold_lengths = torch.tensor([], dtype=torch.float32, device=self._device)

        # Mold state tensors
        self.mold_pos = torch.tensor([], dtype=torch.float32, device=self._device)
        self.mold_quat = torch.tensor([], dtype=torch.float32, device=self._device)
        self.mold_linvel = torch.tensor([], dtype=torch.float32, device=self._device)
        self.mold_angvel = torch.tensor([], dtype=torch.float32, device=self._device)
        self.mold_force = torch.tensor([], dtype=torch.float32, device=self._device)

    def _get_env_yaml_params(self):
        '''Initialize instance variables from YAML files.'''

        # Set factory env config
        cs = hydra.core.config_store.ConfigStore.instance()  # type: ignore
        cs.store(name='factory_schema_config_env', node=FactorySchemaConfigEnv)

        # Env params
        config_path = 'task/FactoryEnvMold.yaml'  # Relative to Hydra search path (cfg dir)
        self.cfg_env = hydra.compose(config_name=config_path)
        self.cfg_env = self.cfg_env['task']  # Strip superfluous nesting

        # Get mold info
        asset_info_path = (
            '../tasks/factory/yaml/factory_asset_info_mold.yaml'
        )  # relative to Gym's Hydra search path (cfg dir)
        self.asset_info_mold = hydra.compose(config_name=asset_info_path)
        self.asset_info_mold = self.asset_info_mold['']['']['']['tasks']['factory']['yaml']  # Strip superfluous nesting

    def update_config(self, sim_config):
        '''[summary]'''

        # configs
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        # Properties
        self._num_envs = self._task_cfg['env']['numEnvs']
        self._num_observations = self._task_cfg['env']['numObservations']
        self._num_actions = self._task_cfg['env']['numActions']
        self._env_spacing = self.cfg_base['env']['env_spacing']

        # Get env params
        self._get_env_yaml_params()

    def create_views(self, scene) -> None:
        '''Create views'''

        # Import FactoryTomoDJView
        if self.hand == 'newhand':
            from omniisaacgymenvs.robots.articulations.views.factory_tomodj_newhand_view import FactoryTomoDJView
        elif self.hand == 'gripper':
            from omniisaacgymenvs.robots.articulations.views.factory_tomodj_view import FactoryTomoDJView

        # Remove existed views if needed
        if self.tomodjs is not None:
            if scene.object_exists('tomodjs_view'):
                scene.remove_object('tomodjs_view', registry_only=True)
            if scene.object_exists('molds_view'):
                scene.remove_object('molds_view', registry_only=True)
            if scene.object_exists('hands_view'):
                scene.remove_object('hands_view', registry_only=True)
            for fingertip_view_name in self.tomodjs.fingertip_views:
                if scene.object_exists(fingertip_view_name):
                    scene.remove_object(fingertip_view_name, registry_only=True)
            if scene.object_exists('fingertip_centereds_view'):
                scene.remove_object('fingertip_centereds_view', registry_only=True)

        # Create TomoDJ and Mold's views
        self.tomodjs = FactoryTomoDJView(prim_paths_expr='/World/envs/.*/tomodj', name='tomodjs_view')
        self.molds = RigidPrimView(prim_paths_expr='/World/envs/.*/plastic_mold', name='molds_view', track_contact_forces=True)

        # Add views to scene
        scene.add(self.molds)
        scene.add(self.tomodjs)
        scene.add(self.tomodjs.hands)
        for fingertip_view in self.tomodjs.fingertip_views.values():
            scene.add(fingertip_view)
        scene.add(self.tomodjs.fingertip_centereds)

    def set_up_scene(
        self,
        scene,
        replicate_physics=False,
        collision_filter_global_paths: List[str] = [],
        filter_collisions=True,
        copy_from_source=False
    ) -> None:
        '''Import assets. Add to scene.'''

        # Increase buffer size to prevent overflow for Place and Screw tasks
        physxSceneAPI = self.world.get_physics_context()._physx_scene_api
        physxSceneAPI.CreateGpuCollisionStackSizeAttr().Set(256 * 1024 * 1024)

        # Import TomoDJ from usd file
        self.import_robot_assets(add_to_stage=True)

        # Create mold's material
        self.create_mold_material()

        # # Create cuboid to check mold grasp and hand grasp poses
        # for i in range(self.cfg_task.rl.num_keypoints + 1):
        #     for obj in ['hand', 'mold']:
        #         VisualCuboid(
        #             prim_path=self.default_zero_env_path + (f'/{obj}_{i}' if i != 0 else f'/{obj}'),
        #             name=f'{obj}_{i}' if i != 0 else obj,
        #             scale=[0.01, 0.01, 0.01],
        #             color=np.array([1.0, 0.0, 0.0] if obj == 'hand' else [0.0, 0.0, 1.0])
        #         )

        # Set up scene
        RLTask.set_up_scene(
            self,
            scene,
            replicate_physics=replicate_physics,
            collision_filter_global_paths=collision_filter_global_paths,
            filter_collisions=filter_collisions,
            copy_from_source=copy_from_source
        )

        # Import Mold from usd file
        self._import_env_assets(add_to_stage=True)

        # Create views
        self.create_views(scene)

        # For get collision contact
        self.physx_interface = get_physx_simulation_interface()

        # Update distals to contact links
        if self.hand == 'newhand':
            self.tomodj_contact_links.append('hand')
        self.tomodj_contact_links += [fingertip_view_name[:-6] for fingertip_view_name in self.tomodjs.fingertip_views]

    def initialize_views(self, scene) -> None:
        '''Initialize views for extension workflow.'''

        super().initialize_views(scene)

        # Ensure scene was fully
        self.import_robot_assets(add_to_stage=False)
        self._import_env_assets(add_to_stage=False)
        self.create_views(scene)

    def create_mold_material(self):
        '''Define mold material.'''

        utils.addRigidBodyMaterial(
            self._stage,
            self.moldPhysicsMaterialPath,
            density=self.cfg_env.env.mold_density,
            staticFriction=self.cfg_env.env.mold_friction,
            dynamicFriction=self.cfg_env.env.mold_friction,
            restitution=0.0
        )

    def _import_env_assets(self, add_to_stage=True):
        '''Set mold asset options. Import assets.'''

        # Mold properties
        mold_heights = []
        mold_widths = []
        mold_lengths = []

        # For import
        mold_translation = [0.0, self.cfg_env.env.mold_lateral_offset, self.cfg_base.env.table_height]
        mold_orientation = [1.0, 0.0, 0.0, 0.0]

        # Add Molds to stage
        for i in range(0, self._num_envs):
            j = np.random.randint(0, len(self.cfg_env.env.desired_subassemblies))
            subassembly = self.cfg_env.env.desired_subassemblies[j]
            components = list(self.asset_info_mold[subassembly])

            # Attributes
            mold_height = self.asset_info_mold[subassembly][components[0]]['height']
            mold_width = self.asset_info_mold[subassembly][components[0]]['width']
            mold_length = self.asset_info_mold[subassembly][components[0]]['length']
            mold_heights.append(mold_height)
            mold_widths.append(mold_width)
            mold_lengths.append(mold_length)

            # Add to stage
            if add_to_stage:

                # Create XForm
                mold_file = self.asset_info_mold[subassembly][components[0]]['usd_path']
                add_reference_to_stage(mold_file, f'/World/envs/env_{i}' + '/plastic_mold')
                XFormPrim(
                    prim_path=f'/World/envs/env_{i}' + '/plastic_mold',
                    translation=mold_translation,
                    orientation=mold_orientation
                )

                # Edit material
                self._stage.GetPrimAtPath(
                    f'/World/envs/env_{i}' + '/plastic_mold' + '/collisions'
                ).SetInstanceable(False)  # This is required to be able to edit physics material
                physicsUtils.add_physics_material_to_prim(
                    self._stage,
                    self._stage.GetPrimAtPath(
                        f'/World/envs/env_{i}'
                        + '/plastic_mold'
                        + '/collisions/collisions'
                    ),
                    self.moldPhysicsMaterialPath
                )

                # Applies articulation settings from the task configuration yaml file
                self._sim_config.apply_articulation_settings(
                    'mold',
                    self._stage.GetPrimAtPath(f'/World/envs/env_{i}' + '/plastic_mold'),
                    self._sim_config.parse_actor_config('mold')
                )

        # For computing body COM pos
        self.mold_heights = torch.tensor(mold_heights, device=self._device).unsqueeze(-1)
        self.mold_widths = torch.tensor(mold_widths, device=self._device).unsqueeze(-1)
        self.mold_lengths = torch.tensor(mold_lengths, device=self._device).unsqueeze(-1)

    def refresh_env_tensors(self):
        '''Refresh Mold tensors.'''

        mold_pos, mold_quat = self.molds.get_world_poses(clone=False)
        self.mold_pos = self.ensure_tensor(mold_pos) - self._env_pos
        self.mold_quat = self.ensure_tensor(mold_quat)
        mold_velocities = self.ensure_tensor(self.molds.get_velocities(clone=False))
        self.mold_linvel = mold_velocities[:, 0:3]
        self.mold_angvel = mold_velocities[:, 3:6]
        self.mold_force = self.molds.get_net_contact_forces(clone=False)
