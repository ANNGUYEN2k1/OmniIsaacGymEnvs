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

'''Factory: class for nut-bolt env.

Inherits base class and abstract environment class. Inherited by nut-bolt task classes. Not directly executed.

Configuration defined in FactoryEnvNutBolt.yaml. Asset info defined in factory_asset_info_nut_bolt.yaml.
'''

from typing import List

import hydra
import numpy as np
import torch
from omni.isaac.core.objects import VisualCuboid  # noqa: F401
from omni.isaac.core.prims import RigidPrimView, XFormPrim
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.nucleus import get_assets_root_path
from omni.physx import get_physx_simulation_interface
from omni.physx.scripts import physicsUtils, utils

import omniisaacgymenvs.tasks.factory.factory_control as fc
from omniisaacgymenvs.robots.articulations.views.factory_franka_view import FactoryFrankaView
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.tasks.factory.factory_base import FactoryBase
from omniisaacgymenvs.tasks.factory.factory_schema_class_env import FactoryABCEnv
from omniisaacgymenvs.tasks.factory.factory_schema_config_env import FactorySchemaConfigEnv


class FactoryEnvNutBolt(FactoryBase, FactoryABCEnv):
    '''[summary]'''

    def __init__(self, name, sim_config, env, offset=None) -> None:
        '''Initialize base superclass. Initialize instance variables.'''

        super().__init__(name, sim_config, env, offset)

        # Get env params
        self._get_env_yaml_params()

        # Views
        self.nuts = None
        self.bolts = None
        self.frankas = None

        # Physics Simulation
        self.physx_interface = None

        # Contact links
        self.franka_contact_links = []

        # Nut Bolt properties
        self.nutboltPhysicsMaterialPath = '/World/Physics_Materials/NutBoltMaterial'
        self.nut_heights = torch.tensor([], dtype=torch.float32, device=self._device)
        self.nut_widths_max = torch.tensor([], dtype=torch.float32, device=self._device)
        self.bolt_widths = torch.tensor([], dtype=torch.float32, device=self._device)
        self.bolt_head_heights = torch.tensor([], dtype=torch.float32, device=self._device)
        self.bolt_shank_lengths = torch.tensor([], dtype=torch.float32, device=self._device)
        self.thread_pitches = torch.tensor([], dtype=torch.float32, device=self._device)

        # Nut state tensors
        self.nut_pos = torch.tensor([], dtype=torch.float32, device=self._device)
        self.nut_quat = torch.tensor([], dtype=torch.float32, device=self._device)
        self.nut_com_pos = torch.tensor([], dtype=torch.float32, device=self._device)
        self.nut_com_quat = torch.tensor([], dtype=torch.float32, device=self._device)
        self.nut_linvel = torch.tensor([], dtype=torch.float32, device=self._device)
        self.nut_angvel = torch.tensor([], dtype=torch.float32, device=self._device)
        self.nut_com_linvel = torch.tensor([], dtype=torch.float32, device=self._device)
        self.nut_com_angvel = torch.tensor([], dtype=torch.float32, device=self._device)
        self.nut_force = torch.tensor([], dtype=torch.float32, device=self._device)

        # Bolt state tensors
        self.bolt_pos = torch.tensor([], dtype=torch.float32, device=self._device)
        self.bolt_quat = torch.tensor([], dtype=torch.float32, device=self._device)
        self.bolt_force = torch.tensor([], dtype=torch.float32, device=self._device)

    def _get_env_yaml_params(self):
        '''Initialize instance variables from YAML files.'''

        # Set factory env config
        cs = hydra.core.config_store.ConfigStore.instance()  # type: ignore
        cs.store(name='factory_schema_config_env', node=FactorySchemaConfigEnv)

        # Env params
        config_path = 'task/FactoryEnvNutBolt.yaml'  # Relative to Hydra search path (cfg dir)
        self.cfg_env = hydra.compose(config_name=config_path)
        self.cfg_env = self.cfg_env['task']  # Strip superfluous nesting

        # Get nut bolt info
        asset_info_path = (
            '../tasks/factory/yaml/factory_asset_info_nut_bolt.yaml'
        )  # relative to Gym's Hydra search path (cfg dir)
        self.asset_info_nut_bolt = hydra.compose(config_name=asset_info_path)
        self.asset_info_nut_bolt = self.asset_info_nut_bolt['']['']['']['tasks']['factory']['yaml']  # Strip superfluous nesting

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

        # Remove existed views if needed
        if self.frankas is not None:
            if scene.object_exists('frankas_view'):
                scene.remove_object('frankas_view', registry_only=True)
            if scene.object_exists('nuts_view'):
                scene.remove_object('nuts_view', registry_only=True)
            if scene.object_exists('bolts_view'):
                scene.remove_object('bolts_view', registry_only=True)
            if scene.object_exists('hands_view'):
                scene.remove_object('hands_view', registry_only=True)
            for fingertip_view_name in self.frankas.fingertip_views:
                if scene.object_exists(fingertip_view_name):
                    scene.remove_object(fingertip_view_name, registry_only=True)
            if scene.object_exists('fingertip_centereds_view'):
                scene.remove_object('fingertip_centereds_view', registry_only=True)

        # Create Franka, Nut and Bolt's views
        self.frankas = FactoryFrankaView(prim_paths_expr='/World/envs/.*/franka', name='frankas_view')
        self.nuts = RigidPrimView(
            prim_paths_expr='/World/envs/.*/nut/factory_nut.*',
            name='nuts_view',
            track_contact_forces=True
        )
        self.bolts = RigidPrimView(
            prim_paths_expr='/World/envs/.*/bolt/factory_bolt.*',
            name='bolts_view',
            track_contact_forces=True
        )

        # Add views to scene
        scene.add(self.nuts)
        scene.add(self.bolts)
        scene.add(self.frankas)
        scene.add(self.frankas.hands)
        for fingertip_view in self.frankas.fingertip_views.values():
            scene.add(fingertip_view)
        scene.add(self.frankas.fingertip_centereds)

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

        # Import Franka from usd file
        self.import_robot_assets(add_to_stage=True)

        # Create nut and bolt's material
        self.create_nut_bolt_material()

        # # Create cuboid to check nut grasp and hand grasp poses
        # for i in range(self.cfg_task.rl.num_keypoints):
        #     VisualCuboid(
        #         prim_path=self.default_zero_env_path + f'/hand_{i}',
        #         name=f'hand_{i}',
        #         scale=[0.01, 0.01, 0.01],
        #         color=np.array([1.0, 0.0, 0.0])
        #     )
        #     VisualCuboid(
        #         prim_path=self.default_zero_env_path + f'/nut_{i}',
        #         name=f'nut_{i}',
        #         scale=[0.01, 0.01, 0.01],
        #         color=np.array([0.0, 0.0, 1.0])
        #     )

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
        self.franka_contact_links = [fingertip_view_name[:-6] for fingertip_view_name in self.frankas.fingertip_views]

    def initialize_views(self, scene) -> None:
        '''Initialize views for extension workflow.'''

        super().initialize_views(scene)

        # Ensure scene was fully
        self.import_robot_assets(add_to_stage=False)
        self._import_env_assets(add_to_stage=False)
        self.create_views(scene)

    def create_nut_bolt_material(self):
        '''Define nut and bolt material.'''

        utils.addRigidBodyMaterial(
            self._stage,
            self.nutboltPhysicsMaterialPath,
            density=self.cfg_env.env.nut_bolt_density,
            staticFriction=self.cfg_env.env.nut_bolt_friction,
            dynamicFriction=self.cfg_env.env.nut_bolt_friction,
            restitution=0.0
        )

    def _import_env_assets(self, add_to_stage=True):
        '''Set nut and bolt asset options. Import assets.'''

        # Nut-Bolt properties
        nut_heights = []
        nut_widths_max = []
        bolt_widths = []
        bolt_head_heights = []
        bolt_shank_lengths = []
        thread_pitches = []

        # For import
        assets_root_path = get_assets_root_path()
        nut_translation = [0.0, self.cfg_env.env.nut_lateral_offset, self.cfg_base.env.table_height]
        nut_orientation = [1.0, 0.0, 0.0, 0.0]
        bolt_translation = [0.0, 0.0, self.cfg_base.env.table_height]
        bolt_orientation = [1.0, 0.0, 0.0, 0.0]

        # Add Nuts and Bolts to stage
        for i in range(0, self._num_envs):
            j = np.random.randint(0, len(self.cfg_env.env.desired_subassemblies))
            subassembly = self.cfg_env.env.desired_subassemblies[j]
            components = list(self.asset_info_nut_bolt[subassembly])

            # Attributes
            nut_height = self.asset_info_nut_bolt[subassembly][components[0]]['height']
            nut_width_max = self.asset_info_nut_bolt[subassembly][components[0]]['width_max']
            bolt_width = self.asset_info_nut_bolt[subassembly][components[1]]['width']
            bolt_head_height = self.asset_info_nut_bolt[subassembly][components[1]]['head_height']
            bolt_shank_length = self.asset_info_nut_bolt[subassembly][components[1]]['shank_length']
            thread_pitch = self.asset_info_nut_bolt[subassembly]['thread_pitch']
            nut_heights.append(nut_height)
            nut_widths_max.append(nut_width_max)
            bolt_widths.append(bolt_width)
            bolt_head_heights.append(bolt_head_height)
            bolt_shank_lengths.append(bolt_shank_length)
            thread_pitches.append(thread_pitch)

            # Add to stage
            if add_to_stage:

                # Create XForm
                nut_file = assets_root_path + self.asset_info_nut_bolt[subassembly][components[0]]['usd_path']
                add_reference_to_stage(nut_file, f'/World/envs/env_{i}' + '/nut')
                XFormPrim(
                    prim_path=f'/World/envs/env_{i}' + '/nut',
                    translation=nut_translation,
                    orientation=nut_orientation,
                )
                bolt_file = assets_root_path + self.asset_info_nut_bolt[subassembly][components[1]]['usd_path']
                add_reference_to_stage(bolt_file, f'/World/envs/env_{i}' + '/bolt')
                XFormPrim(
                    prim_path=f'/World/envs/env_{i}' + '/bolt',
                    translation=bolt_translation,
                    orientation=bolt_orientation,
                )

                # Edit nut-bolt material
                self._stage.GetPrimAtPath(
                    f'/World/envs/env_{i}' + f'/nut/factory_{components[0]}/collisions'
                ).SetInstanceable(False)  # This is required to be able to edit physics material
                physicsUtils.add_physics_material_to_prim(
                    self._stage,
                    self._stage.GetPrimAtPath(f'/World/envs/env_{i}' + f'/nut/factory_{components[0]}/collisions/mesh_0'),
                    self.nutboltPhysicsMaterialPath
                )
                self._stage.GetPrimAtPath(
                    f'/World/envs/env_{i}' + f'/bolt/factory_{components[1]}/collisions'
                ).SetInstanceable(False)  # This is required to be able to edit physics material
                physicsUtils.add_physics_material_to_prim(
                    self._stage,
                    self._stage.GetPrimAtPath(f'/World/envs/env_{i}' + f'/bolt/factory_{components[1]}/collisions/mesh_0'),
                    self.nutboltPhysicsMaterialPath
                )

                # Applies articulation settings from the task configuration yaml file
                self._sim_config.apply_articulation_settings(
                    'nut',
                    self._stage.GetPrimAtPath(f'/World/envs/env_{i}' + '/nut'),
                    self._sim_config.parse_actor_config('nut')
                )
                self._sim_config.apply_articulation_settings(
                    'bolt',
                    self._stage.GetPrimAtPath(f'/World/envs/env_{i}' + '/bolt'),
                    self._sim_config.parse_actor_config('bolt')
                )

        # For computing body COM pos
        self.nut_heights = torch.tensor(nut_heights, device=self._device).unsqueeze(-1)
        self.bolt_head_heights = torch.tensor(bolt_head_heights, device=self._device).unsqueeze(-1)

        # For setting initial state
        self.nut_widths_max = torch.tensor(nut_widths_max, device=self._device).unsqueeze(-1)
        self.bolt_shank_lengths = torch.tensor(bolt_shank_lengths, device=self._device).unsqueeze(-1)

        # For defining success or failure
        self.bolt_widths = torch.tensor(bolt_widths, device=self._device).unsqueeze(-1)
        self.thread_pitches = torch.tensor(thread_pitches, device=self._device).unsqueeze(-1)

    def refresh_env_tensors(self):
        '''Refresh tensors.'''

        # Nut tensors
        nut_pos, nut_quat = self.nuts.get_world_poses(clone=False)
        self.nut_quat = self.ensure_tensor(nut_quat)
        self.nut_pos = self.ensure_tensor(nut_pos) - self._env_pos
        self.nut_com_pos = fc.translate_along_local_z(
            pos=self.nut_pos,
            quat=self.nut_quat,
            offset=self.bolt_head_heights + self.nut_heights * 0.5,
            device=self.device
        )
        self.nut_com_quat = self.nut_quat  # always equal
        nut_velocities = self.ensure_tensor(self.nuts.get_velocities(clone=False))
        self.nut_linvel = nut_velocities[:, 0:3]
        self.nut_angvel = nut_velocities[:, 3:6]
        self.nut_com_linvel = self.nut_linvel + torch.cross(self.nut_angvel, (self.nut_com_pos - self.nut_pos), dim=1)
        self.nut_com_angvel = self.nut_angvel  # always equal
        self.nut_force = self.nuts.get_net_contact_forces(clone=False)

        # Bolt tensors
        bolt_pos, bolt_quat = self.bolts.get_world_poses(clone=False)
        self.bolt_quat = self.ensure_tensor(bolt_quat)
        self.bolt_pos = self.ensure_tensor(bolt_pos) - self._env_pos
        self.bolt_force = self.bolts.get_net_contact_forces(clone=False)
