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
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Factory: abstract base class for task classes.

Inherits ABC class. Inherited by task classes. Defines template for task classes.
"""


from abc import ABC, abstractmethod


class FactoryABCTask(ABC):
    '''[summary]'''

    @abstractmethod
    def __init__(self):
        """Initialize instance variables. Initialize environment superclass."""

    @abstractmethod
    def _get_task_yaml_params(self):
        """Initialize instance variables from YAML files."""

    @abstractmethod
    def _acquire_task_tensors(self):
        """Acquire tensors."""

    @abstractmethod
    def _refresh_task_tensors(self):
        """Refresh tensors."""

    @abstractmethod
    def pre_physics_step(self):
        """Reset environments. Apply actions from policy as controller targets. Simulation step called after this method."""

    @abstractmethod
    def post_physics_step(self):
        """Step buffers. Refresh tensors. Compute observations and reward."""

    @abstractmethod
    def get_observations(self):
        """Compute observations."""

    @abstractmethod
    def calculate_metrics(self):
        """Detect successes and failures. Update reward and reset buffers."""

    @abstractmethod
    def _update_rew_buf(self, curr_successes):
        """Compute reward at current timestep."""

    @abstractmethod
    def _update_reset_buf(self, curr_successes, curr_failures):
        """Assign environments for reset if successful or failed."""

    @abstractmethod
    def reset_idx(self, env_ids, randomize_hand_pose):
        """Reset specified environments."""

    @abstractmethod
    def _reset_robot(self, env_ids):
        """Reset DOF states and DOF targets of robot."""

    @abstractmethod
    def _reset_object(self, env_ids):
        """Reset root state of object."""

    @abstractmethod
    def _reset_buffers(self, env_ids):
        """Reset buffers."""
