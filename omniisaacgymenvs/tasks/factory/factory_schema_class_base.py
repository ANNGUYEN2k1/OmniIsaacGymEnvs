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

"""Factory: abstract base class for base class.

Inherits ABC class. Inherited by base class. Defines template for base class.
"""


from abc import ABC, abstractmethod


class FactoryABCBase(ABC):
    '''[summary]'''

    @abstractmethod
    def __init__(self):
        """Initialize instance variables. Initialize VecTask superclass."""

    @abstractmethod
    def _get_base_yaml_params(self):
        """Initialize instance variables from YAML files."""

    @abstractmethod
    def import_robot_assets(self, add_to_stage):
        """Set robot and table asset options. Import assets."""

    @abstractmethod
    def refresh_base_tensors(self):
        """Refresh tensors."""
        # NOTE: Tensor refresh functions should be called once per step, before setters.

    @abstractmethod
    def parse_controller_spec(self, add_to_stage):
        """Parse controller specification into lower-level controller configuration."""

    @abstractmethod
    def generate_ctrl_signals(self):
        """Get Jacobian. Set robot DOF position targets or DOF torques."""

    @abstractmethod
    def enable_gravity(self, gravity_mag):
        """Enable gravity."""

    @abstractmethod
    def disable_gravity(self):
        """Disable gravity."""
