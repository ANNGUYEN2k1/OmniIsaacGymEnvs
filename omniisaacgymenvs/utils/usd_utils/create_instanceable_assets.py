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

'''Required python modules'''

from typing import Optional, Sequence
import numpy as np

import omni.client
from omni.isaac.core.materials.physics_material import PhysicsMaterial
from omni.isaac.core.materials.visual_material import VisualMaterial
from omni.isaac.core.objects import VisualCuboid
from omni.isaac.core.prims.geometry_prim import GeometryPrim
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.string import find_unique_string_name
import omni.usd as usd
from pxr import Sdf, UsdGeom


def update_reference(source_prim_path, source_reference_path, target_reference_path):
    '''[summary]'''

    stage = usd.get_context().get_stage()
    prims = [stage.GetPrimAtPath(source_prim_path)]
    while len(prims) > 0:
        prim = prims.pop(0)
        prim_spec = stage.GetRootLayer().GetPrimAtPath(prim.GetPath())
        reference_list = prim_spec.referenceList
        refs = reference_list.GetAddedOrExplicitItems()
        if len(refs) > 0:
            for ref in refs:
                if ref.assetPath == source_reference_path:
                    prim.GetReferences().RemoveReference(ref)
                    prim.GetReferences().AddReference(assetPath=target_reference_path, primPath=prim.GetPath())
        prims = prims + prim.GetChildren()


def create_parent_xforms(asset_usd_path, source_prim_path, save_as_path=None):
    """Adds a new UsdGeom.Xform prim for each Mesh/Geometry prim under source_prim_path.
    Moves material assignment to new parent prim if any exists on the Mesh/Geometry prim.

    Args:
        asset_usd_path (str): USD file path for asset
        source_prim_path (str): USD path of root prim
        save_as_path (str): USD file path for modified USD stage. Defaults to None, will save in same file.
    """

    # Copy mesh to mesh usd file
    usd.get_context().open_stage(asset_usd_path)
    stage = usd.get_context().get_stage()
    prims = [stage.GetPrimAtPath(source_prim_path)]
    edits = Sdf.BatchNamespaceEdit()
    while len(prims) > 0:
        prim = prims.pop(0)
        print(prim)
        if prim.GetTypeName() in ["Mesh", "Capsule", "Sphere", "Box"]:
            new_xform = UsdGeom.Xform.Define(stage, str(prim.GetPath()) + "_xform")
            print(prim, new_xform)
            edits.Add(Sdf.NamespaceEdit.Reparent(prim.GetPath(), new_xform.GetPath(), 0))
            continue
        children_prims = prim.GetChildren()
        prims = prims + children_prims
    stage.GetRootLayer().Apply(edits)

    # Save mesh usd file
    if save_as_path is None:
        usd.get_context().save_stage()
    else:
        usd.get_context().save_as_stage(save_as_path)


def convert_asset_instanceable(
    asset_usd_path: str,
    source_prim_path: str,
    save_as_path: Optional[str] = None,
    create_xforms: bool = True
):
    """Makes all mesh/geometry prims instanceable.
    Can optionally add UsdGeom.Xform prim as parent for all mesh/geometry prims.
    Makes a copy of the asset USD file, which will be used for referencing.
    Updates asset file to convert all parent prims of mesh/geometry prims to reference cloned USD file.

    Args:
        asset_usd_path (str): USD file path for asset
        source_prim_path (str): USD path of root prim
        save_as_path (str): USD file path for modified USD stage. Defaults to None, will save in same file.
        create_xforms (bool): Whether to add new UsdGeom.Xform prims to mesh/geometry prims.
    """

    # Create the usd mesh file
    if create_xforms:
        create_parent_xforms(asset_usd_path, source_prim_path, save_as_path)
        if save_as_path is not None:
            asset_usd_path = save_as_path
    instance_usd_path = ".".join(asset_usd_path.split(".")[:-1]) + "_meshes.usd"
    omni.client.copy(asset_usd_path, instance_usd_path)

    # Create instanceable prims
    usd.get_context().open_stage(asset_usd_path)
    stage = usd.get_context().get_stage()
    prims = [stage.GetPrimAtPath(source_prim_path)]
    while len(prims) > 0:
        prim = prims.pop(0)
        if prim:
            if prim.GetTypeName() in ["Mesh", "Capsule", "Sphere", "Box"]:
                parent_prim = prim.GetParent()
                if parent_prim and not parent_prim.IsInstance():
                    parent_prim.GetReferences().AddReference(
                        assetPath=instance_usd_path, primPath=str(parent_prim.GetPath())
                    )
                    parent_prim.SetInstanceable(True)
                    continue
            children_prims = prim.GetChildren()
            prims = prims + children_prims

    # Save usd file with instanceable prims
    if save_as_path is None:
        usd.get_context().save_stage()
    else:
        usd.get_context().save_as_stage(save_as_path)


class FixedCuboid(VisualCuboid):
    """High level wrapper to create/encapsulate a fixed cuboid

    .. note::

        Fixed cuboids (Cube shape) have collisions (Collider API) but no rigid body dynamics (Rigid Body API)

    Args:
        prim_path (str): prim path of the Prim to encapsulate or create
        name (str, optional): shortname to be used as a key by Scene class.
                                Note: needs to be unique if the object is added to the Scene.
                                Defaults to "fixed_cube".
        position (Optional[Sequence[float]], optional): position in the world frame of the prim. shape is (3, ).
                                                        Defaults to None, which means left unchanged.
        translation (Optional[Sequence[float]], optional): translation in the local frame of the prim
                                                        (with respect to its parent prim). shape is (3, ).
                                                        Defaults to None, which means left unchanged.
        orientation (Optional[Sequence[float]], optional): quaternion orientation in the world/ local frame of the prim
                                                        (depends if translation or position is specified).
                                                        quaternion is scalar-first (w, x, y, z). shape is (4, ).
                                                        Defaults to None, which means left unchanged.
        scale (Optional[Sequence[float]], optional): local scale to be applied to the prim's dimensions. shape is (3, ).
                                                Defaults to None, which means left unchanged.
        visible (bool, optional): set to false for an invisible prim in the stage while rendering. Defaults to True.
        color (Optional[np.ndarray], optional): color of the visual shape. Defaults to None, which means 50% gray
        size (Optional[float], optional): length of each cube edge. Defaults to None.
        visual_material (Optional[VisualMaterial], optional): visual material to be applied to the held prim.
                                Defaults to None. If not specified, a default visual material will be added.
        physics_material (Optional[PhysicsMaterial], optional): physics material to be applied to the held prim.
                                Defaults to None. If not specified, a default physics material will be added.

    Example:

    .. code-block:: python

        >>> from omni.isaac.core.objects import FixedCuboid
        >>> import numpy as np
        >>>
        >>> # create a red fixed cube at the given path
        >>> prim = FixedCuboid(prim_path="/World/Xform/Cube", color=np.array([1.0, 0.0, 0.0]))
        >>> prim
        <omni.isaac.core.objects.cuboid.FixedCuboid object at 0x7f7b4d91da80>
    """

    def __init__(
        self,
        prim_path: str,
        name: str = "fixed_cube",
        position: Optional[Sequence[float]] = None,
        translation: Optional[Sequence[float]] = None,
        orientation: Optional[Sequence[float]] = None,
        scale: Optional[Sequence[float]] = None,
        visible: Optional[bool] = None,
        color: Optional[np.ndarray] = None,
        size: Optional[float] = None,
        rest_offset: float = 0.0,
        contact_offset: float = 0.1,
        torsional_patch_radius: float = 1.0,
        min_torsional_patch_radius: float = 0.8,
        visual_material: Optional[VisualMaterial] = None,
        physics_material: Optional[PhysicsMaterial] = None,
    ) -> None:
        '''[summary]'''

        # Set physics params
        set_offsets = False
        if not is_prim_path_valid(prim_path):
            # Set default values if no physics material given
            if physics_material is None:
                static_friction = 0.2
                dynamic_friction = 1.0
                restitution = 0.0
                physics_material_path = find_unique_string_name(
                    initial_name="/World/Physics_Materials/physics_material",
                    is_unique_fn=lambda x: not is_prim_path_valid(x),
                )
                physics_material = PhysicsMaterial(
                    prim_path=physics_material_path,
                    dynamic_friction=dynamic_friction,
                    static_friction=static_friction,
                    restitution=restitution,
                )
            set_offsets = True

        # Create visuals
        VisualCuboid.__init__(
            self,
            prim_path=prim_path,
            name=name,
            position=position,
            translation=translation,
            orientation=orientation,
            scale=scale,
            visible=visible,
            color=color,
            size=size,
            visual_material=visual_material,
        )

        # Create collisions
        GeometryPrim.set_collision_enabled(self, True)
        if physics_material is not None:
            FixedCuboid.apply_physics_material(self, physics_material)
        if set_offsets:
            FixedCuboid.set_rest_offset(self, rest_offset)
            FixedCuboid.set_contact_offset(self, contact_offset)
            FixedCuboid.set_torsional_patch_radius(self, torsional_patch_radius)
            FixedCuboid.set_min_torsional_patch_radius(self, min_torsional_patch_radius)
        return
