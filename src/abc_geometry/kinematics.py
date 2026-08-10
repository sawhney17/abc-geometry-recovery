"""Forward kinematics for the bimanual I2RT YAM used by ABC-130K.

ABC records six arm-joint angles per side in radians.  This module evaluates
those angles against a minimal copy of the official Amazon FAR / I2RT MuJoCo
chain and returns the pose of its ``grasp_site``.

Two output frames are supported:

``arm_local``
    The base frame of the selected YAM arm.  Left and right arms therefore use
    the same kinematic origin.
``shared_bimanual``
    One observed ABC station cohort, with the left arm as origin and the right
    arm translated by ``[0, -0.61, 0]`` metres.  This separation is empirical,
    not declared by the single-arm YAM MJCF.  A second public cohort uses
    ``[0, -0.80, 0]``; callers must not treat this frame as universal.

Returned quaternions use MuJoCo's scalar-first ``[w, x, y, z]`` convention.
The rotation maps vectors in the grasp-site frame into the requested base
frame, so :attr:`Pose.homogeneous_matrix` is ``T_base_grasp``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from threading import Lock
from typing import Literal

import mujoco
import numpy as np
from numpy.typing import NDArray

Arm = Literal["left", "right"]
Frame = Literal["arm_local", "shared_bimanual"]

JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 7))
RIGHT_ARM_SHARED_TRANSLATION = np.array([0.0, -0.61, 0.0], dtype=np.float64)
RIGHT_ARM_SHARED_TRANSLATION.setflags(write=False)
MODEL_SOURCE_REVISION = "amazon-far/abc@6bc6586721cf0c409ccee80f675a28de9b9b2f5e"
SHARED_FRAME_TRANSFORM_VERSION = "abc-public-right-base-minus-0.61-v1"


@lru_cache(maxsize=1)
def model_sha256() -> str:
    """Return the digest of the exact kinematic model used for derivation."""

    model_resource = resources.files("abc_geometry").joinpath("assets", "yam_kinematics.xml")
    return hashlib.sha256(model_resource.read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class Pose:
    """A grasp-site pose expressed in a YAM base frame.

    Attributes:
        position: XYZ translation in metres, shape ``(3,)``.
        quaternion_wxyz: Unit quaternion in scalar-first ``[w, x, y, z]``
            order, shape ``(4,)``.
    """

    position: NDArray[np.float64]
    quaternion_wxyz: NDArray[np.float64]

    def __post_init__(self) -> None:
        position = _validated_vector(self.position, length=3, name="position")
        quaternion = _validated_vector(
            self.quaternion_wxyz,
            length=4,
            name="quaternion_wxyz",
        )
        norm = float(np.linalg.norm(quaternion))
        if norm <= np.finfo(np.float64).eps:
            raise ValueError("quaternion_wxyz must have non-zero norm")

        position.setflags(write=False)
        quaternion = quaternion / norm
        # q and -q encode the same rotation.  A canonical sign makes serialized
        # recovered records stable across MuJoCo versions.
        if quaternion[0] < 0:
            quaternion = -quaternion
        quaternion.setflags(write=False)

        object.__setattr__(self, "position", position)
        object.__setattr__(self, "quaternion_wxyz", quaternion)

    def as_vector(self) -> NDArray[np.float64]:
        """Return ``[x, y, z, qw, qx, qy, qz]`` as a new float64 array."""

        return np.concatenate((self.position, self.quaternion_wxyz))

    @property
    def homogeneous_matrix(self) -> NDArray[np.float64]:
        """Return the 4x4 transform ``T_base_grasp``."""

        rotation_flat = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rotation_flat, self.quaternion_wxyz)
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = rotation_flat.reshape(3, 3)
        matrix[:3, 3] = self.position
        return matrix

    @property
    def matrix(self) -> NDArray[np.float64]:
        """Alias for :attr:`homogeneous_matrix` used by validation adapters."""

        return self.homogeneous_matrix


class YAMKinematics:
    """Cached MuJoCo model/data pair for repeated YAM FK evaluation.

    Construct this class once for a stream or batch.  The compiled
    :class:`mujoco.MjModel` and mutable :class:`mujoco.MjData` are retained for
    every call; a lock makes calls on one instance safe across worker threads.
    """

    def __init__(self) -> None:
        model_resource = resources.files("abc_geometry").joinpath("assets", "yam_kinematics.xml")
        model_xml = model_resource.read_text(encoding="utf-8")
        self._model = mujoco.MjModel.from_xml_string(model_xml)
        self._data = mujoco.MjData(self._model)
        self._lock = Lock()

        joint_ids = np.array(
            [
                mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in JOINT_NAMES
            ],
            dtype=np.int32,
        )
        if np.any(joint_ids < 0):  # Defensive: the bundled model is package data.
            raise RuntimeError("bundled YAM model is missing one or more arm joints")
        self._qpos_addresses = self._model.jnt_qposadr[joint_ids].copy()

        self._grasp_site_id = mujoco.mj_name2id(
            self._model,
            mujoco.mjtObj.mjOBJ_SITE,
            "grasp_site",
        )
        if self._grasp_site_id < 0:
            raise RuntimeError("bundled YAM model is missing grasp_site")

    @property
    def model_id(self) -> str:
        """Stable identifier combining the model source revision and content digest."""

        return f"{MODEL_SOURCE_REVISION}#sha256:{model_sha256()}"

    @property
    def model_digest(self) -> str:
        return model_sha256()

    @property
    def model_source_revision(self) -> str:
        return MODEL_SOURCE_REVISION

    @property
    def shared_frame_transform_version(self) -> str:
        return SHARED_FRAME_TRANSFORM_VERSION

    def grasp_pose(
        self,
        joints: Sequence[float] | NDArray[np.floating],
        *,
        arm: Arm = "left",
        frame: Frame = "arm_local",
    ) -> Pose:
        """Compute the YAM grasp pose from six joint angles in radians.

        Args:
            joints: Joint 1 through joint 6 in the source-record order.
            arm: Which physical arm produced the values.  This affects only the
                shared-frame base translation; both arms use the same chain.
            frame: ``"arm_local"`` or ``"shared_bimanual"``.

        Raises:
            ValueError: If an enum value, shape, or numeric value is invalid.
        """

        if frame not in ("arm_local", "shared_bimanual"):
            raise ValueError(f"frame must be 'arm_local' or 'shared_bimanual', got {frame!r}")

        local, shared = self.grasp_pose_pair(joints, arm=arm)
        return local if frame == "arm_local" else shared

    def grasp_pose_pair(
        self,
        joints: Sequence[float] | NDArray[np.floating],
        *,
        arm: Arm = "left",
    ) -> tuple[Pose, Pose]:
        """Return ``(arm_local, shared_bimanual)`` with one FK evaluation."""

        if arm not in ("left", "right"):
            raise ValueError(f"arm must be 'left' or 'right', got {arm!r}")
        joint_values = _validated_vector(joints, length=6, name="joints")

        with self._lock:
            self._data.qpos[self._qpos_addresses] = joint_values
            mujoco.mj_forward(self._model, self._data)
            position = self._data.site_xpos[self._grasp_site_id].copy()
            rotation = self._data.site_xmat[self._grasp_site_id].copy()

        quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quaternion, rotation)
        local = Pose(position=position, quaternion_wxyz=quaternion)
        if arm == "right":
            shared = Pose(
                position=position + RIGHT_ARM_SHARED_TRANSLATION,
                quaternion_wxyz=quaternion,
            )
        else:
            shared = local
        return local, shared


@lru_cache(maxsize=1)
def _default_kinematics() -> YAMKinematics:
    """Return the process-wide model/data pair used by the convenience API."""

    return YAMKinematics()


def forward_kinematics(
    joints: Sequence[float] | NDArray[np.floating],
    *,
    arm: Arm = "left",
    frame: Frame = "arm_local",
) -> Pose:
    """Compute a YAM grasp pose using a process-wide cached model/data pair."""

    return _default_kinematics().grasp_pose(joints, arm=arm, frame=frame)


def _validated_vector(
    values: Sequence[float] | NDArray[np.floating],
    *,
    length: int,
    name: str,
) -> NDArray[np.float64]:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain {length} real numbers") from error
    if array.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


__all__ = [
    "Arm",
    "Frame",
    "JOINT_NAMES",
    "MODEL_SOURCE_REVISION",
    "Pose",
    "RIGHT_ARM_SHARED_TRANSLATION",
    "SHARED_FRAME_TRANSFORM_VERSION",
    "YAMKinematics",
    "forward_kinematics",
    "model_sha256",
]
