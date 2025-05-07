#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from pygltflib import GLTF2, Node
import trimesh
#from trimesh.scene import Scene
import pyrender
import os

# Helper functions for quaternion and matrix operations

def quaternion_to_matrix(q):
    """Convert quaternion [x, y, z, w] to 4x4 rotation matrix."""
    x, y, z, w = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    return np.array([
        [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy),   0],
        [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx),   0],
        [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy), 0],
        [0,           0,           0,           1]
    ], dtype=np.float32)

def compose_matrix(translation, rotation, scale):
    """Compose translation, rotation, scale into a 4x4 matrix."""
    T = np.eye(4, dtype=np.float32)
    if translation:
        T[:3, 3] = translation

    R = quaternion_to_matrix(rotation or [0, 0, 0, 1])

    S = np.eye(4, dtype=np.float32)
    if scale:
        S[0,0], S[1,1], S[2,2] = scale

    return T @ R @ S

def world_to_local_rotation(target_world_rot, parent_world_rot):
    """
    Convert a world-space quaternion into a local-space one given parent world-space rotation.

    Both quaternions should be in [x, y, z, w] format.
    """
    r_target = R.from_quat(target_world_rot)
    r_parent = R.from_quat(parent_world_rot)

    # Local rotation = inverse(parent_world) * target_world
    r_local = r_parent.inv() * r_target
    return r_local.as_quat()

def direction_vector(p1, p2):
    vec = p2 - p1
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-6 else np.zeros(3)

def quaternion_from_vectors(source, target):
    """
    Compute the quaternion that rotates `source` to align with `target`.
    """
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    rot = R.align_vectors([target], [source])[0]
    return rot.as_quat()


# In[ ]:


# SkeletonHelper is the simple version that focuses on setting joint transforms
class SkeletonHelper:
    def __init__(self, gltf: GLTF2, skin_index=0):
        self.gltf = gltf
        self.skin = gltf.skins[skin_index]
        self.joint_nodes = self.skin.joints #TODO refator to remove duplication
        self.node_to_parent = self.build_parent_map()
        self.joint_indices = self.skin.joints

        # Will store local and world transforms
        self.local_matrices = {}
        self.world_matrices = {}
        self.inverse_bind_matrices = self.load_inverse_bind_matrices()
        self.final_skinning_matrices = {}
        # Build name -> index mapping
        self.joint_name_to_idx = {}
        for joint_idx in self.joint_indices:
            node = gltf.nodes[joint_idx]
            name = node.name or f"Joint_{joint_idx}"
            self.joint_name_to_idx[name] = joint_idx


    def build_parent_map(self):
        """Build a map from node index to its parent node index."""
        node_to_parent = {}
        for idx, node in enumerate(self.gltf.nodes):
            if node.children:
                for child_idx in node.children:
                    node_to_parent[child_idx] = idx
        return node_to_parent

    def load_inverse_bind_matrices(self):
        """Load inverse bind matrices from skin."""
        accessor_idx = self.skin.inverseBindMatrices
        if accessor_idx is None:
            raise ValueError("Skin has no inverseBindMatrices accessor!")

        accessor = self.gltf.accessors[accessor_idx]
        buffer_view = self.gltf.bufferViews[accessor.bufferView]
        buffer = self.gltf.buffers[buffer_view.buffer]

        # Extract raw binary
        raw = self.get_buffer_data()

        # Parse matrices
        stride = buffer_view.byteStride or 64  # 4x4 floats = 64 bytes
        inverse_bind_matrices = []
        for i in range(accessor.count):
            offset = buffer_view.byteOffset + accessor.byteOffset + i * stride
            mat = np.frombuffer(raw, dtype=np.float32, count=16, offset=offset)
            mat = mat.reshape((4,4)).T  # glTF is column-major
            inverse_bind_matrices.append(mat)

        return inverse_bind_matrices

    def get_buffer_data(self, buffer_index=0):
        buffer = self.gltf.buffers[buffer_index]
        uri = buffer.uri

        # Embedded in .glb
        if uri is None:
            return self.gltf.binary_blob()  # Works for .glb files

        # Embedded base64 in .gltf
        if uri.startswith('data:'):
            import base64
            header, encoded = uri.split(',', 1)
            return base64.b64decode(encoded)

        # External .bin file
        else:
            #buffer_path = os.path.join(os.path.dirname(self.gltf), uri)
            buffer_path = "data/ride1/rider_mesh_w_rig.bin"
            with open(buffer_path, 'rb') as f:
                return f.read()

    def set_joint_rotation(self, joint_idx, new_quat):
        """
        Set the rotation of a joint by index.

        Args:
            joint_idx (int): The joint's index.
            new_quat (list or np.array): The new quaternion [x, y, z, w].
        """
        if joint_idx not in self.joint_nodes:
            raise ValueError(f"Joint index {joint_idx} not found!")

        node = self.gltf.nodes[joint_idx]

        # Set rotation (overwrite)
        node.rotation = list(new_quat)
        print(f"Set joint {joint_idx} rotation to {new_quat}")

    def set_joint_translation(self, joint_idx, new_translation):
        """
        Set the translation of a joint by index.

        Args:
            joint_idx (int): The joint's index.
            new_translation (list or np.array): The new translation [x, y, z].
        """
        if joint_idx not in self.joint_nodes:
            raise ValueError(f"Joint index {joint_idx} not found!")

        node = self.gltf.nodes[joint_idx]

        # Set translation (overwrite)
        node.translation = list(new_translation)
        print(f"Set joint {joint_idx} translation to {new_translation}")

    def set_joint_world_rotation(self, joint_idx, new_quat):
        """
        Set the rotation of a joint by index.

        Args:
            joint_idx (int): The joint's index.
            new_quat (list or np.array): The new quaternion [x, y, z, w].
        """
        if joint_idx not in self.joint_nodes:
            raise ValueError(f"Joint index {joint_idx} not found!")

        node = self.gltf.nodes[joint_idx]

        # Set rotation (overwrite)
        node.rotation = list(world_to_local_rotation(new_quat, node.rotation))
        print(f"Set joint {joint_idx} rotation to {new_quat}")

    def set_joint_world_translation(self, joint_idx, new_translation):
        """
        Set the translation of a joint by index.

        Args:
            joint_idx (int): The joint's index.
            new_translation (list or np.array): The new translation [x, y, z].
        """
        if joint_idx not in self.joint_nodes:
            raise ValueError(f"Joint index {joint_idx} not found!")

        node = self.gltf.nodes[joint_idx]

        # Set translation (overwrite)
        node.translation = list(new_translation)
        print(f"Set joint {joint_idx} translation to {new_translation}")    

    def set_joint_rotation_by_name(self, joint_name, new_quat, as_world=False):
        """
        Set the rotation of a joint by name.

        Args:
            joint_name (str): The joint's name.
            new_quat (list or np.array): The new quaternion [x, y, z, w].
        """
        if joint_name not in self.joint_name_to_idx:
            raise ValueError(f"Joint name {joint_name} not found!")

        idx = self.joint_name_to_idx[joint_name]
        if as_world:
            self.set_joint_world_rotation(idx, new_quat)
        else:
            self.set_joint_rotation(idx, new_quat)

    def set_joint_translation_by_name(self, joint_name, new_translation):
        """
        Set the translation of a joint by name.

        Args:
            joint_name (str): The joint's name.
            new_translation (list or np.array): The new translation [x, y, z].
        """
        if joint_name not in self.joint_name_to_idx:
            raise ValueError(f"Joint name {joint_name} not found!")

        idx = self.joint_name_to_idx[joint_name]
        self.set_joint_translation(idx, new_translation)

    def apply_pose(self, pose_points, point_names, bones):
        """
        Apply a full pose update.
        """
        #point_names = GetKeypointNames()
        #bones = GetBones()
        #self #skel = SkeletonHelperGLTF.SkeletonHelper(mesh)
        for bone_name, (kp_start, kp_end) in bones.items(): #TODO: bones should be iterated in tree order
            source_dir = np.array([0, -1, 0])  # assume bone initially points down local Y axis
            target_dir = direction_vector(pose_points[point_names[kp_start]], pose_points[point_names[kp_end]])
            quat = quaternion_from_vectors(source_dir, target_dir)  # x, y, z, w
            self.set_joint_rotation_by_name(bone_name, quat, as_world=True)
        self.update()

    def compute_local_matrices(self):
        """Compute local transform matrices for each joint."""
        for node_idx in self.joint_nodes:
            node = self.gltf.nodes[node_idx]
            translation = node.translation if node.translation else [0, 0, 0]
            rotation = node.rotation if node.rotation else [0, 0, 0, 1]
            scale = node.scale if node.scale else [1, 1, 1]

            self.local_matrices[node_idx] = compose_matrix(translation, rotation, scale)

    def compute_world_matrices(self):
        """Compute world transform matrices hierarchically."""
        def get_world_matrix(node_idx):
            if node_idx in self.world_matrices:
                return self.world_matrices[node_idx]
            try:
                local = self.local_matrices[node_idx]
            except KeyError:
                # If local matrix not computed, compute it
                node = self.gltf.nodes[node_idx]
                translation = node.translation if node.translation else [0, 0, 0]
                rotation = node.rotation if node.rotation else [0, 0, 0, 1]
                scale = node.scale if node.scale else [1, 1, 1]
                self.local_matrices[node_idx] = compose_matrix(translation, rotation, scale)
                local = self.local_matrices[node_idx]
            parent_idx = self.node_to_parent.get(node_idx)
            if parent_idx is not None:
                parent_world = get_world_matrix(parent_idx)
                world = parent_world @ local
            else:
                world = local
            self.world_matrices[node_idx] = world
            return world

        for node_idx in self.joint_nodes:
            get_world_matrix(node_idx)

    def compute_final_skinning_matrices(self):
        """Compute world * inverseBind matrices for skinning."""
        self.final_skinning_matrices = {}
        for i, node_idx in enumerate(self.joint_nodes):
            world = self.world_matrices[node_idx]
            inv_bind = self.inverse_bind_matrices[i]
            self.final_skinning_matrices[node_idx] = world @ inv_bind

    def update(self):
        """Recompute all local and world matrices."""
        self.compute_local_matrices()
        self.compute_world_matrices()
        self.compute_final_skinning_matrices()

    def visualize(self):
        """Visualize the skeleton using pyrender."""
        self.update()  # Ensure transforms are up to date
        scene = pyrender.Scene()

    def save(self, filename="output.glb", path="/mnt/c/Users/janis/Documents/BlenderLiveView/"):
        """Save the modified GLTF to a file."""
        full_p = path + filename
        self.gltf.save_binary(full_p)
        print(f"Saved modified GLTF to {full_p}")


# In[ ]:


def forward_kinematics(joint_positions, parents, local_offsets):
    """
    Compute world-space joint positions from local offsets and parent indices.

    Args:
        joint_positions (dict): Output world-space joint positions.
        parents (dict): Mapping from joint index to parent index.
        local_offsets (dict): Local offset vectors for each joint.

    Returns:
        dict: Updated joint_positions with world-space coordinates.
    """
    joint_positions[0] = local_offsets[0]  # root
    for idx in sorted(local_offsets.keys()):
        if idx == 0:
            continue
        parent_idx = parents[idx]
        joint_positions[idx] = joint_positions[parent_idx] + local_offsets[idx]
    return joint_positions


# In[ ]:


import numpy as np

def fabrik(joint_positions, target, tolerance=1e-3, max_iter=10):
    """
    FABRIK inverse kinematics solver for a single bone chain.

    Args:
        joint_positions (list): List of joint positions as np.array([x, y, z]).
        target (np.array): Target position for the end effector.
        tolerance (float): Distance threshold for stopping.
        max_iter (int): Maximum number of iterations.

    Returns:
        list: New joint positions.
    """
    positions = joint_positions.copy()
    bone_lengths = [np.linalg.norm(positions[i+1] - positions[i]) for i in range(len(positions) - 1)]
    total_length = sum(bone_lengths)

    if np.linalg.norm(positions[0] - target) > total_length:
        # Target is unreachable: stretch toward it
        for i in range(len(positions) - 1):
            r = np.linalg.norm(target - positions[i])
            lambda_ = bone_lengths[i] / r
            positions[i+1] = (1 - lambda_) * positions[i] + lambda_ * target
        return positions

    for _ in range(max_iter):
        # Step 1: Forward reaching
        positions[-1] = target
        for i in reversed(range(len(positions) - 1)):
            r = np.linalg.norm(positions[i+1] - positions[i])
            lambda_ = bone_lengths[i] / r
            positions[i] = (1 - lambda_) * positions[i+1] + lambda_ * positions[i]

        # Step 2: Backward reaching
        positions[0] = joint_positions[0]
        for i in range(len(positions) - 1):
            r = np.linalg.norm(positions[i+1] - positions[i])
            lambda_ = bone_lengths[i] / r
            positions[i+1] = (1 - lambda_) * positions[i] + lambda_ * positions[i+1]

        # Convergence check
        if np.linalg.norm(positions[-1] - target) < tolerance:
            break

    return positions


# In[ ]:


def apply_pose_to_gltf(gltf, pose_dict):
    """
    Apply a pose to a GLTF model.

    Args:
        gltf (GLTF2): The GLTF model.
        pose_dict (dict): Dictionary of joint names and their transformations.
    """
    skeleton_helper = SkeletonHelper(gltf)
    skeleton_helper.apply_pose(pose_dict)
    skeleton_helper.update()
    skeleton_helper.save("output.glb")


# In[3]:


def get_accessor_data(gltf, accessor_index):
    """
    Extracts the data for a given accessor from a glTF2 object into a NumPy array.

    Parameters:
        gltf (GLTF2): Loaded GLTF2 object (from pygltflib).
        accessor_index (int): Index of the accessor to extract.
    Returns:
        numpy.ndarray: Array of shape (count, components) with the accessor data.
    """
    import numpy as np

    # Retrieve accessor and bufferView objects
    try:
        accessor = gltf.accessors[accessor_index]
        buffer_view = gltf.bufferViews[accessor.bufferView]
        buffer = gltf.buffers[buffer_view.buffer]
    except IndexError:
        raise ValueError(f"Invalid accessor index: {accessor_index}")

    # Compute absolute byte offset within the buffer
    offset = (buffer_view.byteOffset or 0) + (accessor.byteOffset or 0)
    count = accessor.count

    # Determine number of components (SCALAR=1, VEC3=3, etc.)
    type_to_num = {
        "SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4,
        "MAT2": 4, "MAT3": 9, "MAT4": 16
    }
    num_components = type_to_num[accessor.type]

    # Map glTF componentType to numpy dtype
    comp_dtype_map = {
        5120: np.int8,
        5121: np.uint8,
        5122: np.int16,
        5123: np.uint16,
        5125: np.uint32,
        5126: np.float32
    }
    dtype = comp_dtype_map.get(accessor.componentType)
    if dtype is None:
        raise ValueError(f"Unsupported componentType: {accessor.componentType}")

    # Load the raw buffer bytes (handle embedded, external, or GLB)
    if buffer.uri:
        if buffer.uri.startswith("data:"):
            raw_bytes = gltf.decode_data_uri(buffer.uri)
        else:
            with open(buffer.uri, "rb") as f:
                raw_bytes = f.read()
    else:
        raw_bytes = gltf.binary_blob()

    # Determine byte stride (default = tightly packed)
    dtype_size = np.dtype(dtype).itemsize
    byte_stride = buffer_view.byteStride or (num_components * dtype_size)
    byte_length = count * byte_stride

    # Extract the relevant byte range for this accessor
    segment = raw_bytes[offset : offset + byte_length]

    # Use numpy to interpret as the dtype and reshape
    arr = np.frombuffer(segment, dtype=dtype)
    arr = arr.reshape((count, byte_stride // dtype_size))

    # Return only the first `num_components` columns (ignore padding)
    return arr[:, :num_components]

def get_joint_matrices(skeleton: SkeletonHelper):
    """Returns a list of final joint matrices: world * inverseBindMatrix."""
    ibms = skeleton.skin.inverseBindMatrices
    accessor = skeleton.gltf.accessors[ibms]
    ibm_data = get_accessor_data(skeleton.gltf)[ibms]  # shape (J, 4, 4)

    joint_matrices = []
    for joint_idx, ibm in zip(skeleton.joint_indices, ibm_data):
        joint_world = skeleton.world_matrices[joint_idx]
        joint_matrix = joint_world @ ibm
        joint_matrices.append(joint_matrix)
    return joint_matrices


# In[4]:


def apply_skinning(skeleton: SkeletonHelper, joint_name) -> trimesh.Trimesh:
    gltf = skeleton.gltf
    skin = skeleton.skin
    mesh_node_idx = [idx for idx, node in enumerate(gltf.nodes) if node.mesh is not None][0]
    mesh_node = gltf.nodes[mesh_node_idx]
    mesh = gltf.meshes[mesh_node.mesh]

    primitive = gltf.meshes[skeleton.joint_name_to_idx[joint_name]].primitives[0]
    pos_accessor_idx = primitive.attributes.POSITION
    accessor_data = get_accessor_data(gltf, pos_accessor_idx)

    # Get base vertex positions
    position_accessor_idx = mesh.primitives[0].attributes.POSITION
    positions = accessor_data[position_accessor_idx]  # (N, 3)

    # Get joint indices and weights
    joints_accessor_idx = mesh.primitives[0].attributes.JOINTS_0
    weights_accessor_idx = mesh.primitives[0].attributes.WEIGHTS_0

    joints = accessor_data[joints_accessor_idx].astype(np.int32)  # (N, 4)
    weights = accessor_data[weights_accessor_idx]  # (N, 4)

    # Calculate final joint matrices: M_skin = world * inverseBindMatrix
    joint_matrices = get_joint_matrices(skeleton)  # list of 4x4 matrices, len = # joints

    # Perform skinning: v' = sum(w_i * M_i * v)
    skinned_vertices = []
    for i in range(len(positions)):
        pos_h = np.append(positions[i], 1.0)  # homogeneous coordinate
        skinned_pos = np.zeros(4)
        for j in range(4):
            joint_idx = joints[i][j]
            weight = weights[i][j]
            M = joint_matrices[joint_idx]
            skinned_pos += weight * (M @ pos_h)
        skinned_vertices.append(skinned_pos[:3])  # drop homogeneous w

    skinned_vertices = np.array(skinned_vertices)

    # Reuse indices from glTF
    indices_accessor_idx = mesh.primitives[0].indices
    faces = accessor_data[indices_accessor_idx].reshape(-1, 3)

    return trimesh.Trimesh(vertices=skinned_vertices, faces=faces, process=False)


# In[5]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def test_skeleton_helper():
    gltf = GLTF2().load("data/raw/human_base_mesh.glb")
    skeleton = SkeletonHelper(gltf)
    print("Joint names and indices:")
    skeleton.joint_name_to_idx = dict(sorted(skeleton.joint_name_to_idx.items()))
    for name, idx in skeleton.joint_name_to_idx.items():
        print(f"{name}: {idx}")
    #skeleton.set_joint_rotation_by_name("hips", [0, 0, 0, 1])
    skeleton.update()
    n_frames = 100
    elbow_joint_name = "DEF-foot.L"
    for frame in range(n_frames + 1):
        # Calculate the new rotation for the elbow joint
        rotation_angle = frame * (360 / n_frames)  # Rotating 360 degrees in n_frames steps
        r = R.from_euler('z', rotation_angle, degrees=True)
        quat = r.as_quat()

        # Apply the rotation to the elbow joint
        skeleton.set_joint_rotation_by_name(elbow_joint_name, quat)

        # Visualize the model at each frame
        print(f"Frame {frame + 1} - Elbow rotation: {rotation_angle}°")
        skeleton.save(filename=f"SkeletonHelperGLTF_{frame}.glb")


# In[6]:


test_skeleton_helper()


# In[ ]:


def test_skeleton_helper2():
    pass


# In[ ]:


def test_skeleton_helper3():
    pass

