import numpy as np
from visual_mpc.policy.policy import Policy
from semiparametrictransfer.utils.general_utils import AttrDict

import visual_mpc.sim.pybullet as bullet
from visual_mpc.sim.pybullet.pybullet_meshes.shapenet_object_lists import GRASP_OFFSETS

class PickPlace(Policy):
    """
    Behavioral Cloning Policy
    """
    def __init__(self, ag_params, policyparams):
        super().__init__()

        self._hp = self._default_hparams()
        self._override_defaults(policyparams)
        self.env = ag_params.env_handle

    def _default_hparams(self):
        dict = AttrDict(
            type=None,
            pick_height_thresh=-0.31,
            xyz_action_scale=1.0,
            pick_point_noise=0.00,
            drop_point_noise=0.00,
            grasp_distance_thresh=0.02,
            robot_grasp_offset=None

        )
        default_dict = super()._default_hparams()
        default_dict.update(dict)
        return default_dict

    def reset(self):
        # self.dist_thresh = 0.06 + np.random.normal(scale=0.01)
        self.object_to_target = self.env.target_object
        self.drop_point = self.env.container_position
        self.drop_point[2] = -0.2
        self.place_attempted = False

        if self._hp.pick_point_noise != 0:
            self.pick_point_randoffset = np.random.normal(0, self._hp.pick_point_noise, 2)
        self.get_pickpoint()

    def get_pickpoint(self):
        self.pick_point = bullet.get_object_position(
            self.env.objects[self.object_to_target])[0]
        if self.object_to_target in GRASP_OFFSETS.keys():
            self.pick_point += np.asarray(GRASP_OFFSETS[self.object_to_target])

        if self._hp.pick_point_noise != 0:
            self.pick_point[:2] += self.pick_point_randoffset

        self.pick_point[2] = -0.32
        if self._hp.robot_grasp_offset is not None:
            self.pick_point += np.asarray(self._hp.robot_grasp_offset)

    def act(self, t):
        if t == 0:
            self.reset()


        ee_pos, _ = bullet.get_link_state(
            self.env.robot_id, self.env.end_effector_index)
        object_pos, _ = bullet.get_object_position(
            self.env.objects[self.object_to_target])
        object_lifted = object_pos[2] > self._hp.pick_height_thresh
        gripper_pickpoint_dist = np.linalg.norm(self.pick_point - ee_pos)
        gripper_droppoint_dist = np.linalg.norm(self.drop_point - ee_pos)
        done = False

        # print('gripper_pickpoint_dist ', gripper_pickpoint_dist )
        # print('self.env.is_gripper_open ', self.env.is_gripper_open)
        if self.place_attempted:
            # print('place attemtped')
            # Avoid pick and place the object again after one attempt
            action_xyz = [0., 0., 0.]
            action_angles = [0., 0., 0.]
            action_gripper = [0.]
        elif gripper_pickpoint_dist > self._hp.grasp_distance_thresh and self.env.is_gripper_open:
            self.get_pickpoint()
            # print('moving near object ')
            # move near the object
            action_xyz = (self.pick_point - ee_pos) * self._hp.xyz_action_scale
            xy_diff = np.linalg.norm(action_xyz[:2] / self._hp.xyz_action_scale)
            if xy_diff > 0.03:
                action_xyz[2] = 0.0
            action_angles = [0., 0., 0.]
            action_gripper = [0.0]
        elif self.env.is_gripper_open:
            # print('peform grasping, gripper open:', self.env.is_gripper_open)
            # near the object enough, performs grasping action
            action_xyz = (self.pick_point - ee_pos) * self._hp.xyz_action_scale
            action_angles = [0., 0., 0.]
            action_gripper = [-0.7]
        elif not object_lifted:
            # print('lift object')
            # lifting objects above the height threshold for picking
            # action_xyz = (self.env.ee_pos_init - ee_pos) * self._hp.xyz_action_scale
            action_xyz = np.array([0., 0., 0.08]) * self._hp.xyz_action_scale
            action_angles = [0., 0., 0.]
            action_gripper = [0.]
        elif gripper_droppoint_dist > 0.02:
            # print('move towards container')
            # lifted, now need to move towards the container
            action_xyz = (self.drop_point - ee_pos) * self._hp.xyz_action_scale
            action_angles = [0., 0., 0.]
            action_gripper = [0.]
        else:
            # already moved above the container; drop object
            # print('drop')
            action_xyz = (0., 0., 0.)
            action_angles = [0., 0., 0.]
            action_gripper = [0.7]
            self.place_attempted = True

        agent_info = dict(place_attempted=self.place_attempted, done=done)
        action = np.concatenate((action_xyz, action_angles, action_gripper))
        return {'actions': action}



# if __name__ == '__main__':
