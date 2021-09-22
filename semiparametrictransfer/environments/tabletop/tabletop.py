from visual_mpc.envs.mujoco_env.base_mujoco_env import BaseMujocoEnv
import numpy as np
import copy
import os
from metaworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv
from semiparametrictransfer.utils.general_utils import AttrDict


class Tabletop(BaseMujocoEnv, SawyerXYZEnv):
    """Tabletop Manip (Metaworld) Env"""
    def __init__(self, env_params_dict, reset_state=None):
        hand_low=(-0.2, 0.4, 0.0)
        hand_high=(0.2, 0.8, 0.05)
        obj_low=(-0.3, 0.4, 0.1)
        obj_high=(0.3, 0.8, 0.3)

        dirname = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self._hp = self._default_hparams()
        self._override_defaults(env_params_dict)

        if self._hp.xml is None:
            if self._hp.textured:
                filename = os.path.join(dirname, "assets/sawyer_xyz/sawyer_multiobject_textured.xml")
            else:
                filename = os.path.join(dirname, "assets/sawyer_xyz/sawyer_multiobject.xml")
        else:
            filename = os.path.join(dirname, self._hp.xml)

        BaseMujocoEnv.__init__(self, filename, self._hp)
        SawyerXYZEnv.__init__(
                self,
                frame_skip=20,
                action_scale=1./10,
                hand_low=hand_low,
                hand_high=hand_high,
                model_name=filename
            )
        goal_low = self.hand_low
        goal_high = self.hand_high
        self._adim = 4
        self._state_dim = 30
        self.liftThresh = 0.04
        self.max_path_length = 100
        self.hand_init_pos = np.array((0, 0.6, 0.0))
        self.num_objects = self._hp.num_objects

    def _default_hparams(self):
        dict = AttrDict({
            'verbose': False,
            'difficulty': None,
            'textured': False,
            'xml': None,
            'randomize_initial_armpos': False,
            'set_object_touch_goal': False,
            'target_obj_id': 2,
            'num_objects':3,
        })
        default_dict = super()._default_hparams()
        default_dict.update(dict)
        return default_dict

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        start_id = 9 + self.targetobj*2
        qpos[start_id:(start_id+2)] = pos.copy()
        qvel[start_id:(start_id+2)] = 0
        self.set_state(qpos, qvel)

    def _set_arm_pos_to_start(self):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[:9] = self._obs_history[0]['qpos'][:9].copy()
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _reset_hand(self, goal=False):
        pos = self.hand_init_pos.copy()
        pos[0] += np.random.uniform(-0.2, 0.2, 1)
        pos[1] += np.random.uniform(-0.2, 0.2, 1)
        for _ in range(10):
            self.data.set_mocap_pos('mocap', pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation([-1, 1], self.frame_skip)
        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        self.init_fingerCOM = (rightFinger + leftFinger) / 2
        self.pickCompleted = False

    def get_site_pos(self, siteName):
        _id = self.model.site_names.index(siteName)
        return self.data.site_xpos[_id].copy()

    def reset(self, reset_state=None):

        if reset_state is not None:
            if isinstance(reset_state, dict):
                target_qpos = reset_state['qpos_all']
                target_qvel = reset_state['qvel_all']
            else:
                target_qpos = reset_state
                target_qvel = np.zeros_like(self.data.qvel)
            self.set_state(target_qpos, target_qvel)
            obj_init_pos = target_qpos[9:].reshape(self.num_objects, 2)
            self.obj_init_pos = obj_init_pos
        else:
            self._reset_hand()

            obj_init_pos = []
            for i in range(3):
                self.targetobj = i
                init_pos = np.random.uniform(
                    -0.2,
                    0.2,
                    size=(2,),
                )
                obj_init_pos.append(init_pos)
                self._set_obj_xyz(init_pos)
                for _ in range(100):
                    self.do_simulation([0.0, 0.0])
            obj_init_pos = np.stack(obj_init_pos)

        self._obs_history = []
        obs = self._get_obs()
        self._reset_eval()

        if self._hp.set_object_touch_goal:
            goal_arm_pos = np.concatenate([obj_init_pos[self._hp.target_obj_id], np.array([self.get_endeff_pos()[2]])])
            self.obj_init_pos = obj_init_pos
            self.set_goal(obj_init_pos.flatten(), goal_arm_pos)

        #Can try changing this
        return obs, self.sim.data.qpos.flat.copy()

    def step(self, action, render=True):
        self.set_xyz_action(action[:3])
        # for i in range(100):
        self.do_simulation([action[-1], -action[-1]])
        obs = self._get_obs(render)
        return obs
  
    def render(self):
        return super().render().copy()

    def set_goal(self, goal_obj_pose, goal_arm_pose):
        print(f'Setting goals to {goal_obj_pose} and {goal_arm_pose}!')
        super(Tabletop, self).set_goal(goal_obj_pose, goal_arm_pose)

    def get_mean_obj_dist(self):
        distances = self.compute_object_dists(self.sim.data.qpos.flat[9:], self._goal_obj_pose)
        return np.mean(distances)

    def get_distance_score(self):
        """
        :return:  mean of the distances between all objects and goals
        """
        mean_obj_dist = self.get_mean_obj_dist()
        # Pretty sure the below is not quite right...
        # print(f'Object distance score is {mean_obj_dist}')
        # print(f'Arm joint distance score is {arm_dist_despos}')
        #return arm_dist_despos

        score = mean_obj_dist
        if self._hp.set_object_touch_goal:
            score = np.linalg.norm(self._goal_arm_pose - self.get_endeff_pos())

            # for debug:
            eef_pos = self.get_endeff_pos()
            obj_arm_dists = []
            for o in range(self.num_objects):
                obj_arm_dists.append(np.array(np.linalg.norm(self.obj_init_pos[o] - eef_pos[:2]))[None])
            self.arm_obj_distance = np.concatenate(obj_arm_dists)
        return score

    def has_goal(self):
        return True

    def compute_object_dists(self, qpos1, qpos2):
        distances = []
        for i in range(3):
            dist = np.linalg.norm(qpos1[i*2:(i+1)*2] - qpos2[i*2:(i+1)*2])
            distances.append(dist)
        return distances

    def goal_reached(self):
        og_pos = self._obs_history[0]['qpos']
        object_dists = self.compute_object_dists(og_pos[9:], self.sim.data.qpos.flat[9:])
        #print('max dist', max(object_dists))
        return max(object_dists) > 0.075

    def _get_obs(self, render=True):
        obs = {}
        #joint poisitions and velocities
        obs['qpos'] = copy.deepcopy(self.sim.data.qpos[:].squeeze())
        obs['qpos_full'] = copy.deepcopy(self.sim.data.qpos)
        obs['qvel'] = copy.deepcopy(self.sim.data.qvel[:].squeeze())
        obs['qvel_full'] = copy.deepcopy(self.sim.data.qvel)

        obs['gripper'] = self.get_endeff_pos()
        obs['state'] = np.concatenate([copy.deepcopy(self.sim.data.qpos[:].squeeze()),
                                       copy.deepcopy(self.sim.data.qvel[:].squeeze())])
        obs['object_qpos'] = copy.deepcopy(self.sim.data.qpos[9:].squeeze())

        #copy non-image data for environment's use (if needed)
        self._last_obs = copy.deepcopy(obs)
        self._obs_history.append(copy.deepcopy(obs))

        #get images
        if render:
            obs['images'] = self.render()
        obs['env_done'] = False
        return obs
  
    def valid_rollout(self):
        return True

    def current_obs(self):
        return self._get_obs()
  
    def get_goal(self):
        return self.goalim
  
    def reset_model(self):
        pass

    @staticmethod
    def get_goal_states_from_obsdict(obs_dict):
        goal_obj_pose = obs_dict['object_qpos'][-1]
        goal_arm_pose = obs_dict['qpos'][-1][:9]
        loaded_traj_info = obs_dict
        return goal_arm_pose, goal_obj_pose, loaded_traj_info

   
if __name__ == '__main__':
    env_params = {
      # resolution sufficient for 16x anti-aliasing
      'viewer_image_height': 192,
      'viewer_image_width': 256,
      'textured': True
      #     'difficulty': 'm',
    }
    env = Tabletop(env_params)
    env.reset()
    env.targetobj = 2
    init_pos = np.array([
        0,
        0.2
    ])
    env.obj_init_pos = init_pos
    env._set_obj_xyz(env.obj_init_pos)
    import ipdb; ipdb.set_trace()

    import matplotlib.pyplot as plt
    for i, coord in enumerate(np.linspace(-0.1, 0.1, 21)):
        env.targetobj = 0
        init_pos = np.array([
            coord,
            0.2
        ])
        env.obj_init_pos = init_pos
        env._set_obj_xyz(env.obj_init_pos)
        img = env.render()[0]
        plt.imsave(f'./examples/im_{i}.png', img)
