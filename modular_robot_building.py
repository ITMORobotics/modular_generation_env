import gym
from gym.utils import seeding

from itmobotics_sim.utils.math import vec2SE3, SE32vec
from itmobotics_sim.pybullet_env.pybullet_world import PyBulletWorld, GUI_MODE
from spatialmath import SE3, SO3, Twist3
from spatialmath import base as sb

import numpy as np
import time
import copy
import os

from builder import serial

import json
from jsonschema import Draft7Validator, validators

def extend_with_default(validator_class):
    validate_properties = validator_class.VALIDATORS["properties"]

    def set_defaults(validator, properties, instance, schema):
        for property, subschema in properties.items():
            if "default" in subschema:
                instance.setdefault(property, subschema["default"])

        for error in validate_properties(
            validator, properties, instance, schema,
        ):
            yield error

    return validators.extend(
        validator_class, {"properties" : set_defaults},
    )
DefaultValidatingDraft7Validator = extend_with_default(Draft7Validator)

class ModularBuildingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, env_config: dict):
        self.__gui_mode = GUI_MODE.SIMPLE_GUI if env_config['simulation']['gui'] else GUI_MODE.DIRECT
        self.__env_config = env_config

        with open('module_sets/serial_builder_env_config_schema.json') as json_file:
            env_schema = json.load(json_file)
            DefaultValidatingDraft7Validator(env_schema).validate(self.__env_config)
        
        if 'random_seed' in self.__env_config['simulation']:
            self._np_random, self._seed = seeding.np_random(self.__env_config['simulation']['random_seed'])
        else:
            self._np_random, self._seed = seeding.np_random(int(time.time()))

        self.__avail_modules_list = list(self.__env_config['robot']['modules'].keys())
        self.__avail_modules_set = {}
        for i in range(0 , len(self.__avail_modules_list)):
            m = self.__avail_modules_list[i]
            self.__avail_modules_set[m] = serial.SerialRobot(
                name=m,
                id=i,
                urdf_filename=os.path.join(os.getcwd(),self.__env_config['robot']['modules'][m]['urdf_filename']),
                connect_link=self.__env_config['robot']['modules'][m]['connect_link'],
                allow_connect=self.__env_config['robot']['modules'][m]['allow_connect'],
                connect_tf=vec2SE3(np.array(self.__env_config['robot']['modules'][m]['connect_tf']))
            )
        self.__max_N_modules = self.__env_config['robot']['max_N_modules']
        self.__N_avail_modules = len(self.__avail_modules_set)

        self.__sim = PyBulletWorld(
            self.__gui_mode,
            time_step = self.__env_config['simulation']['time_step'],
            time_scale = self.__env_config['simulation']['sim_time_scale']
        )

        self.__initial_module = copy.deepcopy(self.__avail_modules_set[self.__env_config['robot']['initial_module']])
        self.__initial_module.urdf_filename = self.__env_config['robot']['output_urdf']
        self.__initial_module.save()
        self.__sim.add_robot(
            urdf_filename=self.__initial_module.urdf_filename,
            base_transform = vec2SE3(np.array(self.__env_config['robot']['mount_tf'])),
            name = self.__env_config['robot']['name']
        )
        self.action_space = gym.spaces.Discrete(self.__N_avail_modules)
        self.observation_space = gym.spaces.box.Box(
            low=-1*np.ones(self.__max_N_modules ,dtype=np.float32),
            high=self.__N_avail_modules*np.ones(self.__max_N_modules, dtype=np.float32)
        )
        print(self.action_space)
        self.reset()
    
    def observation_state_as_vec(self) -> np.ndarray:
        part_of_chain = np.array(self.__initial_module.chain_id, dtype=np.float32)
        return np.concatenate([part_of_chain, -1*np.ones(self.__max_N_modules - part_of_chain.shape[0], dtype=np.float32)])
    
    @property
    def seed(self):
        return self._seed
    
    @property
    def N_modules(self):
        return 

    @seed.setter
    def seed(self, seed: int):
        print("Set Seed = %d"%seed)
        self._np_random, self._seed = seeding.np_random(seed)

    def reset(self) -> np.ndarray:
        self.__initial_module = copy.deepcopy(self.__avail_modules_set[self.__env_config['robot']['initial_module']])
        self.__initial_module.urdf_filename = self.__env_config['robot']['output_urdf']
        self.__initial_module.save()
        self.__sim.reset()
        obs = self.observation_state_as_vec()
        return obs
    
    def render(self, mode: str = 'human', close: bool = False):
        pass

    def step(self, action: np.ndarray):
        done = False
        info = {}

        # Downgrade reward value depends on count of modules
        reward = -1.0
        
        connection_succes=self.__take_action(action)
        if not connection_succes:
            reward += -10.0
            info['termination'] = True

            done = True

        reward += self.__simulate_robot()

        obs = self.observation_state_as_vec()
        print(obs)
        return obs, reward, done, info
    
    def __take_action(self, action: int) -> bool:
        print(self.__avail_modules_set[self.__avail_modules_list[action]].name)
        connection_succes = self.__initial_module.join_module(self.__avail_modules_set[self.__avail_modules_list[action]])
        print(connection_succes, self.__initial_module.chain_id)
        self.__initial_module.save()
        self.__sim.reset()
        return connection_succes
    
    def _sample_random_objects(self):
        for object_name in self.__env_config['world']['world_objects']:
            object_config = self.__env_config['world']['world_objects'][object_name]
            random_tf = self.__sample_random_tf(np.array(object_config['init_tf']), np.zeros(4) )
            self.__sim.add_object(
                object_name,
                object_config['urdf_filename'],
                base_transform = random_tf,
                fixed = object_config['fixed'],
                save = object_config['save'],
                scale_size = object_config['scale_size']
            )
    
    def __sample_random_tf(self, init_tf: np.ndarray, random_variation: np.ndarray) -> SO3:
        pose_variation = (2*self._np_random.random(3) - 1.0)*random_variation[:3]
        random_pose = SE3(*( (init_tf[:3] + pose_variation).tolist()) )

        var_theta = random_variation[3]
        theta_orient_variation = (2*self._np_random.random() - 1.0) * var_theta
        vec_orient_variation = (2*self._np_random.random(3) - 1.0)
        
        orient_variation = SE3(SO3(sb.angvec2r(theta=theta_orient_variation, v=vec_orient_variation), check = False))
        only_rotation_init_tf = init_tf
        only_rotation_init_tf[:3] = 0.0
        random_orient = orient_variation @  Twist3(only_rotation_init_tf).SE3()        
        random_tf = random_pose @ random_orient
        return random_tf

    def __simulate_robot(self) -> float:
        return 0