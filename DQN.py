import json
import time

import numpy as np
from modular_robot_building import ModularBuildingEnv

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DQN
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

def main():
    with open('module_sets/tinybot/tinybot.json', encoding='UTF-8') as json_file:
        env_config = json.load(json_file)
    env = ModularBuildingEnv(env_config)
    # time.sleep(20.0)
    check_env(env)

    env.reset()
    n_actions = env.action_space.shape
    print(n_actions)
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = DQN("MlpPolicy", env, verbose=1)
    number_of_epoch = 2
    for i in range(number_of_epoch):
        model.learn(total_timesteps=100, tb_log_name="dqn")
        model.save("dqn" + str(i))
    del model # remove to demonstrate saving and loading

if __name__ == '__main__':
    main()