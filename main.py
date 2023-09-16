from mpc.mpc import MPC_Chemistry
from env.chemical_env import Chemical
from env.chemistry_env_rl import ColorChangingRL

import numpy as np
import torch

from utils import load_config
from typing import Union
from copy import deepcopy
import argparse
import time
import os
import random
np.set_printoptions(linewidth=np.inf)


global_seed = random.randint(1, 10000)

class DictAsAttributes:
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)

num_objects = 10
graph_type = "full"
graph = f"{graph_type}{num_objects}"
# chemical config
chemical_env_params = DictAsAttributes(dict(
        num_objects= num_objects,
        num_colors= 5,
        continuous_pos= False,
        width_std= 1,
        height_std= 1,
        width= 5,
        height= 5,
        render_image= False,
        render_type= "shapes",
        shape_size= 16,
        movement= "Dynamic",
        use_cuda= False,
        max_steps= 200,
        num_target_interventions= 30,
        g= graph,
        match_type= "all",
        dense_reward= False,
        device='cuda:0',
        seed=global_seed,
        attribute_effective=True,
        reset_num_steps = 10, # reset numsteps 
        deterministic = True, # whether envitonment transition is deterministic
        is_state_grid = True
    ))

# env_gt = Chemical(chemical_env_params)
env_gt = ColorChangingRL(
    test_mode="IID", 
    render_type='shapes', 
    num_objects=num_objects, 
    num_colors=5, 
    movement="Static", 
    max_steps=50,
    seed=global_seed
)
env_gt.set_graph(graph)

env_predictor = ColorChangingRL(
    test_mode="IID", 
    render_type='shapes', 
    num_objects=num_objects, 
    num_colors=5, 
    movement="Static", 
    max_steps=50,
    seed=global_seed
)
env_predictor.set_graph(graph)


env_params = {
    'action_dim': env_gt.action_space.n,
    'num_colors': env_gt.num_colors,
    'num_objects': env_gt.num_objects,
    'width': env_gt.width,
    'height': env_gt.height,
    'state_dim': env_gt.num_colors * env_gt.num_objects * env_gt.width * env_gt.height * 2, # attribute is effective
    'goal_dim': env_gt.num_colors * env_gt.num_objects * env_gt.width * env_gt.height,
    'adjacency_matrix': env_gt.adjacency_matrix, # store the graph 
}


mpc_params = {
    "type": 'Random',
    "horizon": 5,          # should be consistent with the max step
    "popsize": 700,        # how many random samples for mpc
    "gamma": 1,            # reward discount coefficient
    "max_iters": 5,
    "num_elites": 10,
}

episode = 1000
test_episode = 100
mpc_params["env_params"] = env_params

def diff_test(env1: Union[Chemical, ColorChangingRL], env2: Union[Chemical, ColorChangingRL]):
    env1.reset()
    env2.reset()
    env2.step(env2.random_action())

    # 测不一样
    im1, im_goal1  = env1.obs_dict2grid_seperate()
    im2, im_goal2 = env2.obs_dict2grid_seperate()
    print((im1 == im2).all())

    # 测一样
    env2._set_obs_dict_from_grid(im1)
    im2, im_goal2 = env2.obs_dict2grid_seperate()
    print((im1 == im2).all())

    # 测预测结果一样
    action = env2.random_action()

    im0, _ = env1.obs_dict2grid_seperate()
    colors0 = env1.get_current_color()

    env1.step(action)
    colors1 = env1.get_current_color()

    for i in range(10):
        env2.step(env2.random_action())
    colors1_2 = env2.get_current_color()
    env2._set_obs_dict_from_grid(im0)
    colors2 = env2.get_current_color()

    env2.step(action)
    colors3 = env2.get_current_color()

    pass
    

def diff_test2(env1: Union[Chemical, ColorChangingRL], env2: Union[Chemical, ColorChangingRL]):
    env1.reset()
    env2.reset()
    env2.step(env2.random_action())
    
    action = env2.random_action()
    im0, _ = env1.obs_dict2grid_seperate()
    colors0 = env1.get_current_color()
    env1.step(action)
    colors1 = env1.get_current_color()

    for i in range(7):
        env2.step(env2.random_action())
    env2.predict(im0, action)
    colors2 = env2.get_current_color()
    pass


def main():
    mpc_controller = MPC_Chemistry(mpc_params)
    mpc_controller.reset()

    step = 0
    reward = 0
    for e_i in range(test_episode):
        state = env_gt.reset()
        done = False
        original_reward = env_gt.current_reward
        one_train_reward = 0
        while not done:
            action = mpc_controller.act(env_predictor, state)  
            next_state, reward, done, info = env_gt.step(action)
            one_train_reward += reward
            state = deepcopy(next_state)

        print(f"episode:{e_i}, uplift: {one_train_reward} final match: {original_reward + one_train_reward}")

if __name__ == "__main__":
    seed = global_seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    env_gt = Chemical(chemical_env_params) # ps:突然给我报错用随机数生成器导致deepcopy不行，我把Chemical 中的随机数给注释掉
    env_predictor = deepcopy(env_gt)
    # diff_test(env_gt, env_predictor)
    main()
    # diff_test2(env_gt, env_predictor)

        
