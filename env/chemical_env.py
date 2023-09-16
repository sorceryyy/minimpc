"""Gym environment for chnging colors of shapes."""

import numpy as np
import torch
import torch.nn as nn
import re

import gym
from collections import OrderedDict
from dataclasses import dataclass
from gym.utils import seeding
from gym import spaces

import matplotlib as mpl

import skimage
import skimage.draw

from env.drawing import render_cubes, get_colors_and_weights
import random



graphs = {
    'chain3': '0->1->2',
    'fork3': '0->{1-2}',
    'collider3': '{0-1}->2',
    'collider4': '{0-2}->3',
    'collider5': '{0-3}->4',
    'collider6': '{0-4}->5',
    'collider7': '{0-5}->6',
    'collider8': '{0-6}->7',
    'collider9': '{0-7}->8',
    'collider10': '{0-8}->9',
    'collider11': '{0-9}->10',
    'collider12': '{0-10}->11',
    'collider13': '{0-11}->12',
    'collider14': '{0-12}->13',
    'collider15': '{0-13}->14',
    'confounder3': '{0-2}->{0-2}',
    'chain4': '0->1->2->3',
    'chain5': '0->1->2->3->4',
    'chain6': '0->1->2->3->4->5',
    'chain7': '0->1->2->3->4->5->6',
    'chain8': '0->1->2->3->4->5->6->7',
    'chain9': '0->1->2->3->4->5->6->7->8',
    'chain10': '0->1->2->3->4->5->6->7->8->9',
    'chain11': '0->1->2->3->4->5->6->7->8->9->10',
    'chain12': '0->1->2->3->4->5->6->7->8->9->10->11',
    'chain13': '0->1->2->3->4->5->6->7->8->9->10->11->12',
    'chain14': '0->1->2->3->4->5->6->7->8->9->10->11->12->13',
    'chain15': '0->1->2->3->4->5->6->7->8->9->10->11->12->13->14',
    'full3': '{0-2}->{0-2}',
    'full4': '{0-3}->{0-3}',
    'full5': '{0-4}->{0-4}',
    'full6': '{0-5}->{0-5}',
    'full7': '{0-6}->{0-6}',
    'full8': '{0-7}->{0-7}',
    'full9': '{0-8}->{0-8}',
    'full10': '{0-9}->{0-9}',
    'full11': '{0-10}->{0-10}',
    'full12': '{0-11}->{0-11}',
    'full13': '{0-12}->{0-12}',
    'full14': '{0-13}->{0-13}',
    'full15': '{0-14}->{0-14}',
    'tree9': '0->1->3->7,0->2->6,1->4,3->8,2->5',
    'tree10': '0->1->3->7,0->2->6,1->4->9,3->8,2->5',
    'tree11': '0->1->3->7,0->2->6,1->4->10,3->8,4->9,2->5',
    'tree12': '0->1->3->7,0->2->6,1->4->10,3->8,4->9,2->5->11',
    'tree13': '0->1->3->7,0->2->6,1->4->10,3->8,4->9,2->5->11,5->12',
    'tree14': '0->1->3->7,0->2->6,1->4->10,3->8,4->9,2->5->11,5->12,6->13',
    'tree15': '0->1->3->7,0->2->6->14,1->4->10,3->8,4->9,2->5->11,5->12,6->13',
    'jungle3': '0->{1-2}',
    'jungle4': '0->1->3,0->2,0->3',
    'jungle5': '0->1->3,1->4,0->2,0->3,0->4',
    'jungle6': '0->1->3,1->4,0->2->5,0->3,0->4,0->5',
    'jungle7': '0->1->3,1->4,0->2->5,2->6,0->3,0->4,0->5,0->6',
    'jungle8': '0->1->3->7,1->4,0->2->5,2->6,0->3,0->4,0->5,0->6,1->7',
    'jungle9': '0->1->3->7,3->8,1->4,0->2->5,2->6,0->3,0->4,0->5,0->6,1->7,1->8',
    'jungle10': '0->1->3->7,3->8,1->4->9,0->2->5,2->6,0->3,0->4,0->5,0->6,1->7,1->8,1->9',
    'jungle11': '0->1->3->7,3->8,1->4->9,4->10,0->2->5,2->6,0->3,0->4,0->5,0->6,1->7,1->8,1->9,1->10',
    'jungle12': '0->1->3->7,3->8,1->4->9,4->10,0->2->5->11,2->6,0->3,0->4,0->5,0->6,1->7,1->8,1->9,1->10,2->11',
    'jungle13': '0->1->3->7,3->8,1->4->9,4->10,0->2->5->11,5->12,2->6,0->3,0->4,0->5,0->6,1->7,1->8,1->9,1->10,2->11,2->12',
    'jungle14': '0->1->3->7,3->8,1->4->9,4->10,0->2->5->11,5->12,2->6->13,0->3,0->4,0->5,0->6,1->7,1->8,1->9,1->10,2->11,2->12,2->13',
    'jungle15': '0->1->3->7,3->8,1->4->9,4->10,0->2->5->11,5->12,2->6->13,6->14,0->3,0->4,0->5,0->6,1->7,1->8,1->9,1->10,2->11,2->12,2->13,2->14',
    'bidiag3': '{0-2}->{0-2}',
    'bidiag4': '{0-1}->{1-2}->{2-3}',
    'bidiag5': '{0-1}->{1-2}->{2-3}->{3-4}',
    'bidiag6': '{0-1}->{1-2}->{2-3}->{3-4}->{4-5}',
    'bidiag7': '{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}',
    'bidiag8': '{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}->{6-7}',
    'bidiag9': '{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}->{6-7}->{7-8}',
    'bidiag10': '{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}->{6-7}->{7-8}->{8-9}',
    'bidiag11': '{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}->{6-7}->{7-8}->{8-9}->{9-10}',
    'bidiag12': '{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}->{6-7}->{7-8}->{8-9}->{9-10}->{10-11}',
    'bidiag13': '{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}->{6-7}->{7-8}->{8-9}->{9-10}->{10-11}->{11-12}',
    'bidiag14': '{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}->{6-7}->{7-8}->{8-9}->{9-10}->{10-11}->{11-12}->{12-13}',
    'bidiag15': '{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}->{6-7}->{7-8}->{8-9}->{9-10}->{10-11}->{11-12}->{12-13}->{13-14}',
}


def parse_skeleton(graph, M=None):
    """
    Parse the skeleton of a causal graph in the mini-language of --graph.
    
    The mini-language is:
        
        GRAPH      = ""
                     CHAIN{, CHAIN}*
        CHAIN      = INT_OR_SET {-> INT_OR_SET}
        INT_OR_SET = INT | SET
        INT        = [0-9]*
        SET        = \{ SET_ELEM {, SET_ELEM}* \}
        SET_ELEM   = INT | INT_RANGE
        INT_RANGE  = INT - INT
    """
    
    regex = re.compile(r'''
        \s*                                      # Skip preceding whitespace
        (                                        # The set of tokens we may capture, including
          [,]                                  | # Commas
          (?:\d+)                              | # Integers
          (?:                                    # Integer set:
            \{                                   #   Opening brace...
              \s*                                #   Whitespace...
              \d+\s*(?:-\s*\d+\s*)?              #   First integer (range) in set...
              (?:,\s*\d+\s*(?:-\s*\d+\s*)?\s*)*  #   Subsequent integers (ranges)
            \}                                   #   Closing brace...
          )                                    | # End of integer set.
          (?:->)                                 # Arrows
        )
    ''', re.A | re.X)
    
    # Utilities
    def parse_int(s):
        try:    return int(s.strip())
        except: return None
    
    def parse_intrange(s):
        try:
            sa, sb = map(str.strip, s.strip().split("-", 1))
            sa, sb = int(sa), int(sb)
            sa, sb = min(sa,sb), max(sa,sb)+1
            return range(sa,sb)
        except:
            return None
    
    def parse_intset(s):
        try:
            i = set()
            for s in map(str.strip, s.strip()[1:-1].split(",")):
                if parse_int(s) is not None: i.add(parse_int(s))
                else:                        i.update(set(parse_intrange(s)))
            return sorted(i)
        except:
            return None
    
    def parse_either(s):
        asint = parse_int(s)
        if asint is not None: return asint
        asset = parse_intset(s)
        if asset is not None: return asset
        raise ValueError
    
    def find_max(chains):
        m = 0
        for chain in chains:
            for link in chain:
                link = max(link) if isinstance(link, list) else link
                m = max(link, m)
        return m
    
    # Crack the string into a list of lists of (ints | lists of ints)
    graph = [graph] if isinstance(graph, str) else graph
    chains = []
    for gstr in graph:
        for chain in re.findall("((?:[^,{]+|\{.*?\})+)+", gstr, re.A):
            links = list(map(str.strip, regex.findall(chain)))
            assert(len(links) & 1)
            
            chain = [parse_either(links.pop(0))]
            while links:
                assert links.pop(0) == "->"
                chain.append(parse_either(links.pop(0)))
            chains.append(chain)
    
    # Find the maximum integer referenced within the skeleton
    uM = find_max(chains) + 1
    if M is None:
        M = uM
    else:
        assert(M >= uM)
        M = max(M, uM)
    
    # Allocate adjacency matrix.
    gamma = np.zeros((M, M), dtype=np.float32)
    
    # Interpret the skeleton
    for chain in chains:
        for prevlink, nextlink in zip(chain[:-1], chain[1:]):
            if isinstance(prevlink, list) and isinstance(nextlink, list):
                for i in nextlink:
                    for j in prevlink:
                        if i > j:
                            gamma[i, j] = 1
            elif isinstance(prevlink, list) and isinstance(nextlink, int):
                for j in prevlink:
                    if nextlink > j:
                        gamma[nextlink, j] = 1
            elif isinstance(prevlink, int) and isinstance(nextlink, list):
                minn = min(nextlink)
                if minn == prevlink:
                    raise ValueError("Edges are not allowed from " +
                                     str(prevlink) + " to oneself!")
                elif minn < prevlink:
                    raise ValueError("Edges are not allowed from " +
                                     str(prevlink) + " to ancestor " +
                                     str(minn) + " !")
                else:
                    for i in nextlink:
                        gamma[i, prevlink] = 1
            elif isinstance(prevlink, int) and isinstance(nextlink, int):
                if nextlink == prevlink:
                    raise ValueError("Edges are not allowed from " +
                                     str(prevlink) + " to oneself!")
                elif nextlink < prevlink:
                    raise ValueError("Edges are not allowed from " +
                                     str(prevlink) + " to ancestor " +
                                     str(nextlink) + " !")
                else:
                    gamma[nextlink, prevlink] = 1
    
    # Return adjacency matrix.
    return gamma


mpl.use('Agg')


def random_dag(M, N, g=None):
    """Generate a random Directed Acyclic Graph (DAG) with a given number of nodes and edges."""
    if g is None:
        expParents = 5
        idx        = np.arange(M).astype(np.float32)[:, np.newaxis]
        idx_maxed  = np.minimum(idx * 0.5, expParents)
        p          = np.broadcast_to(idx_maxed / (idx + 1), (M, M))
        B          = np.random.binomial(1, p)
        B          = np.tril(B, -1)
        return B
    else:
        gammagt = parse_skeleton(g, M=M)
        return gammagt


class MLP(nn.Module):
    def __init__(self, dims, deterministic=False):
        super().__init__()
        self.deterministic = deterministic
        self.layers = []
        for i in range(1, len(dims)):
            self.layers.append(nn.Linear(dims[i-1], dims[i]))
            torch.nn.init.orthogonal_(self.layers[-1].weight.data, 3.0)
            torch.nn.init.uniform_(self.layers[-1].bias.data, -0.2, +0.2)
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x, mask):

        x = x * mask

        for i, l in enumerate(self.layers):
            if i == len(self.layers) - 1:
                x = torch.softmax(l(x), dim=1)
            else:
                x = torch.relu(l(x))

        if self.deterministic:
            max_prob_index = torch.argmax(x, dim=-1)
            x = torch.nn.functional.one_hot(max_prob_index, num_classes=x.shape[-1])
        else:
            x = torch.distributions.one_hot_categorical.OneHotCategorical(probs=x).sample()

        return x

@dataclass
class Coord:
    x: int
    y: int

    def __add__(self, other):
        return Coord(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Coord(self.x - other.x, self.y - other.y)

    def __eq__(self, other):
        return (self.x == other.x) and (self.y == other.y)


@dataclass
class Object:
    pos: Coord
    color: int


@dataclass
class Object_cont:
    x: float
    y: float
    color: int



def to_numpy(tensor):
    return tensor.cpu().detach().numpy()





class Chemical(gym.Env):
    def __init__(self, params):
        """
        Observation comes in two form:
            1. dict form
            2. numpy form: [colori, xi, yi] * num_obj + [target_colori] * num_obj
        """
        params = params

        self.use_cuda = params.use_cuda
        self.device = device = params.device if self.use_cuda else torch.device("cpu")

        self.width = params.width
        self.height = params.height
        self.continuous_pos = params.continuous_pos
        self.width_std = params.width_std
        self.height_std = params.height_std
        self.attribute_effective = params.attribute_effective
        self.is_state_grid = getattr(params, "is_state_grid", False)

        # add by me, rollout num for reset
        self.reset_num_steps = params.reset_num_steps

        self.render_image = params.render_image
        self.render_type = params.render_type
        if self.render_image:
            assert self.width == self.height
            self.shape_size = shape_size = params.shape_size       # in pixel
            assert (128 - shape_size) % (self.width - 1) == 0
            self.grid_size = (128 - shape_size) // (self.width - 1)             # in pixel

        self.num_objects = params.num_objects
        self.movement = params.movement

        self.num_colors = params.num_colors
        if self.num_colors is None:
            self.num_colors = self.num_objects
        self.num_actions = self.num_objects * self.num_colors
        self.num_target_interventions = params.num_target_interventions
        self.max_steps = params.max_steps

        self.mlps = []
        self.mask = None

        self.colors, _ = get_colors_and_weights(cmap='Set1', num_colors=self.num_colors)
        self.object_to_color = [torch.zeros(self.num_colors, device=device) for _ in range(self.num_objects)]

        self.np_random = None

        self.match_type = params.match_type
        self.dense_reward = params.dense_reward
        if self.match_type == "all":
            self.match_type = list(range(self.num_objects))

        # Initialize to pos outside of env for easier collision resolution.
        self.objects = OrderedDict()

        self.adjacency_matrix = None

        self.deterministic = getattr(params, "deterministic", False)

        mlp_dims = [self.num_objects * self.num_colors, 4 * self.num_objects, self.num_colors] #
        self.mlps = [MLP(mlp_dims, self.deterministic).to(device) for _ in range(self.num_objects)]

        num_nodes = self.num_objects
        num_edges = 0 if num_nodes == 1 else np.random.randint(num_nodes - 1, num_nodes * (num_nodes - 1) // 2 + 1)

        # check g's form
        if params.g in graphs.keys():
            print('INFO: Loading predefined graph for configuration ' + str(params.g))
            params.g = graphs[params.g]


        self.adjacency_matrix = random_dag(num_nodes, num_edges, params.g)
        self.adjacency_matrix = torch.from_numpy(self.adjacency_matrix).to(device).float()

        # Generate masks so that each variable only receives input from its parents.
        self.generate_masks()

        # If True, then check for collisions and don't allow two
        #   objects to occupy the same position.
        self.collisions = True
        self.actions_to_target = []

        self.objects = OrderedDict()

        self.action_dim = self.num_actions

        # self.seed(params.seed)
        self.reset()

        # required in Deep-REinforcement-Learning-Algorithms-with-PyTorch don't know why
        self.env = "Chemical"
        self.reward_threshold = 0.0
        self.trials = 100
        self._max_episode_steps = self.max_steps


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def load_save_information(self, save):
        self.adjacency_matrix = save['graph']
        for i in range(self.num_objects):
            self.mlps[i].load_state_dict(save['mlp' + str(i)])
        self.generate_masks()
        self.reset()

    def set_graph(self, g):
        if g in graphs.keys():
            print('INFO: Loading predefined graph for configuration '+str(g))
            g = graphs[g]
        num_nodes = self.num_objects
        num_edges = np.random.randint(num_nodes, num_nodes * (num_nodes - 1) // 2 + 1)
        self.adjacency_matrix = random_dag(num_nodes, num_edges, g=g)
        self.adjacency_matrix = torch.from_numpy(self.adjacency_matrix).to(self.device).float()
        # print(self.adjacency_matrix)
        self.generate_masks()
        self.reset()

    def get_save_information(self):
        save = {}
        save['graph'] = self.adjacency_matrix
        for i in range(self.num_objects):
            save['mlp' + str(i)] = self.mlps[i].state_dict()
        return save

    def render_grid(self):
        im = np.zeros((3, self.width, self.height))
        for idx, obj in self.objects.items():
            im[:, obj.pos.x, obj.pos.y] = self.colors[obj.color][:3]
        return im

    def render_circles(self):
        grid_size = self.grid_size
        half_grid = grid_size // 2
        im = np.zeros((self.width * grid_size, self.height * grid_size, 3), dtype=np.uint8)
        for idx, obj in self.objects.items():
            rr, cc = skimage.draw.disk(
                (obj.pos.x * grid_size + half_grid, obj.pos.y * grid_size + half_grid), half_grid, shape=im.shape)
            im[rr, cc, :] = self.colors[obj.color][:3]
        return im


    def render_grid_target(self):
        im = np.zeros((3, self.width, self.height))
        for idx, obj in self.objects.items():
            im[:, obj.pos.x, obj.pos.y] = self.colors[torch.argmax(self.object_to_color_target[idx]).item()][:3]
        return im

    def render_circles_target(self):
        grid_size = self.grid_size
        half_grid = grid_size // 2
        im = np.zeros((self.width * grid_size, self.height * grid_size, 3), dtype=np.uint8)
        for idx, obj in self.objects.items():
            rr, cc = skimage.draw.disk(
                (obj.pos.x * grid_size + half_grid, obj.pos.y * grid_size + half_grid), half_grid, shape=im.shape)
            im[rr, cc, :] = self.colors[torch.argmax(self.object_to_color_target[idx]).item()][:3]
        return im

    def render_shapes_target(self):
        return self.render_shapes(target=True)

    def render_cubes(self):
        im = render_cubes(self.objects, self.width)
        return im

    def render(self):
        return dict(
            grid=self.render_grid,
            circles=self.render_circles,
            shapes=self.render_shapes,
            cubes=self.render_cubes,
        )[self.render_type](), dict(
            grid=self.render_grid_target,
            circles=self.render_circles_target,
            shapes=self.render_shapes_target,
            cubes=self.render_cubes,
        )[self.render_type]()

    def get_state(self):
        state = {}
        for idx, obj in self.objects.items():
            if self.continuous_pos:
                state["obj{}".format(idx)] = np.array([obj.color, obj.x, obj.y])
            else:
                state["obj{}".format(idx)] = np.array([obj.color, obj.pos.x, obj.pos.y])
        for idx, color in enumerate(self.object_to_color_target_np):
            state["target_obj{}".format(idx)] = np.array([color])
        if self.render_image:
            image, goal_image = self.render()
            state["image"], state["goal_image"] = (image * 255).astype(np.uint8), (goal_image * 255).astype(np.uint8)
        return state

    def observation_spec(self):
        return self.get_state()

    def observation_dims(self):
        state = {}
        for idx, obj in self.objects.items():
            if self.continuous_pos:
                state["obj{}".format(idx)] = np.array([self.num_colors, 1, 1])
            else:
                state["obj{}".format(idx)] = np.array([self.num_colors, self.width, self.height])
        for idx, color in enumerate(self.object_to_color_target):
            state["target_obj{}".format(idx)] = np.array([self.num_colors])
        return state

    def generate_masks(self):
        mask = self.adjacency_matrix.unsqueeze(-1)
        mask = mask.repeat(1, 1, self.num_colors)
        self.mask = mask.view(self.adjacency_matrix.size(0), -1)

    def generate_target(self, num_steps=10):
        self.actions_to_target = []
        for i in range(num_steps):
            intervention_id = random.randint(0, self.num_objects - 1)
            to_color = random.randint(0, self.num_colors - 1)
            self.actions_to_target.append(intervention_id * self.num_colors + to_color)
            self.object_to_color_target[intervention_id] = torch.zeros(self.num_colors, device=self.device)
            self.object_to_color_target[intervention_id][to_color] = 1
            self.sample_variables_target(intervention_id)

    def reset(self):
        self.cur_step = 0
        num_steps = self.reset_num_steps

        self.object_to_color = [torch.zeros(self.num_colors, device=self.device) for _ in range(self.num_objects)]
        self.object_to_color_target = [torch.zeros(self.num_colors, device=self.device)
                                       for _ in range(self.num_objects)]

        # Sample color for root node randomly
        root_color = np.random.randint(0, self.num_colors)
        self.object_to_color[0][root_color] = 1

        # Sample color for other nodes using MLPs
        self.sample_variables(0, do_everything=True)
        if self.movement == 'Dynamic':
            self.objects = OrderedDict()
            # Randomize object position.
            while len(self.objects) < self.num_objects:
                idx = len(self.objects)
                # Re-sample to ensure objects don't fall on same spot.
                if self.continuous_pos:
                    self.objects[idx] = Object_cont(
                        x=np.random.normal(0, self.width_std),
                        y=np.random.normal(0, self.height_std),
                        color=to_numpy(self.object_to_color[idx].argmax()))
                else:
                    while not (idx in self.objects and self.valid_pos(self.objects[idx].pos, idx)):
                        # width_unit = 128 // self.num_objects
                        # low = width_unit * idx // self.grid_size
                        # high = (width_unit * (idx + 1) - self.shape_size) // self.grid_size + 1
                        # x = np.random.randint(low, high)
                        x = np.random.randint(self.width)
                        y = np.random.randint(self.height)
                        self.objects[idx] = Object(
                            pos=Coord(x=x, y=y),
                            color=to_numpy(self.object_to_color[idx].argmax()))

        for idx, obj in self.objects.items():
            obj.color = to_numpy(self.object_to_color[idx].argmax())

        for i in range(len(self.object_to_color)):
            self.object_to_color_target[i][to_numpy(self.object_to_color[i].argmax())] = 1

        self.generate_target(num_steps)
        self.object_to_color_target_np = [to_numpy(ele.argmax()) for ele in self.object_to_color_target]

        state = self.get_state()
        state = self.obs_dict2array(state)
        if self.is_state_grid:
            return self.obs_dict2grid()
        return state

    def valid_pos(self, pos, obj_id):
        """Check if position is valid."""
        if not 0 <= pos.x < self.width:
            return False
        if not 0 <= pos.y < self.height:
            return False

        if self.collisions:
            for idx, obj in self.objects.items():
                if idx == obj_id:
                    continue

                if pos == obj.pos:
                    return False

        return True

    def is_reachable(self, idx, reached):
        for r in reached:
            if self.adjacency_matrix[idx, r] == 1:
                return True
        return False

    def sample_variables(self, idx, do_everything=False):
        """
        idx: variable at which intervention is performed
        """
        reached = [idx]
        for v in range(idx + 1, self.num_objects):
            if do_everything or self.is_reachable(v, reached):
                reached.append(v)

                inp = torch.cat(self.object_to_color, dim=0).unsqueeze(0)
                mask = self.mask[v].unsqueeze(0)

                out = self.mlps[v](inp, mask)
                self.object_to_color[v] = out.squeeze(0)

    def sample_variables_target(self, idx, do_everything=False):
        """
        idx: variable at which intervention is performed
        """
        reached = [idx]
        for v in range(idx + 1, self.num_objects):
            if do_everything or self.is_reachable(v, reached):
                reached.append(v)

                inp = torch.cat(self.object_to_color_target, dim=0).unsqueeze(0)
                mask = self.mask[v].unsqueeze(0)

                out = self.mlps[v](inp, mask)
                self.object_to_color_target[v] = out.squeeze(0)

    def translate(self, obj_id, color_id):
        """Translate object pixel.

        Args:
            obj_id: ID of object.
            color_id: ID of color.
        """
        color_ = torch.zeros(self.num_colors, device=self.device)
        color_[color_id] = 1
        self.object_to_color[obj_id] = color_
        self.sample_variables(obj_id)
        for idx, obj in self.objects.items():
            obj.color = to_numpy(self.object_to_color[idx].argmax())

    def calculate_matches(self):
        matches = 0.
        for i, (c1, c2) in enumerate(zip(self.object_to_color, self.object_to_color_target)):
            if i not in self.match_type:
                continue
            if torch.argmax(c1).item() == torch.argmax(c2).item():
                matches += 1
        return matches


    def step(self, action: int):
        if not isinstance(action, int) and action.size == self.action_dim:
            action = np.argmax(action)
        old_matches = self.calculate_matches()

        obj_id = action // self.num_colors
        color_id = action % self.num_colors
        self.translate(obj_id, color_id)

        if self.continuous_pos:
            for _, obj in self.objects.items():
                obj.x = np.random.normal(0, self.width_std)
                obj.y = np.random.normal(0, self.height_std)
        else:
            idx = np.random.randint(self.num_objects)
            obj = self.objects[idx]
            while True:
                new_x = np.random.randint(self.width)
                new_y = np.random.randint(self.height)
                new_pos = Coord(x=new_x, y=new_y)
                if self.valid_pos(new_pos, idx):
                    obj.pos = new_pos
                    break

        matches = self.calculate_matches()
        num_needed_match = len(self.match_type)
        if self.dense_reward:
            reward = matches / num_needed_match
        else:
            # reward = float(matches == num_needed_match) # original sparse reward
            reward = (matches - old_matches) / num_needed_match
        info = {"success": matches == num_needed_match}

        self.cur_step += 1

        # check episode finish
        done = False
        if self.cur_step >= self.max_steps:
            done = True
        if matches == self.num_objects:
            done = True # early stop if we achieve the goal

        state = self.get_state()
        state = self.obs_dict2array(state)
        if self.is_state_grid:
            state = self.obs_dict2grid()
        return state, reward, done, info

    def obs_dict2array(self, state:OrderedDict):
        """
        Change dict observation to numpy observation
        """
        effective = self.attribute_effective # effective comes true iff state exclude x, y attribute of objects
        if effective:
            state_list = [np.array([obj[0]]) if idx < self.num_objects else obj for idx, obj in enumerate(state.values()) ]
        else:
            state_list = list(state.values())
        
        state_np = np.concatenate(state_list)
        return state_np

    def obs_dict2grid(self):
        # use 2D grid world to represent the true state
        im, im_target = self.obs_dict2grid_seperate()
        im_one = self._augment_state_with_goal(im, im_target)
        return im_one

    def obs_dict2grid_seperate(self):
        """obtain 1-dimension grid with seperate obj and goal"""
        im = np.zeros((self.num_objects * self.num_colors, self.width, self.height), dtype=np.int32)
        im_target = np.zeros((self.num_objects * self.num_colors, self.width, self.height), dtype=np.int32)
        for idx, obj in self.objects.items():
            im[idx * self.num_colors + obj.color, obj.pos.x, obj.pos.y] = 1
            im_target[idx * self.num_colors + torch.argmax(self.object_to_color_target[idx]).item(), obj.pos.x, obj.pos.y] = 1
        return im, im_target

    def _augment_state_with_goal(self, state, target):
        state = state.reshape(self.num_objects * self.num_colors * self.width * self.height,)
        target = target.reshape(self.num_objects * self.num_colors * self.width * self.height,)
        augmented_state = np.concatenate([state, target], axis=0)
        return augmented_state

    def _set_obs_dict_from_grid(self, state):
        # 将 state 和 target 重新塑造为原始的 3D 形状
        state = state.reshape((self.num_objects * self.num_colors, self.width, self.height))

        # 从 3D 表示中恢复每个对象的状态
        for idx, obj in self.objects.items():
            if idx < self.num_objects:
                for color in range(self.num_colors):
                    coords = np.where(state[idx * self.num_colors + color] == 1)
                    if len(coords[0]) > 0:  # 如果找到了坐标
                        obj.color = color
                        obj.pos.x, obj.pos.y = coords[0][0], coords[1][0]  # TODO: check here!!
                        self.object_to_color[idx] = torch.zeros_like(self.object_to_color[idx])
                        self.object_to_color[idx][color] = 1

        
    def predict(self, state, action):
        """predict the next state s' according to (s, a)"""
        # convert onehot (action_dim,) to action
        next_state = np.zeros_like(state)
        for idx, (single_state, single_action) in enumerate(zip(state, action)):
            self._set_obs_dict_from_grid(single_state)
            single_next_state, reward, done, info = self.step(single_action)
            single_next_state = single_next_state[:self.num_colors * self.num_objects * self.width * self.height] # exclude the target color
            next_state[idx, :] = single_next_state 
        return next_state
        
    @property
    def current_reward(self):
        """aim for get initial reward, 2023.8.23"""
        matches = self.calculate_matches()
        num_needed_match = len(self.match_type)
        # dense reward do not need current reward
        if self.dense_reward:
            return 0
        else:
            return matches / num_needed_match

    @property
    def action_space(self):
        return spaces.Discrete(self.action_dim)

    @property
    def observation_space(self):
        #TODO: change the demension of color to discrete type
        if self.continuous_pos:
            low = np.array([0, float("-inf"), float("-inf")] * self.num_objects + \
                           [0] * self.num_objects)
            high = np.array([self.num_colors, float("inf"), float("inf")] * self.num_objects + \
                           [self.num_colors] * self.num_objects)
        else:
            low = np.zeros(self.num_objects * 4) # action space dim
            high = np.array([self.num_colors, self.width, self.height] * self.num_objects + \
                            [self.num_colors] * self.num_objects)
        return spaces.Box(low = low, high = high)

    def random_action(self):
        action = np.random.randint(0, self.action_space.n)
        return action

    def get_current_color(self):
        colors = []
        for idx, obj in self.objects.items():
            colors.append(obj.color)
        return colors