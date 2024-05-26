from utils.evaluation import evaluate_model, evaluate_log_prob
import time
import torch
from utils.load_data import load_path_feature, load_link_feature, minmax_normalization, load_test_traj
from network_env import RoadWorld
import numpy as np
from model.policy import PolicyCNN

def load_model(model_path):
    CNNMODEL.load_state_dict(torch.load(model_p, map_location=torch.device('cpu')))
    print("BC Model loaded Successfully")

cv = 0  # cross validation process [0, 1, 2, 3, 4]
size = 1000  # size of training data [100, 1000, 10000]
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model_p = "C:/AI/rcm-airl-my-data/RCM-AIRL/trained_models/pnuema/bc_CV%d_size%d.pt" % (cv, size)
test_p = "C:/AI/rcm-airl-my-data/RCM-AIRL/data/cross_validation/test_CV%d.csv" % cv

"""environment"""
edge_p = "C:/AI/rcm-airl-my-data/RCM-AIRL/data/edge.txt"
network_p = "C:/AI/rcm-airl-my-data/RCM-AIRL/data/transit.npy"
path_feature_p = "C:/AI/rcm-airl-my-data/RCM-AIRL/data/feature_od.npy"

"""initialize road environment"""
env = RoadWorld(network_p, edge_p)

"""load path-level and link-level feature"""
path_feature, path_max, path_min = load_path_feature(path_feature_p)
edge_feature, edge_max, edge_min = load_link_feature(edge_p)
path_feature = minmax_normalization(path_feature, path_max, path_min)
path_feature_pad = np.zeros((env.n_states, env.n_states, path_feature.shape[2]))
path_feature_pad[:path_feature.shape[0], :path_feature.shape[1], :] = path_feature
edge_feature = minmax_normalization(edge_feature, edge_max, edge_min)
edge_feature_pad = np.zeros((env.n_states, edge_feature.shape[1]))
edge_feature_pad[:edge_feature.shape[0], :] = edge_feature

"""define policy network"""
CNNMODEL = PolicyCNN(env.n_actions, env.policy_mask, env.state_action, path_feature_pad, edge_feature_pad,
                     path_feature_pad.shape[-1] + edge_feature_pad.shape[-1] + 1, env.pad_idx).to(device)
CNNMODEL.to_device(device)

"""Evaluate model"""
load_model(model_p)
test_trajs, test_od = load_test_traj(test_p)
test_od = torch.from_numpy(test_od).long().to(device)
start_time = time.time()
generated_trajs = evaluate_model(test_od, test_trajs, CNNMODEL, env)
print('test time', time.time() - start_time)

"""Print the action taken at each step for each episode"""
for episode_idx, episode in enumerate(generated_trajs):
    print(f"Episode {episode_idx + 1}:")
    for step_idx, node in enumerate(episode):
        if step_idx < len(episode) - 1:
            current_node = int(node)
            next_node = int(episode[step_idx + 1])
            action_list = env.get_action_list(current_node)
            for action in action_list:
                if env.get_state_transition(current_node, action) == next_node:
                    action_taken = action
                    break
            print(f"Step {step_idx + 1}: Current State: {current_node}, Action Taken: {action_taken}, Next State: {next_node}")
        else:
            print(f"Step {step_idx + 1}: Current State: {int(node)} (Destination)")
    print()

"""Evaluate log prob"""
test_trajs = env.import_demonstrations_step(test_p)
evaluate_log_prob(test_trajs, CNNMODEL)