from utils.evaluation import evaluate_model, evaluate_log_prob, evaluate_train_edit_dist
import time
import torch
from utils.load_data import ini_od_dist, load_path_feature, load_link_feature, \
    minmax_normalization, load_train_sample, load_test_traj
from network_env import RoadWorld
from utils.torch import to_device
import numpy as np
import pandas as pd
from model.policy import PolicyCNN
from model.value import ValueCNN
from model.discriminator import DiscriminatorAIRLCNN

import shap

def load_model(model_path):
    model_dict = torch.load(model_path)
    policy_net.load_state_dict(model_dict['Policy'])
    print("Policy Model loaded Successfully")
    value_net.load_state_dict(model_dict['Value'])
    print("Value Model loaded Successfully")
    discrim_net.load_state_dict(model_dict['Discrim'])
    print("Discrim Model loaded Successfully")

cv = 0  # cross validation process [0, 1, 2, 3, 4]
size = 1000  # size of training data [100, 1000, 10000]
gamma = 0.99  # discount factor
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model_p = "../trained_models/airl_CV%d_size%d.pt" % (cv, size)
test_p = "../data/cross_validation/test_CV%d.csv" % cv

"""environment"""
edge_p = "../data/edge.txt"
network_p = "../data/transit.npy"
path_feature_p = "../data/feature_od.npy"
# train_p = "../data/cross_validation/train_CV%d_size%d.csv" % (cv, size)
# test_p = "../data/cross_validation/test_CV%d.csv" % cv
# # test_p = "../data/cross_validation/train_CV%d_size%d.csv" % (cv, size)
# model_p = "../trained_models/airl_CV%d_size%d.pt" % (cv, size)

train_p = "../data/shortest/shortest_paths.csv"
test_p = "../data/shortest/shortest_paths.csv"
# test_p = "../data/cross_validation/train_CV%d_size%d.csv" % (cv, size)
model_p = "../trained_models/shortest/shortest.pt"

"""initialize road environment"""
od_list, od_dist = ini_od_dist(train_p)
env = RoadWorld(network_p, edge_p, pre_reset=(od_list, od_dist))
"""load path-level and link-level feature"""
path_feature, path_max, path_min = load_path_feature(path_feature_p)
edge_feature, link_max, link_min = load_link_feature(edge_p)
path_feature = minmax_normalization(path_feature, path_max, path_min)
path_feature_pad = np.zeros((env.n_states, env.n_states, path_feature.shape[2]))
path_feature_pad[:path_feature.shape[0], :path_feature.shape[1], :] = path_feature
edge_feature = minmax_normalization(edge_feature, link_max, link_min)
edge_feature_pad = np.zeros((env.n_states, edge_feature.shape[1]))
edge_feature_pad[:edge_feature.shape[0], :] = edge_feature

"""define actor and critic"""
policy_net = PolicyCNN(env.n_actions, env.policy_mask, env.state_action,
                    path_feature_pad, edge_feature_pad,
                    path_feature_pad.shape[-1] + edge_feature_pad.shape[-1] + 1,
                    env.pad_idx).to(device)
value_net = ValueCNN(path_feature_pad, edge_feature_pad,
                    path_feature_pad.shape[-1] + edge_feature_pad.shape[-1]).to(device)
discrim_net = DiscriminatorAIRLCNN(env.n_actions, gamma, env.policy_mask,
                                env.state_action, path_feature_pad, edge_feature_pad,
                                path_feature_pad.shape[-1] + edge_feature_pad.shape[-1] + 1,
                                path_feature_pad.shape[-1] + edge_feature_pad.shape[-1],
                                env.pad_idx).to(device)

def evaluate_rewards(test_traj, policy_net, discrim_net, env):
    device = torch.device('cpu')  # Use CPU device
    policy_net.to(device)  # Move policy_net to CPU
    discrim_net.to(device)  # Move discrim_net to CPU

    reward_data = []
    input_features = []

    for episode_idx, episode in enumerate(test_traj):
        des = torch.LongTensor([episode[-1].next_state]).long().to(device)
        for step_idx, x in enumerate(episode):
            state = torch.LongTensor([x.cur_state]).to(device)
            next_state = torch.LongTensor([x.next_state]).to(device)
            action = torch.LongTensor([x.action]).to(device)
            
            # Get the input features
            neigh_path_feature, neigh_edge_feature, path_feature, edge_feature,next_path_feature, next_edge_feature  = discrim_net.get_input_features(state, des, action, next_state)
            
            # Get the log probability of the action
            log_prob = policy_net.get_log_prob(state, des, action)
            
            # Calculate the reward using the discriminator
            reward = discrim_net.forward_with_actual_features(neigh_path_feature, neigh_edge_feature, path_feature, edge_feature, action, log_prob,next_path_feature, next_edge_feature)
            
            reward_data.append({
                'episode': episode_idx,
                'step': step_idx,
                'state': x.cur_state,
                'action': x.action,
                'next_state': x.next_state,
                'reward': reward.item()
            })
            
            # # Store the input features for SHAP analysis
            # input_features.append((neigh_path_feature, neigh_edge_feature, path_feature, edge_feature, action,log_prob,next_path_feature, next_edge_feature))
                      # Flatten the input features before appending
            # input_features.append((
            #     neigh_path_feature.detach().cpu().numpy().flatten(),
            #     neigh_edge_feature.detach().cpu().numpy().flatten(),
            #     path_feature.detach().cpu().numpy().flatten(),
            #     edge_feature.detach().cpu().numpy().flatten(),
            #     action.detach().cpu().numpy().flatten(),
            #     log_prob.detach().cpu().numpy().flatten(),
            #     next_path_feature.detach().cpu().numpy().flatten(),
            #     next_edge_feature.detach().cpu().numpy().flatten()
            # ))
            input_features.append(np.concatenate((
                neigh_path_feature.detach().cpu().numpy().flatten(),
                neigh_edge_feature.detach().cpu().numpy().flatten(),
                path_feature.detach().cpu().numpy().flatten(),
                edge_feature.detach().cpu().numpy().flatten(),
                action.detach().cpu().numpy().flatten(),
                log_prob.detach().cpu().numpy().flatten(),
                next_path_feature.detach().cpu().numpy().flatten(),
                next_edge_feature.detach().cpu().numpy().flatten()
            )))
    # Convert reward_data to a pandas DataFrame
    reward_df = pd.DataFrame(reward_data)

    return input_features, reward_df

def create_shap_explainer(model, input_features):
    def predict_fn(input_features):
        # # Reverse the original dimensions for each input feature
        # neigh_path_feature = torch.tensor(input_features[:, 0].reshape(-1, 4, 12), dtype=torch.float32)
        # neigh_edge_feature = torch.tensor(input_features[:, 1].reshape(-1, 5, 7), dtype=torch.float32)
        # path_feature = torch.tensor(input_features[:, 2].reshape(-1, 12), dtype=torch.float32)
        # edge_feature = torch.tensor(input_features[:, 3].reshape(-1, 5), dtype=torch.float32)
        # action = torch.tensor(input_features[:, 4], dtype=torch.long)
        # log_prob = torch.tensor(input_features[:, 5], dtype=torch.float32)
        # next_path_feature = torch.tensor(input_features[:, 6].reshape(-1, 12), dtype=torch.float32)
        # next_edge_feature = torch.tensor(input_features[:, 7].reshape(-1, 5), dtype=torch.float32)

        # Determine the number of samples
        num_samples = input_features.shape[0]

        # Initialize an array to store the model outputs
        model_outputs = np.zeros(num_samples)

        # Iterate over the samples
        for i in range(num_samples):
            # Extract the features for the current sample
            neigh_path_feature = torch.tensor(input_features[i, :48].reshape(4, 12), dtype=torch.float32)
            neigh_edge_feature = torch.tensor(input_features[i, 48:76].reshape(4, 7), dtype=torch.float32)
            path_feature = torch.tensor(input_features[i, 76:88], dtype=torch.float32)
            edge_feature = torch.tensor(input_features[i, 88:95], dtype=torch.float32)
            action = torch.tensor(input_features[i, 95], dtype=torch.long)
            log_prob = torch.tensor(input_features[i, 96], dtype=torch.float32)
            next_path_feature = torch.tensor(input_features[i, 97:109], dtype=torch.float32)
            next_edge_feature = torch.tensor(input_features[i, 109:116], dtype=torch.float32)

            # Calculate the model output for the current sample
            model_output = model.forward_with_actual_features(
                neigh_path_feature.unsqueeze(0),
                neigh_edge_feature.unsqueeze(0),
                path_feature.unsqueeze(0),
                edge_feature.unsqueeze(0),
                action.unsqueeze(0),
                log_prob.unsqueeze(0),
                next_path_feature.unsqueeze(0),
                next_edge_feature.unsqueeze(0)
            ).detach().numpy()

            # Store the model output for the current sample
            model_outputs[i] = model_output

        return model_outputs
        
        # neigh_path_feature = torch.tensor(input_features[:, 0], dtype=torch.float32)
        # neigh_edge_feature = torch.tensor(input_features[:, 1], dtype=torch.float32)
        # path_feature = torch.tensor(input_features[:, 2], dtype=torch.float32)
        # edge_feature = torch.tensor(input_features[:, 3], dtype=torch.float32)
        # action = torch.tensor(input_features[:, 4], dtype=torch.long)
        # log_prob = torch.tensor(input_features[:, 5], dtype=torch.float32)
        # next_path_feature = torch.tensor(input_features[:, 6], dtype=torch.float32)
        # next_edge_feature = torch.tensor(input_features[:, 7], dtype=torch.float32)
        # return model.forward_with_actual_features(neigh_path_feature, neigh_edge_feature, path_feature, edge_feature, action, log_prob,next_path_feature, next_edge_feature).detach().numpy()

    # # Extract individual components from input_features
    # neigh_path_feature, neigh_edge_feature, path_feature, edge_feature, action, log_prob,next_path_feature, next_edge_feature = zip(*input_features)

    # # Convert each component to a numpy array
    # neigh_path_feature_array = np.array([npf.detach().cpu().numpy() for npf in neigh_path_feature])
    # neigh_edge_feature_array = np.array([nef.detach().cpu().numpy() for nef in neigh_edge_feature])
    # path_feature_array = np.array([pf.detach().cpu().numpy() for pf in path_feature])
    # edge_feature_array = np.array([ef.detach().cpu().numpy() for ef in edge_feature])
    # action_array = np.array([a.detach().cpu().numpy() for a in action])
    # log_prob_array = np.array([lp.detach().cpu().numpy() for lp in log_prob])
    # next_path_feature_array = np.array([npf.detach().cpu().numpy() for npf in next_path_feature])
    # next_edge_feature_array = np.array([nef.detach().cpu().numpy() for nef in next_edge_feature])

    # # Stack the arrays along a new axis to create the background data
    # background_data = np.stack((neigh_path_feature_array,
    #                             neigh_edge_feature_array,
    #                             path_feature_array,
    #                             edge_feature_array,
    #                             action_array,
    #                             log_prob_array,
    #                             next_path_feature_array,
    #                             next_edge_feature_array), axis=1)
    # # # Create a background dataset by selecting a subset of input features
    # # background_size = min(100, len(input_features))
    # # background_data = shap.sample(combined_array, background_size)
    # Stack the input features along a new axis to create the background data
    input_features_array = np.array(input_features)
    # Create the background dataset
    background_data = shap.sample(input_features_array, 10) 

    explainer = shap.KernelExplainer(predict_fn, background_data)
    return explainer

# def analyze_shap_values(explainer, input_features):
#     input_features_array = np.array(input_features)

#     # Create a mapping between feature indices and their corresponding names
#     feature_mapping = {
#         'shortest_distance': [76],
#         'number_of_links': [77],
#         'number_of_left_turn': [78],
#         'number_of_right_turn': [79],
#         'number_of_u_turn': [80],
#         'freq_road_type_1': [81],
#         'freq_road_type_2': [82],
#         'freq_road_type_3': [83],
#         'freq_road_type_4': [84],
#         'freq_road_type_5': [85],
#         'freq_road_type_6': [86],
#     }

#     # Create a list of real feature names
#     real_feature_names = list(feature_mapping.keys())

#     # Select the desired features from the input features array
#     selected_features_indices = [index for indices in feature_mapping.values() for index in indices]
#     print('selected_features_indices',selected_features_indices)
#     selected_features = input_features_array[:, selected_features_indices]
#     print('selected_features',selected_features)

#     # Calculate SHAP values for the selected features
#     shap_values_selected = explainer.shap_values(selected_features)
#     print('shap_values_selected',shap_values_selected)
#     shap_values_selected_squeezed = np.squeeze(shap_values_selected)

#     # Initialize a DataFrame to store the SHAP values for each feature
#     shap_values_df = pd.DataFrame(columns=real_feature_names)

#     # Assign the SHAP values to the corresponding features
#     for feature_name, indices in feature_mapping.items():
#         feature_indices = [selected_features_indices.index(index) for index in indices]
#         shap_values_df[feature_name] = np.sum(shap_values_selected_squeezed[:, feature_indices], axis=1)

#     # Save the SHAP values to a CSV file
#     shap_values_df.to_csv('shap_values_selected.csv', index=False)

#     # Print the path to the saved file for confirmation
#     print("SHAP values for selected features saved to 'shap_values_selected.csv'")

#     # Reshape the SHAP values to match the expected format for summary_plot
#     shap_values_reshaped = shap_values_df.values.reshape(len(real_feature_names), -1)

#     # Plot the SHAP summary plot with real feature names
#     shap.summary_plot(shap_values_reshaped, plot_type="bar", feature_names=real_feature_names)
import matplotlib.pyplot as plt

def analyze_shap_values(explainer, input_features, feature_indices):
    # Convert input_features to a numpy array
    input_features_array = np.array(input_features)

    # Calculate SHAP values using the explainer
    shap_values = explainer.shap_values(input_features_array)
    shap_values_squeezed = np.squeeze(shap_values)

    # Subset the SHAP values to only analyze selected features
    selected_shap_values = shap_values_squeezed[:, feature_indices]

    # Create feature names for the selected columns
    # selected_feature_names = ['Feature ' + str(i) for i in feature_indices]
        # Map selected feature indices to their names
    selected_feature_names = [feature_names_dict[index] for index in feature_indices]

    # Convert selected SHAP values to a DataFrame
    selected_shap_values_df = pd.DataFrame(selected_shap_values, columns=selected_feature_names)

    # Save the DataFrame to a CSV file
    selected_shap_values_df.to_csv('selected_shap_values.csv', index=False)
    print("Selected SHAP values saved to 'selected_shap_values.csv'")

    # Plot the SHAP summary plot with selected feature names
    shap.summary_plot(selected_shap_values, input_features_array[:, feature_indices], plot_type="bar", feature_names=selected_feature_names, show=False)
    # Save the plot as an image file
    plt.savefig('selected_shap_summary_plot.png')
    plt.close()
    print("Selected SHAP summary plot saved to 'selected_shap_summary_plot.png'")

    # # Create feature names for the columns
    # feature_names = ['Feature ' + str(i) for i in range(shap_values_squeezed.shape[1])]

    # # Convert SHAP values to a DataFrame
    # shap_values_df = pd.DataFrame(shap_values_squeezed, columns=feature_names)

    # # Save the DataFrame to a CSV file
    # shap_values_df.to_csv('shap_values.csv', index=False)

    # # Print the path to the saved file for confirmation
    # print("SHAP values saved to 'shap_values.csv'")

    # # Plot the SHAP summary plot with feature names
    # shap.summary_plot(shap_values_squeezed, input_features_array, plot_type="bar", feature_names=feature_names, show=False)
    # # Save the plot as an image file
    # plt.savefig('shap_summary_plot.png')
    # plt.close()

    # Notify user of plot save
    print("SHAP summary plot saved to 'shap_summary_plot.png'")




"""Evaluate rewards"""
test_trajs = env.import_demonstrations_step(test_p)
input_features, reward_df = evaluate_rewards(test_trajs, policy_net, discrim_net, env)
# print('input_features', input_features)

"""Create SHAP explainer"""
explainer = create_shap_explainer(discrim_net, input_features)

"""Analyze SHAP values"""
feature_names_dict = {
    76: 'shortest_distance',
    77: 'number_of_links',
    78: 'number_of_left_turn',
    79: 'number_of_right_turn',
    80: 'number_of_u_turn',
    81: 'freq_road_type_1',
    82: 'freq_road_type_2',
    83: 'freq_road_type_3',
    84: 'freq_road_type_4',
    85: 'freq_road_type_5',
    86: 'freq_road_type_6',
    # 88: 'link_length',
    # 89: 'road_type_1',
    # 90: 'road_type_2',
    # 91: 'road_type_3',
    # 92: 'road_type_4',
    # 93: 'road_type_5',
    # 94: 'road_type_6',
}

selected_feature_indices = [76,77,78,79,80,81,82,83,84,85,86,] #88,89,90,91,92,93,94
analyze_shap_values(explainer, input_features[0:20],selected_feature_indices )



# def analyze_shap_values(explainer, input_features):
#     input_features_array = np.array(input_features)

#     # Create a mapping between feature indices and their corresponding names
#     feature_mapping = {
#         'shortest_distance': [76],
#         'number_of_links': [77],
#         'number_of_left_turn': [78],
#         'number_of_right_turn': [79],
#         'number_of_u_turn': [80],
#         'freq_road_type_1': [81],
#         'freq_road_type_2': [82],
#         'freq_road_type_3': [83],
#         'freq_road_type_4': [84],
#         'freq_road_type_5': [85],
#         'freq_road_type_6': [86],
#     }

#     # Create a list of real feature names
#     real_feature_names = list(feature_mapping.keys())

#     # Select the desired features from the input features array
#     selected_features_indices = [index for indices in feature_mapping.values() for index in indices]
#     selected_features = input_features_array[:, selected_features_indices]

#     # Calculate SHAP values for the selected features
#     shap_values_selected = explainer.shap_values(selected_features)
#     shap_values_selected_squeezed = np.squeeze(shap_values_selected)

#     # Initialize a DataFrame to store the SHAP values for each feature
#     shap_values_df = pd.DataFrame(columns=real_feature_names)

#     # Assign the SHAP values to the corresponding features
#     for feature_name, indices in feature_mapping.items():
#         feature_indices = [selected_features_indices.index(index) for index in indices]
#         shap_values_df[feature_name] = np.sum(shap_values_selected_squeezed[:, feature_indices], axis=1)

#     # Save the SHAP values to a CSV file
#     shap_values_df.to_csv('shap_values_selected.csv', index=False)

#     # Print the path to the saved file for confirmation
#     print("SHAP values for selected features saved to 'shap_values_selected.csv'")

#     # Plot the SHAP summary plot with real feature names
#     shap.summary_plot(shap_values_df.values, plot_type="bar", feature_names=real_feature_names)