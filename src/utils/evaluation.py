import editdistance
from nltk.translate.bleu_score import sentence_bleu
from scipy.spatial import distance
from nltk.translate.bleu_score import SmoothingFunction
import numpy as np
import torch
import pandas as pd
import csv

smoothie = SmoothingFunction().method1
device = torch.device("cpu")


def create_od_set(test_trajs):
    test_od_dict = {}
    for i in range(len(test_trajs)):
        if (test_trajs[i][0], test_trajs[i][-1]) in test_od_dict.keys():
            test_od_dict[(test_trajs[i][0], test_trajs[i][-1])].append(i)
        else:
            test_od_dict[(test_trajs[i][0], test_trajs[i][-1])] = [i]
    return test_od_dict


def evaluate_edit_dist(test_trajs, learner_trajs, test_od_dict):
    edit_dist_list = []
    for od in test_od_dict.keys():
        idx_list = test_od_dict[od]
        test_od_trajs = set(['_'.join(test_trajs[i]) for i in idx_list])
        test_od_trajs = [traj.split('_') for traj in test_od_trajs]
        learner_od_trajs = [learner_trajs[i] for i in idx_list]
        for learner in learner_od_trajs:
            min_edit_dist = 1.0
            for test in test_od_trajs:
                edit_dist = editdistance.eval(test, learner) / len(test)
                min_edit_dist = edit_dist if edit_dist < min_edit_dist else min_edit_dist
            edit_dist_list.append(min_edit_dist)
    return np.mean(edit_dist_list)


def evaluate_bleu_score(test_trajs, learner_trajs, test_od_dict):
    # print('test_trajs',test_trajs[0])
    # print('learner_trajs',learner_trajs[0])
    # print('test_trajs',test_trajs[1])
    # print('learner_trajs',learner_trajs[1])
    # print('test_trajs',test_trajs[2])
    # print('learner_trajs',learner_trajs[2])
    bleu_score_list = []
    for od in test_od_dict.keys():
        idx_list = test_od_dict[od]
        # get unique reference
        test_od_trajs = set(['_'.join(test_trajs[i]) for i in idx_list])
        test_od_trajs = [traj.split('_') for traj in test_od_trajs]
        learner_od_trajs = [learner_trajs[i] for i in idx_list]
        for learner in learner_od_trajs:
            # print(test_od_trajs)
            # print(learner)
            bleu_score = sentence_bleu(test_od_trajs, learner, smoothing_function=smoothie)
            bleu_score_list.append(bleu_score)

        # Save test and learner trajectories side by side in a CSV file
    with open('trajectories.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Test Trajectory', 'Learner Trajectory'])
        
        for test_traj, learner_traj in zip(test_trajs, learner_trajs):
            test_traj_str = '_'.join(test_traj)
            learner_traj_str = '_'.join(learner_traj)
            csv_writer.writerow([test_traj_str, learner_traj_str])

    return np.mean(bleu_score_list)


def evaluate_dataset_dist(test_trajs, learner_trajs):
    test_trajs_str = ['_'.join(traj) for traj in test_trajs]
    # print('test trajs str len', len(test_trajs_str))
    test_trajs_set = set(test_trajs_str)
    # print('test trajs set len', len(test_trajs_set))
    test_trajs_dict = dict(zip(list(test_trajs_set), range(len(test_trajs_set))))
    test_trajs_label = [test_trajs_dict[traj] for traj in test_trajs_str]
    test_trajs_label.append(0)
    test_p = np.histogram(test_trajs_label)[0] / len(test_trajs_label)

    pad_idx = len(test_trajs_set)
    learner_trajs_str = ['_'.join(traj) for traj in learner_trajs]
    learner_trajs_label = [test_trajs_dict.get(traj, pad_idx) for traj in learner_trajs_str]
    learner_p = np.histogram(learner_trajs_label)[0] / len(learner_trajs_label)
    return distance.jensenshannon(test_p, learner_p)


def evaluate_log_prob(test_traj, model):
    log_prob_list = []
    for episode in test_traj:
        des = torch.LongTensor([episode[-1].next_state]).long().to(device)
        log_prob = 0
        for x in episode:
            with torch.no_grad():
                next_prob = torch.log(model.get_action_prob(torch.LongTensor([x.cur_state]).to(device), des)).squeeze()
            next_prob_np = next_prob.detach().cpu().numpy()

            #change
            log_prob += next_prob_np[int(x.action)]
        log_prob_list.append(log_prob)
    print(np.mean(log_prob_list))
    return np.mean(log_prob_list)


def evaluate_train_edit_dist(train_traj, learner_traj):
    """This function is used to keep the training epoch with the best edit distance performance on the training data"""
    test_od_dict = create_od_set(train_traj)
    edit_dist = evaluate_edit_dist(train_traj, learner_traj, test_od_dict)
    return edit_dist


def evaluate_metrics(test_traj, learner_traj):
    test_od_dict = create_od_set(test_traj)
    edit_dist = evaluate_edit_dist(test_traj, learner_traj, test_od_dict)
    bleu_score = evaluate_bleu_score(test_traj, learner_traj, test_od_dict)
    js_dist = evaluate_dataset_dist(test_traj, learner_traj)
    print('edit dist', edit_dist)
    print('bleu score', bleu_score)
    print('js distance', js_dist)
    return edit_dist, bleu_score, js_dist


def evaluate_model(target_od, target_traj, model, env, n_link=424):
    state_ts = torch.from_numpy(np.arange(n_link)).long().to(device)
    target_o, target_d = target_od[:, 0].tolist(), target_od[:, 1].tolist()
    learner_traj = []

    # Move model parameters to CPU
    for param in model.parameters():
        param.data = param.data.to(device)

    """compute transition matrix for the first OD pair"""
    curr_ori, curr_des = target_o[0], target_d[0]
    des_ts = (torch.ones_like(state_ts) * curr_des).to(device)
    action_prob = model.get_action_prob(state_ts, des_ts).detach().cpu().numpy()  # 714, 8
    state_action = env.state_action[:-1]
    action_prob[state_action == env.pad_idx] = 0.0
    transit_prob = np.zeros((n_link, n_link))
    from_st, ac = np.where(state_action != env.pad_idx)
    to_st = state_action[state_action != env.pad_idx]
    transit_prob[from_st, to_st] = action_prob[from_st, ac]
    """compute sample path for the first OD pair"""
    sample_path = [str(curr_ori)]
    curr_state = curr_ori
    for _ in range(50):
        if curr_state == curr_des: break
        next_state = np.argmax(transit_prob[curr_state])
        sample_path.append(str(next_state))
        curr_state = next_state
    learner_traj.append(sample_path)
    for ori, des in zip(target_o[1:], target_d[1:]):
        if des == curr_des:
            if ori == curr_ori:
                learner_traj.append(sample_path)
                continue
            else:
                curr_ori = ori
        else:
            curr_ori, curr_des = ori, des
            des_ts = (torch.ones_like(state_ts) * curr_des).to(device)
            action_prob = model.get_action_prob(state_ts, des_ts).detach().cpu().numpy()  # 714, 8
            state_action = env.state_action[:-1]
            action_prob[state_action == env.pad_idx] = 0.0
            transit_prob = np.zeros((n_link, n_link))
            from_st, ac = np.where(state_action != env.pad_idx)
            to_st = state_action[state_action != env.pad_idx]
            transit_prob[from_st, to_st] = action_prob[from_st, ac]
        sample_path = [str(curr_ori)]
        curr_state = curr_ori
        for _ in range(50):
            if curr_state == curr_des: break
            next_state = np.argmax(transit_prob[curr_state])
            sample_path.append(str(next_state))
            curr_state = next_state
        learner_traj.append(sample_path)
    evaluate_metrics(target_traj, learner_traj)

    return learner_traj  # Return the generated trajectories


# def save_learner_trajectories(learner_trajs, env):
#     # Create a list to store the trajectory data
#     trajectory_data = []

#     for episode_idx, episode_traj in enumerate(learner_trajs):
#         curr_state = int(episode_traj[0])
#         for step_idx, next_state_str in enumerate(episode_traj[1:], start=1):
#             next_state = int(next_state_str)
#             action = np.where(env.state_action[curr_state] == next_state)[0][0]

#             # Append the trajectory data to the list
#             trajectory_data.append({
#                 'episode': episode_idx,
#                 'step': step_idx,
#                 'learner_trajectory': '_'.join(episode_traj[:step_idx+1]),
#                 'action': action
#             })

#             curr_state = next_state

#     # Save the trajectory data to a CSV file
#     trajectory_df = pd.DataFrame(trajectory_data)
#     trajectory_df.to_csv('learner_trajectories.csv', index=False)

