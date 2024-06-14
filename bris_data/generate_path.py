import pandas as pd
import numpy as np
from yen_ksp import ksp_yen
from context_feature_computation import construct_graph
import random

node_p = "node.txt"
edge_p = "edge.txt"
network_p = "transit.npy"

graph = construct_graph(edge_p, network_p)

# Read the edge data from the "edge.txt" file
edge_data = pd.read_csv(edge_p)

# Extract the unique edge IDs
edge_ids = edge_data['n_id'].unique()

# Load the transit data from the "transit.npy" file
transit_data = np.load(network_p)

# Create an empty list to store the shortest paths
shortest_paths = []

def random_walk(ori, transit_data):
    current_link = ori
    path = [current_link]
    
    while True:
        mask = transit_data[:, 0] == current_link
        if not mask.any():
            break  # No next link found, end the walk
        
        next_link = transit_data[mask, 2][0]
        path.append(next_link)
        current_link = next_link
    
    return path

num_paths = 1  # Number of random paths to collect
print("Starting random walk and shortest path finding for {} paths.".format(num_paths))

for i in range(num_paths):
    print(f"Progress: {i+1}/{num_paths} paths processed ({((i+1)/num_paths)*100:.2f}%)")
    
    # Randomly select an origin edge
    ori = random.choice(edge_ids)
    
    # Perform random walk until there is no way or a dead end
    path = random_walk(ori, transit_data)
    des = path[-1]
    
    # Find the shortest path using ksp_yen
    shortest_path_info = ksp_yen(graph, ori, des, 1)
    
    if shortest_path_info:
        shortest_path = "_".join(map(str, map(int, shortest_path_info[0]['path'])))
        shortest_path_length = shortest_path_info[0]['cost']
    else:
        shortest_path = ""
        shortest_path_length = 0
    
    shortest_paths.append([ori, des, shortest_path, shortest_path_length])

print("Processing complete.")

# Create a DataFrame for shortest paths
shortest_paths_df = pd.DataFrame(shortest_paths, columns=["ori", "des", "shortest_path", "path_length"])

# Convert numeric columns to appropriate data types
shortest_paths_df[["ori", "des", "path_length"]] = shortest_paths_df[["ori", "des", "path_length"]].astype(int)

# Save the DataFrame to a CSV file
shortest_paths_df.to_csv("shortest_paths.csv", index=False)


# ----------------------------------------------------------------------------------------------
# import pandas as pd
# import numpy as np
# from yen_ksp import ksp_yen
# from context_feature_computation import construct_graph
# import random

# node_p = "node.txt"
# edge_p = "edge.txt"
# network_p = "transit.npy"

# graph = construct_graph(edge_p, network_p)

# # Read the edge data from the "edge.txt" file
# edge_data = pd.read_csv(edge_p)

# # Extract the unique edge IDs
# edge_ids = edge_data['n_id'].unique()

# # Load the transit data from the "transit.npy" file
# transit_data = np.load(network_p)

# # Create empty lists to store the path results
# all_paths = []
# shortest_paths = []

# def random_walk(ori, transit_data):
#     current_link = ori
#     path = [current_link]
    
#     while True:
#         mask = transit_data[:, 0] == current_link
#         if not mask.any():
#             break  # No next link found, end the walk
        
#         next_link = transit_data[mask, 2][0]
#         path.append(next_link)
#         current_link = next_link
    
#     return path

# def find_all_paths(ori, des, transit_data):
#     paths = []
#     stack = [(ori, [ori])]
    
#     while stack:
#         current_link, current_path = stack.pop()
        
#         if current_link == des:
#             paths.append(current_path)
#             continue
        
#         mask = transit_data[:, 0] == current_link
#         if not mask.any():
#             continue
        
#         for next_link in transit_data[mask, 2]:
#             if next_link not in current_path:
#                 stack.append((next_link, current_path + [next_link]))
    
#     return paths

# num_paths = 2  # Number of random paths to collect
# print("Starting random walk and path finding for {} paths.".format(num_paths))

# for i in range(num_paths):
#     print(f"Progress: {i+1}/{num_paths} paths processed ({((i+1)/num_paths)*100:.2f}%)")
    
#     # Randomly select an origin edge
#     ori = random.choice(edge_ids)
    
#     # Perform random walk until there is no way or a dead end
#     path = random_walk(ori, transit_data)
#     des = path[-1]
    
#     # Find all possible paths between the origin and destination
#     paths = find_all_paths(ori, des, transit_data)
    
#     for path in paths:
#         path_str = "_".join(map(str, map(int, path)))
#         path_length = len(path) - 1
#         all_paths.append([ori, des, path_str, path_length])
    
#     # Find the shortest path using ksp_yen
#     shortest_path_info = ksp_yen(graph, ori, des, 1)
    
#     if shortest_path_info:
#         shortest_path = "_".join(map(str, map(int, shortest_path_info[0]['path'])))
#         shortest_path_length = shortest_path_info[0]['cost']
#     else:
#         shortest_path = ""
#         shortest_path_length = 0
    
#     shortest_paths.append([ori, des, shortest_path, shortest_path_length])

# print("Processing complete.")

# # Create DataFrames for all paths and shortest paths
# all_paths_df = pd.DataFrame(all_paths, columns=["ori", "des", "path", "len"])
# shortest_paths_df = pd.DataFrame(shortest_paths, columns=["ori", "des", "shortest_path", "path_length"])

# # Convert numeric columns to appropriate data types
# all_paths_df[["ori", "des", "len"]] = all_paths_df[["ori", "des", "len"]].astype(int)
# shortest_paths_df[["ori", "des", "path_length"]] = shortest_paths_df[["ori", "des", "path_length"]].astype(int)

# # Save the DataFrames to CSV files
# all_paths_df.to_csv("all_paths.csv", index=False)
# shortest_paths_df.to_csv("shortest_paths.csv", index=False)


#--------------------------------------------------------------------------------------------------

# import pandas as pd
# import numpy as np
# from yen_ksp import ksp_yen
# from context_feature_computation import construct_graph
# from itertools import permutations
# import random

# node_p = "node.txt"
# edge_p = "edge.txt"
# network_p = "transit.npy"

# graph = construct_graph(edge_p, network_p)

# # Read the edge data from the "edge.txt" file
# edge_data = pd.read_csv(edge_p)

# # Extract the unique edge IDs
# edge_ids = edge_data['n_id'].unique()

# # Generate all possible combinations of origin and destination edges
# od_pairs = list(permutations(edge_ids, 2))

# # Sample 10 random origin-destination pairs
# random_od_pairs = random.sample(od_pairs, 20)
# ori_list, des_list = zip(*random_od_pairs)

# # # Extract the "ori" and "des" lists from the combinations
# # ori_list, des_list = zip(*od_pairs)

# # Load the transit data from the "transit.npy" file
# transit_data = np.load(network_p)

# # Create empty lists to store the path results
# all_paths = []
# shortest_paths = []

# def find_path(ori, des, transit_data):
#     current_link = ori
#     path = [current_link]
    
#     while current_link != des:
#         mask = transit_data[:, 0] == current_link
#         if not mask.any():
#             return None  # No valid path found
        
#         next_link = transit_data[mask, 2][0]
#         path.append(next_link)
#         current_link = next_link
    
#     return path

# total_pairs = len(ori_list)
# print("Starting path finding for {} pairs.".format(total_pairs))

# # Iterate over each pair of origin and destination edges
# for i, (ori, des) in enumerate(zip(ori_list, des_list)):
#     if i % 2 == 0:  # update the progress every 100 iterations
#         print(f"Progress: {i}/{total_pairs} pairs processed ({(i/total_pairs)*100:.2f}%)")
#     # Find a valid path between the origin and destination edges
#     print('ori', ori)
#     print('des', des)
#     path = find_path(ori, des, transit_data)
    
#     if path is None:
#         continue  # Skip if no valid path exists
    
#     # Convert the path to a string representation
#     path_str = "_".join(map(str, map(int, path)))
#     path_length = len(path) - 1
    
#     # Append the path to the all_paths list
#     all_paths.append([ori, des, path_str, path_length])
    
#     # Find the shortest path using ksp_yen
#     shortest_path_info = ksp_yen(graph, ori, des, 1)
    
#     if shortest_path_info:
#         shortest_path = "_".join(map(str, map(int, shortest_path_info[0]['path'])))
#         shortest_path_length = shortest_path_info[0]['cost']
#     else:
#         shortest_path = ""
#         shortest_path_length = 0
    
#     # Append the shortest path to the shortest_paths list
#     shortest_paths.append([ori, des, shortest_path, shortest_path_length])

# print("Processing complete.")

# # Create DataFrames for all paths and shortest paths
# all_paths_df = pd.DataFrame(all_paths, columns=["ori", "des", "path", "len"])
# shortest_paths_df = pd.DataFrame(shortest_paths, columns=["ori", "des", "shortest_path", "path_length"])

# # Convert numeric columns to appropriate data types
# all_paths_df[["ori", "des", "len"]] = all_paths_df[["ori", "des", "len"]].astype(int)
# shortest_paths_df[["ori", "des", "path_length"]] = shortest_paths_df[["ori", "des", "path_length"]].astype(int)

# # Save the DataFrames to CSV files
# all_paths_df.to_csv("all_paths.csv", index=False)
# shortest_paths_df.to_csv("shortest_paths.csv", index=False)