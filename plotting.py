import numpy as np
import torch
import json
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F
import os
import pickle
import textwrap
from tabulate import tabulate
from PIL import Image

#funzionamento attuale SEGNO (NODE)
# def forward(x):
#     nlayers = 7 #iperparametro
#     dt = 1/nlayers
#     #in questo caso T è come se fosse 1 perchè è sempre uguale(10 timestep di distanza) 
#     for i in range(nlayers):
#         x = step(x,dt)

# #funzionamento SEGNO (NODE) generalizzando rispetto a T variabili
# def forward(x, T): #T : numero di timesteps tra input e predizione (default 10)
#     nlayers = 7 #iperparametro
#     #in questo modo si tiene traccia di quando T varia facendo un numero diverso di steps
#     num_steps = nlayers * T  #riduci nlayers (1)
#     #num_steps = T  #ha senso usare solo T non considerando l'iperaparametro nlayers? 
#     dt = 1/num_steps 
#     for i in range(num_steps):
#         x = step(x,dt)

#funzionamento SEGNO (NODE) generalizzando rispetto a T variabili
#opzione 2:mantenere num_steps=n_layers
# def forward(x, T): #T : numero di timesteps tra input e predizione (default 10)
#     nlayers = 7 #iperparametro
#     num_steps = nlayers * T #ha senso usare solo T non considerando l'iperaparametro nlayers? 
#     dt = 1/num_steps #
#     for i in range(num_steps):
#         x = step(x,dt)

def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def repeat_elements_to_exact_shape(tensor_list, n):
    L = len(tensor_list)                    # Number of elements in the list
    repeats_per_element = n // L            # Base number of repeats per element
    remaining_repeats = n % L               # Extra repeats needed to reach exactly `n`
    
    # Repeat each tensor in the list `repeats_per_element` times
    repeated_tensors = [tensor.repeat(repeats_per_element, *[1] * (tensor.dim() - 1)) for tensor in tensor_list]
    
    # Add extra repeats for the first `remaining_repeats` elements in the list
    extra_repeats = [tensor_list[-1] for i in range(remaining_repeats)]
    
    # Concatenate all repeated tensors and the extra repeats
    final_tensor = torch.cat(repeated_tensors + extra_repeats, dim=0)
    
    return final_tensor

#def cumulative_random_tensor_indices_np(n, start, end):
    # Generate the cumulative numpy array as before

# def cumulative_random_tensor_indices(n, start, end):
#     return cumulative_tensor,  random_array


class Obj:
    def __init__(self, name, age):
        self.name = name  # Initialize the name attribute
        self.age = age    # Initialize the age attribute

    def greet(self):
        # Method to greet the person
        print(f"Hello, my name is {self.name}.")

    def display_info(self):
        # Method to display person's information
        print(f"Name: {self.name}")
        print(f"Age: {self.age}")

class Person:
    def __init__(self, name, age):
        self.name = name  # Initialize the name attribute
        self.age = age    # Initialize the age attribute
        self.obj = Obj("al",12)

    def greet(self):
        # Method to greet the person
        print(f"Hello, my name is {self.name}.")

    def display_info(self):
        # Method to display person's information
        self.obj.age = 16
        print(f"Name: {self.name}")
        print(f"Age: {self.age}")
        print(f"Nobj: {self.obj.name}")
        print(f"Aobj: {self.obj.age}")
        
def repeat_elements_to_exact_shape(tensor_list, n,outdims=None):
    L = len(tensor_list)                    # Number of elements in the list
    repeats_per_element = n // L            # Base number of repeats per element
    remaining_repeats = n % L    
    print("rrr")
    print(remaining_repeats)           # Extra repeats needed to reach exactly `n`
    print(tensor_list[-1].shape)
    outdims = outdims if outdims is not None else tensor_list[0].dim()
    # Repeat each tensor in the list `repeats_per_element` times
    repeated_tensors = [tensor.repeat(repeats_per_element,*[1] * (outdims - 1)) for tensor in tensor_list] #
    
    # Add extra repeats for the first `remaining_repeats` elements in the list
    extra_repeats = [tensor_list[-1] for i in range(remaining_repeats)]
    print(repeated_tensors,extra_repeats)
    final_list = repeated_tensors + extra_repeats if remaining_repeats >0 else repeated_tensors
    for i, tensor in enumerate(final_list):
        if tensor.dim() == 0:
            print(tensor)
            tensor = tensor.reshape(-1,)
            print(tensor.shape)
    print(final_list)
    final_tensor = torch.cat(final_list, dim=0)
    print(final_tensor)
    return final_tensor

def random_ascending_tensor(length, min_value=1, max_value=15):
    """
    Generates a random tensor of specified length, in ascending order, with no duplicates.
    
    Args:
    - length (int): Desired length of the output tensor.
    - min_value (int): Minimum possible value (inclusive).
    - max_value (int): Maximum possible value (inclusive).
    
    Returns:
    - torch.Tensor: A 1-D tensor with unique, ascending random values.
    """
    # Generate a sorted list of unique random values
    unique_values = torch.randperm(max_value - min_value + 1)[:length] + min_value
    unique_values = unique_values.sort().values  # Sort the values in ascending order
    
    return unique_values

def cumulative_random_tensor_indices(N, start, end, MAX=100):
    """
    Generates a random integer tensor and adjusts it so that its cumulative sum equals MAX.
    
    Args:
    - N (int): Length of the tensor.
    - start (int): Minimum value for random integers (inclusive).
    - end (int): Maximum value for random integers (exclusive).
    - MAX (int): Desired cumulative sum target (default is 100).
    
    Returns:
    - torch.Tensor: The adjusted random tensor.
    - torch.Tensor: The cumulative sum of the adjusted random tensor.
    """
    # Step 1: Generate a random integer tensor of size N within [start, end)
    random_array = torch.randint(start, end, (N,))
    
    # Step 2: Calculate the initial sum and scale values to approach MAX
    initial_sum = random_array.sum().item()
    
    # If initial sum is zero, reinitialize random_array to avoid division by zero
    while initial_sum == 0:
        random_array = torch.randint(start, end, (N,))
        initial_sum = random_array.sum().item()

    # Scale values to approximate the sum to MAX
    scaled_array = torch.round((random_array.float() / initial_sum) * MAX).int()

    # Step 3: Correct any rounding difference to ensure sum equals MAX
    diff = MAX - scaled_array.sum().item()
    
    if diff != 0:
        # Randomly adjust elements to make the sum exactly MAX
        indices = torch.randperm(N)
        for i in indices:
            # Ensure values stay within the [start, end) range after adjustment
            if start <= scaled_array[i] + diff < end:
                scaled_array[i] += diff
                break  # Exit once sum is corrected
    
    # Step 4: Calculate cumulative sum tensor
    cumulative_tensor = torch.cumsum(scaled_array, dim=0)

    return scaled_array, cumulative_tensor

def compute_correlation_test(tensor1, tensor2):
    """
    Computes the correlation between two tensors of shape (num_samples, T, D) 
    such that the resulting tensor has shape (T,).

    Args:
        tensor1 (torch.Tensor): First tensor of shape (num_samples, T, D).
        tensor2 (torch.Tensor): Second tensor of shape (num_samples, T, D).

    Returns:
        torch.Tensor: A tensor of shape (T,) containing the correlation at each timestep.
    """
    # Reshape tensors to (num_samples * D, T)
    num_samples, T, D = tensor1.shape
    reshaped1 = tensor1.permute(1, 0, 2).reshape(T, -1)  # Shape: (T, num_samples * D)
    reshaped2 = tensor2.permute(1, 0, 2).reshape(T, -1)  # Shape: (T, num_samples * D)
    
    # Compute the mean and standard deviation for each timestep
    mean1 = reshaped1.mean(dim=1, keepdim=True)  # Shape: (T, 1)
    mean2 = reshaped2.mean(dim=1, keepdim=True)  # Shape: (T, 1)
    std1 = reshaped1.std(dim=1, keepdim=True)    # Shape: (T, 1)
    std2 = reshaped2.std(dim=1, keepdim=True)    # Shape: (T, 1)
    
    # Normalize the tensors
    normalized1 = (reshaped1 - mean1) / (std1 + 1e-8)  # Shape: (T, num_samples * D)
    normalized2 = (reshaped2 - mean2) / (std2 + 1e-8)  # Shape: (T, num_samples * D)
    cent1 = (reshaped1 - mean1) # Shape: (T, num_samples * D)
    cent2 = (reshaped2 - mean2)  # Shape: (T, num_samples * D)
    std_x = torch.sqrt((cent1 ** 2).sum(dim=1))
    std_y = torch.sqrt((cent2 ** 2).sum(dim=1))
    #print((std1-std_x).sum(),(std2-std_y).sum())
    print(cent1.shape)
    covariance = (cent1*cent2).sum(dim=1)
    print(covariance.shape)
    # Compute correlation for each timestep
    correlation = (normalized1 * normalized2).mean(dim=1)  # Shape: (T,)
    corr = covariance / (std_x * std_y)
    print(corr.shape,corr)
    return correlation

def bar_plot_models_multiple_metrics(model1_values, model2_values):
    
    metrics = ['A-MSE-traj', 'A-MSE-t1', 'AVG_NSTEPS_CORR'] # change metrics

    #[0.25, 0.35, 0.15]
    #[0.22, 0.30, 0.18]

    x = np.arange(len(metrics)) 
    width = 0.2  

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, model1_values, width, label='EGNO')
    bars2 = ax.bar(x + width/2, model2_values, width, label='SEGNO')

    # Add text annotations for the values
    for bar in bars1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f'{yval:.2f}', ha='center')
    for bar in bars2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f'{yval:.2f}', ha='center')

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Loss Values')
    ax.set_title('Comparison of Different Losses for Both Models')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    plt.tight_layout()
    #plt.show()
    plt.savefig("bar_plot_test.png")


def plot_trajectory_losses(t_losses, path):

    #take the mean
    #losses = np.mean(t_losses,axis=0)

    # Generate an array for timesteps (assuming 0, 1, 2, ...)
    timesteps = np.arange(1,len(t_losses)+1)

    plt.style.use('bmh')

    # Plot the loss values over time
    plt.plot(timesteps, t_losses, marker='o', linestyle='-', color='b')
    plt.yscale('log')
    #plt.yticks([10, 100, 1000, 10000, 100000])

    # Adding titles and labels
    plt.title("Loss Over Timesteps")
    plt.xlabel("Timestep")
    plt.ylabel("Loss Value")
    # Annotate each point with its y-value
    for i, (x_val, y_val) in enumerate(zip(timesteps[::3], t_losses[::3])):
        plt.text(x_val, y_val, f'{y_val:.4f}', fontsize=10, ha='right')
    #plt.xticks(timesteps)

    #plt.savefig('traj_loss_polar-pyramid-17.png', dpi=300, bbox_inches='tight')
    # Show the plot
    plt.savefig(path)
    plt.close()
    #plt.show()

def plot_graphs_from_json(folder_path, model):
    """
    Loads all JSON files in the given folder and plots graphs, saving them with the same name as the JSON file but as .png.

    Args:
        folder_path (str): The path to the folder containing JSON files.
    """
    # Ensure the folder exists
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist.")
        return

    # List all JSON files in the folder
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

    if not json_files:
        print("No JSON files found in the folder.")
        return

    for json_file in json_files:
        json_path = os.path.join(folder_path, json_file)

        # Load the JSON file
        with open(json_path, 'r') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                print(f"Could not decode {json_file}, skipping.")
                continue

        #print(len(data["traj_loss"]),len(data["traj_loss"][0]),len(data["traj_loss"][0][0]))
        if model == "EGNO":
            cut = 20
            d = data['traj_loss'][-1][:]#[:cut]
            #print(len(d))
        else:
            cut = 10
            d = data['traj_loss'][-1][:]#[:cut]

        
        min = 100
        for l in d:
            if len(l) < min:
                min = len(l)
        #print(min)
        # d = d[:][:cut]
        # print(len(d))
        #d = torch.tensor(d)
        #print(d.shape)
        cut = min
        nd = [l[:cut] for l in d]
        traj = np.array(nd)
        b_avg = True
        if b_avg:
            traj = np.mean(traj,axis=0)

        

        # Save the plot with the same name as the JSON file but as .png
        plot_name = os.path.splitext(json_file)[0] + '.png'
        print(f"Saved plot as {plot_name}")
        plot_path = os.path.join(folder_path, plot_name)
        plot_trajectory_losses(traj, path=plot_path)
        



    # path = args.outf + "/" + args.exp_name + "/trajectories"+"_seed="+str(seed)+"_n_part="+str(args.n_balls)+"_n_inputs="+str(args.num_inputs)+"_varDT="+str(varDt)+"_num_timesteps="+str(args.num_timesteps)+"_n_layers="+str(args.n_layers)+"_lr="+str(args.lr)+"_wd="+str(args.weight_decay)+"_.pkl"
    # with open(path, "rb") as f:
    #     loaded_tensor = pickle.load(f)
    # print(loaded_tensor.shape, torch.sum(loaded_tensor[0]-loaded_tensor[1]))
    # exit()


def load_pickle_files_suffix(directory, suffix=".pkl"):
    """
    Load all pickle files with a specific suffix from a directory.

    Args:
        directory (str): Path to the directory containing the pickle files.
        suffix (str): Suffix of the files to load (default is ".pkl").

    Returns:
        dict: A dictionary where keys are filenames and values are the loaded pickle objects.
        list: A list containing all the values in the dictionary
    """
    loaded_data = {}
    
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(suffix):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                loaded_data[filename] = data
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    return loaded_data, list(loaded_data.values())

def compute_mse_mean_std_per_timestep(tensor_list, metric="MSE"):
    
    T = tensor_list[0].shape[2]  # Extract T (number of timesteps)
    mse_losses_list = []  # To store the MSE losses for each tensor
    mae_losses_list = []
    # Iterate over each tensor in the list and compute MSE loss for each timestep
    print(np.array(tensor_list).shape) #(10,2,2000,20,15) = (# runs ,true/pred, num_samples, T, D) D= #nodes *3 (3dimensions)
    for tensor in tensor_list:
        target = tensor[0]  # Shape: (#samples, T, D)
        print(tensor.shape)
        prediction = tensor[1]  # Shape: (#samples, T, D)

        # Compute MSE loss for each timestep: average over samples and dimensions
        mse_per_timestep = torch.mean((prediction - target) ** 2, dim=(0, 2))  # Shape: (T,)
        mse_losses_list.append(mse_per_timestep)
        if metric == "MAE":
            mae_per_timestep = torch.mean(torch.abs(prediction - target), dim=(0, 2))  # Shape: (T,)
            mae_losses_list.append(mae_per_timestep)    
    # Stack all MSE loss vectors from the list to compute mean and std
    mse_losses = torch.stack(mse_losses_list, dim=0)  # Shape: (10, T)
    # Compute mean and standard deviation across the 10 elements (dim=0)
    mean_mse = torch.mean(mse_losses, dim=0)  # Shape: (T,)
    std_mse = torch.std(mse_losses, dim=0)    # Shape: (T,)
    if metric=="MAE":
        mae_losses = torch.stack(mae_losses_list, dim=0)  # Shape: (10, T)
        mean_mae = torch.mean(mae_losses, dim=0)  # Shape: (T,)
        std_mae = torch.std(mae_losses, dim=0)    # Shape: (T,)
        return mean_mse, std_mse, mean_mae, std_mae
    
    return mean_mse, std_mse #, mean_mae, std_mae

def load_trajectory_config(config, model):
    #directoryS = f"{model}_exp\exp_1"
    directoryE = f"{model}_exp\exp_1"

    #trajectories corresponding to different seeds
    #_, trajectoriesS = load_pickle_files_suffix(directoryS, config)
    _, trajectoriesE = load_pickle_files_suffix(directoryE, config)
    print("here print trajs shape")
    print(np.array(trajectoriesE).shape)   # (10,2,2000,20,15) = (# seeds,true/pred, num_samples, T, D) D= #nodes *3 (3dimensions)

    return trajectoriesE

def load_trajs(model="EGNO",t=0.99,n_part=5,num_inputs=1,varDT=False,num_timesteps=10,n_layers=4,lr_=0.0005,onlytest=True, same_t=False, metric="MSE"):
    lr = lr_
    if model.startswith("EGNO"):
        model="EGNO"
        lr = 0.00005 #check
        t_last = -1
        first= num_timesteps
        config = f"n_part={n_part}_n_inputs={num_inputs}_varDT={varDT}_num_timesteps={num_timesteps}_n_layers={n_layers}_lr={lr}_wd=1e-12_.pkl"

    trajs = load_trajectory_config(config, model)

    return trajs

def estimate_velocities(loc, delta_t):
    """
    Estimate velocities from position data using finite differences.

    Parameters:
    - loc: np.array of shape (T, 3, N)
    - delta_t: float, time step between frames

    Returns:
    - vel: np.array of shape (T-1, 3, N)
    """
    vel = (loc[1:] - loc[:-1]) / delta_t
    return vel

def tot_energy(loc, vel, edges, interaction_strength=.1):
        with np.errstate(divide='ignore'):
            K = 0.5 * (vel ** 2).sum()
            U = 0
            for i in range(loc.shape[1]):
                for j in range(loc.shape[1]):
                    if i != j:
                        r = loc[:, i] - loc[:, j]
                        dist = np.sqrt((r ** 2).sum())
                        U += 0.5 * interaction_strength * edges[
                            i, j] * (dist ** 2) / 2
            return U + K

def compute_energy_drift(loc, vels, edges):
    """
    Compute relative energy drift at each timestep from a trajectory.
    
    Parameters:
    - loc: np.array of shape (T, 3, N)
    - vel: np.array of shape (T, 3, N)
    - edges: np.array of shape (N, N)
    
    Returns:
    - energy_drift: np.array of shape (T,)
    """
    delta_t = 0.01  # or whatever your simulation used
    vel = estimate_velocities(loc, delta_t)
    T = loc.shape[0]
    energy_drift = np.zeros(T)

    # Initial energy
    E0 = tot_energy(loc[0], vel[0], edges)

    for t in range(T):
        Et = tot_energy(loc[t], vel[t], edges)
        energy_drift[t] = np.abs((Et - E0) / (E0 + 1e-10))  # epsilon for stability

    return energy_drift


def load_trajectory_for_config(config, model, metric="MSE"):
    #directoryS = f"{model}_exp\exp_1"
    directoryE = f"{model}_exp\exp_1"

    #trajectories corresponding to different seeds
    #_, trajectoriesS = load_pickle_files_suffix(directoryS, config)
    _, trajectoriesE = load_pickle_files_suffix(directoryE, config)
    #print(len(trajectoriesE))
    #MSE
    #mean_losses_S, std_losses_S = compute_mse_mean_std_per_timestep(trajectoriesS)
    if metric=="MAE":
        mean_mse, std_mse, mean_mae, std_mae = compute_mse_mean_std_per_timestep(trajectoriesE, metric)
    else:
        mean_losses_E, std_losses_E = compute_mse_mean_std_per_timestep(trajectoriesE, metric)

    #Correlation
    #mean_corr_S, std_corr_S = compute_correlation_per_timestep(trajectoriesS)
    mean_corr_E, std_corr_E = compute_correlation_per_timestep(trajectoriesE)
    if metric=="MAE":
        return mean_mse, std_mse, mean_corr_E, std_corr_E, mean_mae, std_mae
    
    return mean_losses_E, std_losses_E, mean_corr_E, std_corr_E#mean_losses_S, std_losses_S, mean_losses_E, std_losses_E, mean_corr_S, std_corr_S, mean_corr_E, std_corr_E

def load_trajectory_wandb(config, metric="MSE"):
    
    trajectoriesE = config["data"]
    #print(len(trajectoriesE))
    #MSE
    #mean_losses_S, std_losses_S = compute_mse_mean_std_per_timestep(trajectoriesS)
    if metric=="MAE":
        mean_mse, std_mse, mean_mae, std_mae = compute_mse_mean_std_per_timestep(trajectoriesE, metric)
    else:
        mean_losses_E, std_losses_E = compute_mse_mean_std_per_timestep(trajectoriesE, metric)

    #Correlation
    #mean_corr_S, std_corr_S = compute_correlation_per_timestep(trajectoriesS)
    mean_corr_E, std_corr_E = compute_correlation_per_timestep(trajectoriesE)
    if metric=="MAE":
        return mean_mse, std_mse, mean_corr_E, std_corr_E, mean_mae, std_mae
    
    return mean_losses_E, std_losses_E, mean_corr_E, std_corr_E

def compute_correlation_per_timestep(tensor_list):
    corr_list = []  

    # Iterate over each tensor in the list and compute correlation for each timestep
    for tensor in tensor_list:
        target = tensor[0]  # Shape: (#samples, T, D)
        prediction = tensor[1]  # Shape: (#samples, T, D)
        print(target.shape, prediction.shape)
        # Compute correlation for each timestep
        corr_per_timestep = compute_correlation(target, prediction)  # Shape: (T,)
        corr_list.append(corr_per_timestep)
    
    correlations = torch.stack(corr_list, dim=0)  # Shape: (10, T)

    # Compute mean and standard deviation across the seed dim (dim=0)
    mean_corr = torch.mean(correlations, dim=0)  # Shape: (T,)
    std_corr = torch.std(correlations, dim=0)    # Shape: (T,)

    return mean_corr, std_corr

def compute_correlation(tensor1, tensor2):
    """
    Computes the correlation between two tensors of shape (num_samples, T, D) 
    such that the resulting tensor has shape (T,).

    Args:
        tensor1 (torch.Tensor): First tensor of shape (num_samples, T, D).
        tensor2 (torch.Tensor): Second tensor of shape (num_samples, T, D).

    Returns:
        torch.Tensor: A tensor of shape (T,) containing the correlation at each timestep.
    """
    # Reshape tensors to (num_samples * D, T)
    num_samples, T, D = tensor1.shape
    reshaped1 = tensor1.permute(1, 0, 2).reshape(T, -1)  # Shape: (T, num_samples * D)
    reshaped2 = tensor2.permute(1, 0, 2).reshape(T, -1)  # Shape: (T, num_samples * D)
    
    # Compute the mean and standard deviation for each timestep
    mean1 = reshaped1.mean(dim=1, keepdim=True)  # Shape: (T, 1)
    mean2 = reshaped2.mean(dim=1, keepdim=True)  # Shape: (T, 1)

    std1 = reshaped1.std(dim=1, keepdim=True)    # Shape: (T, 1)
    std2 = reshaped2.std(dim=1, keepdim=True)    # Shape: (T, 1)
    
    # # Normalize the tensors
    # normalized1 = (reshaped1 - mean1) / (std1 + 1e-8)  # Shape: (T, num_samples * D)
    # normalized2 = (reshaped2 - mean2) / (std2 + 1e-8)  # Shape: (T, num_samples * D)

    cent1 = (reshaped1 - mean1) # Shape: (T, num_samples * D)
    cent2 = (reshaped2 - mean2)  # Shape: (T, num_samples * D)
    std_x = torch.sqrt((cent1 ** 2).sum(dim=1))
    std_y = torch.sqrt((cent2 ** 2).sum(dim=1))
    #print((std1-std_x).sum(),(std2-std_y).sum())
    #print(std_x.shape)
    covariance = (cent1*cent2).sum(dim=1)
    #print(covariance.shape)
    # Compute correlation for each timestep
    #correlation = (normalized1 * normalized2).mean(dim=1)  # Shape: (T,)
    corr = covariance / (std_x * std_y)
    #print(corr.shape)
    return corr         #correlation

def test_mean_std_dev_plot(values, std_dev, config, model="EGNO", type="MSE", save=False, plot_title=False, num_steps=10, same_plot=False, sub=False, transparency=False,tran=False):
    # Example data
    T = values.shape[0]
    x = np.arange(T)  # x-axis values (e.g., time or indices)
    step = 3
    
    if model=="SEGNO" and not same_plot:
        x= (x*num_steps) 
        step = 1
        for x_val in x:
            plt.axvline(x=x_val, color='red', linestyle='--')
    elif same_plot and tran:
        x = np.arange(T)
        
        for x_val in x[::num_steps]:
            plt.axvline(x=x_val, color='red', linestyle='--')
    elif same_plot and not transparency:
        x = (x*num_steps) + num_steps
        for x_val in x:
            plt.axvline(x=x_val, color='red', linestyle='--')
   
    
    # Calculate the upper and lower bounds
    upper_bound = values + std_dev
    lower_bound = values - std_dev
    color_m = {"EGNO": "blue", "SEGNO":"green"}
    
    # Plot the main values
    if transparency:
        plt.plot(x, values, marker='o', label=f"{model} 4 layers", color=color_m[model], linewidth=2,alpha=0.5)
    else:                   #fix this after
        plt.plot(x, values, marker='o', label=model, color=color_m[model], linewidth=2) #f"{model} 3 layers"

    # Fill the region between lower and upper bounds
    plt.fill_between(x, lower_bound, upper_bound, color='blue', alpha=0.2, label='± Std Dev')

    if same_plot and not transparency and not tran:
        for i, (x_val, y_val) in enumerate(zip(x, values)):
            plt.text(x_val, y_val, f'{y_val:.4f}', fontsize=10, ha='right')
    # else:
    #     for i, (x_val, y_val) in enumerate(zip(x[::step], values[::step])):
    #         plt.text(x_val, y_val, f'{y_val:.4f}', fontsize=10, ha='right')

    if same_plot:
        return
    
    # Add labels and legend
    plt.xticks(x) 
    plt.xlabel('Timestamps')
    plt.ylabel(type)
    title = model +" : "+type +' averaged among seeds with ± Std Deviation for config: '+ config
    wrapped_title = "\n".join(textwrap.wrap(title, width=40))
    if plot_title:
        plt.title(wrapped_title)
    if type=="MSE":
        plt.yscale('log')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Show the plot
    plt.tight_layout()
    if save:
        plt.savefig('plots/'+model+"_"+type+"_"+config+"_"+str(plot_title)+ '.png')
        plt.close()
    else:
        plt.show()
        plt.close()

def compute_loss_sub(model="EGNO",scale="none",n_part=5,num_inputs=1,varDT=False,num_timesteps=10,n_layers=4,lr_=0.0005,onlytest=True):

    lrE = 0.00005 #check
    lrS = lr_
    configE = f"n_part={n_part}_n_inputs={num_inputs}_varDT={varDT}_num_timesteps={num_timesteps}_n_layers={n_layers}_lr={lrE}_wd=1e-12_.pkl"
    #elif model=="SEGNO":
    configS = f"n_part={n_part}_n_steps={num_timesteps}_n_inputs={num_inputs}_varDT={varDT}_lr{lrS}_wd1e-12_onlytest={onlytest}_.pkl"

    mean_losses_E, std_losses_E, mean_corr_E, std_corr_E = load_trajectory_for_config(configE,"EGNO")
    mean_losses_S, std_losses_S, mean_corr_S, std_corr_S = load_trajectory_for_config(configS,"SEGNO")

    l = mean_losses_E.shape[0] // num_timesteps
    T = mean_losses_E.shape[0]
    if model == "EGNO":
        if scale == "direct":
            sub = mean_losses_E[::num_timesteps]*(torch.arange(11,T+1)[::T-1]) - mean_losses_S[:l]*(torch.arange(11,T+1)[::T-1])
        elif scale == "inverse":
            sub = mean_losses_E[::num_timesteps]*(1/torch.arange(11,T+1)[::T-1]) - mean_losses_S[:l]*(1/torch.arange(11,T+1)[::T-1])
        else:
            sub = mean_losses_E[::num_timesteps] - mean_losses_S[:l]
        sub = torch.sum(sub)#torch.sum(torch.abs())
    if model == "SEGNO":
        if scale == "direct":
            sub = mean_losses_S[:l]*(torch.arange(11,T+1)[::T-1]) - mean_losses_E[::num_timesteps]*(torch.arange(11,T+1)[::T-1])
        elif scale == "inverse":
            sub = mean_losses_S[:l]*(1/torch.arange(11,T+1)[::T-1]) - mean_losses_E[::num_timesteps]*(1/torch.arange(11,T+1)[::T-1])
        else:
            sub = mean_losses_S[:l] - mean_losses_E[::num_timesteps] 
        
        sub = torch.sum(sub)

    return sub

def plot_corr_mse_model(model="EGNO",transparency=False,same_plot=False,save=False,n_part=5,num_inputs=1,varDT=False,num_timesteps=10,n_layers=4,lr_=0.0005,onlytest=True):
    
    if same_plot:
        #if model=="EGNO":
        lrE = 0.00005 #check
        lrS = lr_
        configE = f"n_part={n_part}_n_inputs={num_inputs}_varDT={varDT}_num_timesteps={num_timesteps}_n_layers={n_layers}_lr={lrE}_wd=1e-12_.pkl"
        #elif model=="SEGNO":
        if transparency:
            configS = f"n_part={n_part}_n_inputs={num_inputs}_varDT={varDT}_num_timesteps={num_timesteps}_n_layers={n_layers-1}_lr={lrE}_wd=1e-12_.pkl"
        else:
            configS = f"n_part={n_part}_n_steps={num_timesteps}_n_inputs={num_inputs}_varDT={varDT}_lr{lrS}_wd1e-12_onlytest={onlytest}_.pkl"
        mean_losses_E, std_losses_E, mean_corr_E, std_corr_E = load_trajectory_for_config(configE,"EGNO")
        if transparency:
            mean_losses_S, std_losses_S, mean_corr_S, std_corr_S = load_trajectory_for_config(configS,"EGNO")
        else:
            mean_losses_S, std_losses_S, mean_corr_S, std_corr_S = load_trajectory_for_config(configS,"SEGNO")
        #print(mean_losses_E, std_losses_E) #, mean_corr_E, std_corr_E
        # insert config and type in plot title
        #if model=="EGNO":
        configE = f"n_part={n_part}_n_inputs={num_inputs}_varDT={varDT}_num_timesteps={num_timesteps}_n_layers={n_layers}_lr={lrE}"
        #elif model=="SEGNO":
        if transparency:
            configS = f"n_part={n_part}_n_inputs={num_inputs}_varDT={varDT}_num_timesteps={num_timesteps}_n_layers={n_layers-1}_lr={lrE}_wd=1e-12_.pkl"
        else:
            configS = f"n_part={n_part}_n_steps={num_timesteps}_n_inputs={num_inputs}_varDT={varDT}_lr={lrS}_onlytest={onlytest}"
        type = "MSE"
        if transparency:
            l = mean_corr_E.shape[0]
        else:
            l = mean_corr_E.shape[0]//num_timesteps
        
        if transparency:
            test_mean_std_dev_plot(mean_losses_E.numpy(),std_losses_E.numpy(), config=configE, model="EGNO", type=type,save=save, num_steps=num_timesteps, same_plot=same_plot,transparency=transparency,tran=True)
            test_mean_std_dev_plot(mean_losses_S.numpy(),std_losses_S.numpy(), config=configS, model="EGNO", type=type,save=save, num_steps=num_timesteps, same_plot=same_plot,tran=True)
        else:
            test_mean_std_dev_plot(mean_losses_E[::num_timesteps].numpy(),std_losses_E[::num_timesteps].numpy(), config=configE, model="EGNO", type=type,save=save, num_steps=num_timesteps, same_plot=same_plot,transparency=transparency)
            test_mean_std_dev_plot(mean_losses_S[:l].numpy(),std_losses_S[:l].numpy(), config=configS, model="SEGNO", type=type,save=save, num_steps=num_timesteps, same_plot=same_plot)
        #plot
        x = np.arange(l) 
        x = (x*num_timesteps)+num_timesteps
        if transparency:
            x = np.arange(l)
        
        plt.xticks(x) 
        plt.xlabel('Timestamps')
        plt.ylabel(type)
        #title = model +" : "+type +' averaged among seeds with ± Std Deviation for config: '+ config
        #wrapped_title = "\n".join(textwrap.wrap(title, width=40))
        # if plot_title:
        #     plt.title(wrapped_title)
        if type=="MSE":
            plt.yscale('log')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        # Show the plot
        plt.tight_layout()
        if save:
            plt.savefig('plots/Joined_graph_'+type+"_"+configS+"_"+configE+'_.png')
            plt.close()
        else:
            plt.show()
            plt.close()
        type = "Correlation"
        #test_mean_std_dev_plot(mean_losses_E.numpy(),std_losses_E.numpy(),  config=config, model=model, type=type,save=save, num_steps=num_timesteps,same_plot=same_plot)
        #test_mean_std_dev_plot(mean_losses_E.numpy(),std_losses_E.numpy(),  config=config, model=model, type=type,save=save, num_steps=num_timesteps,same_plot=same_plot)
        if transparency:
            test_mean_std_dev_plot(mean_corr_E.numpy(),std_corr_E.numpy(), config=configE, model="EGNO", type=type,save=save, num_steps=num_timesteps, same_plot=same_plot,transparency=transparency,tran=True)
            test_mean_std_dev_plot(mean_corr_S.numpy(),std_corr_S.numpy(), config=configS, model="EGNO", type=type,save=save, num_steps=num_timesteps, same_plot=same_plot,tran=True)
        else:
            test_mean_std_dev_plot(mean_corr_E[::num_timesteps].numpy(),std_corr_E[::num_timesteps].numpy(), config=configE, model="EGNO", type=type,save=save, num_steps=num_timesteps, same_plot=same_plot,transparency=transparency)
            test_mean_std_dev_plot(mean_corr_S[:l].numpy(),std_corr_S[:l].numpy(), config=configS, model="SEGNO", type=type,save=save, num_steps=num_timesteps, same_plot=same_plot)
        #plot
        
        #plot
        x = np.arange(l) 
        x = (x*num_timesteps)+num_timesteps
        if transparency:
            x = np.arange(l)
        plt.xticks(x) 
        plt.xlabel('Timestamps')
        plt.ylabel(type)
        #title = model +" : "+type +' averaged among seeds with ± Std Deviation for config: '+ config
        #wrapped_title = "\n".join(textwrap.wrap(title, width=40))
        # if plot_title:
        #     plt.title(wrapped_title)
        if type=="MSE":
            plt.yscale('log')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        # Show the plot
        plt.tight_layout()
        if save:
            plt.savefig('plots/Joined_graph_'+type+"_"+configS+"_"+configE+'_.png')
            plt.close()
        else:
            plt.show()
            plt.close()
    else:
        lr = lr_
        if model=="EGNO":
            lr = 0.00005 #check
            config = f"n_part={n_part}_n_inputs={num_inputs}_varDT={varDT}_num_timesteps={num_timesteps}_n_layers={n_layers}_lr={lr}_wd=1e-12_.pkl"
        elif model=="SEGNO":
            config = f"n_part={n_part}_n_steps={num_timesteps}_n_inputs={num_inputs}_varDT={varDT}_lr{lr}_wd1e-12_onlytest={onlytest}_.pkl"

        mean_losses_E, std_losses_E, mean_corr_E, std_corr_E = load_trajectory_for_config(config,model)
        #print(mean_losses_E, std_losses_E) #, mean_corr_E, std_corr_E
        # insert config and type in plot title
        if model=="EGNO":
            config = f"n_part={n_part}_n_inputs={num_inputs}_varDT={varDT}_num_timesteps={num_timesteps}_n_layers={n_layers}_lr={lr}"
        elif model=="SEGNO":
            config = f"n_part={n_part}_n_steps={num_timesteps}_n_inputs={num_inputs}_varDT={varDT}_lr={lr}_onlytest={onlytest}"

        test_mean_std_dev_plot(mean_corr_E.numpy(),std_corr_E.numpy(), config=config, model=model, type="Correlation",save=save, num_steps=num_timesteps)
        test_mean_std_dev_plot(mean_losses_E.numpy(),std_losses_E.numpy(),  config=config, model=model, type="MSE",save=save, num_steps=num_timesteps)
#"trajectories_seed=93_ n_part=5_n_inputs=1_varDT=False_num_timesteps=10_n_layers=5_lr=0.0005_wd=1e-12_.pkl"
# "trajectories_seed=86_n_part=5_n_inputs=1_varDT=False_num_timesteps=5_n_layers=5_lr=0.0005_wd=1e-12_.pkl"  
# "n_part=5_n_steps=5_n_inputs=3_varDT=True_lr5e-05_wd1e-12_onlytest=True_.pkl"
def print_test_loss(model="EGNO",n_part=5,num_inputs=1,varDT=False,num_timesteps=10,n_layers=4,lr_=0.0005,onlytest=True):
    lr = lr_
    if model=="EGNO":
        lr = 0.00005 #check
        config = f"n_part={n_part}_n_inputs={num_inputs}_varDT={varDT}_num_timesteps={num_timesteps}_n_layers={n_layers}_lr={lr}_wd=1e-12_.json"
    elif model=="SEGNO":
        config = f"n_part={n_part}_n_steps={num_timesteps}_n_inputs={num_inputs}_varDT={varDT}_lr{lr}_wd1e-12_onlytest={onlytest}_.json"

    #json_path = f"{model}_exp\exp_1\loss_seed=1_n_part={n_part}_n_inputs={num_inputs}_varDT={varDT}_num_timesteps={num_timesteps}_n_layers={n_layers}_lr={lr}_wd=1e-12_.json"
    directory = f"{model}_exp\exp_1"
    loaded_data = {}

    for filename in os.listdir(directory):
        if filename.endswith(config):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
            loaded_data[filename] = data["test loss"]
    print(np.array(list(loaded_data.values())).shape)
    test_loss = np.mean(np.array(list(loaded_data.values())))
    
    print(test_loss)

    return test_loss

def print_latex_table(headers, data):
    #headers = ["Name", "Age", "Country"]
    # data = [
    #     ["Alice", 25, "USA"],
    #     ["Bob", 30, "UK"],
    #     ["Charlie", 28, "Canada"]
    # ]
    # Generate a LaTeX table
    latex_table = tabulate(data, headers=headers, tablefmt="latex")

    print(latex_table)

def compute_avg_loss_until_t(model="EGNO",t=0.99,n_part=5,num_inputs=1,varDT=False,num_timesteps=10,n_layers=4,lr_=0.0005,onlytest=True, same_t=False, metric="MSE"):
    lr = lr_
    if model.startswith("EGNO"):
        model="EGNO"
        lr = 0.00005 #check
        t_last = -1
        first= num_timesteps
        config = f"n_part={n_part}_n_inputs={num_inputs}_varDT={varDT}_num_timesteps={num_timesteps}_n_layers={n_layers}_lr={lr}_wd=1e-12_.pkl"
    elif model.startswith("SEGNO"):
        model="SEGNO"
        first=0
        config = f"n_part={n_part}_n_steps={num_timesteps}_n_inputs={num_inputs}_varDT={varDT}_lr{lr}_wd1e-12_onlytest={onlytest}_.pkl"
        t_last= 2 if num_timesteps==10 else 4
    
    if metric=="MAE":
        mean_mse, std_mse, mean_corr, std_corr, mean_mae, std_mae = load_trajectory_for_config(config,model, metric=metric)
        if same_t:
            avg_mse = torch.mean(mean_mse[:t_last]) ## ::num_timesteps
            avg_mae = torch.mean(mean_mae[:t_last])
            index = num_timesteps*2 if num_timesteps==10 else num_timesteps*4
            first_mse = mean_mse[first]
            first_mae = mean_mae[first]
            return index, avg_mse, avg_mae, first_mse, first_mae

    else:
        mean_losses, std_losses, mean_corr, std_corr = load_trajectory_for_config(config,model)

    if same_t:
        avg_loss = torch.mean(mean_losses[:t_last]) ## ::num_timesteps
        index = num_timesteps*2 if num_timesteps==10 else num_timesteps*4
        return index, avg_loss

    indeces = torch.where(mean_corr < t)[0]
    if len(indeces) > 0:
        index = indeces[0].item()
    else:
        index = -1
    #print(index)
    avg_loss = torch.mean(mean_losses[:index])
    if index == -1:
        index = mean_losses.shape[0]
    
    index = index*num_timesteps if model=="SEGNO" else index
    return index, avg_loss

def compute_avg_loss_until_t_wandb(config, same_t=True, metric="MAE", std=True):
    
    (t_last, first) = (-1, config["num_timesteps"]) if config["model"] == "EGNO" else (2 if config["num_timesteps"]==10 else 4, 0)

    num_timesteps = config["num_timesteps"]
    if metric=="MAE":
        #TODO check shape of stds and add energy calculation
        mean_mse, std_mse, mean_corr, std_corr, mean_mae, std_mae = load_trajectory_wandb(config, metric=metric)
        ed_mean, ed_std = compute_energy_mean_std_per_timestep(config["energy"])
        if same_t:
            avg_mse = torch.mean(mean_mse[:t_last]) ## ::num_timesteps
            avg_mae = torch.mean(mean_mae[:t_last])
            avg_std_mse = torch.mean(std_mse[:t_last])
            avg_std_mae = torch.mean(std_mae[:t_last])
            avg_ed = torch.mean(ed_mean[:t_last])
            avg_std_ed = torch.mean(ed_std[:t_last])
            index = num_timesteps*2 if num_timesteps==10 else num_timesteps*4
            first_mse = mean_mse[first]
            first_mae = mean_mae[first]
            if std:
                return index, avg_mse, avg_std_mse, avg_mae, avg_std_mae, first_mse, first_mae, avg_ed, avg_std_ed

            return index, avg_mse, avg_mae, first_mse, first_mae

    else:
        mean_losses, std_losses, mean_corr, std_corr = load_trajectory_wandb(config)

    if same_t:
        avg_loss = torch.mean(mean_losses[:t_last]) ## ::num_timesteps
        index = num_timesteps*2 if num_timesteps==10 else num_timesteps*4
        return index, avg_loss

    indeces = torch.where(mean_corr < t)[0]
    if len(indeces) > 0:
        index = indeces[0].item()
    else:
        index = -1
    #print(index)
    avg_loss = torch.mean(mean_losses[:index])
    if index == -1:
        index = mean_losses.shape[0]
    
    index = index*num_timesteps if model=="SEGNO" else index
    return index, avg_loss

def compute_energy_mean_std_per_timestep(tensor_list):
    
    #T = tensor_list[0].shape[2]  # Extract T (number of timesteps)
    energy_drift_list = []  
    # Iterate over each tensor in the list and compute MSE loss for each timestep
    print(np.array(tensor_list).shape) #(10,2,2000,20,15) = (# runs ,true/pred, num_samples, T, D) D= #nodes *3 (3dimensions)
    for tensor in tensor_list:
        print(tensor.shape)

        # Compute MSE loss for each timestep: average over samples and dimensions
        ed_per_timestep = torch.mean((tensor) ** 2, dim=(0, 2))  # Shape: (T,)
        energy_drift_list.append(ed_per_timestep)
    # Stack all MSE loss vectors from the list to compute mean and std
    ed_losses = torch.stack(energy_drift_list, dim=0)  # Shape: (10, T)
    # Compute mean and standard deviation across the 10 elements (dim=0)
    mean_ed = torch.mean(ed_losses, dim=0)  # Shape: (T,)
    std_ed = torch.std(ed_losses, dim=0)    # Shape: (T,)

    return mean_ed, std_ed

def plot_multiple_curves(configurations, plot_func, metric="MSE", save=False, filename=""):
    """
    Plots multiple curves on the same plot by calling an inner function
    that processes each configuration and returns data to plot.
    
    Parameters:
    - configurations: list of dictionaries, each representing a configuration.
    - plot_func: a function that receives one configuration at a time
                 and returns x, y, and label for plotting.
    """
    plt.figure(figsize=(10, 6))  # Create a single figure

    for config in configurations:
        # Call the inner function to get the data for plotting
        x, y, std, label = plot_func(config,metric)
        #plt.plot(x, y, label=label)
        
        if config["model"]=="SEGNO":
            t=np.arange(y.shape[0])
            t= (t*config["num_timesteps"]) + config["num_timesteps"]
            
            for x_val in t[::config["num_timesteps"]]:
                plt.axvline(x=x_val, color='red', linestyle='--')
        else:
            t=np.arange(y.shape[0])
            #(t*config["num_timesteps"]) + config["num_timesteps"]
            for x_val in t[config["num_timesteps"]::config["num_timesteps"]]:
                #print(x_val)
                plt.axvline(x=x_val, color='red', linestyle='--')
    
        
        # Calculate the upper and lower bounds
        upper_bound = y + std
        lower_bound = y - std
        
        # Plot the main values
        if config["baseline"]:
            plt.plot(x, y, marker='o', label=config["model"]+" baseline", color=config["color"], linewidth=2,alpha=0.3)
        else:                   #fix this after
            plt.plot(x, y, marker='o', label=config["label"], color=config["color"], linewidth=2) 

        # Fill the region between lower and upper bounds
        plt.fill_between(x, lower_bound, upper_bound, color=config["color"], alpha=0.2, label='± Std Dev')

        # if config["model"]=="EGNO":
        #     for i, (x_val, y_val) in enumerate(zip(x[::3], y[::3])):
        #         plt.text(x_val, y_val, f'{y_val:.4f}', fontsize=10, ha='right')
        # else:
        #     for i, (x_val, y_val) in enumerate(zip(x, y)):
        #         plt.text(x_val, y_val, f'{y_val:.4f}', fontsize=10, ha='right')
        

    # Finalize the plot
    if metric=="MSE":
        plt.yscale('log')
    plt.xticks(x) 
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlabel("Timestamps")
    plt.ylabel(metric)
    plt.legend()
    # Show the plot
    plt.tight_layout()
    if save:
        plt.savefig(filename+ '.png')
        plt.close()
    else:
        plt.show()
        plt.close()
    return

def plot_multiple_curves_wandb(configurations, plot_func, metric="MSE", save=False, filename=""):
    
    plt.figure(figsize=(10, 6))  # Create a single figure

    for config in configurations:
        # Call the inner function to get the data for plotting
        x, y, std, label = plot_func(config, metric)
        #plt.plot(x, y, label=label)
        
        if config["model"]=="SEGNO":
            t=np.arange(y.shape[0])
            t= (t*config["num_timesteps"]) + config["num_timesteps"]
            
            for x_val in t[::config["num_timesteps"]]:
                plt.axvline(x=x_val, color='red', linestyle='--')
        else:
            t=np.arange(y.shape[0])
            #(t*config["num_timesteps"]) + config["num_timesteps"]
            for x_val in t[config["num_timesteps"]::config["num_timesteps"]]:
                #print(x_val)
                plt.axvline(x=x_val, color='red', linestyle='--')
    
        
        # Calculate the upper and lower bounds
        upper_bound = y + std
        lower_bound = y - std
        
        # Plot the main values
        if config["baseline"]:
            plt.plot(x, y, marker='o', label=config["model"]+" baseline", color=config["color"], linewidth=2,alpha=0.3)
        else:                   #fix this after
            plt.plot(x, y, marker='o', label=config["label"], color=config["color"], linewidth=2) 

        # Fill the region between lower and upper bounds
        plt.fill_between(x, lower_bound, upper_bound, color=config["color"], alpha=0.2, label='± Std Dev')

        # if config["model"]=="EGNO":
        #     for i, (x_val, y_val) in enumerate(zip(x[::3], y[::3])):
        #         plt.text(x_val, y_val, f'{y_val:.4f}', fontsize=10, ha='right')
        # else:
        #     for i, (x_val, y_val) in enumerate(zip(x, y)):
        #         plt.text(x_val, y_val, f'{y_val:.4f}', fontsize=10, ha='right')
        

    # Finalize the plot
    if metric=="MSE":
        plt.yscale('log')
    plt.xticks(x) 
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlabel("Timestamps")
    plt.ylabel(metric)
    plt.legend()
    # Show the plot
    plt.tight_layout()
    if save:
        plt.savefig(filename + '.png')
        plt.close()
    else:
        plt.show()
        plt.close()
    return

# Example inner function that generates data based on a configuration
def plot_function(config,metric):
    """
    Example function to generate data for plotting based on a configuration.
    This function is called once per configuration.
    
    Parameters:
    - config: dictionary containing configuration data.
    
    Returns:
    - x: x-values for the plot
    - y: y-values for the plot
    - label: label for the plot legend
    """
    if config["model"]=="EGNO":
        configuration = f"n_part={config['n_part']}_n_inputs={config['num_inputs']}_varDT={config['varDT']}_num_timesteps={config['num_timesteps']}_n_layers={config['n_layers']}_lr={config['lr']}_wd=1e-12_.pkl"
    elif config["model"]=="SEGNO":
        configuration = f"n_part={config['n_part']}_n_steps={config['num_timesteps']}_n_inputs={config['num_inputs']}_varDT={config['varDT']}_lr{config['lr']}_wd1e-12_onlytest={config['onlytest']}_.pkl"

    mean_losses, std_losses, mean_corr, std_corr = load_trajectory_for_config(configuration, config["model"])
    l = 2 if config["num_timesteps"]==10 else 4
    if config["model"]=="EGNO":
        x = np.arange(mean_losses.shape[0]) +1
    elif config["model"]=="SEGNO":
        x = ((np.arange(mean_losses.shape[0])*config["num_timesteps"]) +config["num_timesteps"])[:l]
   
    if metric=="MSE":
        y = mean_losses
        if config["model"]=="SEGNO":
            y = y[:l]
            std_losses = std_losses[:l]
        return x, y, std_losses, config["model"] 
    
    elif metric=="Correlation":
        y = mean_corr
        if config["model"]=="SEGNO":
            y = y[:l]
            std_corr = std_corr[:l]
        return x, y, std_corr, config["model"] 
    
    return x, y, config["model"] + str(config["n_layers"])

def plot_function_wandb(config, metric):
    
    mean_losses, std_losses, mean_corr, std_corr = load_trajectory_wandb(config, metric=metric)
    l = 2 if config["num_timesteps"]==10 else 4
    if config["model"]=="EGNO":
        x = np.arange(mean_losses.shape[0]) +1
    elif config["model"]=="SEGNO":
        x = ((np.arange(mean_losses.shape[0])*config["num_timesteps"]) +config["num_timesteps"])[:l]
   
    if metric=="MSE":
        y = mean_losses
        if config["model"]=="SEGNO":
            y = y[:l]
            std_losses = std_losses[:l]
        return x, y, std_losses, config["model"] 
    
    elif metric=="Correlation":
        y = mean_corr
        if config["model"]=="SEGNO":
            y = y[:l]
            std_corr = std_corr[:l]
        return x, y, std_corr, config["model"] 
    
    return x, y, config["model"] #+ str(config["n_layers"])

#def load_model():

if __name__ == "__main__":
    #"n_part=5_n_inputs=1_varDT=False_num_timesteps=10_n_layers=4_lr=0.0005_wd=1e-12_.pkl"
    #"n_part=5_n_steps=10_n_inputs=1_varDT=False_lr5e-05_wd1e-12_onlytest=True_.pkl"
    # config = "n_part=5_n_inputs=1_varDT=False_num_timesteps=10_n_layers=4_lr=0.0005_wd=1e-12_.pkl"
    # model = "EGNO"
    # mean_losses_E, std_losses_E, mean_corr_E, std_corr_E = load_trajectory_for_config(config,model)
    # #print(mean_losses_E, std_losses_E) #, mean_corr_E, std_corr_E
    
    # test_mean_std_dev_plot(mean_corr_E.numpy(),std_corr_E.numpy(), model="EGNO", type="corr")
    # test_mean_std_dev_plot(mean_losses_E.numpy(),std_losses_E.numpy(),  model="EGNO", type="mse" )

    # json_path = "SEGNO_exp\exp_1\loss_seed=1_n_part=5_n_steps=2_n_inputs=1_varDT=True_lr0.0005_wd1e-12_onlytest=True_.json"
    # with open(json_path, 'r') as file:
    #     data = json.load(file)
    # print(data["test loss"])
    # a = torch.tensor([3,2])
    # b = torch.tensor([5,1])
    # print(torch.abs(torch.sum(b-a))) #[2,-1]
    # print(torch.sum(torch.abs(b-a)))
    # print(torch.sum(b-a))
    # print(torch.abs(torch.sum(a-b))) #[2,-1]
    # print(torch.sum(torch.abs(a-b)))
    # print(torch.sum(a-b))
    # Example data
    # print(1/torch.arange(11,21)[::9])
    # exit()
    # plot1="plots\EGNO_Correlation_n_part=5_n_inputs=1_varDT=False_num_timesteps=10_n_layers=4_lr=5e-05_False.png"
    # plot2="plots\EGNO_Correlation_n_part=5_n_inputs=1_varDT=False_num_timesteps=10_n_layers=3_lr=5e-05_False.png"
    # plots_transparency(plot1,plot2)
    # a = np.array([1,2,3,4])
    # print(a[1::2])
    # exit()
    #n_part=5,num_inputs=1,varDT=False,num_timesteps=10,n_layers=4,lr_=0.0005,onlytest=True
    """
    {"model": "EGNO", "baseline":True, "n_part": 5,"num_inputs":1,"num_timesteps": 10,
                       "lr":0.00005,"n_layers":3,"varDT":False,"color": "purple"},
                       {"model": "SEGNO", "baseline":True, "n_part": 5,"num_inputs":1,"num_timesteps": 10,
                       "lr":0.0005,"varDT":False,"color": "green", "onlytest": True},
                       {"model": "EGNO", "baseline":False, "n_part": 5,"num_inputs":3,"num_timesteps": 5,
                       "lr":0.00005,"n_layers":3,"varDT":True,"color": "purple"},
                       {"model": "SEGNO", "baseline":False, "n_part": 5,"num_inputs":3,"num_timesteps": 5,
                       "lr":0.0005,"varDT":True,"color": "green", "onlytest": False}
    """
    configurations =[ 
                       {"model": "EGNO", "baseline":False, "n_part": 5,"num_inputs":1,"num_timesteps": 10,
                       "lr":0.00005,"n_layers":3,"varDT":False,"color": "purple","label":"EGNO MI(3)VDT"},
                       {"model": "SEGNO", "baseline":False, "n_part": 5,"num_inputs":3,"num_timesteps": 5,
                       "lr":0.0005,"varDT":True,"color": "green", "onlytest": False,"label":"SEGNO MI(3)VDT"}
                       ]
        
                       #{"model": "SEGNO", "baseline":True, "n_part": 5,"num_inputs":1,"num_timesteps": 10,
                       #"lr":0.0005,"n_layers":3,"varDT":False,"color": "green", "onlytest": True}
    x=590
    if x==1:
        print_test_loss(model="SEGNO")
        exit()
    elif x ==590:
        model = "EGNO"
        num_timesteps=10
        load_trajs(model=model,n_layers=3,lr_=0.0005,num_timesteps=num_timesteps ,num_inputs=3,onlytest=False)
    elif x == 42:
        save=True
        metrics=["MSE","Correlation"]
        for metric in metrics:
            
            saveas = f"plots\Multi_plots_EGNO_SEGNO_3VDT5inputs_{metric}_"
            plot_multiple_curves(configurations,metric=metric, plot_func=plot_function,save=save, filename=saveas)
        exit()
    elif x==2:
        models = ["EGNO","SEGNO","SEGNO"] # "EGNO",
        num_timesteps = 10
        onlytest= False
        num_inputs= 1
        varDT = False
        save = True
        same_plot=True
        transparency=False
        for model in models:
            
            plot_corr_mse_model(model=model,transparency=transparency,save=save,n_layers=4,lr_=0.0005,num_timesteps=num_timesteps,num_inputs=num_inputs,varDT=varDT,onlytest=onlytest,same_plot=same_plot)#,num_inputs=2,onlytest=False
            if model == "SEGNO":
                onlytest=True
            if same_plot:
                break
        exit()
    elif x==3:
        num_timesteps = 10
        model = "EGNO"
        other = "SEGNO"
        sub = compute_loss_sub(model=model,scale="inverse",n_layers=3,lr_=0.0005,num_timesteps=num_timesteps ,num_inputs=3,onlytest=False)
        print(f"{model} - {other} loss : {round(sub.item(),4)}")
        exit()
    elif x==4:
        same_t = True
        thresholds = [0.95,0.8,0.5]
        onlytest = False
        varDT = False
        metric="MAE"
        num_timesteps = 5
        #for threshold in thresholds:
        
        headers = ["model", "avg MSE", "avg MAE", "MSE first step", "MAE first step"]
        data = []
        models = ["EGNO","SEGNO","EGNO MI(2)","SEGNO MI(2)","EGNO MI(3)","SEGNO MI(3)"]
        
        num_inputs=1
        for model in models:
            if model.endswith("MI(2)"):
                num_inputs = 2
                varDT = True
            elif model.endswith("MI(3)"):

                num_inputs = 3
                varDT = True
            else:
                num_inputs=1
                #continue
            if num_timesteps<num_inputs:
                continue
            index, avg_mse, avg_mae, first_mse, first_mae = compute_avg_loss_until_t(model=model,n_layers=3,lr_=0.0005,varDT=varDT,num_timesteps=num_timesteps,num_inputs=num_inputs,onlytest=onlytest,same_t=same_t,metric=metric)
            #print(index, avg_loss)
            data.append([model,round(avg_mse.item(), 4),round(avg_mae.item(), 4),round(first_mse.item(), 4),round(first_mae.item(), 4)])
        print_latex_table(headers,data)
        exit()
    elif x==5:
        index, avg_loss = compute_avg_loss_until_t(model="SEGNO",t=0.95,n_layers=3,lr_=0.0005,num_timesteps=10,num_inputs=1,onlytest=False)
        print(index, avg_loss)
        exit()

    tensor_list = [torch.randn(2, 5, 3, 15) for _ in range(10)]
    mean_losses, std_losses = compute_correlation_per_timestep(tensor_list)

    print("Mean Corr:", mean_losses)
    print("Std Dev Corr:", std_losses)
    exit()
    num_samples, T, D = 5, 3, 15
    tensor1 = torch.randn(num_samples, T, D)
    tensor2 = torch.randn(num_samples, T, D)
    
    # Compute correlation
    correlation = compute_correlation(tensor1, tensor2)
    print(correlation)  # Output: Tensor of shape (T,)
    # t = torch.randint(0, 3, (2,3,4),dtype=float)
    # print(t.shape,t,t.mean(dim=2).shape,t.mean(dim=2))
    exit()
    tensor_list = [torch.randn(2, 5, 20, 15) for _ in range(10)]

    # Compute mean and std deviation of losses
    mean_losses, std_losses = compute_mse_mean_std_per_timestep(tensor_list)

    print("Mean Losses:", mean_losses.shape)
    print("Std Dev Losses:", std_losses.shape)
    exit()
    # model = EGNO(n_layers=4, in_node_nf=1, in_edge_nf=2, hidden_nf=64)  # Create a new instance of the model
    # model.load_state_dict(torch.load("EGNO_exp_1\saved_model_seed=1_n_part=5_n_inputs=4_varDT=True_num_timesteps=5_n_layers=3_lr=0.0005_wd=1e-12.pth"))
    # print(model)
    # exit()
    l = []
    t1 = torch.randint(0,5,(2,3,4),dtype=float)
    t2 = torch.randint(0,5,(2,3,4),dtype=float)
    T, B, N, D = 2, 3, 4, 5 
    if len(l) == 0:
        print("l")
    
    
    t3 = torch.randint(0,5,(2,3,4),dtype=float)
    t4 = torch.randint(0,5,(2,3,4),dtype=float)
    T, B, N, D = 2, 3, 4, 5 
    tt = torch.cat((t1, t3), dim=0)
    tp = torch.cat((t2, t4), dim=0)
    #print(l[0].shape,l[1].shape)
    ten = torch.stack((tt,tp), dim=0)
    #print(torch.sum(ten[0]-l[0]),torch.sum(ten[1]-l[1]))
    print(ten.shape)

    # print(t)
    # t1 = t.reshape(T, B, N, D).permute(1, 0, 2, 3).reshape(B, T, N * D)
    # t2 = t.reshape(T, B, N*D).permute(1, 0, 2) #.reshape(B, T, N * D)
    # print(torch.sum(t1-t2))
    # Save the tensor to a pickle file
    with open("tensor.pkl", "wb") as f:
        pickle.dump(ten, f)
    
    with open("tensor.pkl", "rb") as f:
        loaded_tensor = pickle.load(f)
    print(loaded_tensor.shape)
    print(torch.sum(loaded_tensor-ten))
    exit()
    x = 3
    if x == 1:
        with open('EGNO_exp_1\loss_seed=1_n_part=5_n_inputs=1_varDT=False_num_timesteps=2_n_layers=3_lr=0.0005_wd=1e-12.json', 'r') as file:
            data = json.load(file)

        #data['traj_loss'] : 100,20, (last dim is variable: 20/30/40)
        #print(len(data['traj_loss'][0][0])) 
        # if want to cut the traj at T
        cut = 20
        print(len(data["traj_loss"]),len(data["traj_loss"][0]),len(data["traj_loss"][0][0]))
        
        d = data['traj_loss'][-1][:]#[:cut]
        nd = [l[:cut] for l in d ]

        # d = d[:][:cut]
        # print(len(d))
        #d = torch.tensor(d)
        #print(d.shape)
        d = torch.tensor(nd)
        print(d.shape)
        traj = np.array(nd)
        traj = np.mean(traj,axis=0)
        print(traj.shape, traj)
        avg_bl = torch.mean(d,dim=0)
        print(avg_bl) # calculate mean over batches
        exit()
    elif x == 2:
        # Example usage:
        folder_path = "./SEGNO_exp_1"  # Replace with your folder path
        
        plot_graphs_from_json(folder_path, "SEGNO")

        exit()
    elif x == 3:
        l1, l2 = [0.25, 0.35, 0.15], [0.22, 0.30, 0.18]
        bar_plot_models_multiple_metrics(l1, l2)
        
        exit()
    with open('SEGNO_exp_1\loss_seed=1_n_part=5_n_steps=2_n_inputs=1_varDT=False_lr0.0005_wd1e-12_onlytest=False.json', 'r') as file:
        data = json.load(file)

    #data['traj_loss'] : 100,20, (last dim is variable: 20/30/40)
    #print(len(data['traj_loss'][0][0])) 
    # if want to cut the traj at T
    cut = 20
    print(len(data["traj_loss"]),len(data["traj_loss"][0]),len(data["traj_loss"][0][0]))
    
    d = data['traj_loss'][-1][:]#[:cut]
    nd = [l[:cut] for l in d ]

    # d = d[:][:cut]
    # print(len(d))
    #d = torch.tensor(d)
    #print(d.shape)
    d = torch.tensor(nd)
    print(d.shape)
    traj = np.array(nd)
    traj = np.mean(traj,axis=0)
    print(traj.shape, traj)
    avg_bl = torch.mean(d,dim=0)
    print(avg_bl) # calculate mean over batches

    
    exit()

    l1, l2 = [0.25, 0.35, 0.15], [0.22, 0.30, 0.18]
    bar_plot_models_multiple_metrics(l1, l2)
    
    exit()

    stepx = False
    threshold = 0.05
    with open('loss_test.json', 'r') as file:
        data = json.load(file)

    #data['traj_loss'] : 100,20, (last dim is variable: 20/30/40)
    #print(len(data['traj_loss'][0][0])) 
    # if want to cut the traj at T
    cut = 20
    d = data['traj_loss'][-1][:]#[:cut]

    nd = [l[:cut] for l in d ]

    # d = d[:][:cut]
    # print(len(d))
    #d = torch.tensor(d)
    #print(d.shape)
    d = torch.tensor(nd)
    print(d.shape)
    traj = np.array(nd)
    traj = np.mean(traj,axis=0)
    print(traj.shape, traj)
    avg_bl = torch.mean(d,dim=0)
    print(avg_bl) # calculate mean over batches
    

    """
    average error until a certain point
    do other metrics like error after x steps (eg. 10/20/30 …) 
    or after which step the error goes under a threshold
    slope (derivata prima: (e_t - e_t-1) / delta_t ) dell'errore in media o in quei punti
    """
    #error at step x 
    if stepx:
        traj = traj[1::10]
    # elif threshold>0:
    #     traj

    plot_trajectory_losses(traj)
    

    exit()
    t = torch.tensor([1,3,4])
    print(t[:3])
    exit()
    timesteps = torch.linspace(0, 10 - 1, 5, dtype=int)
    print(timesteps)
    
    exit()
    timesteps = random_ascending_tensor(length=2)
    t_list = [x.unsqueeze(0) for x in timesteps]#.reshape(1,)
    #print(t_list)
    print(timesteps)
    timesteps_rep = repeat_elements_to_exact_shape(t_list, 10)
    print(timesteps, timesteps_rep)

    exit()
    criterion = torch.nn.MSELoss()
    criterionnr = torch.nn.MSELoss(reduction='none')
    pred1 = torch.randint(0,5,(5,4),dtype=float)
    pred2 = torch.randint(0,5,(5,4),dtype=float)
    target1 = torch.randint(0,5,(5,4),dtype=float)
    target2 = torch.randint(0,5,(5,4),dtype=float)
    l1 = [pred1,pred2]
    l2 = [target1,target2]
    loss = criterion(torch.stack(l1),torch.stack(l2))
    print(loss)
    l1 = [pred1,pred2]
    l2 = [target1,target2]
    loss = criterion(torch.cat(l1),torch.cat(l2))
    
    print(loss)
    exit()
    pred = torch.mean(pred)
    target = torch.mean(target)
    loss = criterion(pred,target)
    print(pred,target)
    print(loss)
    exit()
    seed = 42
    torch.manual_seed(seed)

    np.random.seed(seed)
    N = 10
    start = 1
    end = 10
    MAX = 100
    random_tensor, cumulative_tensor = cumulative_random_tensor_indices(N, start, end, MAX)
    print("Random Tensor:", random_tensor)
    print("Cumulative Tensor:", cumulative_tensor)
    print("Final Sum of Random Tensor:", random_tensor.sum())  # This should be equal to MAX
    print("Final Cumulative Sum:", cumulative_tensor[-1])  # 
    exit()
    # Example usage
    timesteps = torch.linspace(0, 15 - 1, 2, dtype=int)
    print(timesteps)
    tensor = random_ascending_tensor(length=2)
    print(tensor)
    tensor = random_ascending_tensor(length=2)
    print(tensor)
    exit()
    T=12
    r, sizes = cumulative_random_tensor_indices(T,0,15)
    print(sizes)
    timesteps = torch.linspace(0, T - 1, 2, dtype=int)#torch.arange(T).to(x[0])
    t_list = [x.unsqueeze(0) for x in sizes]#.reshape(1,)
    #t_list = [torch.tensor(0.),torch.tensor(10.)]
    timesteps = repeat_elements_to_exact_shape(t_list, T)
    exit()
    t = torch.linspace(0, 10 - 1, 3, dtype=int).cpu().tolist()
    t = torch.randint(0,10,(4,3,2))
    n = torch.randint(0,10,(1,2))
    print(torch.arange(2))
    lis = [x for x in t]
    print(torch.cat(lis).shape,torch.stack(lis).shape)
    exit()
    ch = [-1,1]
    m = np.random.choice(ch,size=(5,1),p=[0.5,0.5])
    print(m)
    print(m.transpose())
    print(m.dot(m.transpose()))
    exit()
    lis = [x for x in t][:3]#.unsqueeze(0)
    l(lis[0])
    exit()
    v= t[0].unsqueeze(0).repeat(10, 1, 1)
    print(v.shape,v)
    d = repeat_elements_to_exact_shape(lis,10,3)#.reshape(-1,1,t.shape[-1])
    print(d.shape,d)
    exit()
    # Example usage:
    person1 = Person("Alice", 30) 
    person1.display_info()
    person1.obj.display_info()
    exit()
    timesteps = torch.arange(4)
    timesteps = torch.linspace(start=0,end=6,steps=4,dtype=int)#torch.tensor([0,2,4,6])
    timesteps = torch.tensor([1,4,5,7]) #multiple inputs and variable dt
    print(timesteps)
    t_emb = get_timestep_embedding(timesteps,4)
    print(t_emb)
    exit()
    with open('loss_comfy-sponge-21.json', 'r') as file:
        data = json.load(file)
    d = data['traj_loss'][100][10]
    print(len(data['traj_loss']))
    
    trajs = np.array(d)
    print(trajs.shape)
    plot_trajectory_losses(trajs)
    exit()
    print("test")
    results = {'eval epoch': [1,2],'losses':[], 'val loss': [3,4], 'test loss': [5,6], 'train loss': [7,8]}
    l = torch.tensor([[12],[13]]).tolist()
    results['losses'].append(l)
    results['losses'].append(l)
    
    json_object = json.dumps(results, indent=4)
    with open("loss1.json", "w") as outfile:
        outfile.write(json_object)
    exit()
    
    
    idxs,r = cumulative_random_tensor_indices(10,5,15)
    print(idxs,r)


if __name__ == "__main__":
    li = [torch.randn(3,1) for i in range(2)]
    s = repeat_elements_to_exact_shape(li,4)
    
    exit()