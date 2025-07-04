import numpy as np
import torch
from torch_geometric.utils import to_dense_batch

def reshape_sample(sample):
    """
    Convert (T, N*3) -> (T, 3, N)
    """
    T, N3 = sample.shape
    N = N3 // 3
    sample = sample.reshape(T, N, 3)        # (T, N, 3)
    sample = sample.transpose(0, 2, 1)      # (T, 3, N)
    return sample

def random_ascending_tensor(length, min_value=0, max_value=9):
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

def cumulative_random_tensor_indices(size, start, end):
    # Generate the cumulative numpy array as before
    random_array = torch.randint(start, end, size=(size,))
    
    cumulative_tensor = torch.cumsum(random_array,dim=0)
    
    return cumulative_tensor,  random_array


def cumulative_random_tensor_indices_capped(N, start, end, MAX=100):
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
    return cumulative_tensor, scaled_array



def tot_energy_spring(loc, vel, edges, interaction_strength=.1):
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


def tot_energy_charged(loc, vel, edges, interaction_strength=1):
    """
    loc: np.array of shape (3, N)
    vel: np.array of shape (3, N)
    edges: np.array of shape (N, N) with interaction strengths
    """
    # disables division by zero warning, since I fix it with fill_diagonal
    with np.errstate(divide='ignore'):

        # K = 0.5 * (vel ** 2).sum()
        K = (0.5 * np.linalg.norm(vel, axis=1)**2).sum()
        U = 0
        for i in range(loc.shape[1]):
            for j in range(loc.shape[1]):
                if i != j:
                    r = loc[:, i] - loc[:, j]
                    # dist = np.sqrt((r ** 2).sum())
                    # if overflow occurs, return np.inf
                    dist = np.linalg.norm(r)
                    U += 0.5 * interaction_strength * edges[
                        i, j] / dist
        return U + K


def tot_energy_charged_batch(loc, vel, edges, interaction_strength=1):
    """
    loc, vel: np.array of shape (T, N, 3)
    edges: np.array of shape (N, N) with interaction strengths
    """
    assert loc.shape[-1] == 3, "loc must have shape (T, N, 3)"
    assert edges.shape[-1] == edges.shape[-2] and edges.shape[-1] == loc.shape[1], \
        "edges must have shape (T, N, N)"
    K = 0.5 * np.sum(np.sum(vel**2, axis=-1), axis=-1)  # (T,)

    # calculate pairwise distances on the last dimension
    dist = loc[:, :, np.newaxis, :] - loc[:, np.newaxis, :, :]  # (T, N, N, 3)
    dist = np.linalg.norm(dist, axis=-1)  # (T, N, N)
    dist = np.where(dist == 0, np.inf, dist)  # avoid division by zero
    
    if edges.ndim == 2:
        edges = np.expand_dims(edges, axis=0)
    U = 0.5 * interaction_strength * np.sum(edges / dist, axis=(-1, -2)).view()  # (T,)
    return K + U.squeeze()


def tot_energy_gravity(pos, vel, mass, G=1.0):
    assert pos.shape[1] == 3, "Position must have shape (N, 3)"
    # Kinetic Energy:
    # pos, vel: np.array of shape (N, 3)
    # mass: np.array of shape (N, 1)

    KE = 0.5 * np.sum(np.sum(mass * vel**2))

    # Potential Energy:

    # positions r = [x,y,z] for all particles
    x = pos[:, 0:1]
    y = pos[:, 1:2]
    z = pos[:, 2:3]

    # matrix that stores all pairwise particle separations: r_j - r_i
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z

    # matrix that stores 1/r for all particle pairwise particle separations
    inv_r = np.sqrt(dx**2 + dy**2 + dz**2)
    inv_r[inv_r > 0] = 1.0/inv_r[inv_r > 0]

    # sum over upper triangle, to count each interaction only once
    PE = G * np.sum(np.sum(np.triu(-(mass*mass.T)*inv_r, 1)))
    return KE+PE

def tot_energy_gravity_batch(loc, vel, mass, G=1.0):
    # pos, vel: np.array of shape (T, N, 3) 
    # mass: np.array of shape (T, N, 1)

    assert loc.shape[-1] == 3, "Position must have shape (T, N, 3)"
    KE = np.squeeze(0.5 * np.sum(np.sum(mass * vel**2, axis=-1), axis=-1))  # (T,)

    # Potential Energy:
    # positions r = [x,y,z] for all particles
    x = loc[:, :, 0:1]  # (T, N, 1)
    y = loc[:, :, 1:2]  # (T, N, 1)
    z = loc[:, :, 2:3]  # (T, N, 1)

    dx = x.transpose(0, 2, 1) - x  # (T, N, N)
    dy = y.transpose(0, 2, 1) - y  # (T, N, N)
    dz = z.transpose(0, 2, 1) - z  # (T, N, N)
    inv_r = np.sqrt(dx**2 + dy**2 + dz**2) # (T, N, N)
    inv_r[inv_r > 0] = 1.0 / inv_r[inv_r > 0]  # avoid division by zero

    PE = G * np.sum(np.sum(np.triu(-(mass * mass.transpose(0, 2, 1)) * inv_r, 1), axis=-1), axis=-1)  # (T,)
    return KE + PE

def conserved_energy_fun(dataset, loc, vel, edges, batch=None):
    assert edges.ndim == 2 and edges.shape[1] == 1
    edge_matr = to_dense_batch(edges, batch)[0]
    edge_single = edge_matr.cpu().numpy()
    edge_matr = edge_matr.cpu().numpy().repeat(edge_matr.shape[1], axis=2)
    edge_matr = np.einsum('tij,tji ->tij', edge_matr, edge_matr)

    if loc.ndim == 2:
        loc, _ = to_dense_batch(loc, batch)
        vel, _ = to_dense_batch(vel, batch)
    nploc = loc.cpu().numpy() # (B, N, 3)
    npvel = vel.cpu().numpy()

    # for i in range(nploc.shape[0]):
    if dataset == "gravity":
        energies = tot_energy_gravity_batch(nploc, npvel, edge_single)
    elif dataset == "charged":
        energies = tot_energy_charged_batch(nploc, npvel, edge_matr)
    # elif dataset == "spring":
    #     energies = tot_energy_spring_batch(nploc.transpose(0, 2, 1), npvel.transpose(0, 2, 1), edge_matr)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return energies

def compute_energy_drift(loc, vel, edges):
    """
    Compute relative energy drift at each timestep from a trajectory.
    
    Parameters:
    - loc: np.array of shape (T, 3, N)
    - vel: np.array of shape (T, 3, N)
    - edges: np.array of shape (N, N)
    
    Returns:
    - energy_drift: np.array of shape (T,)
    """
    T = loc.shape[0]
    energy_drift = np.zeros(T)

    # Initial energy
    E0 = tot_energy_charged(loc[0], vel[0], edges)

    for t in range(T):
        Et = tot_energy_charged(loc[t], vel[t], edges)
        energy_drift[t] = np.abs((Et - E0) / (E0 + 1e-10))  # epsilon for stability

    return energy_drift

def compute_energy_drift_batch(loc, vel, edges):
    all_energy_drifts = []

    for i in range(loc.shape[0]):
        loc_i = reshape_sample(loc[i])
        vel_i = reshape_sample(vel[i])
        drift = compute_energy_drift(loc_i, vel_i, edges)
        all_energy_drifts.append(drift)

    # Optionally convert to np.array: shape (num_samples, T)
    all_energy_drifts = np.stack(all_energy_drifts)

    return all_energy_drifts



def pearson_correlation_batch(x, y, N):
    """
    Compute the Pearson correlation for each time step (T) in each batch (B).
    
    Args:
    - x: Tensor of shape (T, B*N, 3), predicted states.
    - y: Tensor of shape (T, B*N, 3), ground truth states.
    
    Returns:
    - correlations: Tensor of shape (B, T), Pearson correlation for each time step in each batch.
    """
    
    # Reshape to (B, T, N*3) 
    
    T = x.shape[0] 
    cut = int(0.4 * T)  # Calculate 40% of the total elements to avoid NaN values
    B = x.size(1) // N
    x = x.reshape( T, B, -1)[:cut].transpose(0,1)  # Flatten N and 3 into a single dimension
    y = y.reshape( T, B, -1)[:cut].transpose(0,1)
    
    
    # Mean subtraction
    mean_x = x.mean(dim=2, keepdim=True)
    mean_y = y.mean(dim=2, keepdim=True)
    
    xm = x - mean_x
    ym = y - mean_y

    # Compute covariance between x and y along the flattened dimensions
    covariance = (xm * ym).sum(dim=2)

    # Compute standard deviations along the flattened dimensions
    std_x = torch.sqrt((xm ** 2).sum(dim=2))
    std_y = torch.sqrt((ym ** 2).sum(dim=2))

    # Compute Pearson correlation for each sample in the batch
    correlation = covariance / (std_x * std_y)

    #number of steps before reaching a value of correlation, between prediction and ground truth for each timesteps, lower than 0.5
    num_steps_batch = []

    for i in range(correlation.shape[0]):
        
        if any(correlation[i] < 0.5):
            num_steps_before = (correlation[i] < 0.5).nonzero(as_tuple=True)[0][0].item()
            
        else:
            num_steps_before = cut
        num_steps_batch.append(num_steps_before)

    # Check if all values along B dimension are >= 0.5 for each T
    mask = torch.all(correlation >= 0.5, dim=0)

    # Convert the boolean mask to int for argmax
    first_failure_index = torch.argmax(~mask.int()).item()
    # If no failures, return the number of columns as the "end"
    if mask.all():
        first_failure_index = correlation.size(1)       
    
    #return the minimum first index along T dimension after which correlation drops below the threshold                                 
    return correlation, torch.mean(torch.Tensor(num_steps_batch)), first_failure_index 