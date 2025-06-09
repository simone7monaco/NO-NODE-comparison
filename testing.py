import numpy as np
import torch

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

def cumulative_random_tensor_indices_np(n, start, end):
    # Generate the cumulative numpy array as before
    
    random_array = np.random.randint(start, end, size=n)
    print(random_array)
    cumulative_array = np.cumsum(random_array)
    print(cumulative_array)
    
    # Convert the cumulative numpy array to a PyTorch tensor
    cumulative_tensor = torch.tensor(cumulative_array, dtype=torch.long)
    
    return cumulative_tensor,  torch.tensor(random_array, dtype=torch.long)

def cumulative_random_tensor_indices(n, start, end):
    # Generate the cumulative numpy array as before
    random_array = torch.randint(start, end, size=(n,))
    print(random_array)
    cumulative_tensor = torch.cumsum(random_array,dim=0)
    
    return cumulative_tensor,  random_array


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
    E0 = tot_energy(loc[0], vel[0], edges)

    for t in range(T):
        Et = tot_energy(loc[t], vel[t], edges)
        energy_drift[t] = np.abs((Et - E0) / (E0 + 1e-10))  # epsilon for stability

    return energy_drift



        
if __name__ == "__main__":

    print("test")
    masses = 1.0 + 0.1 * (2 * np.random.rand(3, 1) - 1)
    print(np.ones((3, 1)).shape,masses.shape)
    masses = np.ones((5, 1))
    edges = masses.dot(masses.T)  # shape: (n, n)
    print(edges)
    exit()
    idxs,r = cumulative_random_tensor_indices(10,5,15)
    print(idxs,r)
    size=idxs[-1]
    
    t = torch.zeros((size,2))
    for i in range(10):
        if i ==0:
            x=torch.randint(0, 5, size=(r[i],2))
            t[:idxs[i]]=x
            print(x)
        else:
            x=torch.randint(0, 5, size=(r[i],2))
            t[idxs[i-1]:idxs[i]]= x
            print(x)
    print("t")
    print(t)
    print(size,t.shape)
    exit()
    start=30
    idxs+=start
    print(idxs)
    print(idxs[-1]-start)
    print((idxs[0]-start)+(idxs[-1]-idxs[0]))
    exit()

    li = [torch.randn(3,1) for i in range(2)]
    s = repeat_elements_to_exact_shape(li,4)
    
    exit()