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


if __name__ == "__main__":

    print("test")
    
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