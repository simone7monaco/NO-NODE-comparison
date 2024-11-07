import torch
import numpy as np
from torch import nn
from motion.dataset import MotionDataset


def collector(batch):
    """
    Rebatch the input and padding zeros for loc, vel, loc_end, vel_end.
    Add the additional mask (B*N, 1) at the last.
    :param batch:
    :return: the re_batched list.
    """
    re_batch = [[] for _ in range(len(batch[0]))]
    for b in batch:
        [re_batch[i].append(d) for i, d in enumerate(b)]

    loc, vel, edge_attr, charges, loc_end, vel_end = re_batch[:6]
    res = []
    padding = [True, True, False, False, True, True, False, False, False]
    for b, p in zip(re_batch[:-1], padding[:len(re_batch) -1]):
        res.append(do_padding(b, padding=p))
    mask = generate_mask(loc)
    res.append(re_batch[-1])
    res.append(mask)
    return res


def collector_simulation(batch):
    """
    Rebatch the input and padding zeros for loc, vel, loc_end, vel_end.
    Add the additional mask (B*N, 1) at the last.
    :param batch:
    :return: the re_batched list.
    """
    re_batch = [[] for _ in range(len(batch[0]))]
    for b in batch:
        [re_batch[i].append(d) for i, d in enumerate(b)]

    assert len(re_batch) == 8
    loc, vel, edges, edge_attr, local_edge_mask, charges, loc_end, vel_end = re_batch
    max_size = max([x.size(0) for x in loc])
    node_nums = torch.tensor([x.size(0) for x in loc])
    mask = generate_mask(loc)
    loc = _padding(loc, max_size)
    vel = _padding(vel, max_size)
    edges = _pack_edges(edges, max_size)
    edge_attr = torch.cat(edge_attr, dim=0)
    local_edge_mask = torch.cat(local_edge_mask, dim=0)
    charges = _padding(charges, max_size)
    loc_end = _padding(loc_end, max_size)
    vel_end = _padding(vel_end, max_size)
    return loc, vel, edges, edge_attr, local_edge_mask, charges, loc_end, vel_end, mask, node_nums, max_size


def collector_simulation_no(batch):
    """
    Rebatch the input and padding zeros for loc, vel, loc_end, vel_end.
    Add the additional mask (B*N, 1) at the last.
    :param batch:
    :return: the re_batched list.
    """
    re_batch = [[] for _ in range(len(batch[0]))]
    for b in batch:
        [re_batch[i].append(d) for i, d in enumerate(b)]

    assert len(re_batch) == 8
    loc, vel, edges, edge_attr, local_edge_mask, charges, loc_end, vel_end = re_batch
    max_size = max([x.size(0) for x in loc])
    node_nums = torch.tensor([x.size(0) for x in loc])
    mask = generate_mask(loc)
    loc = _padding(loc, max_size)
    vel = _padding(vel, max_size)
    edges = _pack_edges(edges, max_size)
    edge_attr = torch.cat(edge_attr, dim=0)
    local_edge_mask = torch.cat(local_edge_mask, dim=0)
    charges = _padding(charges, max_size)
    loc_end = _padding_3(loc_end, max_size)
    vel_end = _padding_3(vel_end, max_size)
    return loc, vel, edges, edge_attr, local_edge_mask, charges, loc_end, vel_end, mask, node_nums, max_size


def _padding(tensor_list, max_size):
    res = [torch.cat([r, torch.zeros([max_size - r.size(0), r.size(1)])]) for r in tensor_list]
    res = torch.cat(res, dim=0)
    return res


def _padding_3(tensor_list, max_size):
    res = [torch.cat([r, torch.zeros([max_size - r.size(0), r.size(1), r.size(2)])]) for r in tensor_list]
    res = torch.cat(res, dim=0)
    return res

def pad_tensor_to_length(tensor, target_length=10, dim=0):
    # Check if the specified dimension size is already target_length or greater
    if tensor.size(dim) >= target_length:
        return tensor.narrow(dim, 0, target_length)  # Truncate if longer than target length
    
    # Calculate the padding size along the specified dimension
    padding_size = target_length - tensor.size(dim)
    
    # Get the last element along the specified dimension
    last_element = tensor.select(dim, tensor.size(dim) - 1).unsqueeze(dim)
    
    # Repeat the last element along the specified dimension to match the padding size
    repeat_sizes = [1] * tensor.dim()
    repeat_sizes[dim] = padding_size
    padding = last_element.repeat(*repeat_sizes)
    
    # Concatenate the original tensor with the padding along the specified dimension
    padded_tensor = torch.cat((tensor, padding), dim=dim)
    
    return padded_tensor

def repeat_elements_to_exact_shape(tensor_list, n, outdims=None):
    L = len(tensor_list)                    # Number of elements in the list
    repeats_per_element = n // L            # Base number of repeats per element
    remaining_repeats = n % L               # Extra repeats needed to reach exactly `n`
    outdims = outdims if outdims is not None else tensor_list[0].dim()
    # Repeat each tensor in the list `repeats_per_element` times
    repeated_tensors = [tensor.repeat(repeats_per_element, *[1] * (outdims - 1)) for tensor in tensor_list]
    
    # Add extra repeats for the first `remaining_repeats` elements in the list
    extra_repeats = [tensor_list[-1] for i in range(remaining_repeats)]
    # for tensor in repeated_tensors + extra_repeats:
    #     print(tensor.shape)
    # Concatenate all repeated tensors and the extra repeats
    final_list = repeated_tensors + extra_repeats
    final_tensor = torch.cat(final_list, dim=0)
    
    return final_tensor

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

def _pack_edges(edge_list, n_node):
    for idx, edge in enumerate(edge_list):
        edge[0] += idx * n_node
        edge[1] += idx * n_node
    return torch.cat(edge_list, dim=1)  # [2, BM]


def do_padding(tensor_list, padding=True):
    """
    Pad the input tensor_list ad
    :param tensor_list: list(B, tensor[N, *])
    :return: padded tensor [B*max_N, *]
    """
    if padding:
        max_size = max([x.size(0) for x in tensor_list])
        res = [torch.cat([r, torch.zeros([max_size - r.size(0), r.size(1)])]) for r in tensor_list]
    else:
        res = tensor_list
    res = torch.cat(res, dim=0)
    return res


def generate_mask(tensor_list):
    max_size = max([x.size(0) for x in tensor_list])
    res = [torch.cat([torch.ones([r.size(0)]), torch.zeros([max_size - r.size(0)])]) for r in tensor_list]
    res = torch.cat(res, dim=0)
    return res


def test_do_padding():
    tensor_list = [torch.ones([2, 3]), torch.zeros([4, 3])]
    res = do_padding(tensor_list)

    # tensor([[1., 1., 1.],
    #         [1., 1., 1.],
    #         [0., 0., 0.],
    #         [0., 0., 0.],
    #         [0., 0., 0.],
    #         [0., 0., 0.],
    #         [0., 0., 0.],
    #         [0., 0., 0.]])


def test_generate_mask():
    tensor_list = [torch.rand([2, 3]), torch.rand([4, 3])]
    res = generate_mask(tensor_list)
    print(res)


def test_collector():
    data_train = MotionDataset(partition='train', max_samples=100, delta_frame=30, data_dir='motion/dataset')
    loader_train = torch.utils.data.DataLoader(data_train, batch_size=2, shuffle=True, drop_last=True,
                                               num_workers=1, collate_fn=collector)
    for batch_idx, data in enumerate(loader_train):
        print(data)


class MaskMSELoss(nn.Module):
    def __init__(self):
        super(MaskMSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction="none")

    def forward(self, pred, target, mask, grouped_size=None):
        """

        :param pred: [N, d]
        :param target: [N, d]
        :param mask: [N, 1]
        :param grouped_size: [B, K], B * K = N
        :return:
        """
        assert grouped_size is None or (mask.size(0) % grouped_size.size(0) == 0)
        loss = self.loss(pred, target)
        # Looks strange, do I miss something?
        loss = (loss.T * mask).T
        if grouped_size is not None:
            loss = loss.reshape([grouped_size.size(0), -1, pred.size(-1)])
            # average loss by grouped_size on dim=1
            loss = torch.sum(loss, dim=1) / grouped_size.unsqueeze(dim=1)
            loss = torch.mean(loss)
        else:
            loss = torch.sum(loss) / (torch.sum(mask) * loss.size(-1))
        return loss


def test_MaskMSELoss():
    input = torch.rand([6, 2])
    target = torch.rand([6, 2])
    mask = torch.tensor([1, 0, 1, 0, 1, 1])
    grouped_size = torch.tensor([1, 1, 2])
    loss = MaskMSELoss()
    print(loss(input, target, mask, grouped_size))
    print(loss(input, target, mask))


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model, master_worker=True):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, master_worker)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, master_worker)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, master_worker=True):
        '''Saves model when validation loss decrease.'''
        if not master_worker:
            return
        if self.verbose and master_worker:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss