import numpy as np
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
    K = 0.5 * np.sum(np.sum(vel**2, axis=-1), axis=-1)  # (T,)

    # calculate pairwise distances on the last dimension
    dist = loc[:, :, np.newaxis, :] - loc[:, np.newaxis, :, :]  # (T, N, N, 3)
    dist = np.linalg.norm(dist, axis=-1)  # (T, N, N)
    dist = np.where(dist == 0, np.inf, dist)  # avoid division by zero
    
    U = 0.5 * interaction_strength * np.sum(np.expand_dims(edges, axis=0) / dist, axis=(-1, -2)).view()  # (T,)
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
    edge_matr, _ = to_dense_batch(edges, batch)
    edge_matr = edge_matr.cpu().numpy().repeat(edge_matr.shape[1], axis=2)
    edge_matr = np.einsum('tij,tji ->tij', edge_matr, edge_matr)

    edge_single = to_dense_batch(edges, batch)[0].cpu().numpy()
    nploc, _ = to_dense_batch(loc, batch)
    npvel, _ = to_dense_batch(vel, batch)
    nploc = nploc.cpu().numpy() # (B, N, 3)
    npvel = npvel.cpu().numpy()

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