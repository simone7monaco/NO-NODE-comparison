import numpy as np

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
        loc: np.array of shape (T, 3, N)
        vel: np.array of shape (T, 3, N)
        edges: np.array of shape (N, N) with interaction strengths
        """
        K = 0.5 * (vel ** 2).sum()
        N = loc.shape[2]
        
        # Pairwise differences: shape (T, 3, N, N)
        diff = loc[:, :, :, None] - loc[:, :, None, :]  # (T, 3, N, N)
        dist = np.sqrt(np.sum(diff ** 2, axis=1))       # (T, N, N)
        np.fill_diagonal(dist[0], np.inf)
        inv_dist = np.where(dist > 0, 1.0 / dist, 0.0)

        # Symmetric interactions (i != j) – sum over upper triangle to avoid double-counting
        triu_mask = np.triu(np.ones((N, N)), k=1)[None, :, :]  # shape (1, N, N)

        U = interaction_strength * np.sum(inv_dist * edges[None, :, :] * triu_mask)
        return U + K


def tot_energy_gravity(pos, vel, mass, G=1.0):
        # Kinetic Energy:
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

        return KE, PE, KE+PE

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