from synthetic_sim import ChargedParticlesSim, SpringSim, GravitySim
import time
import numpy as np
import argparse
from pathlib import Path
from tqdm import trange

"""
nbody_small:   python3 -u generate_dataset.py --simulation=charged --num-train 10000 --seed 43 --suffix small
    python -u generate_dataset.py --simulation=charged --num-train 3000 --seed 43 --suffix small --length 20000 --length_test 20000 --n_balls 20
gravity_small: python3 -u generate_dataset.py --simulation=gravity --num-train 10000 --seed 43 --suffix small
"""

parser = argparse.ArgumentParser()
parser.add_argument('--simulation', type=str, default='charged', choices=['springs', 'charged', 'gravity'],
                    help='What simulation to generate.')
parser.add_argument('--num-train', type=int, default=10000,
                    help='Number of training simulations to generate.')
parser.add_argument('--num-valid', type=int, default=2000,
                    help='Number of validation simulations to generate.')
parser.add_argument('--num-test', type=int, default=2000,
                    help='Number of test simulations to generate.')
parser.add_argument('--length', type=int, default=20000,
                    help='Length of trajectory.')
parser.add_argument('--length_test', type=int, default=52000,
                    help='Length of test set trajectory.')
parser.add_argument('--sample-freq', type=int, default=100,
                    help='How often to sample the trajectory.')
parser.add_argument('--n_balls', type=int, default=5,
                    help='Number of balls in the simulation.')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed.')
parser.add_argument('--initial_vel', type=int, default=1,
                    help='consider initial velocity')
parser.add_argument('--suffix', type=str, default="",
                    help='add a suffix to the name')

args = parser.parse_args()

initial_vel_norm = 0.5
if not args.initial_vel:
    initial_vel_norm = 1e-16

if args.simulation == 'springs':
    sim = SpringSim(noise_var=0.0, n_balls=args.n_balls)
    suffix = '_springs'
elif args.simulation == 'charged':
    sim = ChargedParticlesSim(noise_var=0.0, n_balls=args.n_balls, vel_norm=initial_vel_norm)
    suffix = '_charged'
elif args.simulation == 'gravity':
    sim = GravitySim(noise_var=0.0, n_balls=args.n_balls, vel_norm=initial_vel_norm)
    suffix = '_gravity'
else:
    raise ValueError('Simulation {} not implemented'.format(args.simulation))

suffix += str(args.n_balls) + "_initvel%d" % args.initial_vel + args.suffix
np.random.seed(args.seed)

print(suffix)


def generate_dataset(num_sims, length, sample_freq):
    loc_all = list()
    vel_all = list()
    edges_all = list()
    charges_all = list()
    if hasattr(sim, 'sample_trajectory_batch'):
        # Use batched sampling if available
        batch_size = 50
        num_batches = num_sims // batch_size + (num_sims % batch_size > 0)
        for i in trange(num_batches):
            t = time.time()
            loc, vel, edges, charges = sim.sample_trajectory_batch(T=length,
                                                                  sample_freq=sample_freq,
                                                                  batch_size=batch_size)
            loc_all.append(loc)
            vel_all.append(vel)
            edges_all.append(edges)
            charges_all.append(charges)
            
        loc_all = np.concatenate(loc_all, axis=0)
        vel_all = np.concatenate(vel_all, axis=0)
        edges_all = np.concatenate(edges_all, axis=0)
        charges_all = np.concatenate(charges_all, axis=0)
        return loc_all, vel_all, edges_all, charges_all
    
    for i in trange(num_sims):
        t = time.time()
        loc, vel, edges, charges = sim.sample_trajectory(T=length,
                                                         sample_freq=sample_freq)

        loc_all.append(loc)
        vel_all.append(vel)
        edges_all.append(edges)
        charges_all.append(charges)

        # if i % 100 == 0:
        #     print("Iter: {}, Simulation time: {}".format(i, time.time() - t))

    charges_all = np.stack(charges_all)
    loc_all = np.stack(loc_all)
    vel_all = np.stack(vel_all)
    edges_all = np.stack(edges_all)

    return loc_all, vel_all, edges_all, charges_all


if __name__ == "__main__":

    # vel = np.load("vel_train_charged5_initvel1small.npy")
    # edges = np.load("edges_train_charged5_initvel1small.npy")
    # loc = np.load("loc_train_charged5_initvel1small.npy")
    # charges = np.load("charges_train_charged5_initvel1small.npy")
    # print(vel.shape,edges.shape,loc.shape,charges.shape)
    # exit()

    outdir = Path('data')

    print("Generating {} training simulations".format(args.num_train))
    loc_train, vel_train, edges_train, charges_train = generate_dataset(args.num_train,
                                                                        args.length,
                                                                        args.sample_freq)

    print("Generating {} validation simulations".format(args.num_valid))
    loc_valid, vel_valid, edges_valid, charges_valid = generate_dataset(args.num_valid,
                                                                        args.length,
                                                                        args.sample_freq)

    print("Generating {} test simulations".format(args.num_test))
    loc_test, vel_test, edges_test, charges_test = generate_dataset(args.num_test,
                                                                    args.length_test,
                                                                    args.sample_freq)

    np.save(outdir / f'loc_train{suffix}.npy', loc_train)
    np.save(outdir / f'vel_train{suffix}.npy', vel_train)
    np.save(outdir / f'edges_train{suffix}.npy', edges_train)
    np.save(outdir / f'charges_train{suffix}.npy', charges_train)

    np.save(outdir / f'loc_valid{suffix}.npy', loc_valid)
    np.save(outdir / f'vel_valid{suffix}.npy', vel_valid)
    np.save(outdir / f'edges_valid{suffix}.npy', edges_valid)
    np.save(outdir / f'charges_valid{suffix}.npy', charges_valid)

    np.save(outdir / f'loc_test{suffix}.npy', loc_test)
    np.save(outdir / f'vel_test{suffix}.npy', vel_test)
    np.save(outdir / f'edges_test{suffix}.npy', edges_test)
    np.save(outdir / f'charges_test{suffix}.npy', charges_test)
