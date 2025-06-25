## RUN SWEEP

This code is adapted or borrows heavily from the N-Body implementation of EGNN. Used under the MIT license.

### Dataset generation : 
from:  "SEGNO\nbody\dataset_gravity" folder run command: "python -u generate_dataset.py --simulation=charged --num-train 3000 --seed 43 --suffix small --length 20000 --length_test 20000 --n_balls 20"

### EGNO

- dataset should be in "EGNO\simple" (unzipped with all the files in \simple)
- from: "EGNO" folder run the command: "python EGNO_sweep.py"
- in "SEGNO\nbody\sweep_params.py" are saved the parameters configuration for the sweep

### SEGNO

- dataset should be in "SEGNO\nbody\dataset_gravity" (unzipped with all the files in \simple)
- from: "SEGNO\nbody" folder run the command: "python SEGNO_sweep.py"
- in "SEGNO\nbody\sweep_params.py" are saved the parameters configuration for the sweep


# NEW

How to download an artifact from the sweep:

```python
import wandb
from pathlib import Path
import torch

run = wandb.init()
artifact = run.use_artifact('jet-tagging/Particle-Physics/charged_seed-2_n_part-20_n_inputs-2_varDT-False_num_timesteps-10:v0', type='results')
artifact_dir = artifact.download()

artifact_dir = next(iter(Path(artifact_dir).iterdir()))
results = torch.load(artifact_dir, weight_only=False)
results

# Data(targets=[T, traj_len, N, 3], 
#      preds=[T, traj_len, N, 3], 
#      energy_conservation=[T, traj_len, 1], 
#      test_loss=FLOAT)

```


