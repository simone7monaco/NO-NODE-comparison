## RUN SWEEP

# Dataset generation : 
from:  "SEGNO\nbody\dataset_gravity" folder run command: "python -u generate_dataset.py --simulation=charged --num-train 3000 --seed 43 --suffix small --length 20000 --length_test 20000 --n_balls 20"

# EGNO

- dataset should be in "EGNO\simple"
- from: "EGNO" folder run the command: "python EGNO_sweep.py"
- in "SEGNO\nbody\sweep_params.py" are saved the parameters configuration for the sweep

# SEGNO

- dataset should be in "SEGNO\nbody\dataset_gravity"
- from: "SEGNO\nbody" folder run the command: "python SEGNO_sweep.py"
- in "SEGNO\nbody\sweep_params.py" are saved the parameters configuration for the sweep