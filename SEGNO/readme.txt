** This is a PyTorch implementation of SEGNO. **
** It is also on a private github, we shall publish it right after the review process. **

This code is adapted from the implementation of [EGNN](https://github.com/vgsatorras/egnn) and [SEGNN](https://github.com/RobDHess/Steerable-E3-GNN).

1. unzip SEGNO.zip
2. cd SEGNO

- Dataset generation
cd nbody/dataset_gravity
python3 -u generate_dataset.py --simulation=gravity --num-train 10000 --seed 43

- Run the model
python main.py --dataset=nbody --epochs=500 --max_samples=3000 --layers=8 --hidden_features=64 --norm=none --batch_size=100 --gpu=1 --weight_decay=1e-12