# RelaVDEP

RelaVDEP (**Re**inforcement **L**earning **A**ssisted **V**irtual **D**irected **E**volution for **P**roteins) is a model-based reinforcement learning framework for accelerating virtual directed evolution of proteins.

## Overview

RelaVDEP optimizes protein function through virtual directed evolution. It uses a pre-trained protein fitness predictor as the reward model and a graph neural network to encode structure-aware inter-residue relationships. The framework supports distributed self-play and training, and includes a downstream multi-objective optimization step to construct mutant libraries that balance predicted fitness and sequence diversity.

![](figures/RelaVDEP.svg "Dynamics path")

## Installation

We recommend using Conda to install the dependencies.

```bash
git clone https://github.com/Gonglab-THU/RelaVDEP.git
cd RelaVDEP
conda env create -f environment.yml
conda activate relavdep
```

## Model Parameters

Download the ESM-2 and SPIRED-Fitness model parameters:

```bash
# Download zip archive
curl -o models.zip https://zenodo.org/records/17590929/files/models.zip?download=1
# Unpack and remove zip archive
unzip models.zip -d models/
rm models.zip
```

Alternatively, manually download `models.zip` from [Zenodo](https://doi.org/10.5281/zenodo.17590929) and extract the contents into the `models/` directory under the project root.

## Usage

### Step 1: Data Preparation

Prepare the target sequence and mutation data by following the examples in:

- `relavdep/data/target_sequence/TARGET.fasta`
- `relavdep/data/mutation_data/TARGET.csv`

Here and below, `TARGET` refers to the target protein name.

Prepare the fasta file and optional mutation constraints with:

```bash
python 0_prepare_inputs.py --pdb_id TARGET --sequence SEQUENCE --legal_pos 10,25,64,66 --illegal_mut A10V,G64D
```

This writes the fasta file to `relavdep/data/target_sequence/TARGET.fasta`. If any constraint option is provided, it also writes `relavdep/data/mutation_constraint/TARGET.npz` containing `illegal` and `legal` action arrays.

### Step 2: Reward Model Preparation

Fine-tune the reward model with supervised mutation data:

```bash
python 1_train_reward_model.py --fasta relavdep/data/target_sequence/TARGET.fasta --data relavdep/data/mutation_data/TARGET.csv --output outputs/TARGET
```

Run `python 1_train_reward_model.py -h` to view all optional arguments. Training metrics are logged to Weights & Biases; use `--wandb_project` and `--wandb_mode` to control logging.

Important outputs:

- `outputs/TARGET/TARGET.pth`: fine-tuned reward model parameters.
- `relavdep/data/mutation_constraint/TARGET.npz`: mutation constraints inferred from predicted beneficial single mutants, unless `--constraint` is provided.
- `n_layer`: best MLP layer count, reported when `--cross_val` is enabled.
- `cutoff`: predicted wild-type fitness, used as the library construction cutoff in Step 4.

If you already have a prepared `.npz` file containing `illegal` and `legal` arrays, pass it with `--constraint` to skip beneficial mutation prediction.

### Step 3: Virtual Directed Evolution

Run virtual directed evolution:

```bash
python 2_run_directed_evolution.py \
    --fasta relavdep/data/target_sequence/TARGET.fasta \
    --rm_params outputs/TARGET/TARGET.pth \
    --constraint relavdep/data/mutation_constraint/TARGET.npz \
    --output outputs/TARGET
```

Here, `--rm_params` and `--constraint` are obtained from Step 2.

Useful options:

- `--rm_type {small,large}`: choose the reward model type.
- `--log_interval`: control console progress logging frequency.
- `--warmup_steps`: enable learning-rate warmup before decay.
- `--lr_decay_steps` and `--lr_decay_rate`: control learning-rate decay.
- `--save_buffer`: save `replay_buffer.pkl`; disabled by default.

Run `python 2_run_directed_evolution.py -h` to view all optional arguments.

Important outputs:

- `checkpoint.pth`: latest RelaVDEP training checkpoint.
- `mutants.csv`: mutants obtained during virtual directed evolution. The file includes sequence, predicted fitness, and source information such as `player_id` and `play_id`.
- Weights & Biases run files: training logs.
- `replay_buffer.pkl`: replay buffer, only written when `--save_buffer` is used.

### Step 4: Mutant Library Construction

Before running this step, clone [Dense-Homolog-Retrieval](https://github.com/ml4bio/Dense-Homolog-Retrieval) into the `scripts/` directory and build the `fastMSA` Conda environment following its official instructions. Then download `dhr2_ckpt.zip` from the official repository and extract it into `scripts/Dense-Homolog-Retrieval/` to obtain `dhr_cencoder.pt` and `dhr_qencoder.pt`.

Construct the optimized mutant library:

```bash
python 3_build_mutant_library.py \
    --fasta relavdep/data/target_sequence/TARGET.fasta \
    --mutants outputs/TARGET/mutants.csv \
    --output outputs/TARGET \
    --cutoff CUTOFF
```

Here, `CUTOFF` is the wild-type fitness value reported in Step 2.

Run `python 3_build_mutant_library.py -h` to view all optional arguments.

Important outputs:

- `library.csv`: **Optimized mutant library (the final result containing recommended mutants)**
- `library.png`: Distribution of the selected mutants in 2D space
- `frequency.png`: Mutation frequency of the selected mutants

RelaVDEP uses the zero-shot version of [SPIRED-Stab](https://www.nature.com/articles/s41467-024-51776-x) as a filter to predict stability ($\Delta\Delta G$ and $\Delta T_m$) and foldability ($pLDDT$) for the mutant library. We also recommend using [ESMFold](https://www.science.org/doi/10.1126/science.ade2574) and [OpenFold](https://www.nature.com/articles/s41592-024-02272-z) as additional filters to further improve evaluation reliability.

### Optional: Run Script

For convenience, `bin/run.sh` provides an example command for running Step 3:

```bash
bash bin/run.sh <cuda_device_ids> <run_name>
```

For example:

```bash
bash bin/run.sh 0,1 avGFP_test
```

Please update the target-specific paths and parameters inside `bin/run.sh` before using it for a new protein.

## Acknowledgements

We adapted code from SPIRED-Fitness and other open-source projects. We thank the authors for their impressive work.

1. Chen, Y., Xu, Y., Liu, D., Xing, Y., & Gong, H. (2024). An end-to-end framework for the prediction of protein structure and fitness from single sequence. Nature Communications, 15(1), 7400. doi:10.1038/s41467-024-51776-x
2. Ahdritz, G., Bouatta, N., Floristean, C., Kadyan, S., Xia, Q., Gerecke, W., … AlQuraishi, M. (2024). OpenFold: retraining AlphaFold2 yields new insights into its learning mechanisms and capacity for generalization. Nature Methods, 21(8), 1514–1524. doi:10.1038/s41592-024-02272-z
3. Lin, Z., Akin, H., Rao, R., Hie, B., Zhu, Z., Lu, W., … Rives, A. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. Science (New York, N.Y.), 379(6637), 1123–1130. doi:10.1126/science.ade2574.
4. Duvaud, W., & Hainaut, A. (2019). MuZero General: Open Reimplementation of MuZero. GitHub repository. GitHub. https://github.com/werner-duvaud/muzero-general
