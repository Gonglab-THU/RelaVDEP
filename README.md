# RelaVDEP
RelaVDEP (**Re**inforcement **L**earning **A**ssisted **V**irtual **D**irected **E**volution for **P**roteins) is a model-based reinforcement learning framework for accelerating virtual directed evolution of proteins.

## Overview

RelaVDEP is a model-based Reinforcement Learning framework specifically designed to optimize protein functions through a virtual Directed Evolution process. The framework integrates a high-precision pre-trained protein fitness predictor as its reward model and employs a Graph Neural Network architecture to explicitly encode the structure-aware inter-residue relationships. Built with a distributed computational architecture, RelaVDEP supports a parallelized training process. Additionally, a multi-objective optimization strategy is designed to construct the mutant library that systematically balances functional fitness and sequence diversity.

![](figures/RelaVDEP.svg "Dynamics path")

## Installation
We recommend using conda to install the dependencies.
```
git clone https://github.com/Gonglab-THU/RelaVDEP.git
cd RelaVDEP
conda env create -f environment.yml
conda activate relavdep
```

## Data Access
Download ESM-2 and SPIRED-Fitness model parameters by running the following:

```
# Download zip archive
curl -o models.zip https://zenodo.org/records/17590929/files/models.zip?download=1
# Unpack and remove zip archive
unzip models.zip -d models/
rm models.zip
```

Alternatively, you can manually download the model parameters (`models.zip`) from [Zenodo](https://doi.org/10.5281/zenodo.17590929) and unzip the contents into the `models/` directory within the `RelaVDEP` project folder.

## Usage

### Step 1: Data Preparation
Please strictly refer to `TARGET.fasta` and `TARGET.csv` in the `relavdep/data/target_sequence` and `relavdep/data/mutation_data` directories for target protein sequence and mutation data preparation. Here and below, `TARGET` refers to the target protein name.

### Step 2: Reward Model Preparation

Supervised fine-tune the reward model via:

```
python 1_supervised_training.py --fasta relavdep/data/target_sequence/TARGET.fasta --data relavdep/data/mutation_data/TARGET.csv --output outputs/TARGET
```
Run `python 1_supervised_training.py -h` to view all optional arguments. Upon completion, the following information and files will be obtained:

- Parameters to be used in Step 3:
  - `TARGET.pth`: Parameters of the reward model
  - `TARGET.npz`: Mutation site constraints
  - `n_layer`: MLP Layer Count (Only provided after cross-validation)

- Parameter to be used in Step 4:
  - `cutoff`: Fitness cutoff value of the target protein

### Step 3: Virtual Directed Evolution
Apply the RelaVDEP model to evolve the target protein via:

```
python 2_directed_evolution.py --fasta relavdep/data/target_sequence/TARGET.fasta --rm_params outputs/TARGET/TARGET.pth --constraint relavdep/data/mutation_constraint/TARGET.npz --output outputs/TARGET
```

Here, reward model parameters (`--rm_params`) and mutation constraints (`--constraint`) are obtained in step 2.

Run `python 2_directed_evolution.py -h` to get optional arguments. After completing this step, the following files will be obtained:

- `checkpoint.pth`: Checkpoint during RelaVDEP execution
- `events.out.tfevents.*`: Training logs of RelaVDEP
- `mutants.csv`: All mutants obtained through virtual directed evolution
- `replay_buffer.pkl`: Replay buffer of RelaVDEP

### Step 4: Mutant Library Construction
Before executing this step, please clone the repository `Dense-Homolog-Retrieval.git` into `scripts/` directory and build `fastMSA` environment following the official instructions ([Dense-Homolog-Retrieval](https://github.com/ml4bio/Dense-Homolog-Retrieval)). Then, download the checkpoints (`dhr2_ckpt.zip`) in the official repository and unzip it directly into `scripts/Dense-Homolog-Retrieval/` directory to obtain `dhr_cencoder.pt` and `dhr_qencoder.pt`.

Construct the optimized mutant library via:

```
python 3_construct_library.py  --fasta relavdep/data/target_sequence/TARGET.fasta --mutants outputs/TARGET/mutants.csv --output outputs/TARGET/ --cutoff cutoff
```

Here, the fitness cutoff (`--cutoff`) is obtained in step 2.

Run `python 3_construct_library.py -h` to get optional arguments. After completing this step, the following files will be obtained:

- `library.csv`: **Optimized mutant library (the final result containing recommended mutants)**
- `library.png`: Distribution of the selected mutants in 2D space
- `frequency.png`: Mutation frequency of the selected mutants

We provide the zero-shot version of [SPIRED-Stab](https://www.nature.com/articles/s41467-024-51776-x) as a filter to predict stability ($\Delta\Delta G$ & $\Delta T_m$) and foldability ($pLDDT$) for mutant library. Additionally, we recommend using [ESMFold](https://www.science.org/doi/10.1126/science.ade2574) and [OpenFold](https://www.nature.com/articles/s41592-024-02272-z) as extra filters to further enhance the reliability of evaluation results.

## Acknowledgements
We adapted some codes from SPIRED-Fitness and other projects. We thank the authors for their impressive work.
1. Chen, Y., Xu, Y., Liu, D., Xing, Y., & Gong, H. (2024). An end-to-end framework for the prediction of protein structure and fitness from single sequence. Nature Communications, 15(1), 7400. doi:10.1038/s41467-024-51776-x
2. Ahdritz, G., Bouatta, N., Floristean, C., Kadyan, S., Xia, Q., Gerecke, W., … AlQuraishi, M. (2024). OpenFold: retraining AlphaFold2 yields new insights into its learning mechanisms and capacity for generalization. Nature Methods, 21(8), 1514–1524. doi:10.1038/s41592-024-02272-z
3. Lin, Z., Akin, H., Rao, R., Hie, B., Zhu, Z., Lu, W., … Rives, A. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. Science (New York, N.Y.), 379(6637), 1123–1130. doi:10.1126/science.ade2574.
4. Duvaud, W., & Hainaut, A. (2019). MuZero General: Open Reimplementation of MuZero. GitHub repository. GitHub. https://github.com/werner-duvaud/muzero-general
