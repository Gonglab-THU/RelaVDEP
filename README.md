# RelaVDEP

## Introduction

RelaVDEP is a model-based RL framework specifically designed to optimize protein functions through a virtual DE process. The framework integrates a high-precision pre-trained protein fitness predictor as its reward function and employs a graph neural network (GNN) architecture to explicitly encode the structure-aware inter-residue relationships. Built with a distributed computational architecture, RelaVDEP supports a parallelized training process. Additionally, a multi-objective optimization strategy is designed to construct the mutant library that systematically balances functional fitness and sequence diversity.

![](figures/RelaVDEP.svg "Dynamics path")

## Installation
We recommend using conda to install the dependencies.
```
git clone https://github.com

# create the conda environment
cd RelaVDEP
conda env create -f environment.yml
conda activate relavdep
```

## Usage

### Step 1: Process data
Prepare the wild-type protein sequence (FASTA) and mutation data (CSV). For detailed instructions, see `notebook/1_prepare.ipynb`.

### Step 2: Fine-tune reward model
Select the appropriate model type and fine-tune the reward model using the mutant data. For detailed instructions, see `notebook/2_train_rm.ipynb`.

### Step 3: Run RelaVDEP
Before training, please check if the necessary files exist:

- `TARGE.fasta`: Protein sequence
- `TARGET.npz`: Mutation site constraint
- `TARGET.csv`: Mutation data
- `TARGET.pth`: Parameters of the fine-tuned reward model

Here, `TARGET` refers to the example target protein name. Then use the RelaVDEP to evolve the target protein. 

Arguments:
```
-h, --help            show this help message and exit
--fasta FASTA         Protein sequence
--rm_params RM_PARAMS
                    Supervised fine-tuned reward model parameters
--rm_type {SmallFitness,LargeFitness,SmallStab,LargeStab}
                    Type of the reward model (default: SmallFitness)
--n_layer N_LAYER     Number of downstream MLP layers (default: 1)
--restraint RESTRAINT
                    Restraint file (.npz format)
--data_dir DATA_DIR   Directory for model parameters (default: data/params)
--output OUTPUT       Output directory (default: tasks)
--temp_dir TEMP_DIR   Temporary directory for spilling object store (default: /tmp/ray)
--max_mut MAX_MUT     Maximum mutation counts (default: 4)
--n_players N_PLAYERS
                    Number of self-play workers (default: 6)
--n_sim N_SIM         Number of MCTS simulations (default: 1200)
--train_delay TRAIN_DELAY
                    Training delay (default: 2)
--n_gpus N_GPUS       Number of GPUs (default: 1)
--batch_size BATCH_SIZE
                    Batch size for training (default: 32)
--seed SEED           Random seed (default: 0)
--no_buffer           Skip saving replay buffer during training (default: False)
--unroll_steps UNROLL_STEPS
                    Number of unroll steps (default: None)
--td_steps TD_STEPS   Number of td steps (default: None)
--init_checkpoint INIT_CHECKPOINT
                    Initialized checkpoint (default: None)
--init_buffer INIT_BUFFER
                    Initialized replay buffer (default: None)
```

Example:
```
cd relavdep
python run.py --fasta data/fasta/TARGET.fasta --rm_params supervised/TARGET/TARGET.pth --rm_type SmallFitness --n_layer 5 --restraint data/restraints/TARGET.npz --output tasks/TARGET --n_gpus 4 --no_buffer 
```
Here, `n_layer` is the parameter used in the reward model training process.

### Step 4: Mutant library
Construct a mutant library that balances fitness and diversity, then use multiple filters to screen candidates for wet-lab experimental validation.

#### 1. Generate DHR embeddings
Before executing this step, please install [Dense-Homolog-Retrieval](https://github.com/ml4bio/Dense-Homolog-Retrieval) at first. Next,download the checkpoint file (`dhr2_ckpt.zip`) and unzip it in `Dense-Homolog-Retrieval` directory to obtain `dhr_cencoder.pt` and `dhr_qencoder.pt`. Then, generate sequence embeddings for all mutants as follows:
```
cd evaluate
conda activate fastMSA
python embedding.py --mutants ../tasks/TARGET/mutants.csv --output ../tasks/TARGET
```

#### 2. Construct the library
Perform DHR embedding-based clustering on mutants with fitness values above a given threshold. Subsequently, construct a mutant library that achieves a balance of high fitness and high diversity.

Arguments:
```
-h, --help            show this help message and exit
--fasta FASTA         Input protein sequence
--embedding EMBEDDING
                    DHR embedding of mutants
--output OUTPUT       Output directory path
--cutoff CUTOFF       Fitness cutoff value (default: 0)
--size SIZE           The size of mutant library (default: 10)
--seed SEED           Random seed (default: 42)
--n_cpu N_CPU         Number of cpu using in Ray (default: 10)
```
Example:
```
conda activate relavdep
python library.py --fasta ../data/fasta/TARGET.fasta --embedding ../tasks/TARGET/embeddings.pt --output ../tasks/TARGET
```

#### 3. Filter evaluation
We provide the zero-shot version of [SPIRED-Stab](https://www.nature.com/articles/s41467-024-51776-x) as a filter for evaluating stability and foldability. Run this step as follows:
```
python eval_stab.py --fasta ../data/fasta/TARGET.fasta --library ../tasks/TARGET/library.csv --output ../tasks/TARGET
```
The `library_stab.csv` is the final result file. Additionally, we recommend installing [ESMFold](https://www.science.org/doi/10.1126/science.ade2574)/[OpenFold](https://www.nature.com/articles/s41592-024-02272-z) as extra filters to further enhance the reliability of evaluation results. Here, we provide scripts (`eval_esmfold.py/eval_openfold.py`) for ESMFold/OpenFold inference in `evaluate` directory.

## Acknowledgements
We adapted some codes from SPIRED-Fitness, OpenFold and ESMFold. We thank the authors for their impressive work.
1. Chen, Y., Xu, Y., Liu, D., Xing, Y., & Gong, H. (2024). An end-to-end framework for the prediction of protein structure and fitness from single sequence. Nature Communications, 15(1), 7400. doi:10.1038/s41467-024-51776-x
2. Ahdritz, G., Bouatta, N., Floristean, C., Kadyan, S., Xia, Q., Gerecke, W., … AlQuraishi, M. (2024). OpenFold: retraining AlphaFold2 yields new insights into its learning mechanisms and capacity for generalization. Nature Methods, 21(8), 1514–1524. doi:10.1038/s41592-024-02272-z
3. Lin, Z., Akin, H., Rao, R., Hie, B., Zhu, Z., Lu, W., … Rives, A. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. Science (New York, N.Y.), 379(6637), 1123–1130. doi:10.1126/science.ade2574.
