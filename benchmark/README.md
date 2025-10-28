# Evaluation of the Fitness Predictor


## Overview

## Data Access

* Reference file: 

* DMS assays:

## Usage

### Step 1: Extract sequence embeddings

All sequence embeddings of ProteinGym DMS assays (`embeddings.zip`) can be downloaded from [Zenodo](https://zenodo.org/doi/10.5281/zenodo.15720582). The full zip archive takes up approximately 44GB of storage.

Alternatively, you can extract embeddings for the specific assay data via:

```
python 1_extract_embeddings.py \ 
    cv_splits=multiples
    cv_scheme=fold_rand_multiples
```

### Step 2: Evaluation
To evaluate the fitness predictor on the full benchmark, run the following:

```
python 2_proteingym_benchmark.py --multirun \ 
    cv_splits=singles
    cv_scheme=fold_random_5,fold_modulo_5,fold_contiguous_5
```

To evaluate a single dataset from the ProteinGym assays (e.g., `GFP_AEQVI_Sarkisyan_2016`), run the following:
```
python 2_proteingym_benchmark.py \
    cv_splits=multiples
    cv_scheme=fold_rand_multiples
    dataset=single
    single_id=GFP_AEQVI_Sarkisyan_2016
```

All commands in step 1 and step 2 can refer to the following explanations:

- `cv_splits`: Determines the data splits used for supervised cross-validation. Must be one of the following:
    * `singles`: Only utilizes single mutants (variants with only one mutation).
    * `multiples`: Utilizes all mutants in the dataset (including variants with multiple mutations).

- `cv_scheme`: Defines the method used for generating cross-validation folds.
    * If `cv_splits` is set to `singles`, it must be chosen from [`fold_random_5`, `fold_modulo_5`, `fold_contiguous_5`]
    * If `cv_splits` is set to `multiples`, it must be `fold_rand_multiples`.
