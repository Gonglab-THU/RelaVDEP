#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash run_pipeline.sh --target TARGET [options]

Required:
  --target TARGET              Target name used for default input/output paths.

Options:
  --fasta PATH                 Fasta file. Default: relavdep/data/target_sequence/TARGET.fasta
  --data PATH                  Mutation data CSV. Default: relavdep/data/mutation_data/TARGET.csv
  --output DIR                 Output directory. Default: outputs/TARGET
  --constraint PATH            Prepared mutation constraint NPZ. Skips beneficial mutation prediction in step 1.
  --cutoff VALUE               Fitness cutoff for library construction. If omitted, parsed from step 1 log.
  --library-size INT           Mutant library size passed to step 3. Default: 10
  --seed INT                   Random seed passed to all steps. Default: 42
  --n-gpus INT                 Number of GPUs passed to step 2. Default: 1
  --n-player INT               Number of self-play workers passed to step 2. Default: 6
  --n-sim INT                  Number of MCTS simulations passed to step 2. Default: 1200
  --batch-size INT             Batch size passed to steps 1 and 2. Default: 32
  --cross-val                  Enable cross validation in step 1.
  --wandb-project NAME         Weights & Biases project for step 2. Default: RelaVDEP
  --wandb-mode MODE            Weights & Biases mode: online, offline, or disabled.
  -h, --help                   Show this help message.
EOF
}

target=""
fasta=""
data=""
output=""
constraint=""
cutoff=""
library_size=10
seed=42
n_gpus=1
n_player=6
n_sim=1200
batch_size=32
cross_val=false
wandb_project="RelaVDEP"
wandb_entity=""
wandb_name=""
wandb_mode=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      target="$2"
      shift 2
      ;;
    --fasta)
      fasta="$2"
      shift 2
      ;;
    --data)
      data="$2"
      shift 2
      ;;
    --output)
      output="$2"
      shift 2
      ;;
    --constraint)
      constraint="$2"
      shift 2
      ;;
    --cutoff)
      cutoff="$2"
      shift 2
      ;;
    --library-size)
      library_size="$2"
      shift 2
      ;;
    --seed)
      seed="$2"
      shift 2
      ;;
    --n-gpus)
      n_gpus="$2"
      shift 2
      ;;
    --n-player)
      n_player="$2"
      shift 2
      ;;
    --n-sim)
      n_sim="$2"
      shift 2
      ;;
    --batch-size)
      batch_size="$2"
      shift 2
      ;;
    --cross-val)
      cross_val=true
      shift
      ;;
    --wandb-project)
      wandb_project="$2"
      shift 2
      ;;
    --wandb-mode)
      wandb_mode="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$target" ]]; then
  echo "ERROR: --target is required." >&2
  usage
  exit 1
fi

fasta="${fasta:-relavdep/data/target_sequence/${target}.fasta}"
data="${data:-relavdep/data/mutation_data/${target}.csv}"
output="${output:-outputs/${target}}"
mkdir -p "$output"

step1_log="${output}/1_train_reward_model.log"
step2_log="${output}/2_run_directed_evolution.log"
step3_log="${output}/3_build_mutant_library.log"

echo "============================================================"
echo "RelaVDEP pipeline"
echo "Target: $target"
echo "Fasta: $fasta"
echo "Data: $data"
echo "Output: $output"
if [[ -n "$constraint" ]]; then
  echo "Constraint: $constraint"
fi
echo "============================================================"

step1_cmd=(
  python 1_train_reward_model.py
  --fasta "$fasta"
  --data "$data"
  --output "$output"
  --batch_size "$batch_size"
  --seed "$seed"
)
if [[ "$cross_val" == true ]]; then
  step1_cmd+=(--cross_val)
fi
if [[ -n "$constraint" ]]; then
  step1_cmd+=(--constraint "$constraint")
fi

echo ">>> Step 1/3: supervised reward model training"
"${step1_cmd[@]}" 2>&1 | tee "$step1_log"

rm_params="$(awk '/--rm_params / {print $2}' "$step1_log" | tail -n 1)"
constraint="$(awk '/--constraint / {print $2}' "$step1_log" | tail -n 1)"
parsed_n_layer="$(awk '/--n_layer / {print $2}' "$step1_log" | tail -n 1)"
parsed_cutoff="$(awk '/--cutoff / {print $2}' "$step1_log" | tail -n 1)"
cutoff="${cutoff:-$parsed_cutoff}"

if [[ -z "$rm_params" || -z "$constraint" || -z "$cutoff" ]]; then
  echo "ERROR: Failed to parse --rm_params, --constraint, or --cutoff from $step1_log." >&2
  echo "Please check the step 1 output, or pass --cutoff explicitly." >&2
  exit 1
fi

step2_cmd=(
  python 2_run_directed_evolution.py
  --fasta "$fasta"
  --rm_params "$rm_params"
  --constraint "$constraint"
  --output "$output"
  --n_gpus "$n_gpus"
  --n_player "$n_player"
  --n_sim "$n_sim"
  --batch_size "$batch_size"
  --seed "$seed"
  --wandb_project "$wandb_project"
)
if [[ -n "$parsed_n_layer" ]]; then
  step2_cmd+=(--n_layer "$parsed_n_layer")
fi
if [[ -n "$wandb_entity" ]]; then
  step2_cmd+=(--wandb_entity "$wandb_entity")
fi
if [[ -n "$wandb_name" ]]; then
  step2_cmd+=(--wandb_name "$wandb_name")
fi
if [[ -n "$wandb_mode" ]]; then
  step2_cmd+=(--wandb_mode "$wandb_mode")
fi

echo ">>> Step 2/3: virtual directed evolution"
"${step2_cmd[@]}" 2>&1 | tee "$step2_log"

mutants="${output}/mutants.csv"
if [[ ! -f "$mutants" ]]; then
  echo "ERROR: Expected mutants file not found: $mutants" >&2
  exit 1
fi

echo ">>> Step 3/3: mutant library construction"
python 3_build_mutant_library.py \
  --fasta "$fasta" \
  --mutants "$mutants" \
  --output "$output" \
  --cutoff "$cutoff" \
  --size "$library_size" \
  --seed "$seed" 2>&1 | tee "$step3_log"

echo "============================================================"
echo "Pipeline completed successfully."
echo "Logs:"
echo "  $step1_log"
echo "  $step2_log"
echo "  $step3_log"
echo "Final library:"
echo "  ${output}/library.csv"
echo "============================================================"
