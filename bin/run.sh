set -xe

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <cuda_device_ids> <run_name>"
    exit 1
fi

PROJECT_DIR=$(cd "$(dirname $0)" && pwd)/../

device_ids=$1
run_name=$2
output_dir=${PROJECT_DIR}/outputs/avGFP/${run_name}
mkdir -p ${output_dir}

# params
fasta_fpath=${PROJECT_DIR}/relavdep/data/target_sequence/avGFP.fasta
constraint=${PROJECT_DIR}/relavdep/data/mutation_constraint/avGFP.npz
rm_params=${PROJECT_DIR}/outputs/avGFP/avGFP.pth

export CUDA_VISIBLE_DEVICES=${device_ids}
ngpu=$(echo "$device_ids" | awk -F',' '{print NF}')

# baseline
python -u ${PROJECT_DIR}/2_run_directed_evolution.py \
    --fasta ${fasta_fpath} \
    --rm_params ${rm_params} \
    --constraint ${constraint} \
    --rm_type large \
    --output ${output_dir} \
    --data_dir ${PROJECT_DIR}/models \
    --n_gpus ${ngpu} \
    --n_player 6 \
    --n_sim 1200 \
    --train_delay 2 \
    --log_interval 50 \
    --learning_rate 0.001 \
    --warmup_steps 1000 \
    --lr_decay_rate 0.95 \
    --seed 42 \
