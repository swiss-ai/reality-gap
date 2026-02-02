#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=audio-tok
#SBATCH --reservation=PA-2338-RL
#SBATCH --environment=nemo
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/benchmark-audio-tokenizer/logs/multinode_tok_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/benchmark-audio-tokenizer/logs/multinode_tok_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --partition=normal
#SBATCH --exclusive

set -e

echo "=========================================="
echo "Multi-node Audio Tokenization"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NUM_NODES}"
echo "Node list: ${SLURM_JOB_NODELIST}"
echo "GPUs per node: 4"
echo "=========================================="

# Create logs directory
mkdir -p /iopsstor/scratch/cscs/xyixuan/benchmark-audio-tokenizer/logs

# Get node names
NODES=($(scontrol show hostnames ${SLURM_JOB_NODELIST}))
HEAD_NODE=${NODES[0]}
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" hostname -I | awk '{print $1}')

# Ray port
RAY_PORT=6379
RAY_ADDRESS="${HEAD_NODE_IP}:${RAY_PORT}"
export HEAD_NODE_IP
export HEAD_NODE
export RAY_PORT
export RAY_ADDRESS

echo "Head node: ${HEAD_NODE} (${HEAD_NODE_IP})"
echo "Ray address: ${RAY_ADDRESS}"
echo ""

# Calculate total resources
TOTAL_GPUS=$((SLURM_JOB_NUM_NODES * 4))

echo "Total GPUs: ${TOTAL_GPUS}"
echo ""
 
# Run head/worker and driver under a single srun so the driver is on the head node
srun -N ${SLURM_JOB_NUM_NODES} --tasks-per-node=1 -u bash -c '
    set -e
    export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/benchmark-audio-tokenizer:$PYTHONPATH
    export FFMPEG_PATH=/users/xyixuan/.local/ffmpeg/bin/ffmpeg
    export LD_LIBRARY_PATH=/users/xyixuan/.local/ffmpeg/lib:$LD_LIBRARY_PATH

    # Install torchaudio in the same container instance
    echo "Installing torchaudio on $(hostname -s)..."
    uv pip install --python /opt/venv/bin/python --no-deps --no-build-isolation \
        git+https://github.com/pytorch/audio.git@release/2.9

    if [[ $SLURM_PROCID = 0 ]]; then
        echo "Starting Ray head on ${HEAD_NODE} (${HEAD_NODE_IP})..."
        ray start --head \
            --node-ip-address="${HEAD_NODE_IP}" \
            --port="${RAY_PORT}" \
            --num-cpus=288 \
            --num-gpus=4

        sleep 15

        echo ""
        echo "Verifying Ray cluster..."
        ray status --address="${RAY_ADDRESS}"
        echo ""

        cd /iopsstor/scratch/cscs/xyixuan/benchmark-audio-tokenizer

        echo "=========================================="
        echo "Starting tokenization pipeline..."
        echo "=========================================="

        python -m audio_tokenization.tokenize \
            dataset=audioset_bucketed \
            dataset.bucket.target_bucket=240000 \
            num_gpus='"${TOTAL_GPUS}"' \
            ray.address="'"${RAY_ADDRESS}"'" \
            wandb.name="hf_audio_tokenization_'"${SLURM_JOB_NUM_NODES}"'nodes" \
            wandb.tags="[benchmark,multinode,'"${SLURM_JOB_NUM_NODES}"'nodes]"

        echo ""
        echo "=========================================="
        echo "Pipeline completed!"
        echo "=========================================="

        ray stop --force
    else
        sleep 10
        echo "Starting Ray worker on $(hostname -s)..."
        ray start --address="${RAY_ADDRESS}" \
            --num-cpus=288 \
            --num-gpus=4 \
            --block
    fi
'
