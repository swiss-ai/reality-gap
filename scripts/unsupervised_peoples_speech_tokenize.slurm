#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=audio-tok
#SBATCH --reservation=PA-2338-RL
#SBATCH --environment=nemo
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/benchmark-audio-tokenizer/logs/peoplespeech_wds_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/benchmark-audio-tokenizer/logs/peoplespeech_wds_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --partition=normal
#SBATCH --exclusive

set -e

echo "=========================================="
echo "Multi-node WDS Tokenization (People's Speech)"
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
    export FFPROBE_PATH=/users/xyixuan/.local/ffmpeg/bin/ffprobe
    export LD_LIBRARY_PATH=/users/xyixuan/.local/ffmpeg/lib:$LD_LIBRARY_PATH

    # Install torchaudio + mutagen in the same container instance
    echo "Installing torchaudio and mutagen on $(hostname -s)..."
    uv pip install --python /opt/venv/bin/python --no-deps --no-build-isolation \
        git+https://github.com/pytorch/audio.git@release/2.9
    uv pip install --python /opt/venv/bin/python mutagen

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
        echo "Starting WDS tokenization (1000 shards)..."
        echo "=========================================="

        SHARDS=$(ls \
          /iopsstor/scratch/cscs/xyixuan/audio-datasets/unsupervised_peoples_speech_commercial_wds/audio/*.tar \
          /iopsstor/scratch/cscs/xyixuan/audio-datasets/unsupervised_peoples_speech_commercial_wds/audio2/*.tar \
          2>/dev/null | sort | paste -sd, -)

        python -m audio_tokenization.tokenize \
            dataset=peoples_speech_wds \
            dataset.shards="[${SHARDS}]" \
            batch_size=4 \
            num_gpus='"${TOTAL_GPUS}"' \
            ray.address="'"${RAY_ADDRESS}"'" \
            shard_assignment=shared \
            wandb.enabled=true

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
