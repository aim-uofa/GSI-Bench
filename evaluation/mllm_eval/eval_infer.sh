#!/bin/bash
# vLLM service startup + automated evaluation script
# Usage example:
# bash eval_infer.sh <MODEL_PATH> default <PORT>
#
# source <VENV_PATH>/bin/activate
# cd <WORKING_DIR>

if [ -z "$2" ]; then
    echo "Usage: $0 <model_path> <mode> [port]"
    exit 1
fi

MODEL_PATH="$1"
MODE="${2:-default}"
PORT="${3:-8000}"
HOST="0.0.0.0"

FIRST_PART=$(basename "$(dirname "$(dirname "$(dirname "$MODEL_PATH")")")")
SECOND_PART=$(basename "$(dirname "$(dirname "$MODEL_PATH")")")
MODEL_NAME="${FIRST_PART}_${SECOND_PART}"
echo "MODEL_NAME: $MODEL_NAME"

PARENT_DIR=$(dirname "$MODEL_PATH")  # Get the parent directory of MODEL_PATH

HF_DIR="$MODEL_PATH"
echo "Checking whether the model is already merged: $HF_DIR"

if [ -d "$HF_DIR" ] && ls "$HF_DIR"/*.safetensors >/dev/null 2>&1; then
    echo "✅ .safetensors files detected, skipping model_merger.py"
else
    echo "🔧 Starting model merging..."
    # python <PATH_TO_MODEL_MERGER>/model_merger.py \
    python model_merger.py \
        --local_dir "$PARENT_DIR"
    echo "✅ Model merging completed"
fi

GPU_MEMORY_UTILIZATION=0.9
TENSOR_PARALLEL_SIZE=4

LOG_FILE="vllm_server.log"
INFER_RESULTS_DIR="./infer_results"

echo "Starting vLLM service for model: $MODEL_NAME"
echo "Model path: $MODEL_PATH"
echo "Service address: http://$HOST:$PORT"
echo "Logs will be written to: $LOG_FILE"
echo ""

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: model path does not exist: $MODEL_PATH"
    exit 1
fi

# =============== Step 1: Start vLLM service in background ===============
if lsof -i:"$PORT" -sTCP:LISTEN >/dev/null; then
    echo "⚠️ Port $PORT is already in use. Skipping vLLM startup."
    SERVER_ALREADY_RUNNING=true
else
    echo "Launching vLLM service for $MODEL_NAME..."
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

    nohup vllm serve \
        "$MODEL_PATH" \
        --served-model-name "$MODEL_NAME" \
        --host "$HOST" \
        --port "$PORT" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --trust-remote-code \
        --dtype bfloat16 \
        --disable-log-requests \
        --max-num-seqs 16 > "$LOG_FILE" 2>&1 &

    SERVER_PID=$!
    echo "vLLM service started in background, PID: $SERVER_PID"

    # Wait for service to become ready
    echo "Waiting for service to be ready (up to 120 seconds)..."
    for i in $(seq 1 120); do
        sleep 3
        if curl -s http://$HOST:$PORT/docs >/dev/null; then
            echo "✅ Service is ready!"
            break
        fi
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "❌ Service failed to start. Check log: $LOG_FILE"
            exit 1
        fi
    done
fi

# =============== Step 3: Run inference script ===============
# Skip if predictions for this model already exist under infer_results/
EXISTING_PREDICTIONS=$(find "$INFER_RESULTS_DIR" -maxdepth 1 -name "predictions_infer_*_${MODEL_NAME}_*.json" 2>/dev/null | head -1)
if [ -n "$EXISTING_PREDICTIONS" ]; then
    echo "⚠️ Prediction file already exists: $EXISTING_PREDICTIONS"
    echo "Skipping mllm_eval.py inference..."
else
    echo "Running mllm_eval.py..."
    if [ "$MODE" = "default" ]; then
        python mllm_eval.py --model_name "$MODEL_NAME"
    else
        echo "Wrong mode: $MODE."
        exit 1
    fi
fi

# =============== Step 4: Aggregate AC scores ===============
# Prediction JSONs are written to infer_results/ and can be fed into:
#   python -m eval.aggregate --root-dir <...> --mllm-eval-dir infer_results
# See the root README for details.
echo "✅ Inference complete. Check prediction JSONs in: $INFER_RESULTS_DIR/"
echo "Use 'python -m eval.aggregate --mllm-eval-dir $INFER_RESULTS_DIR' to incorporate AC scores."

# =============== Step 5: Shut down vLLM service ===============
if [ "${SERVER_ALREADY_RUNNING:-false}" = true ]; then
    echo "vLLM service was already running; leaving it untouched."
else
    echo "Shutting down vLLM service (PID: $SERVER_PID)..."
    kill $SERVER_PID
    wait $SERVER_PID 2>/dev/null
fi
echo "All tasks completed successfully."
