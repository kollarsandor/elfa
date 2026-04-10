#!/usr/bin/env bash
set -Eeuo pipefail
echo "=== EFLA Trainer Smoke Test ==="
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. No GPU available."
    echo "Smoke test requires at least 1 NVIDIA GPU."
    exit 1
fi
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1 | tr -d ' ')
if [ "$GPU_COUNT" -lt 1 ]; then
    echo "ERROR: No GPUs detected."
    exit 1
fi
echo "Found ${GPU_COUNT} GPU(s)"
nvidia-smi --query-gpu=name,memory.total --format=csv
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. CUDA toolkit required."
    exit 1
fi
NVCC_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | tr -d ',')
echo "CUDA version: ${NVCC_VERSION}"
if [ ! -f "zig-out/bin/efla-train" ]; then
    echo "Building EFLA trainer..."
    zig build -Doptimize=ReleaseFast
fi
mkdir -p data
if [ ! -f "data/smoke.bin" ]; then
    echo "Creating minimal test data..."
    python3 -c "
import struct
import os
# Create header
magic = 0xEF1AD001
version = 1
num_tokens = 10000
num_shards = 1
data = struct.pack('<IIQI', magic, version, num_tokens, num_shards)
# Add shard metadata
shard_path = b'smoke_shard.bin'
data += struct.pack('<I', len(shard_path))
data += shard_path
data += struct.pack('<Q', num_tokens)
data += bytes(32)  # checksum placeholder
# Add tokens (random)
import random
random.seed(42)
tokens = [random.randint(0, 8191) for _ in range(num_tokens)]
data += struct.pack(f'<{num_tokens}I', *tokens)
with open('data/smoke.bin', 'wb') as f:
    f.write(data)
print(f'Created smoke test data: {len(data)} bytes')
"
fi
echo "Running smoke test..."
zig-out/bin/efla-train smoke-test --config configs/smoke.yaml
echo ""
echo "=== Smoke Test Complete ==="
