#!/usr/bin/env bash
# Usage: bash scripts/train.sh [rho_07|equal_alpha] [--config.FIELD=VALUE ...]
# Run on both nodes (8 GPUs each), with NODE_RANK=0 or NODE_RANK=1.
set -euo pipefail

MODE=${1:-rho_07}
if [[ $# -gt 0 ]]; then shift; fi
case "$MODE" in
    rho_07) CONFIG=sd3_multi_reward_pareto_5r_alpha_ema ;;
    equal_alpha) CONFIG=sd3_ablation_equal_alpha ;;
    *) echo "Usage: $0 [rho_07|equal_alpha] [--config.FIELD=VALUE ...]" >&2; exit 1 ;;
esac

cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${MASTER_ADDR:?Set MASTER_ADDR to the address of node 0}"
: "${NODE_RANK:?Set NODE_RANK to 0 or 1}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}

exec torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="${MASTER_PORT:-29500}" \
    train.py \
    --config "config/nft.py:$CONFIG" \
    "$@"
