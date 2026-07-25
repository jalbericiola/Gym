#!/bin/bash
set -e
set -x  # Enable debug output

# Variables
setup_dir=$SETUP_DIR
miniforge_dir=$MINIFORGE_DIR
openhands_dir=$OPENHANDS_DIR
agent_framework_repo=$AGENT_FRAMEWORK_REPO
agent_framework_commit=$AGENT_FRAMEWORK_COMMIT

cd $setup_dir

# Install miniforge if not properly installed
if [ ! -f "$miniforge_dir/bin/conda" ] || [ ! -f "$miniforge_dir/bin/mamba" ]; then
    echo "Installing miniforge..."
    # Clean up any partial installation
    rm -rf "$miniforge_dir"
    rm -f Miniforge3-*.sh

    echo "Downloading miniforge..."
    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"

    echo "Running miniforge installer..."
    bash Miniforge3-$(uname)-$(uname -m).sh -b -p $miniforge_dir

    echo "Cleaning up installer..."
    rm Miniforge3-$(uname)-$(uname -m).sh
else
    echo "Miniforge already installed at $miniforge_dir"
fi

# Add conda to PATH and source conda setup
echo "Setting up conda environment..."
export PATH="$miniforge_dir/bin:$PATH"
source $miniforge_dir/etc/profile.d/conda.sh
conda activate base

# Verify conda and mamba are available
echo "Verifying conda installation..."
which conda
which mamba
conda --version
mamba --version

# Install required packages
echo "Installing conda packages (this may take 5-10 minutes)..."
mamba install -y --override-channels conda-forge::python=3.12 conda-forge::nodejs conda-forge::poetry conda-forge::tmux conda-forge::git

$miniforge_dir/bin/python -m pip install -q 'packaging==26.0'

# Install jq as a static binary (avoid conda solver changing other package versions)
if [ ! -f "$miniforge_dir/bin/jq" ]; then
    echo "Installing jq static binary..."
    curl -fsSL https://github.com/jqlang/jq/releases/download/jq-1.8.1/jq-linux-amd64 -o "$miniforge_dir/bin/jq"
    chmod +x "$miniforge_dir/bin/jq"
fi

echo "Verifying jq installation..."
which jq
jq --version || true


# Verify installations
echo "Verifying package installations..."
which python
which node
which poetry
which jq

# Clone OpenHands
if [ ! -d "$openhands_dir/.git" ]; then
    echo "Cloning OpenHands..."
    # Clean up any partial clone
    rm -rf "$openhands_dir"
    git clone $agent_framework_repo $openhands_dir
else
    echo "OpenHands already cloned at $openhands_dir"
fi

cd $openhands_dir
echo "Checking out $agent_framework_commit..."
git checkout $agent_framework_commit

# Apply local NeMo-Gym patches on top of the pinned upstream commit. The checkout
# above resets tracked files to $agent_framework_commit, so we re-apply on every
# fresh setup. Currently: thread the off-policy training fields
# (policy_epoch/kv_cache_epoch/num_evictions) through the conversation history so
# NeMoGym's *ForTraining schema validates on every assistant turn, not just the
# final response. Idempotent: skip a patch that is already applied; hard-fail if a
# patch neither applies cleanly nor is already present (prevents silent drift).
# (This loop was dropped in the upstream refactor and restored by the
# swe-parity-port review 2026-07-22 — without it every multi-turn episode 500s
# against the Megatron policy server with a pydantic ValidationError.)
patch_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/patches"
if [ -d "$patch_dir" ]; then
    for patch in "$patch_dir"/*.patch; do
        [ -e "$patch" ] || continue
        if git apply --reverse --check "$patch" >/dev/null 2>&1; then
            echo "Patch already applied, skipping: $(basename "$patch")"
        elif git apply --check "$patch" >/dev/null 2>&1; then
            echo "Applying patch: $(basename "$patch")"
            git apply "$patch"
        else
            echo "ERROR: patch neither applies cleanly nor is already applied: $(basename "$patch")" >&2
            exit 1
        fi
    done
fi

# Build OpenHands
echo "Building OpenHands (this may take 5-10 minutes)..."
export INSTALL_DOCKER=0


# Remove any cached virtualenvs from previous runs
# Use poetry's actual cache dir (respects XDG_CACHE_HOME) instead of hardcoded ~/.cache
echo "Removing any cached poetry virtualenvs..."
poetry_cache_dir="$(poetry config cache-dir 2>/dev/null || echo ~/.cache/pypoetry)"
rm -rf "$poetry_cache_dir"/virtualenvs/openhands-* || true

# CRITICAL: Unset any active virtualenv from the host .venv
# This prevents poetry from getting confused about which venv to use
echo "Unsetting host virtualenv to avoid poetry confusion..."
unset VIRTUAL_ENV
unset PYTHONHOME
# Remove any venv paths from PATH to ensure clean environment
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '\.venv' | tr '\n' ':' | sed 's/:$//')

# Configure poetry to create virtualenv in the project directory (so it's mounted in container)
export POETRY_VIRTUALENVS_IN_PROJECT=true

# Retry `make build` with a timeout guard on the first attempt
MAX_MAKE_BUILD_ATTEMPTS=2
MAKE_BUILD_TIMEOUT_SECONDS=$((2 * 60))
MAKE_BUILD_TIMEOUT_MINUTES=$((MAKE_BUILD_TIMEOUT_SECONDS / 60))

attempt=1
while [ "$attempt" -le "$MAX_MAKE_BUILD_ATTEMPTS" ]; do
    echo "Running make build (attempt $attempt/$MAX_MAKE_BUILD_ATTEMPTS)..."

    if [ "$attempt" -lt "$MAX_MAKE_BUILD_ATTEMPTS" ]; then
        if timeout "$MAKE_BUILD_TIMEOUT_SECONDS" make build; then
            echo "make build completed successfully."
            break
        fi

        exit_code=$?
        if [ "$exit_code" -eq 124 ]; then
            echo "make build timed out after $MAKE_BUILD_TIMEOUT_MINUTES minutes."
        else
            echo "make build failed with exit code $exit_code."
        fi

        echo "Retrying make build after cleanup..."
        make clean || true
        attempt=$((attempt + 1))
        continue
    fi

    if make build; then
        echo "make build completed successfully."
        break
    fi

    exit_code=$?
    echo "make build failed on the final attempt with exit code $exit_code."
done

# Install Python dependencies with poetry
echo "Installing Python dependencies (creating .venv in OpenHands directory)..."
poetry install --no-interaction --no-root

# Install datasets package
echo "Installing datasets package..."

# wandb: evaluation/utils/shared.py imports it while recording results. If it is
# missing every episode raises 'No module named wandb', exhausts its retries and
# returns an EMPTY trajectory -- the agent does all its real work (edits, test
# runs) and then the harness dies on the import, so 128 rollouts come back empty
# and the job dies at prepare_trajectories (observed 2026-07-25 after a clean
# rebuild; the previous long-lived setup happened to have it installed).
# Resolve the venv interpreter explicitly and use it for every install below.
# `poetry run python` is not reliable here (see the nemo_gym note further down).
_NG_VENV_PY="$(pwd)/.venv/bin/python"
if [ ! -x "$_NG_VENV_PY" ]; then
    _NG_VENV_PY="$(poetry env info --path 2>/dev/null)/bin/python"
fi
if [ ! -x "$_NG_VENV_PY" ]; then
    echo "FATAL: cannot locate the OpenHands venv interpreter" >&2; exit 1
fi
echo "OpenHands venv interpreter: $_NG_VENV_PY"

# gprof2dot/pydot are declared nemo_gym deps (pyproject.toml) pulled in by
# nemo_gym.profiling <- nemo_gym.server_utils. They are NOT optional: the agent's
# very first import (NemoGymClient -> ServerClient) walks that chain, so without
# them every episode dies before doing any work. They are installed explicitly
# because the nemo_gym install below uses --no-deps (to avoid clobbering
# OpenHands' own pins).
"$_NG_VENV_PY" -m pip install datasets huggingface_hub packaging==26.0 wandb gprof2dot pydot

# Install the CURRENT tree's nemo_gym into the OpenHands venv.
#
# The agent running inside the sandbox imports nemo_gym from THIS venv, not from
# the Gym tree. Left to pip's own resolution the venv can end up with a stale
# nemo_gym that predates the ServerClient self-heal (bounded connection retries +
# re-resolving a server's address from the head server when a connect fails).
# Without the heal, a policy-server port rebind makes every agent retry a dead
# address 3x and return an EMPTY trajectory -- 128 empty rollouts then kill the
# job at prepare_trajectories with no obvious cause (observed 2026-07-25).
# This MUST be all-or-nothing. An earlier version ran `pip install -q ... || cp
# *.py || echo WARNING`: pip did not target this venv, `-q` hid the error, and the
# cp fallback landed only part of the package. That left a NEW global_config.py
# against an OLD __init__.py, so every episode died on
#   ImportError: cannot import name 'WORKING_DIR' from 'nemo_gym'
# before doing any work -- strictly worse than the stale-but-coherent package it
# replaced (observed 2026-07-25). Hence: the venv's own interpreter (not
# `poetry run`, which resolved elsewhere), no -q, no fallback, and a hard
# post-install import check that fails the setup rather than shipping a half
# package.
_NG_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
if [ -d "$_NG_SRC/nemo_gym" ] && [ -x "$_NG_VENV_PY" ]; then
    echo "Syncing nemo_gym from tree ($_NG_SRC) into the OpenHands venv..."
    "$_NG_VENV_PY" -m pip install --no-deps --force-reinstall --no-cache-dir "$_NG_SRC" || {
        echo "FATAL: nemo_gym install into the OpenHands venv failed" >&2; exit 1; }
    # Verify the EXACT chain the sandboxed agent walks on its first import
    # (codeact_agent -> nemo_gym_client -> ServerClient -> profiling), not just a
    # token module. An earlier check imported only global_config and therefore
    # passed while nemo_gym.profiling was still missing gprof2dot -- the failure
    # then surfaced 36 times per link at runtime instead of once at setup.
    "$_NG_VENV_PY" - <<'PYCHK' || { echo "FATAL: nemo_gym in the OpenHands venv is inconsistent" >&2; exit 1; }
import nemo_gym, nemo_gym.global_config, nemo_gym.server_utils, nemo_gym.profiling
from nemo_gym import CACHE_DIR, PARENT_DIR, RESULTS_DIR, WORKING_DIR
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.server_utils import ServerClient
print("nemo_gym venv sync verified")
PYCHK
else
    echo "FATAL: cannot sync nemo_gym (src=$_NG_SRC venv_py=$_NG_VENV_PY)" >&2; exit 1
fi

mkdir -p evaluation/oh
mkdir -p logs
mkdir -p .eval_sessions

echo "Verifying .venv was created..."
if [ -d .venv ]; then
    echo "✓ .venv created at $(pwd)/.venv"
else
    echo "✗ ERROR: .venv was not created!"
    exit 1
fi

echo "OpenHands setup complete!"
