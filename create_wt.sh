#!/bin/bash
set -euo pipefail
SUFFIX=$(openssl rand -hex 4)
WORKTREE_ID="jacket-smoke-${SUFFIX}"
REPO_ROOT="/home/s2795056/projects/GitHub/WISDEM"
REPO_BASENAME="WISDEM"
REPO_HASH=$(printf '%s' "$REPO_ROOT" | sha256sum | cut -c1-12)
REPO_KEY="${REPO_BASENAME}-${REPO_HASH}"
WORKTREE_SET_DIR="$HOME/.cursor/worktrees/$WORKTREE_ID"
WORKTREE_DIR="$WORKTREE_SET_DIR/$REPO_KEY"
mkdir -p "$WORKTREE_SET_DIR"
if [ -d "$WORKTREE_DIR" ]; then echo "ERROR: exists $WORKTREE_DIR" >&2; exit 1; fi
git -C "$REPO_ROOT" worktree add --detach "$WORKTREE_DIR" HEAD
HEAD_COMMIT=$(git -C "$WORKTREE_DIR" rev-parse HEAD)
echo "WORKTREE_ID=$WORKTREE_ID"
echo "REPO_KEY=$REPO_KEY"
echo "WORKTREE_PATH=$WORKTREE_DIR"
echo "REPO_ROOT=$REPO_ROOT"
echo "HEAD_COMMIT=$HEAD_COMMIT"
echo "WORKTREE_START_REF=HEAD"
