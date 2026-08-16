#!/usr/bin/env bash
set -euo pipefail

build_containers=false
install_host_science=false

for argument in "$@"; do
  case "$argument" in
    --containers)
      build_containers=true
      ;;
    --host-science)
      install_host_science=true
      ;;
    *)
      echo "Unknown option: $argument" >&2
      echo "Usage: $0 [--containers] [--host-science]" >&2
      exit 2
      ;;
  esac
done

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

sync_arguments=(sync --extra dev --extra mcp)
if "$install_host_science"; then
  sync_arguments+=(--extra reasoning --extra biology --extra engineering)
fi
uv "${sync_arguments[@]}"

if "$build_containers"; then
  if command -v docker >/dev/null 2>&1; then
    docker compose build
  elif command -v podman >/dev/null 2>&1; then
    podman compose build
  else
    echo "Docker or Podman is required to build isolated tool images." >&2
    exit 1
  fi
fi

uv run reagents tools doctor
