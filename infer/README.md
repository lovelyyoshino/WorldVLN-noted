# WorldVLN Inference Service

This directory contains the online inference service used to serve WorldVLN as an API. The service keeps the main model weights resident in memory and exposes a deployment-oriented entrypoint for inference workloads.

## Overview

The inference package is centered on a FastAPI service implemented in `infinity_tsformer_api_server.py` and launched through `run_server.sh`.

At a high level, the service:

- loads InfinityStar and latent-to-action weights once at startup
- accepts session-based visual observations
- produces action predictions through the API server

## Key Components

| Path | Purpose |
| --- | --- |
| `run_server.sh` | Public launcher for the online inference service. |
| `infinity_tsformer_api_server.py` | Main API server implementation. |
| `config.json` | Default runtime configuration. |
| `InfinityStar-main/` | Local InfinityStar source tree used by the service. |
| `TSformer-VO-main/` | Local action-module dependency used by the service. |

## Setup

The inference package expects model assets to be supplied at runtime.

| Asset | Variable | Purpose |
| --- | --- | --- |
| InfinityStar checkpoint | `INFINITY_CKPT` | Main WorldVLN / InfinityStar checkpoint used by the service. |
| Stage-2 latent-to-action checkpoint | `STAGE2_LATENT2ACTION_CKPT` | Action prediction checkpoint for latent-to-action mode. |
| Runtime config | `INFINITY_SERVER_CONFIG` | Optional override for the default `config.json`. |
| InfinityStar source root | `INFINITY_REPO_ROOT` | Optional override for the bundled `InfinityStar-main/`. |
| Latent cache root | `INFINITY_LATENT_CACHE_ROOT` | Output cache directory used by the service. |
| Service host and port | `HOST`, `PORT` | Bind address for the API server. |

The root-level WorldVLN backbone weights are documented in the repository homepage. Additional service-specific checkpoints should be placed in your preferred checkpoint directory and passed in through the variables above.

## Running the Service

The default launcher is:

```bash
bash run_server.sh
```

A typical explicit launch looks like this:

```bash
export PYTHON_BIN=$(which python)
export INFINITY_CKPT=/path/to/infinity/global_step_xxx.pth
export STAGE2_LATENT2ACTION_CKPT=/path/to/stage2_latent2action_combined.pt

bash run_server.sh
```

`run_server.sh` starts a Uvicorn server and forwards the relevant runtime environment variables to `infinity_tsformer_api_server.py`.

## Inputs and Outputs

The service is designed for session-based online inference.

- Input: streamed visual observations grouped by session, typically beginning with an initial frame and followed by later observation batches
- Output: action predictions returned by the API server for the current session state

Internally, the service uses InfinityStar to produce latent representations and then applies the latent-to-action head to generate motion outputs.

## Notes

- `config.json` uses repository-relative defaults where possible and is intended to be safe for public release.
- The service can be pointed at alternate local copies of InfinityStar through `INFINITY_REPO_ROOT`.
- Runtime caches should not be committed back into the repository.
