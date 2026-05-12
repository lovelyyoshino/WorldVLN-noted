# Remote Simulator Workflow

The remote_sim path keeps policy inference on the training host and delegates only environment reset, action execution, and rendering to the simulator host.

## Service Endpoints

The simulator-side service is implemented by runtime/infinity_tsformer_client.py in service mode. It exposes:

- GET /health
- POST /reset
- POST /step_actions

## Start the Service

On the simulator machine:

```bash
python runtime/infinity_tsformer_client.py \
  --mode service \
  --host 0.0.0.0 \
  --port 8765 \
  --task_json_root /path/to/UAV-Flow-Eval/test_jsons
```

## Reverse Port Forwarding

If the simulator machine cannot be reached directly from the training host, create a reverse tunnel from the simulator machine to the training host:

```bash
ssh -R 18765:127.0.0.1:8765 user@training-host
```

Then validate from the training host:

```bash
curl --noproxy '*' http://127.0.0.1:18765/health
```

## StageA Settings

The open-source posttrain package keeps only remote_sim for StageA. Use the wrapper command below:

```bash
cd /ML-vePFS/research_gen/vln_uav/opensource/posttrain

unset ALL_PROXY all_proxy
export NO_PROXY=127.0.0.1,localhost

export PATH=/ML-vePFS/research_gen/jmy/jmy_ws/envs_conda/inifnitystar/bin:$PATH
export PYTHON_BIN=/ML-vePFS/research_gen/jmy/jmy_ws/envs_conda/inifnitystar/bin/python

export CUDA_VISIBLE_DEVICES=6
export GRPO_LOCAL_GPU_IDS=6
export NPROC_PER_NODE=1
export NNODES=1
export NODE_RANK=0

export SRC_JSON=/ML-vePFS/research_gen/vln_uav/49frame_dataset/reference_video_full_49f_trajectory_prompts.json
export CHECKPOINTS_DIR=/manifold-obs/vln-uav/checkpointsinf
export INFINITY_CKPT=/manifold-obs/vln-uav/uavflowck/checkpoint_ref49f_repeat11_from19000_8ep/global_step_27000.pth
export ACTIONHEAD_CKPT=/ML-vePFS/research_gen/vln_uav/pipeline_test/TSformer-VO-main/checkpoints/uavflow_plus_ref_ft100_z15to30_rot_warmup_hard3x/checkpoint_last.pth
export ACTIONHEAD_RUN_CONFIG=/ML-vePFS/research_gen/vln_uav/pipeline_test/TSformer-VO-main/checkpoints/uavflow_plus_ref_ft100_z15to30_rot_warmup_hard3x/run_config.json
UAVFLOW_STAGEA_ROLLOUT_BACKEND=remote_sim
UAVFLOW_SIMULATOR_BASE_URL=http://127.0.0.1:18765
export UAVFLOW_SIMULATOR_TIMEOUT_S=120
UAVFLOW_TASK_JSON_ROOT=/path/to/UAV-Flow-Eval/test_jsons
STAGEA_NPROC=1

export RUN_ID=posttrain_remote_sim_smoke_$(date +%Y%m%d_%H%M%S)

bash scripts/run_stagea_collect.sh \
  RUN_ID=${RUN_ID} \
  TOP_N=1 \
  K_CAND=1 \
  STAGEA_NPROC=1 \
  STAGEA_PROGRESS_EVERY_N=1
```

STAGEA_NPROC should remain 1 unless the simulator service is extended to support multiple concurrent sessions.
