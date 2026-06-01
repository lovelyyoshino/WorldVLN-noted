# WorldVLN 代码阅读指南

这份文档回答三个问题：先从哪里看、每条主线在代码里对应什么、哪些实现点需要重点读。建议先按这里的顺序走一遍，不要一开始就进入 `Worldmodel/infinity/models/` 里的大模型细节。

## 1. 先看什么

推荐阅读顺序：

1. `README.md`
   先确认项目入口、依赖权重、在线推理和两阶段训练命令。

2. `infer/server.py`
   这是理解 WorldVLN 在线闭环推理的第一入口。重点看 `/v1/predict_delta_actions` 的请求协议、`TrajectoryState`、`_predict_delta_actions_impl()`、`_infer_latents_for_actions_and_advance_cache()`、`_stage2_predict_16_actions_for_segment_cm_deg()`。

3. `infer/client.py`
   从客户端角度看同一个闭环协议：先发 1 帧预热图像，再按 `step` 上传执行动作后得到的真实新帧。这里也能看到当前 16 个逐帧动作和旧 4 个宏动作的兼容处理。

4. `Worldmodel/runtime/tools/infinity_streaming_session.py`
   这里解释服务端如何维护文本 cache、真实观测 cache 和预测 cache。重点看 `reset()`、`compute_kv_cache_gt()`、`infer_chunk()`、`correction_clear_pred()`。

5. `Worldmodel/action_decoder/src/models/vae96_to_tsformer_adapter.py`
   这是 latent 世界转移到 TimesFormer token 的桥。重点看输入输出 shape：VAE decoder 中间特征 `(B,96,T,H,W)` 变成 TimesFormer patch tokens `(B*T,480,384)`。

6. `train/action_decoder/tools/train_stageA_ddp.py` 和 `train/action_decoder/tools/train_stageB_ddp.py`
   Stage A 做 adapter 蒸馏，Stage B 做 latent 到动作训练。先读两个文件头部注释，再看 loss、VAE hook、TimesFormer 构建和动作标签归一化。

7. `train/scripts/train_from_base.sh`、`train/train.py`、`train/TRAINING.md`
   这条线对应 Stage 1 世界模型监督训练：用语言和真实视频 latent 训练自回归世界模型预测下一段 latent。

8. `Worldmodel/runtime/tools/GRPO/reward_uavflow.py` 和 `action_aware_grpo/`
   这条线对应 Stage 2 Action-aware GRPO：先 rollout，再按动作几何、任务进度、reference/CE 约束组合奖励。
   如果要看“GRPO rollout 服务端”而不是默认 `infer/` 服务端，继续读 `action_aware_grpo/grpo_server.py`。

## 2. 总体架构

WorldVLN 的主线不是“输入图像直接输出动作”，而是：

```text
语言指令 + 真实观测历史
        |
        v
文本编码器 + video VAE 编码
        |
        v
自回归世界模型预测下一段 latent 世界转移
        |
        v
动作解码器把 latent 转移解成 6D 增量动作
        |
        v
客户端执行动作并回传真实新观测
        |
        v
服务端用真实观测覆盖/纠偏下一轮上下文
```

关键点：预测出的 latent 不是为了生成好看的 RGB 视频，而是为了让动作解码器知道“如果这样走，世界会怎样变化”。因此 `summed_codes` 的用途主要是动作决策，而不是展示。

## 2.1 符号、张量和单位速查

读代码时最容易混淆的是“帧时间线、latent 时间线、动作时间线”不完全一样：

| 名称 | 常见形状 / 单位 | 出现位置 | 含义 |
| --- | --- | --- | --- |
| `images_base64` | List[str] | `infer/server.py` 请求体 | 客户端上传的真实 RGB 帧 |
| `frames_cpu` | List[`[3,H,W]`] in `[-1,1]` | `TrajectoryState` | 同一个 session 已收到的真实观测前缀 |
| `points` | `[1,17,33,49,...]` | `InfinityConfig.points()` | segment 边界；默认 step=16 |
| `summed_codes` | `[1,C,T_lat,H,W]` | 世界模型输出 | 预测到的 latent 世界转移 |
| `z_ext` | `[1,64,T_lat,16,16]` | Stage-2 VAE decode | 动作解码器期望的 patchified latent |
| VAE up_block feature | `[B,96,T,H,W]` | decoder hook | Adapter 的输入，不是最终 RGB |
| Adapter token | `[B*T,480,384]` | `Vae96ToTSformerEmbedAdapter` | TimesFormer 可消费的 patch tokens |
| server action | `[dx,dy,dz,droll,dyaw,dpitch]` | API 输出 | 单位为 cm/deg |
| training delta | `[dz,dy,dx,tx,ty,tz]` | Stage B / trajectory labels | 角度通常 rad，位移 m |

经验规则：

- `T_frames=49` 时，动作增量通常是 48 个，因为动作描述的是相邻帧之间的转移。
- `step=16` 时，一轮闭环输出 16 个逐帧动作。
- video VAE 有时间压缩，latent 时间 `T_lat` 小于 RGB 帧数，所以代码里经常出现 `(num_frames - 1)//4 + 1`。

## 2.2 关键公式与行号索引

下面的行号是按当前工作区生成的“函数入口行号”。具体公式写在该函数 docstring 或函数内相邻中文注释里；如果后续继续编辑代码，行号可能会漂移，但函数名仍可用 `rg "def 函数名"` 快速定位。

| 主题 | 公式/规则 | 当前位置 |
| --- | --- | --- |
| 闭环 segment 边界 | `points = [1, 1+step, 1+2*step, ..., num_frames]` | `infer/server.py:199` `_obs_points()`；`infer/client.py:910` `_obs_points()`；`action_aware_grpo/grpo_server.py:187` `_obs_points()` |
| segment readiness | 默认 `ready_default = n >= points[seg+1]`；future 模式可放宽到 `n >= points[seg]` | `infer/server.py:2437` `_predict_delta_actions_impl()`；`action_aware_grpo/grpo_server.py:2173` `_predict_delta_actions_impl()` |
| 帧号到 latent 下标 | `latent_index(frame f) = (f - 1)//temporal_compress_rate + 1`，本项目通常 `temporal_compress_rate=4` | `infer/server.py:1975` `_infer_latents_for_actions_and_advance_cache()`；`action_aware_grpo/grpo_server.py:1585` `_slice_abs_latents_from_summed_codes()` |
| API 动作单位转换 | `meters -> cm: *100`，`radians -> degrees: *180/pi` | `infer/server.py:1095` `_stage2_predict_16_actions_for_segment_cm_deg()` |
| 机体系动作积分 | `x += dx*cos(theta) - dy*sin(theta)`，`y += dx*sin(theta) + dy*cos(theta)`；`theta` 由 yaw_first/trans_first/midpoint 决定 | `infer/client.py:191` `_apply_action_to_pose_with_frame()`；`action_aware_grpo/runtime/client.py:188` `_apply_action_to_pose_with_frame()` |
| Stage A 蒸馏 loss | `loss = w_cos*L_cos + w_mse*L_mse + w_mean*L_mean + w_std*L_std` | `train/action_decoder/tools/train_stageA_ddp.py:76` `compute_distill_loss()` |
| Stage B 标签标准化 | `x_norm = (x - mean) / std`；推理反标准化 `x = x_norm * std + mean`，角度和平移分组处理 | `train/action_decoder/tools/train_stageB_ddp.py:182` `_normalize_delta_bt6()`；`train/action_decoder/tools/predict_pose.py:386` `infer_one_route()` |
| 4 帧滑窗聚合 | `delta[t] = sum(predictions covering t) / count[t]`，并固定 `delta[0]=0` | `infer/server.py:1095` `_stage2_predict_16_actions_for_segment_cm_deg()`；`train/action_decoder/tools/predict_pose.py:386` `infer_one_route()` |
| SE(3) 轨迹积分 | `R_next = R_cur * R_delta`，`p_next = p_cur + R_cur * t_rel` | `train/action_decoder/tools/predict_pose.py:294` `integrate_trajectory_se3()`；`train/action_decoder/tools/eval_endpoints.py:128` `integrate_actions_to_traj()` |
| 端点旋转/航向误差 | `geodesic = acos((trace(R_gt^T R_pred)-1)/2)`；yaw 用 360 度环形差 | `train/action_decoder/tools/eval_endpoints.py:102` `_geodesic_angle_deg()`；`train/action_decoder/tools/eval_endpoints.py:115` `_yaw_error_deg()` |
| Adapter token shape | VAE 特征 `(B,96,T,H,W)` 展平为 TimesFormer token `(B*T,N,D)`；`N` 是空间 patch 数 | `Worldmodel/action_decoder/src/models/vae96_to_tsformer_adapter.py:15` `Vae96ToTSformerEmbedAdapter` |
| 训练 manifest 标签布局 | 统一成 `(T,6)`，若输入是 `(T-1,6)` 则补 `delta[0]=0` | `Worldmodel/action_decoder/src/datasets/latent_traj_manifest.py:44` `LatentTrajManifestDataset` |
| GRPO 动作 reward | `loss = alpha_xyz*mse_xyz + alpha_yaw*mse_yaw + alpha_all6*mse_all6`，再映射为 reward 或组内优势 | `Worldmodel/runtime/tools/GRPO/reward_uavflow.py:131` `_reward_from_mse()`；`Worldmodel/runtime/tools/GRPO/reward_uavflow.py:144` `_act_mse_scalar()` |
| GRPO 组内相对优势 | z-score shaping、LOO：`adv_i = r_i - mean(r_j, j!=i)`，rank shaping 把候选排序映射到相对分数 | `Worldmodel/runtime/tools/GRPO/reward_uavflow.py:153` `_zscore_exp_reward()`；`Worldmodel/runtime/tools/GRPO/reward_uavflow.py:231` `_loo_adv()` |
| GRPO task reward 组合 | `task_reward = dense_weight*dense + success_weight*success_bonus`，再和 action/CE reward 组合 | `Worldmodel/runtime/tools/GRPO/reward_uavflow.py:365` `_compose_task_reward_raw()`；`Worldmodel/runtime/tools/GRPO/reward_uavflow.py:459` `main()` |
| 候选 rollout seed | `seed = seed_base + task_idx*task_seed_stride + cand_idx*candidate_seed_stride` | `Worldmodel/runtime/tools/GRPO/generate_candidate_rollouts.py:30` `_candidate_seed()` |
| streaming cache 日程 | `scale_schedule` 决定每个自回归尺度的 token 范围；`t0/gt_obs/pred` cache 分别代表文本、真实观测和预测上下文 | `Worldmodel/runtime/tools/infinity_streaming_session.py:102` `build_schedule_for_num_frames()`；`Worldmodel/runtime/tools/infinity_streaming_session.py:212` `compute_kv_cache_gt()`；`Worldmodel/runtime/tools/infinity_streaming_session.py:298` `infer_chunk()` |
| 动态分辨率 patch 预算 | `area = scale^2`，`pw = sqrt(area / h_div_w)`，`ph = pw*h_div_w`，再 round 成 VAE 网格尺寸 | `Worldmodel/infinity/schedules/dynamic_resolution.py:15` `get_ratio2hws_video_v2()`；`Worldmodel/infinity/schedules/dynamic_resolution.py:74` `get_ratio2hws_pixels2scales()` |
| schedule 分发入口 | `dynamic_scale_schedule` 先映射到对应 `video_encode/video_decode/get_visual_rope_embeds/get_scale_pack_info`，不支持的 schedule 会直接报错 | `Worldmodel/infinity/schedules/__init__.py:4` `get_encode_decode_func()`；`Worldmodel/runtime/infinity/schedules/__init__.py:4` `get_encode_decode_func()` |
| 时间 embedding 偶数维约束 | 正弦/余弦各占一半维度，所以 `dim` 必须是偶数，否则 cos/sin 拼接无法一一配对 | `Worldmodel/infinity/models/infinity.py:80` `get_timestep_embedding()`；`Worldmodel/runtime/infinity/models/infinity.py:80` `get_timestep_embedding()` |
| schedule packing / RoPE | `scale_pack_info` 把每个尺度的帧范围、空间 patch 范围、RoPE 范围打包给 Transformer | `Worldmodel/infinity/schedules/infinity_elegant.py:50` `get_scale_pack_info()`；`Worldmodel/infinity/schedules/infinity_elegant.py:603` `get_visual_rope_embeds()` |
| Infinity 训练和自回归 | forward() 是 teacher-forcing 训练；`ar_infer_infinity_elegant()` 按 `scale_schedule` 粗到细采样并维护 KV cache | `Worldmodel/infinity/models/infinity.py:395` `forward()`；`Worldmodel/infinity/models/infinity.py:609` `ar_infer_infinity_elegant()`；runtime 镜像在 `Worldmodel/runtime/infinity/models/infinity.py:621` |

## 2.3 函数级数据流导读

这一节按“数据从哪里来、经过哪些函数、变成什么”来读。建议你打开对应文件，对着函数 docstring 和这里的步骤画一张调用链图。

### 2.3.1 在线推理主链路：HTTP 请求到 16 个动作

如果你想先从客户端入口往下读，最短链路是：

```text
infer/client.py::main()
  -> 根据 --mode 分到 dataset/unrealcv
  -> dataset: run_one_route()
  -> unrealcv: run_one_task_unrealcv()
  -> _post_frames()/HTTP /v1/predict_delta_actions
  -> _apply_action_to_pose_with_frame()
  -> 写 *_actions.json / *_poses.json / *_summary.json
```

当前函数入口可用这些位置快速定位：`infer/client.py:1197` `main()`，
`infer/client.py:962` `run_one_route()`，`infer/client.py:542` `run_one_task_unrealcv()`，
`infer/client.py:642` 嵌套 `_post_frames()`，`infer/client.py:191` `_apply_action_to_pose_with_frame()`。
dataset 模式主要是“回放已有真实帧并记录动作/位姿”；unrealcv 模式会把动作真正应用到模拟器位姿，
再把新采集的真实帧送回服务端，形成闭环。

1. `infer/client.py::run_one_task_unrealcv()`
   客户端从任务 JSON 读出 `instruction` 和 `initial_pos`，设置 UnrealCV 位姿，抓取首帧 RGB。
   输出给 server 的核心字段是：
   `session_id`、`instruction`、`images_base64`、`num_frames`、`step`、`action_head_mode`。

2. `infer/server.py::predict_delta_actions()`
   FastAPI 接口只负责接收请求、加锁和异常包装；真正业务逻辑进入 `_predict_delta_actions_impl()`。

3. `infer/server.py::_predict_delta_actions_impl()`
   这是服务端主控函数。它会把 base64 图像解码成 PIL/tensor，按 `session_id` 找到或创建 `TrajectoryState`，
   然后判断当前收到了多少真实帧 `n`，以及哪个 segment 可以输出动作。

4. `infer/server.py::_get_or_create_traj()`
   每条轨迹的状态都存在 `TrajectoryState`：真实帧列表、prompt、streaming session、latent cache、
   已输出 segment、最近一次 `summed_codes` 等都在这里挂住。读这个结构时要把它当作“闭环记忆”。

5. `infer/server.py::_infer_latents_for_actions_and_advance_cache()`
   这一步把真实观测前缀送入 `InfinityStreamingSession`，让世界模型预测下一段 latent。
   输出最关键的是 `summed_codes`，也就是后续动作头要消费的 latent 世界转移。

6. `infer/server.py::_stage2_predict_16_actions_for_segment_cm_deg()`
   这一步把 `summed_codes` 转成动作：
   `summed_codes -> z_ext -> VAE decoder feature -> Adapter tokens -> TimesFormer -> deltas -> cm/deg actions`。
   如果只想理解“为什么一段输出 16 个动作”，先读这个函数，而不是先读完整 VAE 或 TimesFormer。

7. `infer/client.py::_apply_action_to_pose_with_frame()`
   客户端把 server 返回的相对动作应用到当前位姿。世界系动作直接相加；机体系动作要用 yaw 旋到世界系。
   这里就是闭环里“动作执行后产生下一帧真实观测”的地方。

### 2.3.2 世界模型内部链路：真实帧到预测 latent

先区分两条线：

- 概念链路
  `reset() -> compute_kv_cache_gt() -> infer_chunk() -> correction_clear_pred()`
  这是理解 streaming session 设计最直观的一条抽象链。

- 服务端真实主链路
  `_predict_delta_actions_impl() -> _infer_latents_for_actions_and_advance_cache() -> _infer_summed_codes_for_step() -> gen_one_example()/autoregressive_infer()`
  也就是说，`infer_chunk()` 更像“底层语义模板”，而在线服务当前主路径更靠近
  `server.py` 里那几个包装函数。

1. `Worldmodel/runtime/tools/infinity_streaming_session.py::reset()`
   初始化当前 session 的文本条件、分辨率 schedule、prompt token 和 cache 容器。

2. `compute_kv_cache_gt()`
   把真实 RGB 帧编码成 VAE latent，并写入 `gt_obs` cache。这个 cache 是可信观测，用来纠正上一轮预测。

3. `infer_chunk()`
   在已有文本 cache + 真实观测 cache 的条件下，自回归预测下一段 latent。
   输出里最重要的是当前段的 `summed_codes` 和对应帧/latent 边界。

4. `correction_clear_pred()`
   清理上一轮预测 cache，避免模型把“自己想象出来的未来”当成真实历史。这个函数是闭环稳定性的关键。

把这段和服务端文件对应起来时，建议你按下面顺序跳读：

1. `infer/server.py::_predict_delta_actions_impl()`
2. `infer/server.py::_infer_latents_for_actions_and_advance_cache()`
3. `infer/server.py::_infer_summed_codes_for_step()`
4. `Worldmodel/runtime/tools/infinity_streaming_session.py`

这样不容易把“服务端调度逻辑”和“streaming session 底层语义”混成一层。

### 2.3.3 动作解码器链路：latent 到 6D delta

1. `_stage2_patchify_to_z64_BCTHW()`
   统一 latent 布局。如果 `summed_codes` 已经是 64 通道 patchified latent，就直接返回；
   如果是 16 通道未 patchify latent，就用 `pixel_unshuffle(2)` 把 2x2 空间块折到通道维。

2. `_stage2_decode_tokens_tnd()`
   运行 VAE decoder，但不使用最终 RGB 输出。函数注册 forward hook，截取最后一个 up_block feature，
   再送入 `Vae96ToTSformerEmbedAdapter` 得到 `(T,N,D)` token。

3. `_gather_window_tokens()`
   TimesFormer 每次看 4 帧窗口。这个函数按窗口起点把 `(T,N,D)` 展开为 `(K*4,N,D)`。

4. TimesFormer forward
   输出每个窗口内相邻帧的动作 delta。重叠窗口会在 `_stage2_predict_16_actions_for_segment_cm_deg()` 中平均。

5. `_stage2_deltas_to_actions_cm_deg()`
   把训练内部布局 `[dz,dy,dx,tx,ty,tz]` 转成 API 布局
   `[dx_cm,dy_cm,dz_cm,droll_deg,dyaw_deg,dpitch_deg]`。

### 2.3.4 训练数据流：Stage A、Stage B、Stage 1

1. Stage A：`train_stageA_ddp.py`
   输入是 latent + RGB teacher frames。RGB 走 frozen TimesFormer 得到 teacher tokens；
   latent 走 VAE decoder hook + Adapter 得到 student tokens；loss 让 student 对齐 teacher。

2. Stage B：`train_stageB_ddp.py`
   输入是 latent + expert trajectory。latent 仍走 VAE hook + Adapter + TimesFormer；
   label 走 `_normalize_delta_bt6()` 标准化；loss 在归一化空间比较预测动作和专家动作。

3. Stage 1：`train/train.py`
   输入是文本 token + video latent token。训练目标是让 Infinity 在 teacher-forcing 下预测下一段 latent token。
   这个阶段不直接训练动作头，但它决定闭环时 `summed_codes` 的质量。

### 2.3.5 GRPO 数据流：rollout 到 replay 训练

1. `Worldmodel/runtime/tools/GRPO/build_rollout_tasks.py`
   先把来源不统一的任务 JSON 归一化成 task jsonl。这里统一 `video_path`、`caption`、
   `fps`、`gt_pose_json`、`grpo_group_id`、`grpo_clip_id` 等字段，属于“schema 对齐层”。

2. `Worldmodel/runtime/tools/GRPO/generate_candidate_rollouts.py`
   再把每条 task 展开成 K 个候选 rollout 元数据。重点看：
   - `grpo_group_id` 为什么保持不变；
   - `candidate_id` 如何标识组内第几个候选；
   - `traj_id=000123_k02` 为什么会成为后续 `trajectory.json` 的目录名。

3. `Worldmodel/runtime/tools/GRPO/generate_candidate_trajectories_real.py`
   这是“候选元数据真正变成真实轨迹”的关键桥梁。它会：
   - 读取上一步生成的 candidate jsonl；
   - 直接导入 `grpo_server.py` 里的本地动作预测实现；
   - 与远端 simulator 做固定节奏的 `reset -> predict -> step_actions` 三段闭环；
   - 最终为每个 `traj_id` 写出 `trajectory.json`、segment trace 和 old logprob。

   第一遍阅读时重点盯三件事：
   - 为什么 `/reset` 只返回 1 帧，而 `/step_actions` 返回 16 帧；
   - 为什么 `instruction` 只在第 0 段传一次；
   - `sample_logprob_segments` 记录的是 old logprob，不是训练时重算的 new logprob。

4. `action_aware_grpo/grpo_server.py`
   GRPO 服务端复用在线闭环推理，但额外保存 segment trace、采样 token、old logprob 和 candidate metadata。

   两个 GRPO 客户端 `action_aware_grpo/runtime/client.py` 和 `action_aware_grpo/windows_client.py`
   与 `infer/client.py` 的主协议基本一致：`main() -> run_one_route()/run_one_task_unrealcv() -> _post_frames()/HTTP -> 动作执行/积分 -> JSON 输出`。
   它们额外多了 simulator-side `service` 模式：`run_env_service()` 暴露 `/reset` 和 `/step_actions`，
   `_service_reset()` 返回首帧、指令和初始 pose，`_service_step_actions()` 执行一批 6D 动作并返回后续真实帧。
   当前入口位置可用 `action_aware_grpo/runtime/client.py:1515`、`:1286`、`:872`、`:823`、`:646`、`:688` 快速跳转；
   Windows 版同名函数行号一致，但默认 `--out_dir` 是 Windows 路径。

5. `Worldmodel/runtime/tools/GRPO/reward_uavflow.py`
   把 rollout 轨迹转成 clip-level replay row，计算 action reward、task reward、CE/reference shaping，
   并在同一任务候选之间做 LOO/rank 相对优势。

6. `Worldmodel/runtime/infinity/trainer/sft_trainer.py`
   runtime trainer 在 hybrid batch 中区分 SFT 与 GRPO replay。读 `train_step()` 时重点看
   `use_grpo`、old logprob、ratio clip、advantage 和冻结/解冻策略。

## 3. 在线闭环推理

核心文件：`infer/server.py`

`infer/server.py` 在整个系统里是“在线推理编排层”，不是单纯的 HTTP wrapper。它把客户端上传的真实观测、
InfinityStar 世界模型、streaming KV cache、Stage-2 动作解码器和每条轨迹的 session 状态串起来。
读这个文件时可以按职责分成 7 层：

1. API 协议层
   定义 `/health` 和 `/v1/predict_delta_actions`，其中核心请求字段是 `session_id`、
   `instruction/prompt`、`images_base64`、`step/num_frames` 相关配置、`action_head_mode`、
   `allow_future_segments`、`prefix_mode`。

2. 轨迹状态层
   `TrajectoryState` 是同一个 `session_id` 的闭环记忆，保存真实帧前缀 `frames_cpu`、文本 prompt、
   `InfinityStreamingSession`、导出的 `kv_cache`、跨段边界 latent `last_latent_1`、目标尺寸和
   `last_emitted_segment`。

3. 输入预处理层
   `_predict_delta_actions_impl()` 把 `images_base64` 解码成 PIL，再按当前 dynamic schedule 的目标
   `(tgt_h,tgt_w)` resize/normalize 成 `[-1,1]` tensor，追加到 `TrajectoryState.frames_cpu`。

4. segment 调度层
   `InfinityConfig.points()` 把 `num_frames/step` 变成 `[1,17,33,49,...]` 这种 segment 边界；
   `_predict_delta_actions_impl()` 根据 `n >= points[seg+1]` 或 `allow_future_segments` 判断当前段能否输出动作。

5. 世界模型编排层
   `_infer_latents_for_actions_and_advance_cache()` 和 `_infer_summed_codes_for_step()` 负责调用
   InfinityStar 预测 `summed_codes`，并决定什么时候把真实观测写回 `gt_obs` cache。

6. 动作头编排层
   默认 `tsformer_latent` 路径走 `_stage2_predict_16_actions_for_segment_cm_deg()`：
   `summed_codes -> z64 -> VAE decoder 中间特征 -> adapter tokens -> TimesFormer 滑窗 -> 16 个 cm/deg 动作`。
   旧兼容路径 `actionhead_ref_vit` 会先把 latent 解码成预测视频，再跑参考 ViT 动作头。

7. 响应和账本层
   服务端返回 `actions/segment_index/num_received_frames/prefix_latents/done`，同时更新
   `last_emitted_segment`、`last_latent_1`、KV cache 导出副本和可选磁盘 latent cache。

服务协议使用 `session_id` 维护轨迹状态。典型调用节奏是：

```text
第 1 次：上传 1 帧预热图像 + instruction
返回：下一段动作，或只完成预热，取决于 allow_future_segments

第 2 次及以后：上传执行动作后收集到的 step 帧真实 RGB
返回：下一段 step 帧对应的动作
```

默认 `step=16`，`points()` 通常形如 `[1,17,33,49,...]`。代码中的 readiness 逻辑决定什么时候可以输出 segment：

- 默认模式：收到 segment 末端真实帧后才输出。
- `allow_future_segments=True`：收到 segment 起点真实帧后就允许预测下一段动作，用于严格闭环“1 帧 -> 动作 -> 16 帧 -> 动作”。
- `allow_future_last_segment=True`：允许最后一段在没有真实尾帧时用预测补齐。

把一次请求展开成代码链路，大致是：

```text
POST /v1/predict_delta_actions
  -> predict_delta_actions()
     只负责 FastAPI 入参、GPU 全局锁和异常包装

  -> _predict_delta_actions_impl()
     解析 session / prompt / 图片，维护 TrajectoryState，判断 segment 是否 ready

  -> _ensure_traj_infinity_session()
     第一次请求时创建 InfinityStreamingSession，并 reset(prompt) 写入文本 cache

  -> _prepare_firstframe_condition_if_needed()
     首帧预热：准备 i2v first-frame condition，不一定立即输出动作

  -> _infer_latents_for_actions_and_advance_cache()
     针对当前 segment 运行世界模型，得到 SegmentInferResult

  -> _infer_summed_codes_for_step()
     选择 first-frame leak / gt_obs / hybrid 历史注入策略，调用 InfinityStar 生成 summed_codes

  -> _stage2_predict_16_actions_for_segment_cm_deg()
     默认动作头：把 summed_codes 解码成 TimesFormer token，再输出当前 segment 的 16 个动作

  -> PredictDeltaActionsResponse
     返回 actions，并更新 last_emitted_segment / last_latent_1 / kv_cache
```

这条链路里最重要的分界线是：`infer/server.py` 负责“调度和状态”，`Worldmodel/runtime/tools/infinity_streaming_session.py`
负责“KV cache 语义”，`Worldmodel/action_decoder/src/models/vae96_to_tsformer_adapter.py` 和 Stage-2 checkpoint
负责“latent 到动作”。不要把这三层混成一个模型。

### 3.1 session 生命周期与内部别名

这里有一个很容易忽略、但对调试非常关键的机制：

```text
external_session_id
    ->
_SESSION_ALIAS[external_session_id]
    ->
internal session_id
    ->
_TRAJ[internal session_id]
```

对应代码：

- `_make_run_session_id()`
  为一次新的内部 run 生成唯一 session，避免覆盖旧的 latent cache 目录。

- `_get_or_create_traj()`
  通过内部 `session_id` 找到或创建 `TrajectoryState`。

- `_predict_delta_actions_impl()`
  负责决定什么时候沿用旧内部 session，什么时候为同一个 external session 重新开新 run。

几个关键规则：

1. 客户端传过来的 `session_id` 只是 external session id，不一定就是服务端真正保存状态的 key。
2. 当 `reset_session=True` 时，服务端会显式创建新的 internal session id。
3. 默认情况下，只要请求满足“1 帧 + prompt/instruction”，服务端也会把它视为一条新的 run。
4. 这就是为什么你可能看到客户端 session id 没变，但服务端磁盘 cache 目录名变了。

### 3.2 请求/响应状态机

这张表建议你边看 `PredictDeltaActionsRequest/Response`，边对照 `infer/client.py::run_one_route()`。

| 场景 | 客户端发送 | 服务端关键判断 | 典型响应 |
| --- | --- | --- | --- |
| 首次预热 | 1 帧 + instruction | 建立 `TrajectoryState`，可能只做预热 | 常见 `segment_index=-1`，`actions=[]` |
| 正常中间段 | 新增 `step` 帧 | `n >= points[seg+1]` 或允许 future segment | 返回当前 segment 的动作 |
| prefix 模式 | 每次发送 `[1..K]` 完整前缀 | 服务端只追加新增尾帧，避免重复 | 响应字段与普通模式相同 |
| 最后一段 future tail | 可能没有最后 16 帧真实观测 | `allow_future_last_segment=True` | 最后一段动作可提前发射 |

重点字段怎么理解：

- `segment_index=-1`
  这次请求还没有新 segment 可发射，最常见于首帧预热。

- `done`
  当前轨迹是否已经走到配置的 `num_frames` 终点。

- `prefix_latents`
  当前真实前缀在 latent 时间线上的右边界索引，不是 RGB 帧数。

- `prefix_mode`
  客户端每次发完整前缀；服务端内部会自动去重，只保留新尾帧。

- `allow_future_segments`
  允许在只到达 segment 左边界真实帧时就先预测下一段动作。

- `allow_future_last_segment`
  允许最后一段在没有真实尾帧时由预测补齐。

客户端还有一个非常实用的技巧：

- 在 `prefix_mode + allow_future_last_segment` 下，客户端会“重复发送一次最长 prefix”。
  这样即使没有新增真实帧，服务端也会因为状态已经 ready 而发射最后一段动作。

重点读：

- `PredictDeltaActionsRequest`
  请求字段定义，尤其是 `session_id`、`images_base64`、`allow_future_segments`、`prefix_mode`。

- `TrajectoryState`
  记录同一条轨迹的真实帧、文本 prompt、KV cache、上一段 latent 边界和已输出 segment。

- `_predict_delta_actions_impl()`
  在线服务主流程：解码图像、追加真实帧、判断 segment 是否 ready、调用世界模型、调用动作头、返回 6D 动作。

- `_infer_latents_for_actions_and_advance_cache()`
  连接世界模型和闭环 cache 纠偏。只有当真实帧到达 `points[seg+1]` 时，才推进 `gt_obs` cache。

客户端对应文件：`infer/client.py`

- `run_one_route()`
  数据集/离线帧模式。它按 `1, step, step, ...` 或 prefix 模式上传帧，并保存每段 actions/poses JSON。

- `run_one_task_unrealcv()`
  仿真闭环模式。它把服务端返回动作真正应用到 UnrealCV 环境，再采集下一帧真实观测。

注意：当前服务端的 `tsformer_latent` Stage-2 路径输出 16 个逐帧动作。客户端仍保留旧 4 个宏动作兼容逻辑，是为了能读懂旧实验和旧 checkpoint，不代表当前默认路径仍是 4 动作。

## 4. 世界模型 cache 机制

核心文件：`Worldmodel/runtime/tools/infinity_streaming_session.py`

这里可以把 cache 分成三类：

- `t0`
  文本前缀 cache，来自语言指令，标记为 GT，不会在预测纠偏时清除。

- `gt_obs`
  真实观测 cache，来自客户端回传的真实 RGB 帧经 VAE 编码后的 latent。

- Pred cache
  本轮自回归推理产生的临时预测 cache。segment 结束后会通过 `correction_clear_pred()` 清掉。

理解这个文件后，闭环设计就很清楚：模型允许短程预演，但不会把预演当作永久记忆。下一轮必须重新锚定真实观测。

### 4.1 `Worldmodel/infinity` 底层实现怎么读

第一遍不建议从大模型源码开始，但当你已经理解 `infer/server.py` 和 streaming session 后，可以按下面顺序读底层：

先明确这里有两份相似但用途不同的 Infinity 代码：

- `Worldmodel/infinity/`
  更像训练/研究主副本。看 Stage-1 训练、checkpoint 导出、底层 schedule 或模型结构时，优先从这里进入。

- `Worldmodel/runtime/infinity/`
  更像在线服务/GRPO runtime 副本。看 `infer/server.py`、`action_aware_grpo/grpo_server.py`、真实闭环 rollout 或 runtime trainer 时，优先从这里进入。

读法建议：

1. 如果你在调在线推理、KV-cache、GRPO replay，先读 `Worldmodel/runtime/infinity/`。
2. 如果你在调 Stage-1 训练、模型结构、权重导出，先读 `Worldmodel/infinity/`。
3. 碰到同名文件时，不要假设两边完全一致；先确认当前入口的 `sys.path` 和 import 来源。
4. `utils/arg_util.py`、`utils/load.py`、`utils/video_decoder.py` 这类 helper 可以两边对照读：主副本通常保留训练语境，runtime 副本通常多服务端/闭环语境。

1. `Worldmodel/infinity/models/infinity.py`
   这是训练/推理共用的 Infinity Transformer 主体。先看文件头部中文导读，再看 `Infinity` 类：
   `forward()` 是训练路径，输入 ground-truth latent/token，目标是学习下一 token；
   `ar_infer_infinity_elegant()` 和 `ar_infer_infinity_star_interact()` 是推理路径，按 schedule 逐尺度自回归采样。

2. `Worldmodel/runtime/infinity/models/infinity.py`
   这是 runtime copy，服务端实际可能从这里 import。读法和主副本相同，重点看流式 KV-cache 相关方法：
   `set_cache_write_is_pred()`、`clear_pred_cache()`、`export_kv_cache()`、`import_kv_cache()`。

3. `Worldmodel/infinity/schedules/dynamic_resolution.py`
   这里决定不同帧数、宽高比、patch 数下的动态分辨率表。读它时只抓住三件事：
   `pn` 表示 patch budget，`h_div_w_template` 表示高宽比模板，`scale_schedule` 表示从粗到细的生成尺度。

4. `Worldmodel/infinity/schedules/infinity_elegant.py`
   这是当前主力 schedule。它把 `scale_schedule` 展开成模型能消费的 `scale_pack_info`、
   RoPE 区间、attention 可见性和 video encode/decode 的 latent packing 规则。

5. `Worldmodel/infinity/schedules/infinity_star_interact.py`
   这是交互/视频场景的 schedule 变体。重点看它如何处理视频帧、空间 patch、噪声注入和视觉 RoPE。

6. `Worldmodel/infinity/schedules/infinity_elegantold.py`
   这是旧版 schedule。读它主要是为了理解旧 checkpoint 或旧实验，不建议作为当前主线。

几个关键词：

- `summed_codes`
  自回归采样后得到的 latent code 总和，是服务端传给 Stage-2 动作头的核心张量。

- `scale_schedule`
  多尺度生成顺序。粗尺度先定大结构，细尺度再补细节。

- `scale_pack_info`
  把“当前尺度要读/写哪些 token、哪些帧、哪些空间块”打包给模型和 RoPE 的辅助结构。

- `RoPE cache`
  视觉 token 的位置编码缓存，必须和当前 schedule 的帧/空间范围一致，否则会出现形状不匹配。

如果只想读闭环推理，不需要先读完整 Transformer block；先看 schedule 和 KV-cache 辅助函数，已经足够理解服务端为什么能分段续写上下文。

### 4.2 分辨率与 schedule 怎样落到实际输入尺寸

很多人第一次会困惑：为什么原始数据集帧可能是 `256x256`，但服务端最后送进模型的并不是这个尺寸？

关键链路是：

```text
h_div_w_template
   -> 选择最接近的宽高比模板
   -> pt = (num_frames - 1)//4 + 1
   -> pt2scale_schedule[pt]
   -> scale_schedule[-1]
   -> tgt_h = scale_schedule[-1][1] * 16
   -> tgt_w = scale_schedule[-1][2] * 16
```

也就是说：

1. 模型先根据 `h_div_w_template` 选一套最接近的宽高比模板。
2. 再根据当前 `num_frames` 对应的 latent 时间长度 `pt` 取出一条 `scale_schedule`。
3. 最后一个尺度决定真正的目标输入尺寸 `tgt_h/tgt_w`。

所以“原图是什么分辨率”和“模型最终吃进去的分辨率”是两件事。
原图只是在进入 `infinity_transform()` 之前的外部观测；真正送进模型前会被 resize 到训练期约定模板。

### 4.3 读 schedule、RoPE 和 cache 报错时怎么定位

这几个文件里的报错信息现在都保留了英文技术 token，但解释改成中文。读错时不要只看异常文本，要顺着变量名回到数据流：

1. `dynamic_scale_schedule 暂未实现`
   先看 `Worldmodel/runtime/infinity/schedules/__init__.py::get_encode_decode_func()`。它只负责把 schedule 名称分发到具体实现；如果这里报错，说明配置名和当前代码支持列表不一致，还没有进入模型推理。

2. `dynamic_scale_schedule=... 暂未实现`
   再看 `Worldmodel/runtime/infinity/schedules/dynamic_resolution.py::get_ratio2hws_pixels2scales()`。这里负责生成 `pt2scale_schedule`。如果 schedule 名称能分发但不能生成动态分辨率表，通常是训练配置、runtime 配置或 checkpoint 期望的 schedule 不一致。

3. `former_clip_features 的时间维 T 期望为 ...`
   这来自 `Infinity.ar_infer_infinity_elegant()` 的 `second_v_clip` 分支。它要求上一段 clip 的特征包含 1 个边界帧加 `frames_inner_clip` 个内部帧；如果 T 不匹配，后面的 RoPE 和插值都会对不上。

4. `embedding 维度 dim 必须是偶数`
   这来自 `get_timestep_embedding()`。时间 embedding 用一半维度放 cos、一半维度放 sin，所以 `dim` 不能是奇数。

### 4.4 支撑工具：序列并行、性能统计和 VAE 小模块

这些文件不是主业务入口，但调训练、显存或多卡问题时经常会碰到。建议按“是否影响数值结果”来区分：

1. `Worldmodel/runtime/infinity/utils/sequence_parallel.py:10`
   `SequenceParallelManager` 只管理序列并行（sequence parallel）的进程组状态。重点看 `init_sp()`：
   它先检查 `sp_size > 1`，再要求 `world_size % sp_size == 0`。如果这里报错，通常是启动进程数和配置的 `sp_size` 不匹配，还没有进入模型前向。

2. `Worldmodel/runtime/infinity/models/videovae/utils/context_parallel.py:10`
   `ContextParallelUtils` 管理 VideoVAE 的上下文并行（context parallel）。它和序列并行类似，都先建通信组，再让后续 gather/send/recv 复用这个组；区别是它更贴近 VAE 视频时间/上下文切分。

3. `Worldmodel/runtime/infinity/utils/profile/py_profiler.py:46`
   `PythonProfiler.__exit__()` 只打印 CPU profiler 的调用次数、自身耗时和累计耗时，不改变模型输出。
   `Worldmodel/runtime/infinity/utils/profile/torch_profiler.py:33` 负责把 PyTorch profiler trace 导出成 Chrome trace 文件。
   如果你只是学习数据流，可以跳过 profiler；如果你在定位慢函数，先看累计耗时最大的函数名，再回到对应模块读调用链。

4. `Worldmodel/runtime/infinity/utils/amp_opt.py:39`
   `AmpOptimizer` 封装 mixed precision 训练里的 `scale -> backward -> unscale -> clip -> step`。
   读 `backward_clip_step()` 时要注意：只有 `unscale_()` 之后梯度才是真实数值，才能安全做梯度裁剪和范数统计。

5. `Worldmodel/runtime/infinity/utils/freeze_utils.py:86`
   `apply_stageb_partial_freeze()` 只改变 `requires_grad`，不改变 forward 公式。它会先找 `block_chunks` 或 `blocks`，再按前缀 chunk 数冻结；如果模型没有这两个容器，说明当前模型结构不支持这个局部冻结工具。

6. `Worldmodel/runtime/infinity/models/rope.py:18`
   `precompute_rope2d_freqs_grid()`、`precompute_rope3d_freqs_grid()`、`precompute_rope4d_freqs_grid()` 分别准备 2D/3D/4D RoPE 频率表。读 RoPE 时先确认 `dim` 是否能被频率拆分规则整除，再确认 schedule 给出的帧/空间范围是否和 cache 对齐。

7. `Worldmodel/runtime/infinity/models/basic.py:153`
   `SelfAttention` 是 Infinity/Qwen 主干里的注意力层。它会根据可用 kernel 走不同 flash-attn 路径，但核心输入输出仍围绕 `(batch_size, seqlen, nheads, headdim)`；调形状错误时先看这里。

8. `Worldmodel/runtime/infinity/models/videovae/modules/normalization.py:41`
   `Normalize` 是 VAE 里的归一化包装器。`group/batch/no` 是模式名，不是三条不同业务链路：
   `group` 用 GroupNorm，`batch` 用 SyncBatchNorm，`no` 直接 Identity。

9. `Worldmodel/runtime/infinity/models/videovae/modules/quantizer/multiscale_bsq.py:128`
   `get_latent2scale_schedule()` 把 `(T,H,W)` latent 网格拆成一串从粗到细的量化尺度。
   这里的 `original/dynamic/dense/same*` 都是保留的配置枚举；读它时重点看返回的 `(t,h,w)` 列表如何决定残差量化顺序。

10. `Worldmodel/runtime/infinity/models/videovae/modules/quantizer/lookup_free_quantization.py:68`
    `LFQ` 把连续特征压成 bit-codebook；`finite_scalar_quantization.py:59` 的 `FSQ` 则按有限标量级别量化。两者都属于 VAE token 化的一部分，不直接输出动作。

11. `Worldmodel/runtime/infinity/models/videovae/utils/init_models.py:194`
    `load_unstrictly()` 和 `load_cnn()` 是兼容旧 checkpoint 的加载工具。读到 `missing_keys/unexpected_keys/loaded_keys` 时不要立刻当成错误；先看代码是否有意跳过旧权重或扩展卷积权重。

## 5. 动作解码器

核心文件：

- `Worldmodel/action_decoder/src/models/vae96_to_tsformer_adapter.py`
- `infer/server.py` 中 `_stage2_decode_tokens_tnd()`、`_stage2_predict_16_actions_for_segment_cm_deg()`

动作解码链路：

```text
summed_codes
  -> patchified z_ext
  -> VAE decoder
  -> hook decoder up_block feature
  -> Vae96ToTSformerEmbedAdapter
  -> TimesFormer sliding windows
  -> 6D 增量动作
```

这里最容易误读的一点是：VAE decoder 在推理动作时不一定为了输出 RGB。代码注册 forward hook 捕获中间特征，再把它变成 TimesFormer token。真正进入动作头的是 token，不是视频帧。

输出动作 API 约定：

```text
[dx_cm, dy_cm, dz_cm, droll_deg, dyaw_deg, dpitch_deg]
```

训练内部常见布局是：

```text
[dz, dy, dx, tx, ty, tz]
```

所以读动作相关代码时要特别注意单位和维度转换。

### 5.1 为什么 16 动作只看左上下文，不看右未来帧

当前默认 `tsformer_latent` 路径里，动作头会为当前 segment 额外取左侧最多 3 帧上下文：

```text
ctx_start_abs = max(1, clip_abs_start - 3)
clip_abs_start = obs_len + 1
clip_abs_end   = next_obs_len
```

原因是：

1. TimesFormer 使用 4 帧窗口；
2. 预测“到达某一帧”的动作时，最多只需要它左边的 3 帧上下文；
3. 如果去看右侧未来帧，就会造成未来信息泄漏。

实现上最容易看混的是：

- 绝对帧号是 1-based；
- Python tensor slice 是 0-based，且右边界是开区间；
- 所以代码里常出现 `s0 = ctx_start_abs - 1`、`e0 = clip_abs_end` 这种写法。

### 5.2 `tsformer_latent` 和 `actionhead_ref_vit` 的区别

| 模式 | 主要输入 | 中间桥接 | 动作来源 | 是否先解码 RGB |
| --- | --- | --- | --- | --- |
| `tsformer_latent` | `summed_codes` | VAE decoder feature -> Adapter token | Stage-2 latent-to-action TimesFormer | 不必须 |
| `actionhead_ref_vit` | 预测视频帧 | 4 帧滑窗 ViT | reference-video action head | 需要 |

所以：

- `tsformer_latent` 更接近“直接从 latent 世界转移恢复动作”；
- `actionhead_ref_vit` 更接近“先得到预测视频，再从视频外观恢复动作”。

### 5.3 跨 segment latent 是怎么拼起来的

这里最关键的变量是：

- `latent5_input`
  当前动作头真正消费的 5 个 latent 时间步输入。

- `last_latent_1`
  上一段末尾保留下来的 1 个边界 latent。

拼接规则：

1. `seg0`
   直接取 `[end-4 .. end]`，得到完整 5 个 latent。

2. `seg>0`
   只从当前段新切出 `new4 = (prev_end+1 .. cur_end)` 这 4 个新 latent，
   再和上一段保留的 `last_latent_1` 拼成：

```text
latent5_input = [last_latent_1, new4]
```

这么做的原因是：

- 动作头看的是连续 5 个 latent 时间步；
- 相邻 segment 之间必须共享 1 个边界 latent，才能保持时间连续，不会在段间断开。

动作解码器训练数据入口：`Worldmodel/action_decoder/src/datasets/latent_traj_manifest.py`

这个文件把 manifest 中的三类路径统一成训练样本：

```json
{
  "latent_path": "path/to/latents.pt",
  "traj_json_path": "path/to/preprocessed_logs.json",
  "images_dir": "path/to/images"
}
```

读它时重点看：

- `_load_latents()`
  支持 raw Tensor 或 `{"latents": Tensor}` 两种 checkpoint 格式。

- `_load_traj()`
  把 list pose、dict wrapper、`action6` 等格式统一成 `(T,6)` 标签。若输入是 `(T-1,6)`，会补 `delta[0]=0`。

- `_load_frames()`
  Stage A 才需要 RGB 帧，用于 teacher token distillation。

- `__getitem__()`
  返回 `z_ext`、`traj`、可选 `frames_rgb`，并支持坏样本返回 empty tensor。

## 6. 从 VLA 到 WAM

VLA 通常是：

```text
f(o_t, instruction) -> action
```

WorldVLN 更接近：

```text
p(z_{t+1:t+K} | z_{\le t}, instruction) -> D_phi(z_{t+1:t+K}) -> actions
```

差异不只是多一个模块，而是控制范式变化：VLA 更像反应式控制，WAM 先预测世界转移，再从转移中恢复动作。无人机场景中视角变化快、累计误差明显，因此这种短程预演 + 真实观测纠偏的协议很重要。

## 7. 训练路线

> 在跳进每一阶段的细节之前，强烈建议先扫一眼下面这张「三阶段速览」。
> 很多人第一次会把 *Stage 1 / Stage A / Stage B / GRPO* 弄混，因为「Stage 1 vs Stage 2」是「世界模型 vs 动作解码器」的大阶段划分，
> 而「Stage A vs Stage B」是 Stage 2 内部的两步。它们是**两个独立维度**，不是同一条线性流程的四段。

### 7.0 三阶段速览：到底在监督什么？

把整条训练流水线按「**输入 → 监督信号 → 监督什么 → 产出**」摊开，你会发现这其实是 4 套独立训练，每一套都有自己的 GT 和 loss 形式：

```text
Stage 1（世界模型监督训练）
   输入：自然语言指令 + 真实视频 latent（teacher-forcing）
   监督：CrossEntropy on next-token / next-bit
   GT  ：video VAE BSQ / LFQ 量化器输出的真实 bit code
   产出：InfinityStar transformer checkpoint

Stage 2 = Stage A + Stage B + GRPO
   ┌─ Stage A（Adapter 蒸馏）
   │     输入：真实 RGB 帧 + 同段 latent
   │     监督：让 student token 对齐 teacher token
   │     GT  ：frozen TimesFormer 看真实 RGB 得到的 teacher 分布
   │     产出：Adapter checkpoint
   │
   ├─ Stage B（latent → 动作监督）
   │     输入：z_ext (latent) + 专家轨迹 JSON
   │     监督：MSE 预测相邻帧 6D delta
   │     GT  ：专家轨迹相邻帧的 [dz,dy,dx,tx,ty,tz]（rad+m，再做归一化）
   │     产出：stage2_latent2action_combined checkpoint
   │
   └─ GRPO（强化学习微调）
         输入：rollout 轨迹 + reward + old_logprob
         监督：策略梯度（GRPO/PPO 风格的 ratio·advantage）
         GT  ：reward（动作 MSE + 任务进度 + CE shaping），不是 token 标签
         实际更新：Stage 1 的世界模型权重（动作头通常冻结或低 lr）
```

**核心区别一句话：**

- **Stage 1** 教模型「看图说下一帧」——和动作完全无关，是一个可控视频生成式的世界模型。
- **Stage A** 教 Adapter「翻译」——不学动作，只让 latent 路径出来的 token 在 TimesFormer 眼里和真实 RGB 路径出来的 token「长一样」。
- **Stage B** 才是「动作监督」——把 Stage 1 的 latent + Stage A 的对齐 token 翻译成具体 6D 动作。
- **GRPO** 把世界模型从「会模仿真实视频」升级成「会生成能解出好动作的 latent」。监督信号从 token 标签换成了 reward。

#### 7.0.1 监督信号对照表

| 维度 | **Stage 1** | **Stage A** | **Stage B** | **GRPO** |
|---|---|---|---|---|
| **训练对象（哪个权重在更新）** | InfinityStar transformer | Adapter | TimesFormer + Adapter | InfinityStar transformer（再训） |
| **监督信号本质** | 下一 bit code 概率 | teacher token 分布 | 专家动作 delta | rollout reward |
| **Loss 类型** | CrossEntropy (label_smooth) | `cos + MSE + mean + std` 复合 | 加权 MSE（旋转/水平/垂直分组） | 策略梯度 (ratio·advantage) |
| **GT 来源** | video VAE 量化 bit code | frozen TSformer 看真实 RGB | 专家轨迹 JSON | 模拟器 + reward 函数 |
| **是否需要动作标签** | ❌ | ❌ | ✅ | ❌（用 reward 代替） |
| **是否需要真实 RGB 帧** | 仅训练阶段编码用 | ✅（teacher 分支） | latent 即可 | ❌（rollout 自采样） |
| **是否需要专家轨迹** | ❌ | ❌ | ✅ | reward 计算时可作参考 |
| **教会模型什么能力** | 给定指令和历史，想象未来 latent | 让 latent token ≈ RGB token | 把 latent 翻成 6D 动作 | 让 latent 翻出的动作真的能完成任务 |
| **入口脚本** | `train/train.py` | `train_stageA_ddp.py` | `train_stageB_ddp.py` | `action_aware_grpo/scripts/run_stageb_partialfreeze.sh` |
| **关键 loss 文件:行号** | `Worldmodel/runtime/infinity/trainer/sft_trainer.py:131` | `train/action_decoder/tools/train_stageA_ddp.py:76` | `train/action_decoder/tools/train_stageB_ddp.py:459` | `Worldmodel/runtime/tools/GRPO/reward_uavflow.py:131` + sft_trainer.py:506 |

#### 7.0.2 数据/标签到底长什么样

很多人对「监督什么」感到模糊，是因为没搞清楚每一阶段**真正喂给 loss 函数的张量**是什么。下面把 4 个阶段的"loss 输入"列清楚：

```text
Stage 1
    pred  : transformer 输出的 logits  shape (B, L, V)，V = codebook 规模
    target: video VAE 量化得到的 bit code 索引 shape (B, L)
    loss  : CrossEntropy(pred, target)  按 scale 可加权

Stage A
    student_tokens: (B*T, N=480, D=384)  ← latent → VAE hook → Adapter
    teacher_tokens: (B*T, N=480, D=384)  ← 真实 RGB → frozen TimesFormer.patch_embed
    loss = w_cos*L_cos + w_mse*L_mse + w_mean*L_mean + w_std*L_std

Stage B
    pred  : (B, K, 18)   ← TimesFormer 4 帧滑窗对每个窗口输出 3 个相邻帧 6D delta
    target: (B, K, 18)   ← 专家轨迹相邻帧 delta，先 _normalize_delta_bt6() 标准化
    loss = w_rot*MSE_rot + w_xy*MSE_xy + w_z*MSE_z

GRPO
    new_logprob: 当前策略对 rollout 时采样到的 token 重新打分
    old_logprob: rollout 时刻冻结的旧策略概率
    advantage  : 组内 LOO/rank 优势（公式见 §2.2 GRPO 行）
    loss = -E[ exp(new_logprob - old_logprob) * advantage ]
    （可选 + KL 约束 / + aux SFT 项；trace_replay 模式下默认 PG-only）
```

#### 7.0.3 一图流：四阶段如何串成训练流水线

```text
              ┌─── Stage 1：教世界模型「想象」────────────┐
              │ 输入：文本 + 真实视频 latent              │
              │ 监督：CE on next bit code                  │
              └────────────────────┬───────────────────────┘
                                   │ 产出 InfinityStar ckpt
                                   ▼
              ┌─── Stage A：教 Adapter「翻译」───────────┐
              │ 输入：同段 RGB + latent                   │
              │ 监督：student token ≈ teacher token       │
              └────────────────────┬───────────────────────┘
                                   │ 产出 Adapter ckpt
                                   ▼
              ┌─── Stage B：教动作头「执行」─────────────┐
              │ 输入：latent + 专家轨迹                   │
              │ 监督：MSE on 6D delta                      │
              └────────────────────┬───────────────────────┘
                                   │ 产出 stage2_combined ckpt
                                   ▼
              ┌─── GRPO：让想象出的未来「能完成任务」────┐
              │ 输入：rollout + reward + old_logprob      │
              │ 监督：策略梯度（组内相对优势）            │
              │ 实际更新：Stage 1 的世界模型权重          │
              └────────────────────────────────────────────┘
```

#### 7.0.4 容易踩的坑

1. **"Stage 1 训完动作就会了？"**——不会。Stage 1 完全不监督动作，它只是个会按指令生成视频 latent 的世界模型。要拿到动作能力必须再走 Stage A + Stage B。
2. **"Stage A 不是已经在监督动作了吗？"**——不是。Stage A 只对齐 token 分布，不看任何动作标签。它的存在是为了让 Stage B 能复用 TSformer-VO 的预训练动作能力。
3. **"GRPO 训练的是动作头？"**——一般不是。GRPO 主要把世界模型当成策略来 fine-tune；动作解码器在 GRPO 阶段通常被冻结或给极低学习率，因为它只是「latent → 动作」的翻译器，不是策略本体。
4. **"Stage B 的 loss 直接是 m / rad 吗？"**——不是。专家 delta 会先经 `_normalize_delta_bt6()` 用训练集的 mean/std 归一化，再和模型输出比对。推理时反向操作（先反归一化，再做单位换算到 cm/deg）。详见 §7.1。
5. **"GRPO 的 reward 是单一标量？"**——不是。它是 `λ_act·r_act + λ_task·r_task + λ_ce·r_ce` 的复合 reward，每一项再各自有自己的子结构（动作 MSE 分 xyz/yaw/all6；任务 reward 分 dense/success；CE shaping 用 reference policy）。详见 §2.2 GRPO 行。

读完上面这一节，你应该可以闭着眼回答："这一阶段的 loss 在比对什么张量？为什么需要这一阶段？"。然后再往下看 7.1+ 的具体实现。

---

Stage 1：监督训练世界模型

- 入口：`train/scripts/train_from_base.sh`
- 主程序：`train/train.py`
- 文档：`train/TRAINING.md`

默认设置包括 Python 3.10、49 帧、fps 16、分辨率预设 `0.40M`、模型 `infinity_qwen8b`、学习率 `1e-5`。这一步让世界模型学习：

```text
语言条件 + 历史真实 latent -> 下一段真实 latent
```

Stage A：adapter 蒸馏

- 入口：`train/action_decoder/scripts/train_stageA_ddp.sh`
- 主程序：`train/action_decoder/tools/train_stageA_ddp.py`

目标是让 Adapter 学会把 VAE decoder 中间特征翻译到 TimesFormer patch token 空间。

Stage A 阅读检查点：

- `train_stageA_ddp.py` 的 `compute_distill_loss()`
  看 student/teacher token 如何用 cosine/MSE/均值/方差对齐。

- `main()` 中 teacher branch
  RGB 帧走 TimesFormer patch_embed，teacher 冻结。

- `main()` 中 student branch
  `z_ext` 走 VAE decoder hook，Adapter 训练。

Stage A 建议按这条函数链跟读：

```text
main()
  -> build_tsformer()
  -> load_tsformer()
  -> load_infinitystar_vae()
  -> LatentTrajManifestDataset(load_frames=True)
  -> collate_fn()
  -> _decode_teacher_student_tokens()
  -> compute_distill_loss()
  -> _atomic_torch_save()
```

这条链的核心形状关系是：

```text
z_ext:           (B,64,T_lat,16,16)
frames_teacher:  (B,3,T_rgb,192,640)
VAE hook feature:(B,96,T_dec,H_feat,W_feat)
Adapter tokens:  (B,T_dec,N=480,D=384)
Teacher tokens:  (B,T_rgb,N=480,D=384)
```

其中 `T_dec` 通常由 video VAE 时间上采样决定，近似：

```text
T_rgb = 4 * (T_lat - 1) + 1
```

所以 `latent_chunk_len=3` 时默认对应 9 帧 RGB teacher。实际训练里如果 VAE slicing、样本缺帧或数据裁剪导致长度不一致，代码会按 student/teacher 较短的一侧对齐后再算 loss。

Stage A 的学习目标不是预测动作，也不是重建好看的 RGB，而是让：

```text
VAE decoder 中间特征 -> Adapter -> TimesFormer token 空间
```

这一步对齐后，Stage B 才能把同一套 latent token 接到动作头上训练。

Stage A 教师-学生优化训练细读：

一句话：这里不是在重新训练 VAE，也不是让 VAE 重建 RGB；这里训练的是 `Vae96ToTSformerEmbedAdapter`。Teacher 给出目标 token，student 用 latent 经过冻结 VAE 的中间特征生成 token，然后只让 Adapter 向 teacher token 靠近。这个设计的目的，是把“世界模型会输出的 latent”接到“原 TimesFormer 动作头熟悉的 patch token 空间”。

```text
teacher:
  真实 RGB PNG
    -> resize / ToTensor / KITTI mean-std normalize
    -> frozen TimesFormer.patch_embed
    -> tok_t

student:
  z_ext 或 z_sub
    -> frozen InfinityStar VAE.decode
    -> hook decoder 最后一个 up block feature
    -> trainable Adapter
    -> tok_s

optimize:
  loss(tok_s, tok_t) -> backward -> update Adapter only
```

先看三个角色：

- Teacher 是冻结的 TimesFormer。`main()` 里先 `build_tsformer()`、`load_tsformer()`，然后 `eval()` 并对所有参数 `requires_grad_(False)`。它不学习，只把真实 RGB 帧转成稳定的 `tok_t` 目标。

- Student 的可训练部分只有 Adapter。`Vae96ToTSformerEmbedAdapter` 接收 VAE decoder 中间 feature，输出 TimesFormer 风格 token。`AdamW` 的参数只来自 Adapter；DDP 情况下取 `adapter.module.parameters()`。

- VAE 是冻结特征抽取器。代码通过 `_VaeDecodeOnly.forward()` 调 `vae.decode(z_ext)`，但最终 RGB 输出会被丢弃，真正使用的是 `vae.decoder.up_blocks[-1]` 的 hook feature。

为什么冻结 VAE 还要 decode：Stage B 和推理阶段拿到的是世界模型预测出的 latent，不是真实 RGB。如果直接拿 RGB 训练动作头，Stage B 仍然依赖 `TimesFormer.patch_embed(RGB)`，就绕不开真实图像。Stage A 做的是中间桥接：

```text
world model latent
  -> frozen VAE decoder hidden feature
  -> Adapter
  -> TimesFormer-compatible tokens
```

这样 Stage B 才能在没有真实 RGB patch_embed 的情况下，复用 TimesFormer 风格的 token 表征继续学习动作。

数据侧的对齐方式：

- `LatentTrajManifestDataset(load_frames=True)` 同时取 `z_ext` 和 `frames_rgb`。`z_ext` 走 student 分支，`frames_rgb` 走 teacher 分支。

- Teacher 的 PNG 帧会通过 `_build_png_transform()` 统一成 `(3,T,192,640)`，并使用 KITTI mean/std normalize，保证 teacher 分支输入分布匹配 TimesFormer 训练时的图像统计。

- `collate_mode=per_sample` 是默认稳妥模式。它不强行把变长样本拼成一个大 batch，而是把每个样本单独放进 `samples` 列表；坏样本、缺帧样本、过短 latent 可以被单独跳过，不会拖垮整个 batch。

- `collate_mode=crop` 会把 batch 内 latent 和 RGB 都裁到公共最短长度再拼 batch，吞吐更好，但对数据长度一致性要求更高。

- video VAE 的时间关系近似是 `T_frames = 4*(T_lat-1)+1`。所以 `latent_chunk_len=3` 对应 9 帧 teacher，`latent_chunk_len=5` 对应 17 帧 teacher。这不是动作窗口长度，而是 latent 时间步 decode 后对应的 RGB 帧数。

- 当 `latent_use_full=True` 时，代码直接使用整条 latent 序列，`expected_T=None`，teacher 也使用整段 RGB。这个最直观，覆盖最完整，但显存最高。

- 当关闭 `latent_use_full` 时，代码会构造 `z_sub_list`。如果 `latent_cover_all=True`，按 `latent_stride` 枚举所有 latent 窗口；否则每条样本只取一个窗口，`latent_chunk_random=True` 时随机起点，关闭时固定从 0 开始。

- 对非 0 起点窗口，teacher 帧起点用 `f0 = 4*st_lat - 3` 近似对齐。这是在匹配 video VAE 前缀累计帧数和 latent 边界重叠关系，不是简单的 `4*st_lat`。

`_decode_teacher_student_tokens()` 是这段优化的关键函数：

- 它先注册 `vae.decoder.up_blocks[-1].register_forward_hook(hook)`。hook 只负责把最后一个 up block 的 5D feature 收集起来，形状类似 `(B,96,T_dec,H_feat,W_feat)`。

- VAE decode 放在 `torch.no_grad()` 和 AMP 里运行。这样不会为冻结 VAE 保存反传图，显存压力会小很多。

- hook 里保存的是 `hs.detach()`。这一步很关键：VAE feature 只是 Adapter 的输入，不应该继续连着 VAE decoder 的计算图。

- Adapter 不能放在 VAE 的 `no_grad()` 作用域里执行。正确做法是先用 hook 拿到 detached feature，再在启用梯度的上下文中跑 Adapter；否则 loss 可能没有连到 Adapter，`AdamW` 的 optimizer state 会一直为空。

- Adapter 前向按时间维用 `adapter_frames_chunk` 切块。Adapter 内部没有跨时间混合，所以把 `(B,96,T,H,W)` 拆成几段再拼回 `(B,T,N,D)`，数学结果等价，但反传要保留的激活更少。

- `adapter_use_checkpoint=True` 时，Adapter 前向用 `torch.utils.checkpoint(..., use_reentrant=False)`。这里必须用 non-reentrant，因为输入 feature 已经 detach，不需要输入梯度；默认 reentrant checkpoint 可能因为输入都不带 grad 而不建图，导致 loss 没有 `grad_fn`。

- Teacher token 用 `tsformer.patch_embed(frames_teacher)` 得到，并在 `torch.no_grad()` 下运行。Teacher 和 student 最后都整理成 `(B,T,N=480,D=384)`，若帧数不一致，就按较短的 `T` 截齐后算 loss。

一个 optimizer step 内部可以按这条链路理解：

```text
DataLoader batch
  -> optimizer.zero_grad(set_to_none=True)
  -> 遍历 batch 里的每个 sample
  -> 为 sample 选择一个或多个 latent window: z_sub
  -> 根据 st_lat 裁 teacher RGB: fr_sub
  -> _decode_teacher_student_tokens(z_sub, fr_sub)
  -> compute_distill_loss(tok_s, tok_t)
  -> scaler.scale(loss_i).backward()
  -> 多个 window 的梯度按 backward_count 求平均
  -> grad clip
  -> scaler.step(optimizer)
  -> scaler.update()
```

这里最重要的优化是“即时 backward”。代码不是先把一个 batch 里所有 sample/window 的 loss 全部累起来，再一次 backward；而是每得到一个有效 `loss_i` 就立刻 `backward()`。这样可以尽早释放当前窗口的 Adapter 计算图，避免完整 batch、完整视频、多个窗口的图全部堆在显存里。

为什么要除以 `backward_count`：如果 `latent_cover_all=True`，一条样本可能产生多个 latent window。如果不把梯度除以窗口数，那么窗口越多，等效学习率越大；训练效果会被采样策略影响。代码在同一个 optimizer step 内多次 backward 后，把所有 Adapter 参数梯度除以 `backward_count`，让“枚举多个窗口”更接近“对多个窗口取平均 loss”。

为什么有 DDP-safe zero backward：分布式训练里，各 rank 需要保持相同的 backward/step 节奏。如果某个 rank 因为坏样本、缺帧、VAE decode 空输出导致没有任何有效 loss，它也必须参与一次 backward。代码用可训练参数求和再乘 0，构造一个带图的零 loss：

```text
loss_batch = sum(trainable_params) * 0.0
```

这个 loss 不改变梯度数值，但能让 DDP 的通信和 optimizer step 结构保持一致，避免某些 rank 在等待梯度同步时卡住。

loss 设计：

```text
L_total = w_cos*L_cos + w_mse*L_mse + w_mean*L_mean + w_std*L_std
```

- `L_cos = mean(1 - cos(student, teacher))`，先让 token 方向对齐。对蒸馏来说，方向通常比绝对数值更稳定。

- `L_mse = mean((student - teacher)^2)`，让逐元素数值也贴近 teacher。它对尺度更敏感，前期权重太大可能让训练不稳定。

- `L_mean` 对齐整批 token 每个维度的均值，约束 student token 的分布中心。

- `L_std` 对齐整批 token 每个维度的标准差，约束 student token 的分布宽窄。

默认 loss schedule 是 `piecewise_linear`：前若干 epoch 保持起始权重，之后线性切到结束权重。默认意图是前期 `cosine` 权重大、`MSE` 权重小，先学“方向像不像”；后期逐步降低 `cosine`、提高 `MSE`，再学“数值准不准”。这比一开始就强压 MSE 更稳。

显存和吞吐相关参数可以这样读：

- `global_batch_size` 会按 DDP `world_size` 自动换算每卡 `batch_size`，避免启动脚本里手动算错。

- `adapter_frames_chunk` 越小越省显存，但 Adapter 前向循环次数更多，速度更慢。

- `adapter_use_checkpoint` 省显存但会重算 Adapter 前向，训练变慢；当视频长、batch 大或显存紧张时通常值得开。

- `latent_use_full=True` 覆盖最完整但显存最高；关闭后用 `latent_chunk_len`、`latent_cover_all`、`latent_stride`、`latent_max_windows` 控制覆盖率和显存。

- `vae_disable_slicing`、`vae_disable_tiling` 主要用于排查 VAE 行为。禁用后行为更直观，但显存更高。

- `vae_num_sample_frames_batch_size` 用来覆盖 VAE 内部一次处理多少帧，可用于调 VAE decode 的峰值显存。

- `amp`、`GradScaler` 和 `grad_clip` 是数值稳定和显存吞吐优化。AMP 降低显存和加速，`grad_clip` 防止蒸馏早期梯度尖峰。

checkpoint 和导出逻辑：

- 常规保存文件是 `stage1_adapter_e{epoch}.pt` 和 `stage1_adapter_last.pt`，里面有 `adapter_state_dict`、`optimizer_state_dict`、`scaler_state_dict`、`args`、`global_step`。

- `export_combined=True` 时，会额外导出 `infinitystar_up3_plus_adapter_latent2tokens.pt`，里面打包 `vae_state_dict` 和 `adapter_state_dict`。这个文件是给 Stage B 用的：Stage B 可以直接拿“VAE + Adapter”作为 latent 到 TimesFormer token 的前端。

读源码时建议按这个顺序跳：

- `compute_distill_loss()`：先看 student/teacher token 到底怎么被约束。

- `_loss_weights_for_epoch()`：看 `cosine` 和 `MSE` 的 epoch 级权重调度。

- `_build_png_transform()`：看 teacher RGB 如何变成 TimesFormer 熟悉的输入分布。

- `collate_fn()`：看坏样本、变长 latent、缺帧样本如何被处理。

- `main()` 的 teacher/student/VAE 初始化段：确认 TimesFormer 和 VAE 冻结，只有 Adapter 进 optimizer。

- `_decode_teacher_student_tokens()`：重点看 no-grad VAE decode、hook、detach、Adapter 分块、checkpoint 和 student/teacher 对齐。

- `main()` 的 sample/window 训练循环：看 latent 窗口怎么选、teacher 帧怎么裁、为什么每个 window 立刻 backward。

- checkpoint/export 段：看 Stage A 产物如何交给 Stage B。

这套优化训练的核心取舍是：用冻结 VAE 和冻结 TimesFormer 保持目标空间稳定，只训练轻量 Adapter；用 hook、detach、no-grad、时间切块、checkpoint、即时 backward 把显存压下来；用 loss schedule 从方向对齐过渡到数值对齐，让 Adapter 逐步学成可给 Stage B 使用的 latent-to-token 前端。

Stage B：latent 到动作训练

- 入口：`train/action_decoder/scripts/train_stageB_ddp.sh`
- 主程序：`train/action_decoder/tools/train_stageB_ddp.py`

目标是从真实视频片段 latent 恢复专家动作。也就是先对齐表征，再学习动作输出。

Stage B 阅读检查点：

- `main()` 里的 argparse 参数分组
  现在 CLI help 已补充到可以当作“运行参数索引”来读：manifest/checkpoint、VAE 结构与加载、
  batch/window/DataLoader、optimizer/loss、freeze/train VAE、导出组合 checkpoint 等选项都有中文解释。
  如果你不确定某个开关控制哪条训练分支，先运行 `python train/action_decoder/tools/train_stageB_ddp.py --help`
  或直接看 `train/action_decoder/tools/train_stageB_ddp.py:703` 之后的参数定义。

- `traj_abs_to_delta()` / `delta_to_delta()` / `traj_to_delta()`
  先看这三个函数如何把不同来源的标签统一成同一套布局和单位。
  这是 Stage B 最容易卡住的地方，比 label stats 还更前置。

- `_try_load_label_stats()`
  判断动作标签是否需要按原 TSFormer checkpoint 的统计量归一化。

- `_normalize_delta_bt6()`
  把 `(B,T,6)` 动作标签转到 checkpoint 期望分布。

- `build_tsformer()`
  构造 4 帧窗口 TimesFormer，输出 18 维，对应 3 个相邻动作增量。

- `main()` 的 optimizer groups
  区分 backbone、head、adapter、VAE 的学习率和冻结策略。

### 7.1 Stage B 标签在 normalize 之前怎么统一

Stage B 的 GT 标签可能有三种来源：

1. 绝对位姿轨迹 `abs_pose`
   形状通常是 `(T,6)`，布局 `[x,y,z,roll,yaw,pitch]`。

2. 已经是逐步 delta 的 6 维标签
   可能是 `(T-1,6)` 或 `(T,6)`。

3. 只含 yaw 的简化 delta
   可能是 `(T-1,4)` 或 `(T,4)`，布局近似 `[tx,ty,tz,dyaw]`。

统一目标都是：

```text
[dz, dy, dx, tx, ty, tz]
```

并且单位统一为：

- 角度：radians
- 平移：meters

对应函数关系：

- `traj_abs_to_delta()`
  绝对位姿 -> 相对 delta，核心公式是
  `R_rel = R_prev^T @ R_cur`，
  `p_rel = R_prev^T @ (p_cur - p_prev)`。

- `delta_to_delta()`
  把已经是 delta 的输入规范成 `(T,6)`，必要时补 `delta[0]=0`，并统一 dyaw 单位和位移单位。

- `traj_to_delta()`
  根据 `traj_mode` 统一调度前两者。

### 7.2 Stage B 的窗口监督、loss 权重和冻结时序

Stage B 里最值得跟读的是下面这条链：

```text
z_ext
  -> _decode_tokens_full_T()
  -> _sample_all_window_starts()
  -> _gather_window_tokens()
  -> _gather_window_targets()
  -> TimesFormer
  -> compute_loss_mse()
```

几个关键点：

1. 为什么 4 帧窗口监督 3 个动作
   因为动作描述的是相邻帧之间的转移，所以 4 帧只会产生 3 个 delta。

2. `crop` 和 `per_sample` 的差异
   - `crop`：裁到 batch 内最短长度后一起训练；
   - `per_sample`：逐条样本训练，不裁剪，最保真但更慢。

3. loss 权重怎么分
   `compute_loss_mse()` 会把旋转、水平平移、垂直平移分开加权：
   `loss = w_rot * L_rot + w_xy * L_xy + w_z * L_z`

4. 训练时序怎么分
   - `freeze_adapter_epochs`：前几个 epoch 先冻结 Adapter；
   - `freeze_backbone_epochs`：可选冻结 TimesFormer backbone；
   - `train_vae_after_epoch`：到指定 epoch 后再开始训练 VAE decoder 相关部分。

Stage 2：Action-aware GRPO

- Rollout 入口：`action_aware_grpo/scripts/run_stagea_collect.sh`
- 训练入口：`action_aware_grpo/scripts/run_stageb_partialfreeze.sh`
- 奖励构建：`Worldmodel/runtime/tools/GRPO/reward_uavflow.py`

GRPO 奖励组合关注三件事：

- 动作/轨迹几何是否接近专家。
- 当前 clip 或最终任务目标是否更接近成功。
- 新策略不要过度偏离参考/旧策略分布。

GRPO 阅读检查点：

- `generate_candidate_rollouts.py`
  给同一任务生成 K 个候选 rollout seed，使 GRPO 有同组候选可比较。

- `action_aware_grpo/grpo_server.py`
  这是 Action-aware GRPO 专用在线服务。它复用 InfinityStar streaming session，但会额外保存
  segment trace、sample/old logprob、latent cache 和 reference-video ActionHead 路径，方便 StageB
  用 rollout 结果做 replay/策略优化。

- `reward_uavflow.py` 的 `output_mode="clip"` 分支
  把一条 49 帧轨迹拆成 3 个 16 动作 clip，每个 clip 单独成为 replay row。

- `reward_uavflow.py` 的 group LOO/rank shaping
  同一任务同一 clip 的候选互相比较，形成相对优势。

- `Worldmodel/runtime/scripts/GRPO_stageB_partialfreeze_train.sh`
  设置 KL、ratio clip、冻结前缀、辅助 SFT 等训练保护项。

### 7.3 `Worldmodel/runtime/tools/` 目录怎么读

这个目录更像 runtime 工具箱，不是所有文件都应该作为第一入口。建议先看以下几类：

1. `Worldmodel/runtime/tools/run_infinity.py`
   通用推理工具。重点看 `load_tokenizer()`、`load_transformer()`、`gen_one_example()`、
   `save_video()`：服务端会复用这些函数完成文本编码、模型加载和 latent/video 生成。

2. `Worldmodel/runtime/tools/closed_loop_streaming_infer_480p_81f.py`
   离线/监听式闭环 streaming 调试脚本。它和服务端逻辑相互印证：先写真实观测 cache，再推理
   下一段预测视频，随后用新真实帧覆盖 cache。

3. `Worldmodel/runtime/tools/prompt_rewriter.py`
   prompt 改写辅助脚本。它不是闭环控制主线，但会影响视频生成 prompt 的规范性。
   建议重点看 `OpenAIGPTModel.__init__()`、`_creat_client()`、`__call__()`：
   - 它如何在 `ak` 为空时回退到环境变量/API key；
   - 它如何统计 token 并在 client 池里轮转；
   - 它最终如何把原始 prompt 包装成 rewrite 请求。

4. `Worldmodel/runtime/tools/GRPO/generate_candidate_trajectories_real.py`
   如果你在追 GRPO 的“真实闭环执行”而不是普通视频生成，这个脚本应该紧跟在
   `generate_candidate_rollouts.py` 后面读。重点看 `_run_remote_sim_rollout()` 的固定节奏：
   `1 帧 reset 观测 -> 1 段动作 -> 16 帧真实观测 -> 下一段动作`。

5. `Worldmodel/runtime/train.py`
   runtime 下的训练入口，保留了 GRPO replay/hybrid batch 字段。读它时可和 `train/train.py`
   对照，重点看 `train_one_epoch()` 传入 trainer 的 GRPO 字段。

### 7.4 `train/` 目录怎么读

`train/` 里有三类代码：Stage-1 世界模型训练、Stage-2 动作头训练/推理评估、checkpoint 导出发布。

第一遍建议按这个顺序读：

1. `train/scripts/train_from_base.sh`
   这是 Stage-1 启动脚本。先看它如何设置 `PYTHONPATH`、训练数据路径、49 帧配置、基础 checkpoint 和分布式参数。

2. `train/train.py`
   重点读 `build_everything_from_args()`、`build_dataset()`、`main_train()`、`train_one_epoch()`。
   这里已经额外标注了训练 batch 字段、T5 compact KV 条件、梯度累积、checkpoint 保存和 `trainer.train_step()` 的职责边界。
   真实执行链可以先记成：
   `arg_util.init_dist_and_get_args() -> main_train() -> build_everything_from_args() -> build_dataset() -> train_one_epoch() -> trainer.train_step()`。
   其中世界模型 loss、teacher forcing、下一段 latent target 的主体已经下沉到 `trainer.train_step()`，
   所以如果你在 `train_one_epoch()` 里找不到“真正的 loss 公式”，不是漏读，而是边界就在 trainer 里。

3. `train/TRAINING.md`
   看命令参数和数据准备说明。它更像操作手册，和 `train.py` 对照读。

4. `train/action_decoder/tools/train_stageA_ddp.py`
   看 Adapter distillation。核心问题是：VAE decoder 中间特征如何对齐 frozen TimesFormer teacher token。

5. `train/action_decoder/tools/train_stageB_ddp.py`
   看 latent-to-action 训练。重点是 4 帧窗口、18 维输出、动作标签归一化和 optimizer group。

6. `train/action_decoder/tools/predict_pose.py`
   这是 Stage-2 离线批量推理。它读取 `latents.pt`，通过 VAE hook + Adapter + TimesFormer 输出逐帧动作，再积分成轨迹。
   建议按这条链读：
   `main()` 里的 checkpoint/route 准备
   -> `infer_one_route()`
   -> `_decode_tokens_full_T()`
   -> `_gather_window_tokens()`
   -> 4 帧滑窗平均
   -> `integrate_trajectory_se3()`
   -> `pred_actions.json / pred_path.json / trajectory_m_deg.json`。
   这里最容易漏掉的是 `label_stats` 反标准化：如果 Stage-2 训练时标准化过动作标签，推理必须先还原到 rad/m，后面的轨迹积分和评估才有意义。

   更细一点看，`predict_pose.py` 的核心数据链是：

```text
main()
  -> 读取 Stage-2 checkpoint
  -> build_tsformer() / load_infinitystar_vae() / Adapter.load_state_dict()
  -> route 列表
  -> infer_one_route()
      -> _load_latents()
      -> _decode_tokens_full_T()
      -> _sample_all_window_starts()
      -> _gather_window_tokens()
      -> TimesFormer
      -> 重叠窗口平均成逐帧 delta
      -> label_stats 反标准化
      -> integrate_trajectory_se3()
      -> 写 pred_actions.json / pred_path.json / trajectory_m_deg.json
```

   关键形状可以先记成：

```text
latents.pt       -> z_ext:        (1,64,T_lat,16,16)
VAE hook+Adapter -> tokens_tnd:   (T_dec,N=480,D=384)
4 帧滑窗         -> patch_tokens: (K*4,N,D)
TimesFormer      -> pred:         (K,18) = (K,3,6)
窗口聚合后       -> deltas:       (T_dec,6)
```

   `K = T_dec - 4 + 1`。每个 4 帧窗口预测 3 个相邻转移，多个窗口覆盖同一帧间动作时取平均；第 0 个 delta 固定为 0，表示“第一帧没有前一帧可相减”。

7. `train/action_decoder/tools/eval_endpoints.py`
   这是 Stage-2 离线评估。它不跑模型，只比较 `pred_actions.json` / `pred_path.json` 和 GT 终点。
   默认优先读 `pred_actions.json`，并从 GT 第 0 帧重新积分一遍预测轨迹；只有没有动作文件时才退回 `pred_path.json`。
   终点是否合格的判定是：
   `qualified = (dist_m <= dist_thr_m) and (ang_deg <= ang_thr_deg)`，
   其中 `ang_deg` 是 SO(3) geodesic 误差，`yaw_err_deg` 只是单独拿出来帮助你定位航向角偏差。

8. `train/shard_worldvln_checkpoint.py`
   用于把单文件 `.pth` 训练 checkpoint 转成可复原的 sharded safetensors。适合理解 checkpoint 内部结构。

9. `train/export_worldvln_hf_repo.py`
   用于把 GPT/VAE 拆成 Hugging Face 友好的 `gpt/`、`vae/` 子目录，并生成 `README.md`、`load_weights.py`、manifest。

### 7.5 `infer/` 目录怎么读

`infer/` 是在线闭环推理入口，建议先看启动脚本，再看服务端，最后看客户端。

1. `infer/run_server.sh`
   看环境变量如何连接服务端配置、runtime 代码、latent cache、Stage-2 动作头 checkpoint。
   已额外注释 `INFINITY_SERVER_CONFIG`、`INFINITY_REPO_ROOT`、`INFINITY_REQUIRE_TGT_HW`、`STAGE2_LATENT2ACTION_CKPT` 等变量。

2. `infer/config.json`
   这是默认服务配置。重点看 `num_frames`、`step`、采样参数、world-model checkpoint 和动作头配置。

3. `infer/server.py`
   这是主入口。第一遍不要从模型加载开始读，先读这些点：
   `PredictDeltaActionsRequest`、`TrajectoryState`、`_obs_points()`、`_get_or_create_traj()`、
   `_predict_delta_actions_impl()`、`_infer_latents_for_actions_and_advance_cache()`、
   `_stage2_predict_16_actions_for_segment_cm_deg()`。

4. `infer/client.py`
   从调用者视角理解服务协议。重点看预热 1 帧、后续每轮上传真实执行后的 `step` 帧、以及 16 个逐帧动作的保存方式。
   建议按这条函数链读：

```text
main()
  -> 读取 --config_json/--num_frames/--step
  -> dataset 分支: _discover_routes() -> run_one_route()
      -> _chunks_stream() 或 _chunks_prefix_from_points()
      -> _http_post_json("/v1/predict_delta_actions")
      -> 保存 session_id_summary.json / segXX_actions.json / segXX_poses.json
  -> unrealcv 分支: _make_unrealcv_env() -> run_one_task_unrealcv()
      -> 嵌套 _post_frames()
      -> _apply_action_to_pose_with_frame()
      -> 采集新 RGB 帧并继续上传
```

   读 JSON 输出时记住三类文件：`summary` 记录本次 session 配置和帧数，
   `actions` 记录服务端顺序 `[dx,dy,dz,droll,dyaw,dpitch]`、客户端兼容顺序和累计动作，
   `poses` 记录动作积分后的绝对位姿点。

5. `infer/config.local_bestrecord.json`
   本地实验配置示例。它不是通用默认值，但适合对照 `config.json` 看哪些字段经常被实验覆盖。

6. `infer/OPEN_SOURCE_CLEANUP.md`
   开源整理记录。读代码时如果遇到路径/权重缺失，可以先看这里判断是否是有意清理。

## 8. 不建议优先深挖的路径

这些路径很重要，但不适合作为第一遍阅读入口：

- `Worldmodel/infinity/models/infinity.py`
- `Worldmodel/runtime/infinity/models/infinity.py`
- `Worldmodel/infinity/models/videovae/`
- `Worldmodel/runtime/infinity/models/videovae/`

它们是底层模型实现，代码量大、抽象多。第一遍只需要知道它们提供两个能力：video VAE 编解码 latent、自回归 Transformer 预测 latent。
等你理解服务协议和动作解码链路后，再按 4.1 节的顺序回头看 `Infinity.forward()`、自回归推理函数、schedule 和 KV-cache，会更省时间。

## 9. 本次已额外标注的重点文件

我已经在这些文件里加入中文导读式注释：

补充说明：本次二次检查按 AST 扫描了 `train/`、`infer/`、`action_aware_grpo/` 和 `Worldmodel/`
重点 Python 路径，对缺少函数/类 docstring 的位置补了函数级中文说明。也就是说，阅读 `train/*.py`、
`train/action_decoder/tools/*.py`、`infer/server.py`、`infer/client.py`、`Worldmodel/action_decoder/src/`
和 `Worldmodel/runtime/tools/` 时，函数入口处应能直接看到
“这个函数负责什么、输入输出是什么、为什么这样做”的短说明。

- `infer/server.py`
  标注在线闭环协议、`TrajectoryState`、segment readiness、session 创建、配置解析、权重加载、first-frame condition、latent 时间线切片、FastAPI schema/endpoint、`summed_codes` 到动作的 Stage-2 路径。
  本轮继续中文化了 core service 中的 Stage-2/actionhead 长度不匹配、latent/frame 切片越界、目标分辨率检查、
  `summed_codes` 形状断言和离线 precomputed latent 评估提示，方便按报错直接定位数据流断点。

- `infer/client.py`
  标注客户端如何按 `1, step, step, ...` 上传真实观测、dataset/UnrealCV 两种模式、route 发现、HTTP 请求、JSON 落盘，以及当前 16 动作输出和旧 4 macro 输出的兼容关系。

- `infer/run_server.sh`
  标注在线服务启动时各环境变量的含义，尤其是配置文件、runtime 根目录、输入分辨率、latent cache 和 Stage-2 checkpoint。

- `action_aware_grpo/runtime/client.py`、`action_aware_grpo/windows_client.py`
  标注 GRPO rollout 客户端如何复用同一套闭环协议，并补充 `service` 模式的 `/reset`、`/step_actions`
  请求处理、session 校验、动作执行和 CLI help。实际读法可以把它们看作 `infer/client.py` 的 GRPO/remote-sim 版本；
  本轮继续中文化了 service HTTP handler、active session/payload 校验、unknown path、访问日志和 simulator 端参数 help。

- `Worldmodel/action_decoder/src/datasets/latent_traj_manifest.py`
  标注 manifest、latent、trajectory、RGB 帧如何组成 Stage A/B 的训练样本。

- `Worldmodel/runtime/tools/infinity_streaming_session.py`
  标注 `t0`、`gt_obs`、Pred cache 的含义，以及为什么每轮要清理预测 cache。

- `Worldmodel/action_decoder/src/models/vae96_to_tsformer_adapter.py`
  标注 Adapter 如何把 VAE decoder feature 翻译成 TimesFormer tokens。

- `train/action_decoder/tools/train_stageA_ddp.py`
  标注 Stage A 的 teacher/student token 对齐目标，并补充 DDP 工具、teacher 加载、VAE hook、collate、decode wrapper 等函数级说明。

- `train/action_decoder/tools/train_stageB_ddp.py`
  标注 latent-to-action 训练链路，并补充 DDP 工具、label stats、Adapter/TimesFormer/VAE 加载、滑窗 token/label 收集、loss、VAE 解冻等函数级说明；
  CLI 参数 help 也已覆盖 manifest、checkpoint、VAE、窗口、loss、冻结/解冻和导出设置，方便边运行边对照源码。

- `train/scripts/train_from_base.sh`
  标注 Stage 1 监督训练入口和默认训练目标。

- `train/train.py`
  标注 Stage 1 组装逻辑、dataset 字段、T5 文本条件、epoch/iteration 循环、梯度累积和世界模型训练目标。

- `train/shard_worldvln_checkpoint.py`
  标注单文件 checkpoint 如何拆成 sharded safetensors，以及如何通过 manifest 复原嵌套结构。

- `train/export_worldvln_hf_repo.py`
  标注 GPT/VAE 分开导出、HF 目录支持文件、alias metadata 和逐 tensor 校验；
  生成的 Hugging Face 模型卡标题/说明、加载提示和 standalone safetensors 报错也已中文化。

- `train/action_decoder/tools/predict_pose.py`
  标注 Stage-2 离线推理的 VAE hook、4 帧滑窗、动作聚合、SE(3) 积分和输出 JSON 格式；
  CLI help、`label_stats` 加载、`run_config.json` 回退、route 跳过提示和 checkpoint 兼容日志也已中文化。

- `train/action_decoder/tools/eval_endpoints.py`
  标注端点评估的 GT/预测读取、相对动作积分、终点距离/姿态/yaw 指标和图表输出。

- `Worldmodel/runtime/tools/GRPO/reward_uavflow.py`
  标注 Action-aware GRPO 的奖励来源和最终组合公式。
  本轮继续中文化了严格模式缺失 `trajectory.json` / `sample_logprob_total` 的报错，方便排查 replay meta 与 rollout 产物是否对齐。

- `Worldmodel/runtime/tools/GRPO/*.py`
  对候选任务构建、候选轨迹/rollout 生成、replay dataset、hybrid replay meta 和结果汇总脚本补了函数级说明，
  便于从 rollout 数据生产链路读到 StageB 训练输入；候选数、噪声、负优势、重复组、输入格式校验等常见 CLI/报错文本也已中文化。

- `action_aware_grpo/grpo_server.py`
  标注 GRPO 专用服务端配置、模型初始化、session/latent cache、segment trace、old_logprob trace_ce、
  ActionHead reference-video 模式、FastAPI schema/endpoint、self-test 和离线 precomputed latent 评估；
  本轮继续中文化了 FastAPI schema 描述、future segment 协议说明、离线评估 help/报错和 JSON note。
  也同步清理了 actionhead reference-video 模式的依赖报错、预测视频/`summed_codes` 形状错误、
  pre-resize schema 说明和 gt_obs 前缀推进注释，便于把普通在线服务与 GRPO 服务端对照阅读。

- `Worldmodel/runtime/tools/run_infinity.py`
  标注文本编码、单样本生成、checkpoint 加载、视频保存和通用参数注册，方便理解服务端复用的 runtime API。

- `Worldmodel/runtime/tools/closed_loop_streaming_infer_480p_81f.py`
  标注 checkpoint 布局识别、闭环 points、参数构造、离线 segment 保存和 watch 模式。

- `Worldmodel/runtime/tools/prompt_rewriter.py`
  标注 prompt rewrite client 池、token 统计、Azure OpenAI 调用封装，以及 `ak` 为空时回退到环境变量 API key 的构造逻辑。

- `Worldmodel/runtime/tools/GRPO/generate_candidate_trajectories_real.py`
  标注真实 simulator 闭环 rollout 的 `reset -> predict -> step_actions` 主链、
  `trajectory.json` 字段、old logprob/trace 落盘和 shard 级失败重试逻辑。
  本轮继续中文化了 `Infinity repo`、`StageB/action head checkpoint`、`actionhead runtime repo`、`remote_sim`
  和 UAV-Flow 任务 json 相关 CLI help，保留关键模式名和路径术语，便于按命令行参数理解 rollout 数据流。

- `Worldmodel/runtime/train.py`
  标注 runtime 训练入口、GRPO batch 字段、hybrid stepping、checkpoint 恢复和 `trainer.train_step()` 的职责边界。

- `Worldmodel/action_decoder/src/build_model.py`
  标注 TimesFormer/VisionTransformer 动作头构建、预训练/恢复路径和 patch embedding 输入输出。

- `Worldmodel/action_decoder/src/datasets/latent_traj_manifest.py`
  进一步中文化 manifest、latent、trajectory、frame transform、I/O timeout 和 `on_error` 相关校验信息；
  读 Stage A/B 数据集时可直接对照 `(1,64,T_lat,16,16)` latent、`(T,6)` delta label 和 RGB teacher 帧的流向。

- `Worldmodel/action_decoder/actionhead_runtime/models/vae96_to_tsformer_adapter.py`
  与 `src/models/vae96_to_tsformer_adapter.py` 同步补充 Adapter 中文说明，避免读 runtime copy 时缺上下文。

- `Worldmodel/infinity/models/infinity.py` 和 `Worldmodel/runtime/infinity/models/infinity.py`
  标注训练 `forward()`、自回归推理、KV-cache 导入导出、CFG、RoPE cache、文本/视觉 token 前向传播和 checkpoint 兼容加载。

- `Worldmodel/infinity/trainer/sft_trainer.py` 和 `Worldmodel/runtime/infinity/trainer/sft_trainer.py`
  补充/中文化 VAE token cache、TSFormer 初始化/解码/导出、GRPO trace replay、trace CE、PPO/teacher-forcing 和 FSDP resume 相关说明。

- `Worldmodel/infinity/utils/save_and_load.py`、`Worldmodel/runtime/infinity/utils/save_and_load.py`
  中文化 checkpoint 保存、自动恢复、断点合并、resume 信息和缺失路径提示，方便看训练/服务恢复链路。

- `Worldmodel/infinity/models/videovae/**`、`Worldmodel/runtime/infinity/models/videovae/**` 和 `Worldmodel/runtime/infinity/utils/load.py`
  补充/中文化 VAE 初始化、EMA 权重加载、missing/unexpected 权重提示、并行 size 警告、模型摘要、显存统计和不支持 VAE 类型的错误说明；
  本轮还补齐了 LPIPS 下载/加载日志、discriminator/optimizer-state 日志、WAN VAE block 类型校验、可学习参数打印、BSQ/LFQ 维度与 spherical 校验、entropy loss 不支持提示和 quantizer 实用注释。
  继续清理了 `NanDetector` 的 NaN/Inf 梯度诊断、VideoVAE `misc.py` 的显存/参数/张量结构打印、`bytedance.ndtimeline` 版本报错、
  context parallel 初始化断言/警告，以及 WAN VAE loader 的 EMA 权重提示。
  还补齐了通用 VAE 因果卷积 padding/broadcast 注释和 attention 中保持 `q/k` 参与计算图的 dummy-op 注释。
  多尺度 quantizer 继续补齐了 `multiscale_bsq.py`、`multiscale_bsq_tp.py`、`multiscale_fsq_tp.py`
  及 runtime 镜像中的日程生成、由粗到细残差量化、语义/细节双塔、随机翻转/丢层和索引重建说明。

- `Worldmodel/infinity/utils/{dist,sequence_parallel,save_and_load}.py`、`Worldmodel/runtime/infinity/utils/{dist,sequence_parallel,save_and_load}.py`
  中文化分布式初始化 fallback、sequence/context parallel 警告、checkpoint 合并耗时、profile trace 输出和 MFU/TFLOPs 诊断提示。

- `Worldmodel/infinity/schedules/*.py` 和 `Worldmodel/runtime/infinity/schedules/*.py`
  标注动态分辨率、`scale_schedule`、`scale_pack_info`、video encode/decode、attention 可见性、RoPE 生成、旧版 schedule 兼容逻辑；
  本轮还补齐了 `InfinityStar-Interact` 未知 mode 报错和 WAN causal padding/broadcast 注释。

- `Worldmodel/infinity/models/videovae/utils/diffdist/testing.py` 和 runtime 副本
  中文化 diffdist 通信测试的 reduce_scatter/all_gather/scatter/gather/broadcast/send-recv 输出，方便调分布式通信时看懂每个 rank 的收发和梯度检查。

- `Worldmodel/infinity/models/videovae/utils/diffdist/{extra_collectives,functional,functions,modules}.py` 和 runtime 副本
  补充可求导分布式通信包装的中文说明，把 `send/recv/gather/scatter/all_gather/reduce_scatter`
  的 forward 通信和 backward 梯度回传关系写清楚；这些文件适合在调 context parallel 或自定义通信梯度时再读。

- `Worldmodel/action_decoder/src/timesformer/**`
  对 TimeSformer 底层配置、数据变换、模型构建、ViT/ResNet/Nonlocal block、feature hook、checkpoint helper、benchmark/parser、SSV2/Kinetics loader、AVA 评估工具等补充了函数/类级中文 docstring 和实用日志说明；
  本轮还补齐了 decoder 失败提示、mAP/AVA 评估日志、multigrid 日程日志、AVA numpy box/mask/metrics 校验报错、
  图像 crop/normalize 断言、pathway 数量断言、SyncBatchNorm 空输入、feature hook 类型、模型摘要/GPU 显存日志和 AVA label/id 校验提示。
  这些文件多数是底层依赖，不建议第一遍逐行读；当你需要理解动作头 backbone 或旧 checkpoint 兼容逻辑时再进入。

- `Worldmodel/action_decoder/actionhead_runtime/timesformer/**`
  这是动作头 runtime 使用的另一份 TimeSformer 副本，也已同步补充中文 docstring/注释、checkpoint helper、benchmark/parser 和数据集 loader 说明。
  本轮同步清理了与 `src/timesformer/**` 对应的 transform、head/stem、batchnorm、features、misc、distributed、multigrid 和 AVA evaluation 残留英文提示。
  读法与 `src/timesformer/**` 相同：第一遍不必深挖，只有调试 runtime 动作头、旧 checkpoint 或 backbone 差异时再进入。

## 10. 快速定位清单

想看“服务端怎么连续推进同一个 session”：读 `infer/server.py` 的 `_predict_delta_actions_impl()`。

想看“客户端怎么按闭环协议上传帧并执行动作”：读 `infer/client.py` 的 `run_one_task_unrealcv()`。

想看“真实观测怎么覆盖预测状态”：读 `infinity_streaming_session.py` 的 `compute_kv_cache_gt()` 和 `correction_clear_pred()`。

想看“latent 怎么变成动作”：读 `infer/server.py` 的 `_stage2_predict_16_actions_for_segment_cm_deg()`。

想看“Adapter 为什么存在”：读 `vae96_to_tsformer_adapter.py` 和 `train_stageA_ddp.py`。

想看“动作头怎么训练”：读 `train_stageB_ddp.py`。

想看“监督训练怎么启动”：读 `train/scripts/train_from_base.sh` 和 `train/TRAINING.md`。

想看“Stage-1 每个 batch 里到底传了什么”：读 `train/train.py` 的 `train_one_epoch()`。
想看“Stage-1 真正的 loss/反传为什么不在 train.py 里”：继续顺着 `train_one_epoch() -> trainer.train_step()` 往下读。

想看“训练 checkpoint 怎么变成可发布权重”：读 `train/shard_worldvln_checkpoint.py` 和 `train/export_worldvln_hf_repo.py`。

想看“Stage-2 动作头离线怎么跑完整 route”：读 `train/action_decoder/tools/predict_pose.py`。
如果你第一次读就迷路，先只看 `infer_one_route()` 的 3 段：
`latents -> tokens_tnd -> window_deltas -> deltas -> trajectory`。

想看“Stage-2 端点评估怎么算”：读 `train/action_decoder/tools/eval_endpoints.py`。
先抓住两点：默认更信任 `pred_actions.json`，以及最终只看最后一帧的 `dist_m/ang_deg/yaw_err_deg`。

想看“GRPO 奖励怎么算”：读 `reward_uavflow.py` 中 clip 模式的 `grpo_reward` 组合。
想看“GRPO rollout 元数据最开始怎么组织”：先读 `build_rollout_tasks.py`，再读 `generate_candidate_rollouts.py`。

想看“GRPO 服务端怎么把 rollout trace/logprob 存下来”：读 `action_aware_grpo/grpo_server.py` 的
`_infer_summed_codes_for_step()`、`_infer_latents_for_actions_and_advance_cache()`、`_predict_delta_actions_impl()`。

想看“runtime 工具怎么加载 Infinity checkpoint 并生成 summed_codes”：读 `Worldmodel/runtime/tools/run_infinity.py` 的
`load_transformer()` 和 `gen_one_example()`。

想看“离线闭环 streaming 怎么复现服务端行为”：读 `Worldmodel/runtime/tools/closed_loop_streaming_infer_480p_81f.py`。

## 11. 第一遍阅读建议

第一遍不要逐行读完整仓库，按下面顺序做笔记即可：

1. 在 `infer/server.py` 里画出 request -> `TrajectoryState` -> `summed_codes` -> actions 的调用链。
2. 在 `infinity_streaming_session.py` 里标出 `t0`、`gt_obs`、Pred cache 三种状态。
3. 在 `vae96_to_tsformer_adapter.py` 里写下输入输出 shape。
4. 在 `latent_traj_manifest.py` 里确认 manifest 的三类路径和返回字段。
5. 在 `train_stageA_ddp.py` 里区分 teacher token 和 student token。
6. 在 `train_stageB_ddp.py` 里区分 label layout、label normalization 和 TimesFormer 输出窗口。
7. 在 `reward_uavflow.py` 里看 clip 展开、组内相对奖励和最终 reward 组合。

第二遍再进入 `Worldmodel/runtime/infinity/` 的底层模型。否则容易被大模型实现细节淹没，看不到 WorldVLN 的闭环协议主线。

## 12. 对着注释到底怎么读

如果你是第一次系统看这类项目，建议不要一上来盯着“模型有多少层”。更有效的方法是：每读一个函数，都回答下面 6 个问题。

1. 这个函数的输入从哪里来。
   先看调用它的上一个函数，再看参数名里的 shape、单位、是否是绝对帧号还是局部帧号。

2. 这个函数的输出要交给谁。
   例如 `summed_codes` 会流向 Stage-2 动作头，`actions` 会流向客户端执行，`gt_obs cache` 会流回下一轮世界模型推理。

3. 这个函数有没有改变“状态”。
   比如 `TrajectoryState`、KV cache、`last_latent_1`、磁盘里的 latent cache 文件，这些都不是简单返回值，而是后续流程会继续读取的状态。

4. 这个函数是在“真实观测路径”还是“模型预测路径”上。
   `compute_kv_cache_gt()` 属于真实观测路径；`infer_chunk()` 属于预测路径；`correction_clear_pred()` 属于预测后清理路径。

5. 这个函数处理的是哪一条时间线。
   是 RGB 帧时间线、latent 时间线，还是动作时间线。很多看不懂，本质都是把这三条时间线混了。

6. 这个函数里最关键的公式是什么。
   只要看到 `//4 + 1`、`*100`、`*180/pi`、滑窗平均、SE(3) 积分，就要停下来确认单位和含义。

推荐你边读边记一个最小表格：

| 函数 | 输入 | 输出 | 状态副作用 | 最关键公式 |
| --- | --- | --- | --- | --- |
| `_predict_delta_actions_impl()` | HTTP 请求、真实新帧 | 16 个动作 | 更新 `TrajectoryState` | segment readiness |
| `_infer_latents_for_actions_and_advance_cache()` | 真实观测前缀 | `summed_codes`、`latent5_input` | 更新 `gt_obs` cache、`last_latent_1` | `latent_index(f)=(f-1)//4+1` |
| `_stage2_predict_16_actions_for_segment_cm_deg()` | `summed_codes` | 16 个 cm/deg 动作 | 无 | 滑窗平均、rad->deg、m->cm |
| `integrate_trajectory_se3()` | 相对动作 | 绝对轨迹 | 无 | `p+=R@t_rel`, `R=R@R_rel` |

## 13. 单次请求端到端跟读路线

如果你想对着代码真正走一遍“1 帧输入如何变成 16 个动作”，按下面顺序读，不要跳进所有子函数：

| 顺序 | 当前位置 | 读什么 | 重点确认 |
| --- | --- | --- | --- |
| 1 | `infer/client.py:1197` `main()` | 客户端命令行入口 | 当前是 `dataset`、`unrealcv` 还是其他模式 |
| 2 | `infer/client.py:962` `run_one_route()` 或 `infer/client.py:542` `run_one_task_unrealcv()` | 准备首帧、prompt、session_id | 哪些帧是真实观测，哪些动作会被真正执行 |
| 3 | `infer/client.py:642` `_post_frames()` | 把图片转成 base64 并发 HTTP | 请求体里的 `images_base64/num_frames/step/prefix_mode` |
| 4 | `infer/server.py:2437` `_predict_delta_actions_impl()` | 服务端主状态机 | 新帧如何进入 `TrajectoryState`，segment 是否 ready |
| 5 | `infer/server.py:1975` `_infer_latents_for_actions_and_advance_cache()` | 调世界模型并推进 latent/cache | `latent_index(f)=(f-1)//4+1`，以及 `gt_obs` cache 何时更新 |
| 6 | `infer/server.py:1583` `_infer_summed_codes_for_step()` | 真正生成 `summed_codes` | 世界模型输出的是 latent transition，不是最终动作 |
| 7 | `infer/server.py:1095` `_stage2_predict_16_actions_for_segment_cm_deg()` | Stage-2 latent-to-action | `summed_codes -> VAE hook feature -> Adapter -> TimesFormer -> cm/deg actions` |
| 8 | `infer/client.py:191` `_apply_action_to_pose_with_frame()` | 客户端执行相对动作 | body/world 坐标系、yaw 更新顺序、位姿积分 |

这条链路的核心闭环是：

```text
真实 RGB 帧
  -> server 解码并追加到 TrajectoryState
  -> Infinity 预测 summed_codes
  -> Stage-2 动作头输出 16 个动作
  -> client 执行动作并采集新真实帧
  -> 下一次请求把真实新帧发回 server 纠偏
```

这里最重要的判断点是第 4 步和第 5 步：

1. 第 4 步决定“现在是否可以发射一个 segment 的动作”。
2. 第 5 步决定“真实观测前缀在 latent 时间线上对应到哪里”。
3. 如果你只看第 7 步，会误以为动作头独立工作；实际上它吃的是前面世界模型产生的 `summed_codes`。

## 14. 最容易看混的 10 个点

1. `num_frames=49` 不代表会输出 49 个动作。
   相邻帧之间才有动作，所以通常是 48 个 delta；闭环默认每段输出 16 个动作。

2. `summed_codes` 不是最终视频。
   它首先是世界模型预测出的 latent 世界转移；只有调试或参考视频路径时才会被解成可视化视频。

3. `vae.decode()` 在动作头里不等于“为了出 RGB”。
   这里真正要的是 decoder 中间层 feature，RGB 只是副产品，很多时候根本不会使用。

4. `gt_obs` 和 Pred cache 不能混。
   `gt_obs` 是真实观测；Pred cache 是临时预测结果。闭环稳定性的关键就是每轮结束后清掉 Pred。

5. `clip_abs_start/clip_abs_end` 和 tensor 切片下标不是一套坐标。
   代码里的绝对帧号通常是 1-based；Python tensor 切片是 0-based，而且右边界是开区间。

6. `latent5_input` 不是“5 帧 RGB”，而是“5 个 latent 时间步”。
   对默认压缩率 4 来说，5 个 latent 时间步通常对应 17 帧左右的视觉跨度。

7. 动作布局经常变。
   API 常用 `[dx,dy,dz,droll,dyaw,dpitch]`；训练内部常见 `[dz,dy,dx,tx,ty,tz]`。读代码时必须先认布局，再看单位。

8. “Stage 1 / Stage A / Stage B / GRPO” 不是一条线上的同一步。
   Stage 1 训练世界模型；Stage A 对齐表征；Stage B 做 latent-to-action；GRPO 在此基础上继续做策略优化。

9. `actionhead_ref_vit` 和 `tsformer_latent` 不是同一路径。
   前者偏参考视频动作头；后者是当前默认的 Stage-2 latent-to-action 主路径。

10. `Worldmodel/` 和 `Worldmodel/runtime/` 有镜像副本。
   不是所有阅读都该从其中一个开始。服务端实际 import 哪份，取决于 runtime 路径和 `sys.path` 选择。

## 15. 这一轮建议你直接跟读的代码段

如果你现在已经读过 `infer/server.py` 和 `infinity_streaming_session.py`，下一步最值得跟读的是下面三段：

1. `train/action_decoder/tools/train_stageB_ddp.py`
   重点看：
   `main()`、`traj_to_delta()`、`_decode_tokens_full_T()`、`_gather_window_tokens()`、`_gather_window_targets()`、训练循环里的 `patch_tokens.grad -> grad_tokens -> tokens_btnd.backward(...)`。

   你要带着这几个问题看：
   - GT 轨迹是怎样从绝对位姿变成逐步 delta 的？
   - 为什么 4 帧窗口最终监督 18 维输出？
   - 为什么这里不是直接对整个 TimesFormer 一次 forward/backward，而是先对窗口 chunk 反传，再把梯度 scatter 回整段 token？
   - 为什么 `tsformer` 和 `adapter` 没直接包进 DDP，而要手动 `_allreduce_grads()`？

2. `infer/client.py`
   重点看：
   `run_one_route()`、`run_one_task_unrealcv()`、`_apply_action_to_pose_with_frame()`、`_obs_points()`。

   你要带着这几个问题看：
   - dataset 模式和 unrealcv 模式的数据流差在哪？
   - 客户端什么时候只是“回放真实帧”，什么时候是在“真的执行动作”？
   - `action_frame=world/body` 会怎样改变位姿积分？
   - 为什么默认闭环节奏是 `1, step, step, ...`，而不是一次上传全部 49/81 帧？

3. `Worldmodel/runtime/infinity/models/infinity.py`
   重点看：
   `forward()`、`prepare_text_conditions()`、`ar_infer_infinity_elegant()`、`set_cache_write_is_pred()`、`clear_pred_cache()`、`export_kv_cache()`、`import_kv_cache()`。

   你要带着这几个问题看：
   - 训练 `forward()` 和推理 `ar_infer_infinity_elegant()` 的根本差异是什么？
   - 文本条件为什么要先写成 `t0` cache？
   - 为什么推理是“先文本、再视觉尺度、再逐尺度采样 bit label、再累积成 summed_codes”？
   - CFG/APG 是在哪一步进入流程的？

把这三段连起来，你会看到一条完整主线：

```text
Stage-1 Infinity 预测 summed_codes
        ->
Stage-B 动作头把 summed_codes 解成 16 个动作
        ->
infer/client 执行动作并拿回真实新帧
        ->
server/streaming session 用真实帧纠偏下一轮
```

如果你只想先搞懂一个最关键的“梯度和数据怎么流”的例子，就选 `train_stageB_ddp.py`：

```text
z_ext
  -> VAE decoder hook
  -> tokens_btnd
  -> 按窗口 gather patch_tokens
  -> TimesFormer 预测窗口动作
  -> patch_tokens.grad
  -> scatter 回 grad_tokens
  -> tokens_btnd.backward(...)
  -> Adapter / VAE 收到梯度
```

这段看懂后，你再回头看在线推理链路，会明显轻松很多。
