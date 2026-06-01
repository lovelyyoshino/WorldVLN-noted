# WorldVLN 阅读路线图（小白版）

> 这份文档是 **「我刚 clone 完代码，第一次打开仓库，应该按什么顺序读」** 的入门导览。
> 想要细节、行号、公式索引，请配合 [CODE_READING_GUIDE.zh-CN.md](./CODE_READING_GUIDE.zh-CN.md) 一起看。

---

## 0. 三句话讲清楚 WorldVLN 是什么

1. **任务**：无人机/机器人按自然语言指令在 3D 环境里飞行/行走，输出 6 自由度增量动作。
2. **方法**：先用一个**自回归世界模型**（InfinityStar）预测「世界下一段会怎么变」（latent），再用一个**动作解码器**（VAE→Adapter→TimesFormer）把 latent 翻译成 16 个相邻帧的 6D 动作。
3. **闭环**：客户端把动作执行后看到的真实新帧再发回来，覆盖上一轮预测，避免"靠想象越走越偏"。

> 如果一句话总结："**预测一段未来 latent，从中读出动作，再用真实观测纠偏**"。

---

## 1. 阅读顺序（强烈建议照这个走）

按"从外向内、从协议向模型"的顺序，三天上手：

### 🌱 Day 1：先搞懂"系统在做什么"

| 顺序 | 文件 | 重点 |
|---|---|---|
| 1 | [README.md](../README.md) | 入口、依赖、命令；30 分钟 |
| 2 | [infer/client.py](../infer/client.py) | 客户端如何发请求；先看模块 docstring 和 `main()` |
| 3 | [infer/server.py](../infer/server.py) | 服务端的 `predict_delta_actions()` → `_predict_delta_actions_impl()` |
| 4 | [docs/CODE_READING_GUIDE.zh-CN.md §2](./CODE_READING_GUIDE.zh-CN.md) | 「总体架构」+「张量/单位速查」 |

读完 Day 1 你应该能回答：
- 一次 HTTP 请求里客户端发了什么？服务端返回了什么？单位是什么？
- `session_id` 为什么必须保持？为什么第一次只发 1 帧、后面发 16 帧？

### 🌿 Day 2：搞懂世界模型在做什么

| 顺序 | 文件 | 重点函数 |
|---|---|---|
| 5 | [Worldmodel/runtime/tools/infinity_streaming_session.py](../Worldmodel/runtime/tools/infinity_streaming_session.py) | `reset()` → `compute_kv_cache_gt()` → `infer_chunk()` → `correction_clear_pred()` 四步机 |
| 6 | [infer/server.py](../infer/server.py) 的 `_infer_latents_for_actions_and_advance_cache()` 与 `_infer_summed_codes_for_step()` | 服务端如何调度 streaming session |
| 7 | [Worldmodel/action_decoder/src/models/vae96_to_tsformer_adapter.py](../Worldmodel/action_decoder/src/models/vae96_to_tsformer_adapter.py) | latent 特征 → TimesFormer token 的桥；理解形状 `(B,96,T,H,W) → (B*T,480,384)` |
| 8 | [infer/server.py](../infer/server.py) 的 `_stage2_predict_16_actions_for_segment_cm_deg()` | 怎么把 latent 解码成 16 个 cm/deg 动作 |

读完 Day 2 你应该能回答：
- 三类 cache（`t0` / `gt_obs` / `Pred`）各自存什么？为什么必须分开？
- "为什么不直接把 RGB 喂给 TimesFormer"？答：因为 latent 已经包含了"世界会怎么变"的信息，省一次 VAE encode。

### 🌳 Day 3：搞懂训练在做什么

| 顺序 | 文件 | 重点 |
|---|---|---|
| 9 | [train/action_decoder/tools/train_stageA_ddp.py](../train/action_decoder/tools/train_stageA_ddp.py) | Adapter **蒸馏**：让 latent 走出来的 token 对齐 RGB 走出来的 token |
| 10 | [train/action_decoder/tools/train_stageB_ddp.py](../train/action_decoder/tools/train_stageB_ddp.py) | latent → action 监督训练；重点看 1037 行附近"班长收作业"那段比喻 |
| 11 | [train/train.py](../train/train.py) | Stage 1：自回归世界模型本身的监督训练 |
| 12 | [Worldmodel/runtime/tools/GRPO/reward_uavflow.py](../Worldmodel/runtime/tools/GRPO/reward_uavflow.py) | Stage 2 GRPO：用奖励函数微调 |

读完 Day 3 你应该能区分：
- **Stage A（蒸馏）** vs **Stage B（监督）** vs **Stage 1（世界模型）** vs **Stage 2（GRPO）**——一共有 4 套训练，别混了。

---

## 2. 整体调用链全景图（可打印贴墙）

```text
                        ┌────────────────────────────────────────┐
        客户端          │           服务端 infer/server.py         │
   ─────────────        │                                        │
   client.py            │  predict_delta_actions  (FastAPI 入口)  │
   run_one_route()      │            │                           │
   _post_frames()  ───► │   _predict_delta_actions_impl()         │
                        │            │                           │
                        │   _ensure_traj_infinity_session()       │
                        │   _infer_latents_for_actions_and_       │
                        │     advance_cache()                     │
                        │            │                           │
                        │            ▼                           │
                        │   ┌──────────────────────────────┐     │
                        │   │  InfinityStreamingSession     │     │
                        │   │  • reset()  写 t0 (text)     │     │
                        │   │  • compute_kv_cache_gt()     │     │
                        │   │     写 gt_obs (real frames)  │     │
                        │   │  • infer_chunk()             │     │
                        │   │     输出 summed_codes (latent)│     │
                        │   │  • correction_clear_pred()   │     │
                        │   └──────────────┬───────────────┘     │
                        │                  │ summed_codes        │
                        │                  ▼                     │
                        │   ┌──────────────────────────────┐     │
                        │   │  Stage-2 动作解码器           │     │
                        │   │  summed_codes (16 ch)        │     │
                        │   │   → z_ext  (64 ch via 2x2)   │     │
                        │   │   → VAE decoder hook         │     │
                        │   │       中间特征 (B,96,T,H,W)   │     │
                        │   │   → Vae96→TSformer Adapter   │     │
                        │   │       tokens (B*T,480,384)   │     │
                        │   │   → TimesFormer 4 帧滑窗     │     │
                        │   │   → 6D delta * 3 帧 = 18 维  │     │
                        │   │   → 累加 + 单位换算           │     │
                        │   │       cm/deg 16 个动作       │     │
                        │   └──────────────┬───────────────┘     │
                        │                  │                     │
                  ◄─────│   actions, segment_index, done         │
   _apply_action_       │                                        │
   to_pose_with_frame() └────────────────────────────────────────┘
   写 *_actions.json
   *_poses.json
```

---

## 3. 重点公式速查（按"出现频率"排序）

读这些函数时务必盯着公式：

### 3.1 单位换算（最容易踩的坑）

```
公式：训练用 m/rad，API 输出用 cm/deg
   meters → cm:    *100
   radians → deg:  *180/π
```
出现位置：
- `infer/server.py::_stage2_predict_16_actions_for_segment_cm_deg()`
- `train/action_decoder/tools/predict_pose.py::infer_one_route()`

### 3.2 4 帧滑窗聚合

```
公式：delta[t] = ( Σ_{窗口w 覆盖 t} pred_w[t] ) / count[t]
约束：delta[0] = 0
```
窗口长度 = 4，每个窗口预测 3 个相邻帧的 delta；时间点 t 会被多个窗口覆盖，所以最后取平均。

### 3.3 机体系动作积分（client 把 delta 应用到位姿）

```
公式（单步）：
   theta = yaw_at_application_time
   x_world += dx_body·cos(theta) − dy_body·sin(theta)
   y_world += dx_body·sin(theta) + dy_body·cos(theta)
   z_world += dz
   yaw     += dyaw    (注意环形 [-180, 180])
```
出现位置：`infer/client.py::_apply_action_to_pose_with_frame()`
顺序参数 `yaw_first / trans_first / midpoint` 决定 `theta` 取**应用前 yaw**、**应用后 yaw**、还是**中点 yaw**。

### 3.4 SE(3) 轨迹积分

```
公式：
   R_next = R_cur · R_delta            (旋转用矩阵乘)
   p_next = p_cur + R_cur · t_rel      (平移在当前姿态系下)
```
出现位置：`train/action_decoder/tools/predict_pose.py::integrate_trajectory_se3()`

### 3.5 Stage B 标签归一化

```
公式：
   x_norm_angle = (x_angle - mean_angles) / std_angles    # rad
   x_norm_trans = (x_trans - mean_t)      / std_t         # m
反归一化方向相反。
```
出现位置：`train/action_decoder/tools/train_stageB_ddp.py::_normalize_delta_bt6()`

### 3.6 端点旋转误差

```
公式：geodesic_deg = arccos( (trace(R_gt^T · R_pred) - 1) / 2 ) · 180/π
yaw 误差：(dy + 180) mod 360 - 180
```
出现位置：`train/action_decoder/tools/eval_endpoints.py`

### 3.7 GRPO 奖励组合

```
公式：
   action_mse = α_xyz·MSE(xyz) + α_yaw·MSE(yaw) + α_all6·MSE(all6)
   z-score:   z = (r - mean) / std
   LOO 优势:  adv_i = r_i - mean(r_j ; j ≠ i)
   task_reward = w_dense·dense + w_success·success_bonus
```
出现位置：`Worldmodel/runtime/tools/GRPO/reward_uavflow.py`

### 3.8 候选 rollout 种子

```
公式：seed = seed_base + task_idx · task_seed_stride + cand_idx · candidate_seed_stride
```
出现位置：`Worldmodel/runtime/tools/GRPO/generate_candidate_rollouts.py::_candidate_seed()`

---

## 4. 容易混淆的"双胞胎"目录

WorldVLN 仓库里有几对**同名但用途不同**的目录，很容易看花眼：

| 双胞胎 | 训练副本 | 运行时副本 |
|---|---|---|
| `infinity/` | `Worldmodel/infinity/` — 训练、checkpoint 导出 | `Worldmodel/runtime/infinity/` — 在线服务、GRPO |
| `action_decoder/` | `Worldmodel/action_decoder/src/` — 训练用 | `Worldmodel/action_decoder/actionhead_runtime/` — 推理用 |
| `train.py` | `train/train.py` — 顶层启动入口 | `Worldmodel/runtime/train.py` — runtime 镜像 |

**判断你当前在哪边的方法：** 看文件最上面的 `sys.path.insert(...)` 和 `from xxx import yyy`，import 的根决定了"这一份代码属于哪个副本"。**两边碰到同名文件不要假设完全一致。**

---

## 5. 张量形状速查表（看代码时贴在旁边）

| 名字 | 形状 | 含义 |
|---|---|---|
| `images_base64` | `List[str]` | 客户端上传的真实 RGB 帧 |
| `frames_cpu` | `List[(3,H,W)]`, 范围 `[-1, 1]` | 服务端缓存的真实观测 |
| `points` | `[1, 17, 33, 49, ...]` | segment 边界（默认 step=16） |
| `summed_codes` | `(1, C, T_lat, H, W)` | 世界模型预测的 latent 转移 |
| `z_ext` | `(1, 64, T_lat, 16, 16)` | Stage-2 动作头消费的 patchified latent |
| VAE up_block 中间特征 | `(B, 96, T, H, W)` | Adapter 的输入（不是最终 RGB） |
| Adapter token | `(B*T, 480, 384)` | TimesFormer 可消费的 patch tokens（480 = 24×20，384 = embed_dim） |
| Server action | `[dx, dy, dz, droll, dyaw, dpitch]` | API 输出，单位 cm/deg |
| Training delta | `[dz, dy, dx, tx, ty, tz]` | Stage B 内部布局，角度 rad、平移 m |

**关键关系：**
- 帧数 vs latent 数：`T_lat = (T_frames - 1) // 4 + 1`（VAE 时间压缩 4×）
- 帧数 vs 动作数：49 帧 → 48 个相邻动作（动作描述相邻帧之间的转移）
- 一段动作数：`step = 16` 时一轮闭环输出 16 个

---

## 6. 训练 4 个阶段对照表

很多人第一次看会被"Stage 1/2 vs Stage A/B"绕晕。**它们是两个独立维度。**

```text
┌─────────────────────────────────────────────────────────────┐
│  Stage 1：世界模型监督训练                                   │
│  入口：train/train.py                                        │
│  目标：教 InfinityStar 在 teacher-forcing 下预测下一段 latent │
│                                                              │
│      ↓ 产出 InfinityStar checkpoint                         │
│                                                              │
│  Stage 2 = (Stage A + Stage B) + GRPO                       │
│                                                              │
│  ├─ Stage A：Adapter 蒸馏                                    │
│  │  入口：train/action_decoder/tools/train_stageA_ddp.py     │
│  │  目标：让 latent 走出的 token ≈ RGB 走出的 token           │
│  │     （teacher = frozen TSformer 看真实 RGB                │
│  │      student = VAE hook + Adapter）                       │
│  │  loss = w_cos·L_cos + w_mse·L_mse + w_mean + w_std        │
│  │                                                           │
│  ├─ Stage B：latent → 动作 监督                              │
│  │  入口：train/action_decoder/tools/train_stageB_ddp.py     │
│  │  目标：把 latent token 翻译成专家轨迹的相邻动作 delta      │
│  │  loss = w_rot·MSE_rot + w_xy·MSE_xy + w_z·MSE_z           │
│  │                                                           │
│  └─ Stage 2 GRPO：组内相对优势优化                            │
│     入口：action_aware_grpo/grpo_server.py +                 │
│           Worldmodel/runtime/tools/GRPO/                     │
│     目标：rollout 多个候选 → 计算奖励 → 用 GRPO loss 微调     │
└─────────────────────────────────────────────────────────────┘
```

**人话版："世界模型只管想象未来，动作头负责把未来翻译成动作，蒸馏让动作头听得懂世界模型。"**

---

## 7. 调试遇到问题时按这个顺序找

### 7.1 在线推理出错？

| 现象 | 先去哪查 |
|---|---|
| HTTP 报错 / segment_index = -1 不出动作 | `infer/server.py::_predict_delta_actions_impl()`：`points()`、`ready_default` 判断 |
| 服务端 OOM | `_stage2_predict_16_actions_for_segment_cm_deg()` 的 `windows_chunk` 调小 |
| 动作单位不对（×100 或 ×180/π 漂移） | `_stage2_predict_16_actions_for_segment_cm_deg()` 的最后单位换算段 |
| 动作越走越偏 | `correction_clear_pred()` 是否被调用？看 streaming session 的 `is_pred` 标记 |
| 客户端轨迹画出来不对 | `infer/client.py::_apply_action_to_pose_with_frame()` 的 `yaw_first / trans_first / midpoint` 选择 |

### 7.2 训练 loss 不收敛？

| 现象 | 先去哪查 |
|---|---|
| Stage A loss 不降 | `compute_distill_loss()` 的 4 个权重；teacher 是否真的 frozen |
| Stage B loss NaN | `_normalize_delta_bt6()` 的 mean/std 是否加载；是否用了正确的 `label_stats_json` |
| Stage B 多卡梯度漂移 | 看 `train_stageB_ddp.py:1037` 注释，确认 `_allreduce_grads()` 在 step 前被调用 |
| GRPO 优势全 0 / 全负 | `reward_uavflow.py::_zscore_exp_reward()` 和 `_loo_adv()`；候选 K 是否 ≥ 2 |

### 7.3 闭环 rollout 异常？

| 现象 | 先去哪查 |
|---|---|
| simulator 不响应 | `action_aware_grpo/grpo_server.py::_service_reset()/_service_step_actions()` |
| candidate 全相同 | `_candidate_seed()` 的 `task_seed_stride` 和 `candidate_seed_stride` |

---

## 8. "我只想看一个东西" 极简索引

| 我想看…… | 直接去 |
|---|---|
| HTTP 请求格式 | `infer/server.py` 顶部 `PredictDeltaActionsRequest/Response` |
| 一段 16 个动作怎么算出来的 | `infer/server.py::_stage2_predict_16_actions_for_segment_cm_deg()` |
| 世界模型 KV cache 怎么管 | `Worldmodel/runtime/tools/infinity_streaming_session.py` |
| latent 怎么变成 token | `Worldmodel/action_decoder/src/models/vae96_to_tsformer_adapter.py` |
| TimesFormer 的 divided space-time attention | `Worldmodel/action_decoder/src/timesformer/models/vit.py` |
| 动作怎么积分成位姿 | `train/action_decoder/tools/predict_pose.py::integrate_trajectory_se3()` |
| 端点误差怎么算 | `train/action_decoder/tools/eval_endpoints.py` |
| GRPO 奖励 | `Worldmodel/runtime/tools/GRPO/reward_uavflow.py::_compose_task_reward_raw()` |

---

## 9. 想更细？

- **完整版导读：** [docs/CODE_READING_GUIDE.zh-CN.md](./CODE_READING_GUIDE.zh-CN.md)（1600+ 行，含函数级行号索引）
- **训练命令：** [train/TRAINING.md](../train/TRAINING.md)
- **GRPO 远端模拟器：** [action_aware_grpo/docs/remote_sim.md](../action_aware_grpo/docs/remote_sim.md)
- **Stage-2 推理协议细节：** README.md §"自回归 I/O 约定"

---

## 10. 一图流总结

```
┌─────────────────────────────────────────────────────────────┐
│                     WorldVLN 一图流                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  自然语言指令 + 真实观测前缀                                   │
│           │                                                  │
│           ▼                                                  │
│  ┌───────────────────────────────┐                           │
│  │   InfinityStar 世界模型        │  ← Stage 1 训练            │
│  │   (KV cache: t0 / gt_obs)     │                           │
│  └───────────────┬───────────────┘                           │
│                  │ summed_codes (latent)                     │
│                  ▼                                           │
│  ┌───────────────────────────────┐                           │
│  │   VAE decoder hook            │                           │
│  │   ↓                           │                           │
│  │   Adapter (96→384)            │  ← Stage A 蒸馏            │
│  │   ↓                           │                           │
│  │   TimesFormer 4-frame window  │  ← Stage B 监督            │
│  │   ↓                           │  ← Stage 2 GRPO            │
│  │   16 × [dx,dy,dz,droll,dyaw,  │                           │
│  │           dpitch]  cm/deg     │                           │
│  └───────────────┬───────────────┘                           │
│                  │ 动作下发                                   │
│                  ▼                                           │
│        客户端执行 → 采集真实新帧 → 回传服务端 → 闭环             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

读完这份导览，再回去看 [CODE_READING_GUIDE.zh-CN.md](./CODE_READING_GUIDE.zh-CN.md) 就不会迷路了。
