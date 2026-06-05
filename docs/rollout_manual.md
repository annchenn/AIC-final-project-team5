# Rollout Manual — Moving Cube Grasp (GlowsAI L40S)

## 目的

1. 視覺確認訓練結果（cube 是否出現、arm 是否追蹤、是否成功放入籃子）
2. 量化 eval，量測 proposal 定義的三個指標
3. 找出訓練可優化的地方

## Proposal 評估指標

| 指標 | 定義 | 目標 |
|---|---|---|
| **Capture rate** | 有把 cube 抬過 0.12m 門檻的 episodes / 總 episodes | ≥ 60% |
| **Placement rate** | 成功放入籃子 / 有 capture 的 episodes | ≥ 80% |
| **End-to-end rate** | 成功放入籃子 / 總 episodes | — |

三個速度等級：

| Level | 速度 | 說明 |
|---|---|---|
| Slow | 0.08 m/s | OOD（低於訓練範圍 0.10–0.18） |
| Medium | 0.12 m/s | in-distribution |
| Fast | 0.18 m/s | in-distribution 上限 |

---

## Step 1：進 repo，啟動 container

```bash
nvidia-smi              # 確認 GPU 正常（L40S）
cd ~/aicapstone         # 必須在 repo 根目錄（有 Makefile）
make launch-isaaclab-glowsai-l40s
```

成功後 terminal 會進入 container shell（prompt 改變）。**之後所有指令都在 container 裡跑。**

---

## Step 2：下載 checkpoint

Repo：`yun0523/aic-finalproject-team5`（只有 `main` branch，不需要 `--revision`）

HuggingFace 上的結構：
```
checkpoints/
├── 020000/pretrained_model/
├── 040000/pretrained_model/
├── 060000/pretrained_model/
├── 080000/pretrained_model/
└── 100000/pretrained_model/   ← 最新
```

**下載指定 step（以 100000 為例）：**

```bash
huggingface-cli download yun0523/aic-finalproject-team5 \
    --include "checkpoints/100000/*" \
    --local-dir /workspace/aicapstone
```

下載後路徑：`/workspace/aicapstone/checkpoints/100000/pretrained_model/`

**換成其他 step（例如 080000）：**

```bash
huggingface-cli download yun0523/aic-finalproject-team5 \
    --include "checkpoints/080000/*" \
    --local-dir /workspace/aicapstone
```

---

## Step 3：單一 episode 視覺觀察

先不跑量化，看 sim 畫面確認基本行為。按 `R` 可手動 reset，Ctrl+C 結束。

```bash
cd /workspace/aicapstone
export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/torch/lib:$LD_LIBRARY_PATH

python scripts/rollout.py \
    --task HCIS-MovingCubeGrasp-SingleArm-v0 \
    --policy_type lerobot-diffusion \
    --policy_checkpoint_path checkpoints/100000/pretrained_model \
    --policy_action_horizon 1 \
    --eval_rounds 0 \
    --episode_length_s 30 \
    --device cuda \
    --enable_cameras
```

**換 checkpoint：** 只改 `--policy_checkpoint_path`，例如：
```
--policy_checkpoint_path checkpoints/080000/pretrained_model
```

**重點觀察：**

| 現象 | 代表什麼 |
|---|---|
| Cube 沒出現，只有籃子 | cube init_state 問題（已修） |
| Arm 完全不動 | Policy load 失敗或 observation 接錯 |
| Arm 往固定位置移動，不追 cube | Policy 沒學到動態追蹤 |
| Arm 追了 cube 但抓不住 | Grasp timing 或 Z offset 問題 |
| 抓住但放不進籃子 | 放置 phase 問題 |
| Terminal 印 `captured=True` 但沒 success | 抓到了但放置失敗 |
| 全部成功 → 進 Step 4 | ✅ |

---

## Step 4：測試不同 action_horizon

`--policy_action_horizon` 影響對移動目標的反應速度。

```bash
# horizon=1（最快反應）
python scripts/rollout.py \
    --task HCIS-MovingCubeGrasp-SingleArm-v0 \
    --policy_type lerobot-diffusion \
    --policy_checkpoint_path checkpoints/100000/pretrained_model \
    --policy_action_horizon 1 \
    --eval_rounds 0 --episode_length_s 30 \
    --device cuda --enable_cameras

# horizon=4
# （同上，改 --policy_action_horizon 4）

# horizon=8
# （同上，改 --policy_action_horizon 8）
```

選出視覺上最流暢且能追到 cube 的值，帶到 Step 5。

---

## Step 5：量化 eval（單速度，多 rounds）

用上一步選出的 `<HORIZON>`：

```bash
# Medium speed（in-distribution，先確認基本成功率）
python scripts/rollout.py \
    --task HCIS-MovingCubeGrasp-SingleArm-v0 \
    --policy_type lerobot-diffusion \
    --policy_checkpoint_path checkpoints/100000/pretrained_model \
    --policy_action_horizon <HORIZON> \
    --eval_rounds 10 \
    --episode_length_s 30 \
    --cube_speed 0.12 \
    --device cuda --enable_cameras
```

**換 checkpoint：** 改 `--policy_checkpoint_path`
**換速度：** 改 `--cube_speed`（0.08 / 0.12 / 0.18）

Terminal 會在每個 episode 後印：
```
[Evaluation] running  capture=3/5  placement=2/3 (0.667)  end-to-end=2/5
```

最後印三行 Final 結果：
```
[Evaluation] Final capture rate:    0.700  [7/10]
[Evaluation] Final placement rate:  0.857  [6/7]  (placed | captured)
[Evaluation] Final end-to-end rate: 0.600  [6/10]
```

---

## Step 6：三速度 sweep（對照 proposal 目標）

```bash
python eval/moving_cube_grasp_eval.py \
    --checkpoint checkpoints/100000/pretrained_model \
    --rounds 10 \
    --action_horizon <HORIZON>
```

預設跑 3 個速度：Slow 0.08 / Medium 0.12 / Fast 0.18。

**換 checkpoint：** 改 `--checkpoint`
```bash
python eval/moving_cube_grasp_eval.py \
    --checkpoint checkpoints/080000/pretrained_model \
    --rounds 10 \
    --action_horizon <HORIZON>
```

輸出：
- `eval/results/speed_<X>.txt` — 每個速度的完整 log
- `eval/speed_vs_success.png` — 三條線（capture / placement / end-to-end）vs 速度，含 proposal 目標虛線

---

## Step 7：診斷 observation 問題（如有異常）

```bash
python scripts/rollout.py \
    --task HCIS-MovingCubeGrasp-SingleArm-v0 \
    --policy_type lerobot-diffusion \
    --policy_checkpoint_path checkpoints/100000/pretrained_model \
    --policy_action_horizon 1 \
    --eval_rounds 0 --episode_length_s 30 \
    --device cuda --enable_cameras \
    --debug_policy_shapes
```

確認 `observation.images.wrist` 和 `observation.images.front` 的 shape 與訓練時一致。

---

## 根據結果的優化方向

| 觀察到的問題 | 可能原因 | 優化方法 |
|---|---|---|
| Capture rate 低（arm 不追 cube） | `n_obs_steps` 太小，policy 沒看到連續幀 | 重訓時加大 `--policy.n_obs_steps`（建議 3–4） |
| Capture rate 高但 placement rate 低 | 放置 phase demo 品質或數量問題 | 確認 datagen phase 4–6 成功率；增加 demo 數 |
| Slow（0.08）成功但 Fast（0.18）失敗 | 高速下 policy 反應不及 | 降低 action horizon；或重訓加更多高速 episode |
| 全速段 capture rate 低 | Demo 數量不足 | 加 `--resume` 重跑 datagen 累積更多 episode |

---

## 快速換 checkpoint 對照

| 想測什麼 | 改動 |
|---|---|
| 換到 step 100000 | `--policy_checkpoint_path checkpoints/100000/pretrained_model` |
| 換到 step 080000 | `--policy_checkpoint_path checkpoints/080000/pretrained_model` |
| 換到 step 060000 | `--policy_checkpoint_path checkpoints/060000/pretrained_model` |
| 比較多個 step | 對每個 step 各跑一次 Step 6，比較圖表 |
