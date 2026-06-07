哈哈哈哈
哈哈哈哈

### 如何 run 一遍 FSM的測試：


要先 `git switch 0531-moving-mouse` 進入到進階勞贖的branch

使用 `make launch isaac-lab` 進入isaac sim 的 container後執行
```
python scripts/datagen/generate.py \
    --task HCIS-MovingCubeGrasp-SingleArm-v0 \
    --num_envs 1 --device cuda --enable_cameras \
    --object_poses data/moving_cube_demo/object_poses.json
```

### 0603開train by 黃
- 沒辦法選0.49 NVIDIA GeForce RTX 4090 pending還沒到就會release
- 不要選0.55那個+snapshot要1.95/h
- L40s一直遇到virtualGL安裝不完整的問題 看不到isaac(但是可以跑 有影片&方塊位置可debug 不知道是不是用windowsVNC的關係)
- 有改object_poses.json(變成80筆資料)
    - 他的邏輯是如果第一輪在八個位置都成功 第二輪不會跑已經成功的資料所以不能只設八個不然永遠只有八筆資料
- 進度：完成datagen 可下載 if有人要用可以用(77筆成功資料)

    ```
    export HF_LEROBOT_HOME=/home/glows/AIC-final-project-team5/datasets
    hf download yun0523/aic-finalproject-moving-cube-v1 --repo-type dataset --local-dir $HF_LEROBOT_HOME/yun0523/aic-finalproject-moving-cube-v1
    ```
- 覺得我一直在當智障好耗credits...

### 0604真的開train by 黃
- 今天用0.84/h的等pending等了一個小時...
- 進度：train完一發 未改任何參數(就是用script跑)
    ```
    hf download yun0523/aic-finalproject-team5 \
        --repo-type model \
        --local-dir checkpoints/moving_cube_run1
    ```
- 新增eval/moving_cube_grasp_eval.py跑不同速度的eval
    ```
    python eval/moving_cube_grasp_eval.py \
        --checkpoint checkpoints/moving_cube_run1/checkpoints/100000/pretrained_model \
        --rounds 10
    ```
- 如果跑不出來可以先測試
    ```
    export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/torch/lib:$LD_LIBRARY_PATH

    python scripts/rollout.py \
        --task HCIS-MovingCubeGrasp-SingleArm-v0 \
        --policy_type lerobot-diffusion \
        --policy_checkpoint_path checkpoints/moving_cube_run1/checkpoints/100000/pretrained_model \
        --eval_rounds 3 \
        --episode_length_s 30 \
        --cube_speed 0.10 \
        --device cuda --enable_cameras
    ```
- 測試了一下都不會成功 想debug但我還是開不起來isaacGUI環境我明天用linux試試看
- 可能問題
    - 一開始的object_poses生成位置都太近資料不夠?(太隨機生成又很容易掉下去)
    - 跑eval的時候 可能要把方塊掉下去的狀況剔除或是方塊一掉下去那一個測試就重來
    - 可以考慮加資料
    - 不過我看train的曲線loss超低的 感覺也可以用其他checkpoint跑跑(目前是用最大的)
    - ![image](https://hackmd.io/_uploads/SJu2aPybGe.png)
- train+eval一次大概6個credits 感覺improve個三四次就下班(希望)

## 0605用黃的 checkpoints 跑 rollout - Ethel
下載 checkpoint 指令 (看要載哪個 checkpoint 再改數字)
![image](https://hackmd.io/_uploads/H10Wy4gZze.png)

```bash=
# 只下載 100000 這個 step（最省時間，16GB 全下太慢）
huggingface-cli download yun0523/aic-finalproject-team5 \
    --include "checkpoints/100000/*" \
    --local-dir /workspace/aicapstone
```
make 進去 container 之後會到 root 是正常得，然後要先載下面的
```bash=
cd /workspace/aicapstone
pip install -e packages/simulator
```
run 
```bash=
cd /workspace/aicapstone
export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/torch/lib:$LD_LIBRARY_PATH

python scripts/rollout.py \
    --task HCIS-MovingCubeGrasp-SingleArm-v0 \
    --policy_type lerobot-diffusion \
    --policy_checkpoint_path checkpoints/060000/pretrained_model \
    --policy_action_horizon 1 \
    --eval_rounds 100 \
    --episode_length_s 20 \
    --device cuda \
    --enable_cameras

```
### Rollout script
- 初始時間、速度的邏輯沿用 FSM 
- 初始位置目前先從 `object_poses.json`拿，期望是訓練資料會更容易正確，但為了更 generalized 後續可能會改
### Rollout 觀察
- 爪子夾起來的時機會在很靠近桌面的時候，可以更早一些夾，目前有時會差一點點就能夾起來
- 機器人不管cube怎麼跑，似乎都有往前伸的傾向，好像是因為FSM實際出來的資料更多是cube往前跑，可能可以新增更多dataset
```
FSM 的方向分布不是全方向隨機。


bearing（cube → workspace center）:
  範圍：48.5° ~ 143.4°（80筆資料全在這裡）
  平均：91.8° ≈ 純 +y 方向（遠離機械臂）

  +45° ~ +90°:  38筆 ████████████████████
  +90° ~ +135°: 37筆 ████████████████████
 +135° ~ +180°:  5筆 ████
所有訓練 episode 的 cube 都往 +y 方向（遠離機械臂） 滑，bearing 幾乎都接近 90°。

加上 ±60° jitter 後理論範圍是 -11° ~ 203°，但 實際上幾乎每個 episode 都是往前偏斜方向（有 +y 分量）。所以 policy 學到的策略就是：往前追。

你看到的現象就是這個原因：

Policy 訓練時 cube 幾乎都往 +y（前方）滑
Policy 學到「不管怎樣先往前移動」這個強烈 prior
當 cube 剛好往斜向或 x 方向走，arm 還是往前衝 → 錯過
這是訓練資料本身的分布問題，不是 bug。

有幾個方向可以改善：

收窄 jitter 到 ±30°，讓 rollout 時方向更集中在 policy 擅長的範圍
重新 datagen 時故意加入更多斜向 episode（改 jitter 範圍 or 隨機起始方向）
接受這個限制，只統計 cube 往 +y 走的 episode 的成功率
```
- 目前沒有重試的設定，夾一次失敗就放棄然後跑到盒子上面抖抖抖、、
# Daily Report — 2026-06-05

Branch: `0605-moving-mouse-rollout-test`  
Task: `HCIS-MovingCubeGrasp-SingleArm-v0`  
Checkpoint: `checkpoints/100000/pretrained_model`

---

## 問題發現 & 根本原因分析

### 1. Cube 在 rollout 時不會移動
**根本原因：** `ManagerBasedRLEnv.step()` 不存在 `_pre_physics_step` 這個 hook，導致原本的速度注入邏輯（覆寫 `_pre_physics_step`）永遠不會被呼叫。

**確認方式：** 閱讀 IsaacLab 的 `ManagerBasedRLEnv.step()` 原始碼，確認 step loop 中沒有任何地方呼叫 `_pre_physics_step`。

---

### 2. Cube 每次都往同一個方向跑
**根本原因：** `init_state` 把 cube 固定 spawn 在 `(0.35, -0.35)` = workspace 正中心。bearing 計算是 `atan2(cy - y0, cx - x0)`，當 cube 在中心時 `atan2(0, 0) = 0`（永遠朝 +x 方向），jitter ±60° 只覆蓋右半圓。

**確認方式：** 計算 workspace center `(0.35, -0.35)` 與 cube spawn point 完全相同，bearing 退化為 0。

---

### 3. Success termination 誤觸發（arm 抓著方塊懸在籃子上方就算成功）
**根本原因：** `cube_in_basket` 的 `z_range=(-0.05, 0.20)` 範圍太寬，arm 夾著 cube 停在籃子上方約 10~15 cm 也會滿足條件。

**確認方式：** 對照籃子實際幾何（牆高 `_BASKET_HEIGHT=0.08m`），cube 真正放入時 dz ≈ 0.035m，0.20m 上界遠大於此。

---

### 4. Cube 掉出桌子後 episode 不會結束
**根本原因：** `eval_rounds=0` 時 rollout.py 會把 `time_out` termination 關掉，而原本只有 `success` 和 `time_out` 兩個條件，沒有「cube 掉落」的 termination。

---

### 5. Rollout 方向分布與訓練分布不符
**根本原因：** 從 `object_poses.json` 的 80 個訓練起始位置分析：
- bearing（cube → workspace center）範圍：**48.5° ~ 143.4°**，平均 **91.8°**（幾乎純 +y 方向）
- 原始 FSM jitter ±60° 讓 rollout 有效方向達 -11.5° ~ 203.4°，包含訓練時從未出現的 x-dominant 方向
- Policy 學到「往前（+y）追」的強烈 prior，遇到斜向 cube 就錯過

---

## 改動內容

### Commit 1 — `c3ad617`
**fix: use MovingCubeGraspEnv as rollout entry_point**
- `moving_cup_grasp/__init__.py`：gym registration 從 `ManagerBasedRLEnv` 改為 `MovingCubeGraspEnv`

---

### Commit 2 — `851946f`
**feat: add MovingCubeGraspEnv with FSM-style velocity injection and JSON spawn positions**

新增 `MovingCubeGraspEnv(ManagerBasedRLEnv)` subclass，覆寫 `step()` 而非不存在的 `_pre_physics_step`：

| 功能 | 實作 |
|---|---|
| 速度注入 | `step()` 前呼叫 `_inject_cube_velocity()`，每步寫入目標 vx/vy，保留 vz 和 angular velocity |
| 20 步 settle delay | 對應 FSM `pre_step` 的前 20 步等待邏輯，讓 cube 先落桌再給速度 |
| Rejection sampling | 完整移植 FSM：bearing ± jitter，50 次嘗試，workspace bound 檢查 |
| JSON spawn 位置 | 從 `data/moving_cube_demo/object_poses.json` 載入 80 個訓練位置，每次 reset 隨機抽一個（fallback: uniform random） |

---

### Commit 3 — `d3dd3d2`
**fix: tighten success z_range and add cube_fallen termination**
- `z_range` 從 `(-0.05, 0.20)` 收緊為 `(-0.01, 0.10)`，只有 cube 真正在籃子內才算 success
- 新增 `cube_fallen` termination：cube z < -0.1m 時立即 reset

---

### Commit 4 — `58800ff`
**fix: narrow cube velocity jitter from ±60° to ±30°**
- 根據訓練 bearing 分布（48.5°~143.4°），±60° 會產生訓練未見過的 x-dominant 方向
- 收窄至 ±30° 讓有效範圍縮到 18°~173°，全部保有 +y 分量

---

### Commit 5 — `e70c8d9`
**fix: narrow cube jitter further to ±15°**
- ±30° 仍有明顯 x 分量造成 policy 追蹤困難，再收窄至 ±15°（π/12）
- 有效方向範圍：33°~158°

---

### Commit 6 — `82b8379`
**test: set cube jitter to 0 to verify policy tracks dominant training direction**
- 暫時移除所有 jitter，cube 永遠直線朝 workspace center（+y dominant）移動
- 目的：驗證 policy 失敗是方向問題還是訓練量不足

---

## 待驗證

- [ ] `jitter=0` 時 arm 能否成功追蹤並夾取 cube
  - 若可以 → 方向是主因，可保持 0 或小 jitter 進行量化評估
  - 若不行 → 訓練量不足（80 episodes / 100k steps 對動態追蹤任務可能不夠），需重新 datagen + 訓練

---

## 訓練 vs Rollout 配置確認

| 參數 | 訓練時（datagen 0604） | 現在 Rollout |
|---|---|---|
| cube 速度 | `(0.10, 0.18)` m/s | `(0.10, 0.18)` m/s ✓ |
| workspace x | `(0.05, 0.65)` | `(0.05, 0.65)` ✓ |
| workspace y | `(-0.65, -0.05)` | `(-0.65, -0.05)` ✓ |
| cube 起始位置 | `object_poses.json` 80 個位置 | 同 JSON（新增） ✓ |
| jitter | ±60°（FSM 原始） | 0°（測試中） |
| step_hz | 60 Hz | 60 Hz ✓ |


# 0607 hsu

- 修 `rollout.py`：不能把所有 `reset_terminated` 都當 success，`cube_fallen` 也會觸發 termination。現在只把 `termination_manager.get_term("success")` 算成功，其它 termination 算失敗。
- 觀察：畫面上 `captured=False` 但印 successful，是上述統計 bug，不代表真的成功。
- FSM/datagen：cube 在 phase 2 合夾後會停止注入速度，之後靠摩擦慢下來；rollout env 目前是 cube 被抬高前持續注入速度。
- 目前現象：爪子到 cube 上方時 cube 常變慢，可能是接觸/推擠造成，不一定是 env 主動讓它停。
- `jitter=0` rollout 結果（修 success bug 前跑的，placement/end-to-end 會被 `cube_fallen` 灌水）：
  - Final capture rate: 0.100 [1/10]
  - Final placement rate: 10.000 [10/1]
  - Final end-to-end rate: 1.000 [10/10]
  - 先只參考 capture rate：10 次只有 1 次真的抬起 cube。

