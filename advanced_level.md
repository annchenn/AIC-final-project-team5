# Advanced Level — Moving Cube Grasp

本文件記錄我們 AI Capstone Final Project **Advanced Level** 的任務設計、實作細節與目前進度，避免後續重複詢問。

> Entry Level：`HCIS-ToyBlocksCollection-SingleArm-v0`（已完成）
> Advanced Level：`HCIS-MovingCubeGrasp-SingleArm-v0`（本文件主題）

---

## 1. 任務定義

| 項目 | 說明 |
|---|---|
| Task ID | `HCIS-MovingCubeGrasp-SingleArm-v0` |
| Scene | 沿用 Entry 的 living-room（`LIVING_ROOM_CFG`） |
| 物件 | 單一藍色立方體，邊長 5 cm，質量 0.05 kg，低摩擦（dynamic friction 0.2） |
| 機械臂 | Franka Panda（與 Entry 相同模板） |
| 動態行為 | Cube 在桌面上以**等速直線運動**滑動，速度方向 / 大小隨機 |
| 成功條件 | Cube 被舉起 ≥ 0.12 m（`cube_lifted` termination） |
| Episode length | 20 秒（比 Entry 任務長，需追擊時間） |



---

## 2. 程式檔案結構

| 檔案 | 角色 |
|---|---|
| [packages/simulator/src/simulator/tasks/moving_cup_grasp/moving_cube_grasp_env_cfg.py](packages/simulator/src/simulator/tasks/moving_cup_grasp/moving_cube_grasp_env_cfg.py) | Env / Scene config：cube 定義、success 條件、speed range 等 knobs |
| [packages/simulator/src/simulator/tasks/moving_cup_grasp/__init__.py](packages/simulator/src/simulator/tasks/moving_cup_grasp/__init__.py) | 註冊 gym task id |
| [packages/simulator/src/simulator/datagen/state_machine/moving_cube_grasp.py](packages/simulator/src/simulator/datagen/state_machine/moving_cube_grasp.py) | FSM planner：注入 cube 速度 + 追擊/抓取/抬起 |
| [packages/simulator/src/simulator/tasks/__init__.py](packages/simulator/src/simulator/tasks/__init__.py) | 透過 `from . import moving_cup_grasp` 把任務註冊進去 |
| [scripts/datagen/generate.py](scripts/datagen/generate.py) | 在 `TASK_REGISTRY` 中對應 `MovingCubeGraspStateMachine` |
| [data/moving_cube_demo/object_poses.json](data/moving_cube_demo/object_poses.json) | 8 個合成 episodes 的 cube 初始位置（純 synthetic，不經 UMI） |

---

## 3. 實作細節

### 3.1 場景與物件（`moving_cube_grasp_env_cfg.py`）

```python
class MovingCubeGraspSceneCfg(SingleArmFrankaTaskSceneCfg):
    scene = LIVING_ROOM_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")
    cube = RigidObjectCfg(
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            mass_props=MassPropertiesCfg(mass=0.05),
            physics_material=RigidBodyMaterialCfg(
                static_friction=0.3, dynamic_friction=0.2, restitution=0.0),
            visual_material=PreviewSurfaceCfg(diffuse_color=(0.1, 0.3, 0.9)),
        ),
    )
```

額外 knobs（供 FSM 讀取，方便後續做 evaluation sweep）：

```python
cube_linear_speed_range: tuple[float, float] = (0.05, 0.10)  # m/s
cube_lift_threshold: float = 0.12                            # m
```

Success termination：

```python
def cube_lifted(env, cube_cfg, lift_height):
    cube_pos = env.scene[cube_cfg.name].data.root_pos_w - env.scene.env_origins
    return cube_pos[:, 2] > (OBJECT_Z + lift_height)
```

### 3.2 動態行為：等速直線運動

由 `MovingCubeGraspStateMachine` 在每個 tick 注入：

- **Episode 開始**抽一次速度：
  - 大小：`uniform(0.05, 0.10)` m/s
  - 方向角：`uniform(π/4, 3π/4)`（偏 +y，避免滑出工作區）
  - `vz = 0`
- **`pre_step`** 每 tick 呼叫 `cube.write_root_velocity_to_sim(vel)`，把同一組 `(vx, vy, 0)` 重新寫回，**抵消摩擦力 → 完全等速**。
- **Phase 2（合夾）之後停止注入**，cube 改由 physics 接管（抓住跟著手走 / 沒抓到才會被摩擦力減速）。

### 3.3 FSM 四階段（60 Hz）

繼承自 `ToyBlocksCollectionStateMachine`，改成單物件 4 phases：

| Phase | 步數 | 行為 |
|---|---|---|
| 0. hover | 120 | 從 rest 位置平滑插值到 cube 正上方 |
| 1. approach | 140 | 下降到抓取高度，用「剩餘步數 × cube 速度」做 **predictive lead** |
| 2. grasp | 20 | 合夾 |
| 3. lift | 80 | 直上抬起 |

FSM 是**特權專家**（privileged expert）：直接讀 `cube.data.root_pos_w`（ground-truth 座標）+ 自己注入的速度，**不看任何鏡頭**，所以能用簡單算式做完美預判。

---

## 4. 鏡頭與 Observation

繼承自 [single_arm_franka_cfg.py](packages/simulator/src/simulator/tasks/template/single_arm_franka_cfg.py)，共 **2 顆 RGB camera**：

| Camera | 位置 | 解析度 |
|---|---|---|
| `wrist` | 掛在 `panda_hand`（手腕，前向 4 cm） | 640×480 |
| `front` | 場景固定鏡頭，桌前上方俯瞰 | 640×480 |

Policy observation 內容：
- `wrist` RGB + `front` RGB
- `joint_pos` / `joint_vel` / `joint_pos_rel` / `joint_vel_rel`
- `last_action` / `joint_pos_target`

> **FSM 不看鏡頭**；**Policy 不看 cube ground-truth**。兩者完全互不重疊，這是 imitation learning 的 teacher-student 設計。

---

## 5. 訓練 Pipeline

跟 Entry 一模一樣，只換 task id 和 dataset 名稱。

### 5.1 Step 1 — Data Generation

FSM 自動跑、只存成功 episode（`EXPORT_SUCCEEDED_ONLY`）。**每次 episode FSM 重抽速度**，所以同 8 個初始位置可累積多樣化資料。

```bash
export HF_USER=ann0000000
export DATASET_NAME=aic-finalproject-moving-cube-v1
export HF_LEROBOT_HOME=/workspace/aicapstone/datasets/lerobot_cache

# 第一輪：不要加 --resume
python scripts/datagen/generate.py \
    --task HCIS-MovingCubeGrasp-SingleArm-v0 \
    --num_envs 1 --device cuda --enable_cameras \
    --record --use_lerobot_recorder \
    --lerobot_dataset_repo_id ${HF_USER}/${DATASET_NAME} \
    --object_poses data/moving_cube_demo/object_poses.json

# 後續輪：加 --resume 累積資料
for i in $(seq 1 5); do
  python scripts/datagen/generate.py \
    --task HCIS-MovingCubeGrasp-SingleArm-v0 \
    --num_envs 1 --device cuda --enable_cameras \
    --record --resume --use_lerobot_recorder \
    --lerobot_dataset_repo_id ${HF_USER}/${DATASET_NAME} \
    --object_poses data/moving_cube_demo/object_poses.json
done
```

### 5.2 Step 2 — Policy Training（Diffusion Policy）

```bash
export HF_USER=ann0000000
export DATASET_NAME=aic-finalproject-moving-cube-v1
export RUN_NAME=moving_cube_run1

lerobot-train \
  --dataset.repo_id=${HF_USER}/${DATASET_NAME} \
  --dataset.root=./datasets/lerobot_cache/${HF_USER}/${DATASET_NAME} \
  --policy.type=diffusion \
  --output_dir=./checkpoints/${RUN_NAME} \
  --job_name=${RUN_NAME} \
  --policy.device=cuda \
  --wandb.enable=true
```

**關鍵超參**（動態任務建議調整）：
- `--policy.n_obs_steps`：≥ 2，讓 model 從連續幀隱式估速度
- `--policy.horizon` / `--policy.n_action_steps`：放大讓 policy 有預測能力
- `--batch_size` / `--steps` / `--policy.optimizer_lr`

### 5.3 Step 3 — Rollout & Evaluation

```bash
python scripts/rollout.py \
  --task HCIS-MovingCubeGrasp-SingleArm-v0 \
  --enable_cameras \
  --policy_type lerobot-diffusion \
  --policy_checkpoint_path ./checkpoints/moving_cube_run1/<ckpt> \
  --eval_rounds 20
```

建議寫 `eval/moving_cube_grasp_eval.py`，對不同 `cube_linear_speed_range`（0.05 / 0.08 / 0.10 / 0.12 m/s）各跑 N 次，畫**速度 vs. 成功率**曲線。

---

## 6. Policy 如何學會追動態物件

1. **Datagen**：FSM 用 ground-truth 算 perfect action（含預判），同時把兩顆鏡頭畫面 + joint state 一起錄進 LeRobot dataset。
2. **Training**：Diffusion policy 學 `(wrist RGB, front RGB, joint state) → action chunk` 的映射，完全不碰 cube ground-truth。
3. **隱式估速度**：靠 `n_obs_steps` 堆疊多幀，網路從畫面中 cube 的位移自己推出運動方向。
4. **Inference**：FSM 整個丟掉，只剩 policy + 兩顆鏡頭。

---

## 7. 目前進度

### 已完成
- [x] Task 註冊（`HCIS-MovingCubeGrasp-SingleArm-v0`）
- [x] Env / Scene config（`moving_cube_grasp_env_cfg.py`）
- [x] FSM planner（`moving_cube_grasp.py`），含速度注入 + predictive lead
- [x] Success termination（`cube_lifted`）
- [x] 合成 `object_poses.json`（8 個初始位置）
- [x] 鏡頭設置（沿用 template 的 wrist + front）

### 待處理
- [ ] 新增 `cmd/datagen_moving_cube.sh` / `cmd/train_moving_cube.sh` 專用 script
- [ ] 跑 datagen 累積足量 successful episodes（目標 ≥ 100）
- [ ] 訓練 Diffusion Policy checkpoint
- [ ] 撰寫 `eval/moving_cube_grasp_eval.py`，做速度 sweep 評估
- [ ] 整理 Advanced Report（含 motivation、evaluation procedure、results）
- [ ] 匯出 `configurations folder`（`docs/standalone_env_config_export.md`）
- [ ] 撰寫 `README.txt` 重現步驟（Advanced 必繳）

---

## 8. 與 Entry 的差異（可直接寫進 Report）

| 面向 | Entry (Toy Blocks Collection) | Advanced (Moving Cube Grasp) |
|---|---|---|
| 物件數 | 3 個積木 + 1 籃子 | 1 個方塊 |
| 物件狀態 | 靜態 | **等速直線運動** |
| FSM | 多物件 pick-and-place | 單物件追擊 + predictive lead |
| Termination | 放入籃子 | 舉起 ≥ 12 cm |
| Episode 長度 | 預設 15 s | **20 s** |
| 資料來源 | UMI 真實採集 + sim | **純合成 object_poses**（UMI 無法蒐集動態任務） |
| Policy 挑戰 | 定位 + 抓取 | **隱式估速度 + 預測** |
| 新增 knobs | — | `cube_linear_speed_range`、`cube_lift_threshold` |

---

## 9. 後續可擴展方向

若時間允許可探討（也適合寫進 Report 的 Discussion）：

- **變速運動**：每 N tick 重抽速度，或加週期性加速度
- **曲線運動**：加入 `wz` 角速度，或用 `sin` 調制 vx/vy
- **碰撞反彈**：不每 tick 覆蓋速度，讓桌邊 / 障礙物自然影響軌跡
- **多速度範圍訓練**：訓練時 randomize speed range，評估 generalization
- **觀測消融實驗**：拿掉 wrist 或 front 任一顆鏡頭，比較成功率，驗證鏡頭重要性
