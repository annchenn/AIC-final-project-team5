# AICapstone
This repository relies mainly on IsaacSim, IsaacLab to perform simulations tasks
The documentation should aim for users without any technical background

## Course Project: AI Capstone Spring 2026
**Deadline: 2026/06/07 (Sun) 23:59** — no late submissions accepted.

### Tiers
- **Entry (80%, mandatory group):** Complete one of three pick-and-place tasks using the provided pipeline.
- **Advanced (+20%, optional group):** Define new tasks in simulation; must complete Entry first.
- **Independent Study (+30%, optional individual):** HCIS Lab research collaboration.

### Assigned Tasks (pick one)
1. Cup Stacking (Kitchen) — pick blue cup, stack on pink cup
2. Cutlery Arrangement (Dining Room) — knife right, fork left of plate
3. Toy Block Collection (Living Room) — collect 3 blocks into basket

### Pipeline
1. Real-World Data Collection with UMI (ED305 classroom)
2. Reconstruction in Isaac Sim (object poses only, not raw trajectories)
3. Robot Motion Generation (FSM planner or keyboard teleoperation)
4. Policy Training (LeRobot ecosystem; default: Diffusion Policy)
5. Inference & Evaluation (leaderboard + local rollout)

**Only modifiable at Entry level:** training data and policy model (architecture + hyperparams).

### Submission (to Google Drive)
- `Team{TEAM_ID}_presentation.mp4` (≤15 min, English)
- `Team{TEAM_ID}_project_entry_report.pdf`
- Checkpoint folder, README.txt
- Advanced only: advanced report, configs folder, custom `.usd` assets

## Advanced Level Proposal — Group 5: "Catch Moving Mouse"

**Task:** Robotic arm must intercept a continuously moving block (simulating a mouse) and deposit it into a basket. Extends Entry-level Toy Block Collection by introducing dynamic target unpredictability.

**Four core capabilities:**
1. Perception — infer target motion from RGB cameras (not ground-truth state) during rollout
2. Motion Planning — adaptive interception to reach target before it escapes
3. Grasping — stable grasp under motion uncertainty
4. Task Completion — full intercept → grasp → transport → release pipeline

**Dataset:** Auto-generated in Isaac Sim via FSM expert policy (has access to privileged target position/velocity). Moving block speed, direction, and starting position are randomized per episode. Only successful episodes stored.

**Simulation Environment:**
- Platform: NVIDIA Isaac Sim
- Moving block with randomized initial position, speed, direction
- All USD assets independently created/downloaded — no Entry-level USD files reused

**Methods:**
- Policy: Diffusion Policy trained via imitation learning on FSM demonstrations
- Observation space: RGB images (wrist + front cameras) + robot joint states
- Action space: robot end-effector trajectory (LeRobot format)
- Key hyperparameter: shorter action horizon for faster reaction to moving targets

**Evaluation Targets:**
- Capture success rate: ≥ 60%
- Basket placement rate (after capture): ≥ 80%
- Generalization: tested across 3 speed levels

**Key Challenges & Mitigations:**
- FSM demo quality → start with slow/simple motions, increase difficulty gradually
- Slow policy reaction → tune action/prediction horizon
- Block escapes before arm arrives → FSM moves toward predicted interception point, not current position
- Poor generalization → randomize speed, direction, start pos, path pattern
- Sim-to-real gap → domain randomization on block mass, speed, friction

### Key Links
- LeRobot: https://huggingface.co/lerobot
- Isaac Sim: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/index.html
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/

# "uv" as the main package manager
- uv run python to spawn python shell

# Workspace design
Workspaces organize large codebases by splitting them into multiple packages with common dependencies. Think: a FastAPI-based web application, alongside a series of libraries that are versioned and maintained as separate Python packages, all in the same Git repository.

In a workspace, each package defines its own pyproject.toml, but the workspace shares a single lockfile, ensuring that the workspace operates with a consistent set of dependencies.

As such, uv lock operates on the entire workspace at once, while uv run and uv sync operate on the workspace root by default, though both accept a --package argument, allowing you to run a command in a particular workspace member from any workspace directory.

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
