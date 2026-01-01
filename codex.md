# Codex Conversation Log

保存时间：当前线程结束前  
范围说明：以下内容保存了本线程中用户与助手的主要可见对话。未包含隐藏的 system/developer 指令、工具原始输出、图片二进制内容，以及没有形成单独答复的内部中间推理。

## Turn 1

### User

read CLAUDE.md and the repo to get a fully understanding to the project. Then write a script the play the lastest trained model

### Assistant

The repo is an Isaac Lab external extension centered on `Template-Salmon-Swim-Direct-v0`: PPO trains a salmon articulation with analytic hydrodynamics, and recent runs in this workspace were trained with `with_deformable: true`. Training logs land under `logs/rl_games/<config.name>/<timestamp>/`, but the RL-Games config still uses the template name `cartpole_direct`, so “latest salmon model” has to be inferred from each run’s saved `params/env.yaml`, not the folder name.

I added [play_latest_salmon_swim.sh](${FISH_ROOT}/scripts/play_latest_salmon_swim.sh:1). It scans `logs/rl_games/*/*/params/env.yaml`, filters for salmon-swim runs via `per_bone_hydro_path`, picks the newest valid checkpoint, and launches `scripts/rl_games/play.py`. By default it opens a GUI run on `cuda:1`, uses `NUM_ENVS=1`, and auto-falls back to skeleton-only playback when the latest checkpoint was trained with the soft body, because low-env soft-body play is unstable. I also patched [scripts/rl_games/play.py](${FISH_ROOT}/scripts/rl_games/play.py:10) so it bootstraps `source/` on `sys.path`, matching `train_ppo.py` and making direct env-python play more reliable.

Verified locally: the launcher resolves the current latest saved salmon checkpoint as `logs/rl_games/cartpole_direct/2026-06-24_05-23-59/nn/cartpole_direct.pth`.

Use it like this:

```bash
bash scripts/play_latest_salmon_swim.sh
```

For a closer soft-body replay of the training runtime:

```bash
SOFT=1 NUM_ENVS=64 bash scripts/play_latest_salmon_swim.sh
```

If you want, I can also add a matching `play_latest_salmon_swim_headless.sh` or fix the RL-Games experiment name so future runs stop logging under `cartpole_direct`.

## Turn 2

### User

set up the play script following: dt=1/960(same as training),softbody on, 1 env, latest ckpt

### Assistant

[play_latest_salmon_swim.sh](${FISH_ROOT}/scripts/play_latest_salmon_swim.sh:1) is set up with these defaults now:

- `NUM_ENVS=1`
- `SOFT=1`
- `CHECKPOINT_MODE=last`
- `env.sim.dt=1/960`
- `env.decimation=32`
- `env.episode_length_s=50.0`

It also now resolves the newest `last_*.pth` by file modification time instead of filename sort, so “latest ckpt” is correct. In the current workspace, the dry run resolves to:

`logs/rl_games/cartpole_direct/2026-06-24_05-23-59/nn/last_cartpole_direct_ep_100_rew_291.39133.pth`

Run it with:

```bash
bash scripts/play_latest_salmon_swim.sh
```

I only did a dry run with `PY=/bin/echo`; I did not launch Isaac Sim. Soft-body with 1 env is still flagged as numerically fragile, but the script now defaults to exactly what you asked for.

## Turn 3

### User

recover the original red marker in the play scene. in the terminal, add a success count accross eps. save the trajectory for each eps in a folder named play trajectories

### Assistant

The play path is updated.

[scripts/play_latest_salmon_swim.sh](${FISH_ROOT}/scripts/play_latest_salmon_swim.sh:181) now restores the original red marker by forcing `env.marker_success_sphere=false`, keeps trajectory recording on, saves every episode, and routes play trajectories to `${FISH_ROOT}/demo_out/play trajectories`. Each run still gets its own timestamped subfolder under that root.

[scripts/rl_games/play.py](${FISH_ROOT}/scripts/rl_games/play.py:53) now accepts trajectory overrides and prints a cumulative terminal summary whenever an episode ends:
`[play] episodes completed=... successes=... (this step .../... , ...%)`

[source/FISH/FISH/tasks/direct/fish/salmon_swim_env.py](${FISH_ROOT}/source/FISH/FISH/tasks/direct/fish/salmon_swim_env.py:718) now exposes per-env `success` in `infos`, so the play counter is based on the actual success condition rather than inference.

Run it the same way:

```bash
bash scripts/play_latest_salmon_swim.sh
```

Trajectories will go under:
`${FISH_ROOT}/demo_out/play trajectories/run_YYYYmmdd_HHMMSS/`

I verified shell/Python syntax and dry-ran the launcher. I did not start Isaac Sim itself.

## Turn 4

### User

从某一个step开始鱼开始抽动但是没有爆炸。他的表现和之前以及训练完全不一样

### Assistant

结论先说清楚：这更像是运行条件变了，不是我刚加的红 marker / success count / trajectory 保存把策略弄坏了。那几处改动只动了可视化和日志，不动 policy、hydro 或 FEM 积分。

现在和“之前/训练”不一样，主要有 3 个原因：

1. 你现在跑的是 `soft body + 1 env`。这个组合在这个 repo 里本来就不代表训练时的稳定工况。  
2. 你现在默认加载的是 `latest last checkpoint`，不是旧脚本的 `best checkpoint`。  
3. 现在 play 还开了 `env.reset_deformable=true`，这主要影响 reset 后，不是 mid-episode 抽动的主因。

所以，当前现象不是一个“新 bug 把训练策略毁了”，而是你要求的这组运行条件本身就在训练分布外。

你可以这样快速验证：

```bash
# 看是不是 soft-body 单 env 的问题
SOFT=0 CHECKPOINT_MODE=last bash scripts/play_latest_salmon_swim.sh
```

如果这样就不抽了，根因基本就是 `soft body + 1 env`。

如果你要尽量贴近训练行为，应该这么跑：

```bash
SOFT=1 NUM_ENVS=64 CHECKPOINT_MODE=best \
bash scripts/play_latest_salmon_swim.sh env.debug_print=true
```

如果你坚持 `latest ckpt`，也至少把 env 数提上去：

```bash
SOFT=1 NUM_ENVS=64 CHECKPOINT_MODE=last \
bash scripts/play_latest_salmon_swim.sh env.debug_print=true
```

## Turn 5

### User

现在的ckpt是用dt1/960训练的，num_envs=512。这个num_envs不应该影响play policy因为环境之间互不影响。另外，terminal里没有success count， trajectory也没有save成功。另外，任何情况下不要试着关掉软体

### Assistant

你这点我同意一半：

- 在 policy 层面，`num_envs` 不会改变网络的输入输出映射，env 之间也不共享状态。
- 但在这个项目里，`soft body + PhysX GPU` 的数值行为确实会随 batch 规模变化。这个不是 policy coupling，而是 simulator numerics。

我已经按你的要求改了 3 件事：

1. 永远不关软体。launcher 现在如果你传 `SOFT=0` 会直接报错退出。  
2. success count 不是没写，而是只会在 episode 结束时打印。  
3. trajectory 保存打开了 live trajectory 和每 episode 保存。

另外我把 env 的 `success` 明确信号暴露给 play 侧了。

现在你再跑：

```bash
bash scripts/play_latest_salmon_swim.sh
```

我没有实际启动 Isaac Sim，只做了 dry-run 和语法校验。

## Turn 6

### User

为什么现在的episode逻辑和训练不一样？训练是1500step是一个ep

### Assistant

不是不一样，实际逻辑和训练是一样的。你看到的是 terminal 里的标签把两种不同的“step”混在一起了。

真实 episode 逻辑在这里：

- `decimation = 32`
- `episode_length_s = 1500 * decimation * SIM_DT`
- `SIM_DT = 1/960`
- env 的 control step 是 `32 * 1/960 = 1/30 s`
- 一集就是 `50 s / (1/30 s) = 1500` 个 control steps

Isaac Lab 的 `DirectRLEnv.step()` 每调用一次，只给 `episode_length_buf += 1` 一次，但内部先跑 `decimation=32` 个 physics substeps。

done 逻辑也没变：

- timeout 条件是 `episode_length_buf >= max_episode_length - 1`
- success / blowup / out_of_bounds 也都会提前终止

所以结论很简单：

- 训练和 play 的 episode 长度逻辑没变
- 还是 `1500` 个 control steps 一集
- 一集内部对应 `1500 * 32 = 48000` 个 physics substeps

## Turn 7

### User

看看200多步的时候发生了什么，鱼突然开始抽搐乱跑

### Assistant

按代码排查，`200` 左右没有任何“episode 逻辑切换”。

关键点：

- 一集仍然是 `1500` 个 control steps，不是 `200`
- timeout 也是按这个 `episode_length_buf` 判的
- 在 `~200` step 附近唯一按步数触发的东西只有：
  - `traj_live_every=150` 的 live trajectory 文件刷新
  - 每 `200` 步打印一次 `blowup-rate` 日志

这两个都不会改 action、obs、reward、reset，也不会改 physics state。

所以，“200 多步突然抽搐乱跑”更像是数值/策略轨迹本身进入了坏区域，而不是 step-based branch。最可疑的真实差异是：

- 当前 launcher 默认加载的是最新 `last_*.pth`
- 旧 GUI 脚本默认加载的是 best `cartpole_direct.pth`

如果你要继续往下查，建议直接加一个窄窗口 trace，只在 `global_step 180..230` 打：

- action 均值/最大值
- joint vel 最大值
- root lin/ang vel
- distance to target
- success / terminated 标志

## Turn 8

### User

加载Start time
June 24th, 2026 1:24:01 AM
Runtime
1d 12h 25m 54s
Run path
ANON-ENTITY/fish_articulation_analytic_water/nhslidej

Hostname
UR2080319S512539
OS
Linux-5.15.0-181-generic-x86_64-with-glibc2.35
Python version
CPython 3.11.15
Git repository
git clone git@github.com:anon/FISH.git

Git state
git checkout -b "salmon_swim_debug" c3dccf361a7b411aea47ce89de08089e61ddd6dd

Python executable
${FISH_PYTHON}

Command
${FISH_ROOT}/scripts/rl_games/train_ppo.py env.with_deformable=true env.record_video=false env.debug_print=true

Reproduce this run

System Hardware
CPU count	112
Logical CPU count	224
GPU count	2
GPU type	NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
W&B CLI Version
0.25.0
0.28.0 available这个run最好的模型

### Assistant

这个 run `nhslidej` 对应的本地训练目录是：

[logs/rl_games/cartpole_direct/2026-06-24_05-23-59](${FISH_ROOT}/logs/rl_games/cartpole_direct/2026-06-24_05-23-59)

它的 `best` 模型是：

[cartpole_direct.pth](${FISH_ROOT}/logs/rl_games/cartpole_direct/2026-06-24_05-23-59/nn/cartpole_direct.pth)

同一个 run 的 `latest last` checkpoint 是：

[last_cartpole_direct_ep_100_rew_291.39133.pth](${FISH_ROOT}/logs/rl_games/cartpole_direct/2026-06-24_05-23-59/nn/last_cartpole_direct_ep_100_rew_291.39133.pth)

如果你要复现“这个 run 最好的模型”，应该加载 `cartpole_direct.pth`，不是 `last_*.pth`。

## Turn 9

### User

ok

### Assistant

默认已经改成 `best`。

现在直接运行：

```bash
bash scripts/play_latest_salmon_swim.sh
```

会加载：

[cartpole_direct.pth](${FISH_ROOT}/logs/rl_games/cartpole_direct/2026-06-24_05-23-59/nn/cartpole_direct.pth)

如果你之后还想临时切回最新快照，用：

```bash
CHECKPOINT_MODE=last bash scripts/play_latest_salmon_swim.sh
```

## Turn 10

### User

看demo_out/play trajectories/run_20260625_193155/traj_live_env0.png，明显从某一个step开始乱跑了

（用户附带了一张 `traj_live_env0.png` 图像）

### Assistant

我看了这条 live trajectory，结论很明确：不是在 `200` 整突然切换，而是在接近目标后没进成功半径，然后在 `255-278` 这段开始持续偏航和下潜。

我从 `traj_live_env0.obj` 里算出来的关键点是：

- 离目标最近在 `step 255`
- 成功半径是 `0.5`
- 它没有触发 success/reset
- `z < 0.8` 第一次出现在 `step 272`
- 从 `step 278` 开始，已经是在持续远离最佳点了

所以“乱跑”的真实起点更像是 `255-278`，不是某个硬编码的 `200`。代码里也没有在 `200` 或 `256` 步触发会改动力学/动作的分支。

我的判断是：这是一次“接近目标但没成功”的坏分支。原因更像是：

- play 现在是 `--stochastic`
- PPO 配置保留了明显的 action noise
- reward 里 `heading=0.0`，只靠 progress + success

如果你要，我下一步可以直接加一个窄窗口 trace，只在 `step 230-300` 打：

- `distance`
- `z`
- `root lin/ang vel`
- `joint vel max`
- `action mean/max`

## Turn 11

### User

原policy success rate 是100%，我觉得更像是类blow up的物理错误。你把每个step下policy network输出，joint参数，state保存到log，给我命令重启一个play，用deterministic policy，我们看看问题出在哪

### Assistant

（这一轮没有单独给出最终答复，随后用户继续要求先检查 reset 逻辑。）

## Turn 12

### User

检查一下有没有可能是reset逻辑的问题。出问题的step里目标非常近，有可能是出发了success reset，而reset有bug

### Assistant

结论：不是 `success reset`。

- success 半径还是 `0.5m`
- 我直接算了 `traj_live_env0.obj`：`900` 个连续点里，最近目标是 `step 255`，距离 `0.5616m`
- 从来没有进过 success 球
- 轨迹没有 reset 那种大跳变
- run 目录里只有 `traj_live_env0.*`，没有 `traj_ep...`

我已经把 debug 链补全了：

- env 现在可以把 `reset` 之前的真实状态塞进 `extras["debug_step"]`
- play 现在会把每一步的 `policy mu/sigma/sample/final action`、env 实际 action、reward、done/success/time_out、reset 前 root/joint/target/torque/state 全写进 JSONL
- launcher 现在支持 deterministic/stochastic 和 step log 开关

重启命令用这个：

```bash
REAL_TIME=0 PLAY_STEP_LOG=1 POLICY_MODE=deterministic bash scripts/play_latest_salmon_swim.sh
```

## Turn 13

### User

跑完了，看看log，我从GUI看到的就是200步的时候突然开始乱动

### Assistant

看完了，结论很明确：不是 `success reset`，而是一次没有被 done 逻辑捕获的 root-velocity 类物理失稳。

我看的 log 是 `play_deterministic_20260625_200646.jsonl`。里面：

- 到 `206` 步为止，`env0` 一次都没 `done`
- 真正的突变在 `199-200`
- `joint_vel_max` 从 `2.86` 跳到 `13.13`
- 更关键的是 observation 里的 `root_ang_vel` 和 `root_lin_vel` 先爆了

具体看 actor 输入：

- obs 里 `root_lin_vel`、`root_ang_vel` 分别按 `0.5` 和 `0.2` 缩放
- RL-Games 又把 observation clip 到 `[-5, 5]`

所以这说明可见的“突然乱动”不是 reset，而是 root 速度先进入了 blow-up 区域，然后 obs 被 clip，policy 开始在饱和输入上工作，动作就变得像乱翻。

一句话总结：

不是 reset bug。是 **root 速度类的物理失稳**，而当前 done 条件没有及时终止它。

## Turn 14

### User

检查simulator，env的设定除了num envs以外是不是全部跟训练一模一样

### Assistant

不是。除了 `num_envs`，当前 play 和训练还有几处差异。

一致的关键物理设置包括：

- `sim.dt = 1/960`
- `decimation = 32`
- `sim.render_interval = 32`
- `episode_length_s = 50.0`
- `gravity = (0,0,0)`
- `with_deformable = true`
- FEM material、actuator gains、hydro 参数、target、success radius、joint/root limits、初始随机化都一致

不一致的地方：

- `scene.num_envs`: 训练是 `512`，play 是 `1`
- `sim.device`: 训练是 `cuda:0`，play 默认是 `cuda:1`
- `reset_deformable`: 训练是 `false`，play 被强制成 `true`

不会改 mid-episode physics、只是日志/可视化差异的包括：

- `debug_print`
- `traj_dir`
- `traj_save_every_n_episodes`
- `step_debug_extras`

## Turn 15

### User

行试试吧

### Assistant

[scripts/play_latest_salmon_swim.sh](${FISH_ROOT}/scripts/play_latest_salmon_swim.sh:15) 现在支持 `TRAIN_MATCH=1`。这个模式会从训练 run 的 `params/env.yaml` 继承这些设置：

- `sim.device`
- `scene.num_envs`
- `sim.dt`
- `decimation`
- `episode_length_s`
- `reset_deformable`
- `debug_print`
- `record_video`
- `record_trajectory`
- `traj_dir / traj_save_every_n_episodes / traj_live_every`
- `marker_success_sphere`

如果你要保留 deterministic + step log 来继续查问题，用这个：

```bash
REAL_TIME=0 TRAIN_MATCH=1 POLICY_MODE=deterministic PLAY_STEP_LOG=1 bash scripts/play_latest_salmon_swim.sh
```

如果你要尽量贴近训练 rollout，而不是 debug：

```bash
REAL_TIME=0 TRAIN_MATCH=1 POLICY_MODE=stochastic PLAY_STEP_LOG=0 bash scripts/play_latest_salmon_swim.sh
```

## Turn 16

### User

我run了 REAL_TIME=0 TRAIN_MATCH=1 POLICY_MODE=stochastic PLAY_STEP_LOG=1 bash scripts/play_latest_salmon_swim.sh看看log。为什么208就全部complete了

### Assistant

不是 `208` 步 env0 complete 了。当前这份 log 说明的是另一件事。

我看的是 `play_stochastic_20260625_202406.jsonl`。里面：

- 这次是 `512` env、`stochastic`、`cuda:0`
- log 只跑到了 `global_step=221`
- 到 `221` 为止，`env0` 一次都没 `done`
- `env0` 在 `global_step=208` 时还是：
  - `done_env0=false`
  - `success_env0=false`
  - `distance_to_target_env0=0.9808`
  - `joint_vel_max_env0=2.6272`

你看到 terminal 里的 `208`，不是 “第 208 步 episode complete”。旧打印逻辑里：

- `done_count` 是“这一控制步里，512 个 env 里有多少个 ended”
- `episodes_completed` 是“到目前为止，512 个 env 累计 ended 了多少个 episode”

所以你看到的是 **跨所有并行 env 的累计/瞬时 ended 数**，不是 env0 的 episode 步数。

我也把 terminal 输出改清楚了，后面会打印成：

`global_step=... episodes ended across 512 envs: ...`

## Turn 17

### User

不要再跑了。现在是这个512的run里有些环境可以成功，有些环境会出现类blow up。你已经收集了足够的数据，现在分析问题，为什么会这样，怎么解决

### Assistant

结论：现在这个 `512 env` 的 mixed outcome，不是 reset bug，也不是“同一配置下 simulator 随机坏掉”。更像是当前策略和环境一起存在两个 rollout 分支：

- 一类 sample path 能顺利到 target
- 另一类 sample path 会进入 root 刚体的失稳分支，表现成你看到的“类 blow up”

从现有数据看，真正先坏掉的不是 joint limit，也不是 success reset，而是 **root 的线速度/角速度**。

为什么会这样：

1. `512` 个 env 的差异主要来自 stochastic policy 本身  
2. 当前 failure mode 是 root-speed runaway，不在 done 条件里  
3. 一旦 root velocity 太大，policy 看到的是被 clip 后的观测  
4. reward 没有显式抑制这个分支  
5. 训练里的 `success_rate=100%` 可能高估了稳定性

怎么解决：

1. 给 done 增加 root velocity 终止条件
2. 重新训练时加入稳定性约束
3. 把“坏但未死”的 env 单独记指标

一句话总结：

现在的问题不是“有些 env 莫名 reset 错了”，而是：
**stochastic action 的某些 sample path 会把 fish 推进一个 root-spin 的失稳分支，而当前 env 既没有及时终止这个分支，也没有在 reward 上强力惩罚它。**

## Turn 18

### User

把play的dt改成1440，修改其他参数已确保动作频率和训练一样都是30hz，跑一个env num=1的play测试，打开log以及轨迹保存

### Assistant

这次测试按 `dt=1/1440` 跑了，并把 `decimation` 改成 `48`，所以控制频率仍然是 `30Hz`。实际启动时也确认了：

- `Physics step-size = 0.0006944444444444445`
- `Environment step-size = 0.03333333333333333`

测试是 `num_envs=1`、`stochastic`、`step log on`、`trajectory on`，用的是 headless 运行。最终有效产物是：

- step log: `play_stochastic_20260625_205730.jsonl`
- trajectory live image: `traj_live_env0.png`
- trajectory live mesh: `traj_live_env0.obj`

这次短测跑到了 `33` 个 control steps；为了让 live trajectory 真正落盘，这次 run 的 `PLAY_TRAJ_LIVE_EVERY` 临时设成了 `30`，不影响物理。结果上：

- `done_env0 = false`
- `max_joint_vel_env0 = 3.18`
- `min_distance_to_target_env0 = 1.3118` at step `33`
- live trajectory 保存了 `30` 个点

另外还修了 play logger 的一个 bug，不然 `done_count=0` 时会在第一步就因为 `success_count` 未初始化而退出。

## Turn 19

### User

给我命令我跑一个

### Assistant

用这条：

```bash
REAL_TIME=0 TRAIN_MATCH=1 NUM_ENVS=1 SIM_DT=0.0006944444444444445 DECIMATION=48 EPISODE_LENGTH_S=50.0 POLICY_MODE=stochastic PLAY_STEP_LOG=1 PLAY_TRAJ_LIVE_EVERY=30 bash scripts/play_latest_salmon_swim.sh
```

这条会跑：

- `dt = 1/1440`
- `decimation = 48`
- 控制频率仍然是 `30Hz`
- `num_envs = 1`
- `stochastic policy`
- 打开 `step log`
- 打开 trajectory 保存，并且每 `30` 个 control steps 刷一次 `traj_live_env0`

输出会在：

- log: `${FISH_ROOT}/demo_out/play_step_logs/`
- trajectory: `${FISH_ROOT}/demo_out/swim_trajectories/`

如果你想看 GUI，就在前面再加 `HEADLESS=0`。

## Turn 20

### User

把这个对话完整保存到codex.md

### Assistant

已将本线程的主要可见对话按时间顺序保存到仓库根目录的 `codex.md`。
