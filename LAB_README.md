# OpenPI 具身策略实验台

从公开机器人示范读取状态和动作，审计轨迹质量，训练 PyTorch 行为克隆策略，再进入 LIBERO 仿真比较不同动作执行长度。项目保留数据来源、训练划分、逐轮评测和模拟器回放，便于复核每项结果。

**任务**　抓起餐盘与小碗之间的黑碗，并把它放进餐盘。数据集中的任务编号是 34，对应模拟器的 `libero_spatial/0`。数据子集包含 33 条完整示范，共 3,231 条状态和动作记录。每条轨迹的记录数已与上游 episode 元数据核对。 [数据来源与哈希](lab_data/libero_spatial_0.provenance.json) · [公开 LeRobot 数据集](https://huggingface.co/datasets/lerobot/libero)

## 打开演示

打开 [已运行的离线演示](lab_runs/demo_verified/index.html)。页面展示数据审计、训练误差、真实 LIBERO 模拟器回放和受控配对对照。直接从本地读取数据和回放。

本地可在 PowerShell 中复跑。需要 Docker Desktop 的 Linux 容器后端。脚本优先复用 `openpi-trajectory-lab:cpu` 镜像，缺少镜像时会构建。训练和模拟阶段读取仓库内的数据子集，并在无网络的容器中执行。

```powershell
.\run_lab_demo.ps1 -Trials 3
```

结果写到 `lab_runs/<运行编号>/index.html`。默认每个设定使用 3 个基准初态。第一次构建镜像会安装 CPU 版 PyTorch 和模拟器依赖。

## 本机实测

按 episode 划分为 26 条训练示范和 7 条验证示范。通过 CPU 训练 25 个 epoch 后，验证集的动作 MAE 为 **0.1313**，训练集均值动作基线为 **0.3246**。两次 CPU 训练的逐轮验证记录一致。 [数据审计](lab_runs/demo_verified/audit.json) · [训练记录](lab_runs/demo_verified/bc/training.json)

| 策略 | 每次执行步数 K | LIBERO 成功数 | 平均模型调用次数 |
| --- | ---: | ---: | ---: |
| 行为克隆 | 5 | 2/3 | 25.00 |
| 行为克隆 | 10 | 2/3 | 12.67 |
| 上游空动作对照 | 5 | 0/3 | 44.00 |

这组对照固定了任务、三个初态、仿真种子和最多 220 步的执行预算。两组行为克隆在同样两个初态上成功。结果描述的是单任务、3 个初态下的状态输入基线。另有 30 次一维受控仿真用于验证配对逻辑，包含目标中途移动的场景。8 项自动测试通过，包括使用本地测试服务核对 OpenPI WebSocket 报文的测试。当前机器上的推理结果来自 CPU 行为克隆，官方 π₀.₅ 权重评测需在支持的 GPU 环境中单独运行。 [逐轮结果与回放](lab_runs/demo_verified/closed_loop/report.md) · [验收核对](lab_runs/demo_verified/validation.json)

## 从数据到评测

| 层 | 代码 | 可复核的行为 |
| --- | --- | --- |
| 数据 | [prepare_sample.py](trajectory_lab/prepare_sample.py) · [audit.py](trajectory_lab/audit.py) | 固定源数据版本，对照上游元数据筛出完整的同任务 episode。逐批读取 Parquet，检查帧序号、时间戳、状态和动作维度、数值异常及跨数据划分的 episode 重复。 |
| 训练 | [behavior.py](trajectory_lab/behavior.py) | 把同一 episode 切成 10 步动作窗口，尾部缺失步用掩码排除。按整条 episode 划分训练与验证，并用训练子集统计量归一化。拟合两层 MLP，记录其验证 MAE 与训练集均值动作基线。 |
| 仿真 | [libero_backend.py](trajectory_lab/libero_backend.py) · [evaluation.py](trajectory_lab/evaluation.py) | 对同一任务、初态和仿真种子重置环境。模型预测动作片段，先执行前 K 步，接收新观测后再推理。记录成功、超时、策略错误和环境错误。各组的逐轮 ID 可以配对核查。 |
| 推理 | [libero_backend.py](trajectory_lab/libero_backend.py) · [原版观测预处理](examples/libero/main.py) | 共享上游的主相机、腕相机和状态预处理。模型服务器可通过 OpenPI WebSocket 协议接入。收到超时或无效响应后关闭连接，避免下一次误读旧响应。 |
| 报告 | [render_demo.py](trajectory_lab/render_demo.py) | 保存机器可读 JSON、逐轮 CSV 和浏览器可打开的本地页面。GIF 回放来自每个设定的首个模拟器初态。 |

轨迹审计覆盖 Parquet 记录列。原始相机视频的完整性和同步精度属于下一层审计。仓库中的数据子集只包含用于该基线的状态、动作和索引列。

## 接入预训练 π₀.₅

有合适的 NVIDIA GPU 主机时，可按 OpenPI 官方文档启动 LIBERO 默认推理服务器。默认模型为官方 `pi05_libero` checkpoint。 [部署要求](https://github.com/Physical-Intelligence/openpi#requirements) · [模型服务器源码](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/scripts/serve_policy.py) · [LIBERO 官方示例](https://raw.githubusercontent.com/Physical-Intelligence/openpi/215abfb217dbac7d5f1273282331b9b1866c0479/examples/libero/README.md)

```bash
uv run scripts/serve_policy.py --env LIBERO
```

如果模型服务运行在 Docker Desktop 宿主机，在项目目录打开 PowerShell，调用同一套评测器。模型服务在其他机器时改用可达的 WebSocket 地址。

```powershell
$mount = "type=bind,source=$((Get-Location).Path),target=/app"
docker run --rm --mount $mount openpi-trajectory-lab:cpu python -m trajectory_lab libero --policy openpi --server ws://host.docker.internal:8000 --checkpoint-label pi05_libero --trials 3 --execute-steps 5 10 --record-video --output lab_runs/pi05_comparison
```

测量时保持权重、任务、初态、仿真种子和步数预算一致。报告中的远程推理时间包含观测预处理、通信与服务计算。当前客户端固定仿真种子，服务端模型采样由模型服务自身管理。扩大实验样本时，建议同时记录服务版本和采样随机性。

## 面试讲法

> 这个项目关注具身策略从数据到闭环的验证。我基于 OpenPI 做了一个 LIBERO 实验台，把公开示范按任务和 episode 正规化检查，再训练仅输入机器人自身状态的行为克隆策略。训练时按 episode 做验证集划分，避免重叠窗口把同一段轨迹带到两边。仿真里固定任务、初态和种子，只改变一次推理后执行多少步动作，记录成功率、耗时和回放。我可以沿代码讲清一条动作窗口如何生成，以及为什么要单独报告离线动作误差和闭环成功率。

面试时可结合你已有的 **ROS2-CoAudit** 讲运行时序和失败分析，结合 **Gazebo Actor-Critic** 讲对照变量与指标，结合 **APT 导入和 PDNS 周任务** 讲数据去重、版本与失败重跑。

### 常见代码追问

- **为什么要按 episode 划分？** 相邻动作窗口大量重叠。按帧随机划分会让同一次演示的画面和动作落入训练与验证两边，导致误差偏乐观。
- **K 和预测长度 H 有何关系？** 本项目一次预测 H=10 步。K 决定执行第几步后读新观测并推理，必须满足 `1 ≤ K ≤ H`。改变 K 可在固定权重下单独衡量重新推理频率与任务表现。
- **怎样识别推理失败？** 动作片段太短、维度不对、包含非有限值或服务超时都会记为策略错误。报告把策略错误、仿真错误、步数预算用尽和成功分别记录。
- **怎样接更多数据？** 先固定来源版本，用 episode ID 标识数据，执行字段和时间窗口审计；训练抽样和仿真评测分别记录配置。当前 CPU 训练器在内存中组织小数据集，审计器以批为单位读取 Parquet。

## 来源

- 上游 OpenPI 锁定在 `215abfb217dbac7d5f1273282331b9b1866c0479`，运行使用的 LIBERO 子模块锁定在 `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`。本地功能位于 `trajectory_lab/`，观测预处理与原版评测复用。 [OpenPI 上游提交](https://github.com/Physical-Intelligence/openpi/commit/215abfb217dbac7d5f1273282331b9b1866c0479) · [上游许可](LICENSE)
- 数据源为 `lerobot/libero` 的不可变版本 `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`，选取清单、输入文件哈希和数据子集哈希见本地来源记录。 [数据来源与哈希](lab_data/libero_spatial_0.provenance.json)
