# trainTuning

trainTuning 是一个面向 Windows 单机环境的 YOLO 训练调优桌面应用。C# / WPF
负责项目界面、任务控制和结果展示，Python Worker 负责数据审计、参数推荐、
Ultralytics 训练、指标分析和可确认的训练中调整。

## 已实现

- 中文 WPF 调优工作台
- Detect、Segment、Pose、Classify、OBB 任务入口
- 默认使用 `yolov8n.pt`，并支持 YOLO11、YOLO26 自动规格选择
- 根据显存、数据规模、目标尺寸、类别失衡和标注质量生成参数
- 支持选择已有任务权重，以 AdamW、小学习率、弱增强方式保留检测头微调
- 区分固定工业相机与通用场景；方向敏感任务自动关闭翻转和旋转
- 训练前审计本地 YOLO 数据集、显式负样本、标签格式、类别失衡和划分泄漏风险
- 生成 Baseline 与单变量候选组成的受控实验计划
- 每个参数附带推荐依据和风险提示
- 生成可复制的 Ultralytics CLI 命令
- C# 与 Python 通过逐行 JSON 协议通信
- Python 训练进程独立运行，可由 C# 主程序强制停止
- 实时展示 mAP50-95、mAP50、Precision、Recall、loss 和学习率
- 验证指标进入平台期时提出降低学习率或提前停止建议，由用户接受或跳过
- 训练完成后记录最佳轮次，明确区分部署用 `best.pt` 与续训用 `last.pt`
- 每个真实训练目录写入 `train_tuning_history.json`，保存配置、逐轮指标和调优决策
- 未安装 Ultralytics 时提供包含人工确认流程的通信模拟模式
- 无第三方 NuGet 包，可离线构建 WPF 主程序

## 项目结构

```text
src/trainTuning.App/ C# WPF 主程序
engine/             Python 推荐和训练 Worker
tests/              Python 单元与协议测试
scripts/            开发运行和发布脚本
packaging/          Inno Setup 安装包配置
```

## 开发运行

要求：

- Windows 10/11 x64
- .NET 10 SDK
- Python 3.11 或兼容版本

构建：

```powershell
dotnet build trainTuning.slnx
```

运行：

```powershell
.\scripts\run-dev.ps1
```

如果 `python.exe` 不在 PATH 中，可以指定训练环境：

```powershell
$env:TRAINTUNING_PYTHON = "D:\AI\envs\yolo\python.exe"
.\scripts\run-dev.ps1
```

运行测试：

```powershell
python -m unittest discover -s tests -v
```

## 准备真实训练环境

先根据目标电脑的显卡和驱动安装合适的 CUDA 版 PyTorch，然后安装 Worker 依赖：

```powershell
python -m pip install -r engine\requirements.txt
```

若数据集路径有效且环境中可以导入 `ultralytics`，点击“启动验证训练”会直接启动真实训练。
否则 Worker 会清楚报告错误；只有完全缺少 Ultralytics 时才进入界面通信模拟。

训练中提出建议时，Python Worker 会暂停在当前 Epoch 结束点，直到界面选择“接受并继续”
或“跳过并继续”。实时调整只包括能够安全作用于当前训练的学习率和提前停止。模型规模、
输入尺寸、batch 和数据增强必须作为下一组独立实验运行，避免一次改变多个因素后无法判断
提升来源。

## 发布

仅发布自包含的 .NET 主程序：

```powershell
.\scripts\publish.ps1
```

将准备好的独立 Python 环境一同放入发布目录：

```powershell
.\scripts\publish.ps1 -PythonHome "D:\AI\portable-yolo-env"
```

应用会优先查找 `runtime\python\python.exe`，因此目标电脑不需要安装 Python 或 .NET。
目标电脑仍需要安装与 PyTorch 兼容的 NVIDIA 驱动。

发布完成后，可用 Inno Setup 编译 `packaging\trainTuning.iss` 生成安装程序。

## 推荐结果的含义

规则推荐提供的是高质量基线，不宣称对未知数据集直接得到数学意义上的“全局最优”。
可靠的最佳参数需要在固定验证集上通过数据审计、短周期试训、单变量受控实验和最终完整
训练来确认。主比较指标是 mAP50-95；训练集预测不能代替独立验证或测试结果。固定工业
缺陷检测还应加入真实 OK 图片及同名空标签，以便测量现场误报率。
