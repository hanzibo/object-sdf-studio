# Object SDF Studio

**RGB-D object capture → SAM3 masks → BundleSDF reconstruction → FoundationPose tracking**, in a native PySide6 desktop application. Built with SAM 3.

面向自定义物体的桌面工作流：使用 Orbbec Gemini 336L 同步采集 RGB-D，首帧前景/背景点选，SAM3 连续分割，再用 BundleSDF 建模，并把带纹理模型用于 FoundationPose 离线位姿估计与跟踪。

当前界面以中文为主，面向 Linux + NVIDIA CUDA。相机实时采集已集成；FoundationPose 当前为已录制序列的离线跟踪，接相机实时跟踪尚未实现。

```text
Gemini 336L → 对齐 RGB / Depth / K → SAM3 点选与分段跟踪
                                           ↓
                         验收、连续片段导出 → BundleSDF → OBJ + MTL + 纹理
                                           ↓                  ↓
                      新录制序列 + SAM3 首帧 mask → FoundationPose 跟踪
```

## 功能

- 同步 RGB-D 录制、相机内参获取、预览刷新率控制，工作区清理。
- SAM3 首帧正负点提示、完整序列分割、独立分段、补点和失败段重跑；只在全部片段完成后发布完整 masks。
- 数据合同验收、连续有效片段导出；桌面训练启动器提供阶段、位姿帧数、日志与结果入口。
- 自动查找重建 OBJ；FoundationPose 首帧初始化、离线预览、暂停/继续、停止和重新初始化。
- 历史管理：用途分类、缩略图、时间、文件量、硬链接感知的预计释放空间、多选删除与释放反馈。

## 开始使用

完整步骤见 [安装与权重](docs/INSTALL.md)。已有可用环境可直接配置，无需重装或复制环境：

```bash
git clone https://github.com/hanzibo/object-sdf-studio.git
cd object-sdf-studio
cp config.example.json config.local.json
# 编辑各 Python 路径、上游目录和 SAM3 权重位置
python3 tools/doctor.py
./launch.sh
# 独立的 BundleSDF 训练窗口
./launch_trainer.sh
# 可选：安装指向当前目录的桌面图标
python3 tools/install_desktop.py
```

项目路径请使用不含空格的英文路径，父目录避免小写 `rgb`。这是 BundleSDF 原始读取器全路径替换及 shell 路径处理带来的限制，启动器会在训练前检查。

## 工作流与存储

[数据结构和操作说明](docs/WORKFLOW.md) · [代码、环境与目录盘点](docs/PROJECT_MAP.md) · [验证范围](docs/VALIDATION.md)

`datasets/` 保存可编辑的采集/导入工作数据；`exports/` 保存验收后的训练片段。导出优先使用硬链接，因此目录文件量之和不等于实际磁盘占用。已导出的文件应视为只读，修改内容请回到工作数据并重新生成。训练结果默认在 `training_runs/<run>/reconstruction/`，FoundationPose 结果在 `foundationpose_runs/`。

仓库只包含应用代码、适配器、兼容性补丁和文档。模型权重、私人采集数据、日志、Conda 环境、上游完整源码均不提交。MASt3R 匹配器实验没有纳入，BundleSDF 保持 LoFTR 路径。

## 许可与引用

本项目独立编写的应用/适配代码采用 [MIT](LICENSE)。**完整工作流不是可不受限制商用的 MIT 软件包**：BundleSDF、FoundationPose 使用 NVIDIA 非商业研究/评估许可，SAM3 代码与权重使用独立 SAM 许可。补丁保留其上游许可。使用、修改、再分发和下载权重前请阅读 [第三方声明](THIRD_PARTY_NOTICES.md) 及 [LICENSES](LICENSES/)。

感谢 [BundleSDF](https://github.com/NVlabs/BundleSDF)、[FoundationPose](https://github.com/NVlabs/FoundationPose)、[SAM3](https://github.com/facebookresearch/sam3) 和 [Orbbec SDK](https://github.com/orbbec/pyorbbecsdk)。本项目为独立社区集成，与上游机构无隶属关系；算法贡献及论文引用归各原作者。
