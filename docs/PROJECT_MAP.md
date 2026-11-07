# 项目与环境盘点

## 发布代码

| 模块 | 职责 |
|---|---|
| `app.py` | PySide6 主界面、相机/分割页、工作区生命周期 |
| `worker.py` | 独立进程相机采集、SAM3 分割、导入与验收 |
| `integrations/sam3_tracker.py` | 官方 SAM3 backbone + tracker，32 帧有界历史，保留初始条件帧 |
| `integrations/orbbec_frames.py` | 相机颜色格式转换、深度比例到毫米 |
| `data_ops.py`, `segment_plan.py` | 数据合同、硬链接导出、片段签名与缓存恢复 |
| `trainer.py`, `train_job.py` | 独立训练 GUI、BundleSDF CLI 监督、全局优化及贴图阶段检测 |
| `pose_ui.py`, `pose_data.py` | 新序列首帧 SAM3 初始化、跟踪结果管理 |
| `scripts/run_foundationpose_offline.py` | 使用现有 mesh 的离线注册、跟踪和叠加预览 |
| `mesh_browser.py` | 查找最终带纹理 OBJ、文件对话框快捷位置 |
| `history_store.py`, `history_ui.py` | 分类预览、引用检查、硬链接空间核算及受控删除 |
| `data_lock.py`, `gpu_lock.py` | 进程间数据访问和 GPU 排他协调 |
| `patches/bundlesdf/` | CUDA/CMake/PCL/Trimesh/LoFTR 兼容性修改；上游 NVIDIA 许可 |

SAM3 的文字提示、检测器完整应用和 SAM 3D Body 不在本仓库范围内。这里使用点提示实例跟踪，不需要为每帧人工画 mask。Orbbec 原示例可保留为独立项目，本仓库运行不再依赖其 `examples/common.py` 的个人路径。

## 环境隔离

| 配置键 | 默认环境 | 工作 |
|---|---|---|
| `gui_python`, `camera_python` | `object-studio`, Python 3.11 | Qt、SDK2、CPU 数据检查 |
| `sam_python` | `sam3`, Python 3.12 | SAM3 CUDA 推理 |
| `bundle_python` | `bundlesdf`, Python 3.11 | BundleSDF + 原生 CUDA/C++ 扩展 |
| `pose_python` | `foundationpose`, Python 3.11 | FoundationPose + 渲染/注册扩展 |

原来能工作的相机环境可通过配置继续使用；不同后端不要求合并成一个 Conda 环境。环境名称和源码目录独立，Python 可执行文件才是实际选择依据。GUI 通过 QProcess/子进程调用后端，避免 Qt、PyTorch 和二进制依赖互相污染。

## 本地数据

| 目录（默认当前仓库根目录下） | 内容 |
|---|---|
| `datasets/` | 录制/导入原始数据及可修改提示、分割缓存 |
| `exports/` | 验收生成的连续 BundleSDF 训练片段 |
| `training_runs/` | 每次训练的独立结果、配置和完整日志 |
| `foundationpose_runs/` | 离线跟踪、位姿和渲染结果 |
| `runtime/` | 界面设置、临时预览、进程锁 |
| `third_party/` | 独立克隆的上游代码，不随本仓库提交 |
| `checkpoints/` | 单独下载的权重，不随本仓库提交 |

原有工作目录不自动迁移或清理。新目录默认为空；可在 GUI 中打开旧数据集，或通过配置引用原上游和权重。不要把旧数据软链接成新工作区并在历史页试删，以免误解数据归属。
