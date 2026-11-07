# 验证范围

2026-11-07，在已有 Linux + RTX 4060 Laptop 8GB 工作站验证发布目录；使用已有 Conda 环境，未重装/改写原项目、权重或个人数据。

已完成：

- 13 项自动检查通过：真实合成 PNG 的 RGB-D/mask 合同与导出、错误深度拒绝、配置覆盖、分段缓存失效、硬链接核算、删除边界、活动任务保护、部分失败及符号链接行为。
- Qt 离屏工作区测试通过：卸载数据不删除文件、预览/录制交接、迟到事件、分割预览状态、忙任务保护，以及侧栏焦点与滚动稳定。相机进程为确定性替代程序。
- 本机相机 SDK、Qt、PyTorch/CUDA、BundleSDF `mycuda.common` 和 `my_cpp` 导入通过。
- SAM3 在 5 帧已有 RGB-D 的私有工作副本上实际 CUDA 推理成功，生成 5 张非空 0/255 masks，并发布完整 masks。仅验证分割链路，没有像素真值精度评估；原始图像和结果均未提交。
- BundleSDF 兼容性补丁在固定版本的干净临时 checkout 上应用成功；重复应用识别为已应用，`git diff --check` 通过。
- 公开仓库归档在不同临时路径、无本机 JSON 配置条件下运行 CPU/Qt 测试通过。

边界：这次没有重新训练完整 BundleSDF 序列，没有重新验证 FoundationPose 全序列精度，没有重新连接相机做物理采集，也没有在干净系统上完整重建所有 CUDA 扩展。已有环境的导入检查不能代替这些验证。GitHub CI 仅执行 CPU/Qt 检查，不下载模型、不运行硬件和 GPU 推理。

本地复查：

```bash
# 在有 NumPy/OpenCV 的 Python 环境内
python -m unittest discover -s tests -p 'test_*.py' -v
QT_QPA_PLATFORM=offscreen python tests/workspace_reset.py
python3 tools/doctor.py --imports
```

离屏测试生成的 UI 截图/报告写到忽略的 `evidence/`，测试数据只在临时目录内创建和删除。
