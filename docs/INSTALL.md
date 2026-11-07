# 安装与权重

目标：Linux x86_64 + NVIDIA GPU。开发参考机为 8GB 显存 RTX 4060；长序列仍可能 OOM，分割和训练应顺序执行。以下环境版本是已用配置的参考，不保证所有驱动/发行版可直接复现。新机器完整原生编译尚未在干净系统重装验证。

## 1. 复用已有环境（推荐）

复制 `config.example.json` 为 `config.local.json`，填入现有各环境的 Python 可执行文件、三个上游源码目录和 SAM3 权重文件。SAM3 目录必须直接包含 `sam3/` Python 包及 `pyproject.toml`，不能指向外层个人工作目录。

```bash
python3 tools/doctor.py
python3 tools/doctor.py --imports
./launch.sh
```

配置优先级：`STUDIO_<键的大写>` 环境变量 > `config.local.json` > 默认值。例如 `STUDIO_SAM_CHECKPOINT`。`STUDIO_CONFIG` 可指定其他 JSON；JSON 相对路径相对于该 JSON 目录。`config.local.json` 被忽略，不提交机器专属路径。

## 2. 新建 GUI / 相机环境

```bash
conda env create -f environments/object-studio.yml
conda activate object-studio
python -c 'import PySide6, cv2, pyorbbecsdk; print("imports OK")'
```

使用 **pyorbbecsdk2 2.1.2**，导入名称仍为 `pyorbbecsdk`，不要与 SDK v1 混装。如 PyPI 无对应平台 wheel，请按 [Orbbec 官方 SDK2 文档](https://github.com/orbbec/pyorbbecsdk/tree/v2-main) 安装匹配 Python/系统架构的 wheel、udev 规则和设备权限。本项目不会自动改系统设备规则。

## 3. 获取上游和兼容性补丁

```bash
python3 tools/fetch_backends.py
python3 tools/apply_bundle_patch.py
```

版本由 `backends.lock.json` 固定，拉取到忽略的 `third_party/`。工具拒绝不匹配/已有修改的目录，不更改通过本地配置引用的旧项目。所有源码和模型分别受上游许可约束。完整下载可能较大；也可只选一个：`python3 tools/fetch_backends.py sam3`。

## 4. SAM3 环境与权重

```bash
conda env create -f environments/sam3.yml
conda activate sam3
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e third_party/sam3
```

此组合需要支持 CUDA 12.8 的驱动。按 [SAM3 官方安装/模型访问说明](https://github.com/facebookresearch/sam3#installation) 接受许可并自行获取访问权限，将 `sam3.pt` 放到 `checkpoints/sam3/sam3.pt` 或设置 `sam_checkpoint`。不在仓库中分发权重、登录凭据或访问令牌。适配器使用同一官方 checkpoint 中的 tracker 与共享视觉 backbone，并严格检查权重键；不是使用 SAM2 冒充 SAM3。

## 5. BundleSDF 环境与本地编译

```bash
conda env create -f environments/bundlesdf-base.yml
conda activate bundlesdf
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

这个 YAML 只是构建基础，不等于后端安装完成。继续依据 [BundleSDF 官方部署](https://github.com/NVlabs/BundleSDF#installation) 安装 Python 依赖、CUDA 工具链、Eigen、PCL/FLANN、yaml-cpp、cppzmq/ZeroMQ、Boost、OpenGL 及上游需要的 GPU 扩展。参考本地 Python 依赖为 NumPy 1.26.4、Kornia 0.6.12、Trimesh 5.1.0。原项目 Docker 定义也提供完整依赖参考。

BundleTrack 需要原生 CUDA OpenCV（含 contrib 的 rgbd、cudafeatures2d、cudaimgproc、cudaoptflow）；pip 的 `opencv-python` 不提供这些 C++ CUDA 库。将你编译的 OpenCV CMake 包目录作为 `OpenCV_DIR`。不要用替换成 CPU 匹配来绕过缺少 CUDA 模块。

完成依赖后可使用四个 worker 编译本地扩展：

```bash
# 在已激活 bundlesdf 环境、仓库根目录执行；89 是 RTX 4060 的示例
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=8.9
python -m pip install --no-build-isolation -e third_party/BundleSDF/mycuda
cmake -S third_party/BundleSDF/BundleTrack -B third_party/BundleSDF/BundleTrack/build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" -DOpenCV_DIR="$OpenCV_DIR" \
  -DPython_EXECUTABLE="$CONDA_PREFIX/bin/python"
cmake --build third_party/BundleSDF/BundleTrack/build --target my_cpp --parallel 4
(cd third_party/BundleSDF && python -c 'from mycuda import common')
# my_cpp 位于 BundleTrack/build，运行上游脚本时由其导入路径加载
```

架构号、编译器以及 CUDA toolkit 必须匹配本机。补丁用于已知 CMake/PCL/Trimesh 等兼容问题，不代替所有平台适配。避免直接运行含固定系统 Python 路径和 `rm -rf build` 的旧 `build.sh`。

按 [BundleSDF 权重说明](https://github.com/NVlabs/BundleSDF#download-pretrained-models) 下载所需匹配模型到上游要求的位置（检查 `loftr_wrapper.py`，默认 `BundleTrack/LoFTR/weights/outdoor_ds.ckpt`）；不需要下载官方 demo 才能处理自采数据。

## 6. FoundationPose

```bash
conda env create -f third_party/FoundationPose/environment.yml
conda activate foundationpose
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
# CUDA toolkit/nvcc 与编译器就绪后，按上游说明安装 pytorch3d、nvdiffrast 等
python -m pip install -r third_party/FoundationPose/requirements.txt
```

按固定版本的 [FoundationPose README](https://github.com/NVlabs/FoundationPose) 完成 GPU 扩展、`mycpp` 编译，并下载 refiner/scorer 到上游要求的 `weights/` 子目录。上游 `build_all_conda.sh` 使用 `nproc`；如需限制为四个 worker，可逐条运行其中 CMake 配置后使用 `cmake --build ... --parallel 4`。本工作流使用已重建 mesh，不依赖 FoundationPose 自身 model-free NeRF 重建。

## 7. 检查和启动

```bash
python3 tools/doctor.py --imports
./launch.sh
./launch_trainer.sh
python3 tools/install_desktop.py
```

路径检查和 import 成功不等于推理/硬件验证。首次使用建议短序列验证 SAM3 mask、深度单位、模型尺度及 FoundationPose 注册，再跑全序列。运行 `scripts/render_pose_demo.py` 还需要其解释器中安装 Trimesh、Pillow、OpenCV，以及系统 FFmpeg/Noto CJK 字体。
