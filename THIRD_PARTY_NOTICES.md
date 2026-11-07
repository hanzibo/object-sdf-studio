# Third-party notices and publication scope

This repository's MIT license covers independently written Studio application, orchestration, and integration code only. It does not relicense any upstream project, model, dataset, or patch derived from upstream code. Built with SAM 3.

| Component | Distribution here | Applicable license |
|---|---|---|
| BundleSDF / BundleTrack | Compatibility patch; source fetched separately | NVIDIA Source Code License, non-commercial research/evaluation; full text in `LICENSES/NVIDIA-BundleSDF.txt` |
| FoundationPose | API orchestration; source fetched separately | NVIDIA Source Code License, non-commercial research/evaluation; full text in `LICENSES/NVIDIA-FoundationPose.txt` |
| SAM3 | Point/online tracking adapter; source and weights fetched separately | SAM License; full text in `LICENSES/SAM3.txt`; upstream model terms also apply |
| Orbbec SDK2 | Python API calls; SDK wheel installed separately | See the official SDK distribution for its license |
| Camera frame conversion helpers | Author's integration from hanzibo/orbbec-gemini-336l-python-examples, revision 0d6047f9d5c45fd717fe9905dec14182052b4368 | Independently authored integration, MIT here; no SDK implementation vendored |

`patches/bundlesdf/local-compatibility.patch` contains modified NVIDIA code and context. It is governed by the BundleSDF NVIDIA license, not the root MIT license. Preserve the original notices when applying/redistributing it. Its modifications address CMake dependencies and architecture selection, pointer compatibility, Trimesh API changes, LoFTR batch memory limits and array conversion.

The complete integrated workflow remains subject to the applicable upstream restrictions. Do not infer commercial permission from the root MIT badge or license detector. Weights are not redistributed. Any third-party dependencies downloaded by setup tools retain their own notices and licenses.

Upstream projects and papers:

- [BundleSDF: Neural 6-DoF Tracking and 3D Reconstruction of Unknown Objects](https://github.com/NVlabs/BundleSDF), CVPR 2023.
- [FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects](https://github.com/NVlabs/FoundationPose), CVPR 2024.
- [SAM 3](https://github.com/facebookresearch/sam3), Meta.
- [Orbbec Python SDK](https://github.com/orbbec/pyorbbecsdk).

Pinned revisions are recorded in `backends.lock.json`. Please cite the original papers when using their algorithms in research. This integration is not an official release of any of these projects.
