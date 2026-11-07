"""Bounded online instance tracking using the official SAM3 backbone and tracker."""
from pathlib import Path
import sys
import gc

from config import SAM_PROJECT, SAM_CHECKPOINT
sys.path.insert(0, str(SAM_PROJECT))
import cv2
import numpy as np
import torch
from sam3.model_builder import build_tracker


class OnlineTracker:
    def __init__(self, history=32, compile_mode=None):
        if history < 32:
            raise ValueError("Retain at least 32 frames for temporal memories and pointers")
        self.history = history
        # Fixed temporal memory avoids a growing backwards search on lost targets.
        self.model = build_tracker(apply_temporal_disambiguation=False, with_backbone=True, compile_mode=compile_mode)
        checkpoint = torch.load(str(SAM_CHECKPOINT), map_location="cpu", weights_only=True, mmap=True)
        checkpoint = checkpoint.get("model", checkpoint)
        weights = {}
        for key, value in checkpoint.items():
            if key.startswith("tracker."):
                weights[key.removeprefix("tracker.")] = value
            elif key.startswith("detector.backbone.vision_backbone."):
                weights[key.removeprefix("detector.")] = value
        # No silent missing weights: this is the same shared vision backbone + tracker.
        self.model.load_state_dict(weights, strict=True)
        del weights, checkpoint
        gc.collect()
        self.model.cuda().eval()
        self.state = None
        self.index = 0

    def _features(self, image, index):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.model.image_size, self.model.image_size))
        tensor = torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).cuda().float() / 255
        tensor = (tensor - 0.5) / 0.5
        features = self.model.forward_image(tensor)
        self.state["cached_features"] = {index: (tensor, features)}

    @torch.inference_mode()
    def initialize(self, image, box):
        height, width = image.shape[:2]
        self.state = self.model.init_state(video_height=height, video_width=width, num_frames=1, offload_state_to_cpu=True)
        self.index = 0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            self._features(image, 0)
            x,y,w,h = box
            points = [[x/width,y/height],[(x+w)/width,(y+h)/height]]
            self.model.add_new_points_or_box(self.state, frame_idx=0, obj_id=1, points=points, labels=[2,3], rel_coordinates=True)
            self.model.propagate_in_video_preflight(self.state)
            return self._propagate(0)

    def _propagate(self, index):
        result = None
        for result in self.model.propagate_in_video(self.state, start_frame_idx=index, max_frame_num_to_track=0, reverse=False, tqdm_disable=True):
            pass
        if result is None or result[0] != index:
            raise RuntimeError("Tracker did not return current frame")
        _, ids, _, masks, scores = result
        if ids != [1]:
            raise RuntimeError(f"Unexpected object IDs: {ids}")
        mask = (masks[0,0] > 0).cpu().numpy()
        score_logit = float(scores.reshape(-1)[0].float().cpu())
        if score_logit <= 0:
            mask[:] = False
        return mask, score_logit

    @torch.inference_mode()
    def step(self, image):
        if self.state is None:
            raise RuntimeError("Select a target first")
        if image.shape[:2] != (self.state["video_height"], self.state["video_width"]):
            raise ValueError("Camera resolution changed; reselect target")
        self.index += 1
        self.state["num_frames"] = self.index + 1
        with torch.autocast("cuda", dtype=torch.bfloat16):
            self._features(image, self.index)
            result = self._propagate(self.index)
        cutoff = self.index - self.history + 1
        # Keep the original conditioning frame; remove all references to old outputs.
        mappings = [self.state["output_dict"]["non_cond_frame_outputs"], self.state["frames_already_tracked"]]
        for item in self.state["output_dict_per_obj"].values():
            mappings.append(item["non_cond_frame_outputs"])
        for mapping in mappings:
            for key in list(mapping):
                if key != 0 and key < cutoff:
                    del mapping[key]
        return result

    def state_size(self):
        return {"cached_features": len(self.state["cached_features"]), "conditioning_frames": len(self.state["output_dict"]["cond_frame_outputs"]), "history_frames": len(self.state["output_dict"]["non_cond_frame_outputs"]), "tracked_metadata": len(self.state["frames_already_tracked"])}
