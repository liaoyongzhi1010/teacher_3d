from __future__ import annotations

import torch
import torch.nn.functional as F


class NullTeacherAdapter:
    def __call__(self, batch):
        return {}


class OracleTeacherAdapter:
    def __call__(self, batch):
        return {
            "visible_depth": batch["teacher_visible_depth"],
            "visible_normals": batch["teacher_visible_normals"],
            "visible_confidence": batch["teacher_visible_confidence"],
            "hidden_depth": batch["teacher_hidden_depth"],
            "hidden_confidence": batch["teacher_hidden_confidence"],
        }


class VGGTPrecomputedTeacherAdapter:
    REQUIRED_BATCH_KEYS = {
        "teacher_visible_depth",
        "teacher_visible_confidence",
        "teacher_hidden_depth",
        "teacher_hidden_confidence",
    }

    OPTIONAL_BATCH_KEYS = {
        "teacher_visible_normals",
        "teacher_rendered_target",
        "teacher_hidden_confidence_raw",
        "teacher_hidden_count",
        "teacher_hidden_gap",
    }

    def __init__(self, config) -> None:
        teacher_cfg = config.teacher
        self.enable_hidden_layered_targets = bool(getattr(teacher_cfg, "enable_hidden_layered_targets", False))
        self.hidden_target_support_radius = int(
            getattr(teacher_cfg, "hidden_target_support_radius", getattr(teacher_cfg, "hidden_target_dilation_radius", 0))
        )
        self.hidden_target_mining_strength = float(
            getattr(teacher_cfg, "hidden_target_mining_strength", getattr(teacher_cfg, "hidden_target_dilation_strength", 0.0))
        )
        self.hidden_target_gate_visible = bool(
            getattr(teacher_cfg, "hidden_target_gate_visible", getattr(teacher_cfg, "hidden_target_dilation_gate_visible", True))
        )
        self.hidden_target_gap_min = float(
            getattr(
                teacher_cfg,
                "hidden_target_gap_min",
                getattr(
                    teacher_cfg,
                    "hidden_target_dilation_gap_min",
                    getattr(teacher_cfg, "depth_margin", 0.05),
                ),
            )
        )
        self.hidden_target_mining_mode = str(
            getattr(teacher_cfg, "hidden_target_mining_mode", getattr(teacher_cfg, "hidden_target_dilation_mode", "replace"))
        )
        self.hidden_target_mining_strategy = str(getattr(teacher_cfg, "hidden_target_mining_strategy", "dilation"))
        self.hidden_target_boundary_radius = int(
            getattr(teacher_cfg, "hidden_target_boundary_radius", max(self.hidden_target_support_radius, 1))
        )
        self.hidden_target_boundary_threshold = float(
            getattr(teacher_cfg, "hidden_target_boundary_threshold", self.hidden_target_gap_min)
        )
        self.hidden_target_boundary_rel_threshold = float(
            getattr(teacher_cfg, "hidden_target_boundary_rel_threshold", 0.0)
        )
        self.hidden_target_boundary_min_score = float(
            getattr(teacher_cfg, "hidden_target_boundary_min_score", 0.0)
        )
        default_local_gap = max(self.hidden_target_gap_min * 3.0, 0.15)
        self.hidden_layered_boundary_min_score = float(
            getattr(teacher_cfg, "hidden_layered_boundary_min_score", 0.15)
        )
        self.hidden_layered_local_gap_max = float(
            getattr(teacher_cfg, "hidden_layered_local_gap_max", default_local_gap)
        )
        self.hidden_layered_use_raw_support = bool(
            getattr(teacher_cfg, "hidden_layered_use_raw_support", True)
        )
        self.hidden_layered_raw_confidence_scale = float(
            getattr(teacher_cfg, "hidden_layered_raw_confidence_scale", getattr(teacher_cfg, "conf_threshold", 1.5))
        )
        self.hidden_layered_raw_count_scale = float(
            getattr(teacher_cfg, "hidden_layered_raw_count_scale", 1.0)
        )

    def _compute_boundary_support(self, visible_depth, visible_confidence):
        abs_threshold = max(self.hidden_target_boundary_threshold, 0.0)
        rel_threshold = max(self.hidden_target_boundary_rel_threshold, 0.0)
        valid = visible_confidence > 0.5
        boundary_score = torch.zeros_like(visible_depth)

        def pair_threshold(depth_a, depth_b):
            threshold = torch.full_like(depth_a, abs_threshold)
            if rel_threshold > 0.0:
                rel_base = torch.minimum(depth_a, depth_b).clamp_min(1e-3)
                threshold = torch.maximum(threshold, rel_base * rel_threshold)
            return threshold.clamp_min(1e-6)

        horiz_valid = valid[:, :, :, 1:] & valid[:, :, :, :-1]
        horiz_diff = (visible_depth[:, :, :, 1:] - visible_depth[:, :, :, :-1]).abs()
        horiz_thresh = pair_threshold(visible_depth[:, :, :, 1:], visible_depth[:, :, :, :-1])
        horiz_score = torch.where(horiz_valid, (horiz_diff / horiz_thresh).clamp(0.0, 1.0), torch.zeros_like(horiz_diff))
        boundary_score[:, :, :, 1:] = torch.maximum(boundary_score[:, :, :, 1:], horiz_score)
        boundary_score[:, :, :, :-1] = torch.maximum(boundary_score[:, :, :, :-1], horiz_score)

        vert_valid = valid[:, :, 1:, :] & valid[:, :, :-1, :]
        vert_diff = (visible_depth[:, :, 1:, :] - visible_depth[:, :, :-1, :]).abs()
        vert_thresh = pair_threshold(visible_depth[:, :, 1:, :], visible_depth[:, :, :-1, :])
        vert_score = torch.where(vert_valid, (vert_diff / vert_thresh).clamp(0.0, 1.0), torch.zeros_like(vert_diff))
        boundary_score[:, :, 1:, :] = torch.maximum(boundary_score[:, :, 1:, :], vert_score)
        boundary_score[:, :, :-1, :] = torch.maximum(boundary_score[:, :, :-1, :], vert_score)

        radius = self.hidden_target_boundary_radius
        if radius > 0:
            kernel = 2 * radius + 1
            boundary_score = F.max_pool2d(boundary_score, kernel_size=kernel, stride=1, padding=radius)
        return boundary_score.clamp(0.0, 1.0)

    def _mine_hidden_targets(self, targets):
        radius = self.hidden_target_support_radius
        strength = self.hidden_target_mining_strength
        if radius <= 0 or strength <= 0.0:
            return None

        hidden_confidence = targets["hidden_confidence"].float()
        hidden_depth = targets["hidden_depth"].float()
        visible_depth = targets["visible_depth"].float()
        visible_confidence = targets["visible_confidence"].float()

        positive_mask = hidden_confidence > 0.5
        if not bool(positive_mask.any()):
            return None

        kernel = 2 * radius + 1
        support_mask = F.max_pool2d(positive_mask.float(), kernel_size=kernel, stride=1, padding=radius) > 0.5
        strategy = self.hidden_target_mining_strategy
        mining_weight = torch.ones_like(hidden_confidence)
        if strategy == "dilation":
            candidate_mask = support_mask
        elif strategy == "boundary":
            boundary_support = self._compute_boundary_support(visible_depth, visible_confidence)
            min_score = max(0.0, min(1.0, self.hidden_target_boundary_min_score))
            if min_score > 0.0:
                candidate_mask = support_mask & (boundary_support >= min_score)
                mining_weight = ((boundary_support - min_score) / max(1.0 - min_score, 1e-6)).clamp(0.0, 1.0)
            else:
                candidate_mask = support_mask & (boundary_support > 0.0)
                mining_weight = boundary_support
        else:
            raise ValueError(f"Unsupported hidden_target_mining_strategy: {strategy}")

        if self.hidden_target_gate_visible:
            candidate_mask = candidate_mask & (visible_confidence > 0.5)
        new_mask = candidate_mask & (~positive_mask)
        if not bool(new_mask.any()):
            return None

        strength = max(0.0, min(1.0, strength))
        valid_hidden_depth = torch.where(positive_mask, hidden_depth, torch.full_like(hidden_depth, 1e6))
        pooled_min_hidden_depth = -F.max_pool2d(-valid_hidden_depth, kernel_size=kernel, stride=1, padding=radius)
        fallback_hidden_depth = visible_depth + self.hidden_target_gap_min
        pooled_min_hidden_depth = torch.where(
            pooled_min_hidden_depth < 5e5,
            pooled_min_hidden_depth,
            fallback_hidden_depth,
        )
        pooled_min_hidden_depth = torch.maximum(pooled_min_hidden_depth, fallback_hidden_depth)

        mined_target = new_mask.float() * strength

        return {
            "positive_mask": positive_mask.float(),
            "new_mask": new_mask.float(),
            "mined_target": mined_target,
            "soft_target": torch.maximum(hidden_confidence, mined_target),
            "depth_target": torch.where(new_mask, pooled_min_hidden_depth, hidden_depth),
            "boundary_weight": mining_weight,
        }

    def _apply_hidden_layered_targets(self, targets):
        if not self.enable_hidden_layered_targets:
            return targets

        hidden_confidence = targets["hidden_confidence"].float()
        hidden_depth = targets["hidden_depth"].float()
        visible_depth = targets["visible_depth"].float()
        visible_confidence = targets["visible_confidence"].float()
        raw_confidence = targets.get("hidden_confidence_raw")
        raw_count = targets.get("hidden_count")
        raw_gap = targets.get("hidden_gap")

        boundary_score = self._compute_boundary_support(visible_depth, visible_confidence)
        boundary_mask = boundary_score >= self.hidden_layered_boundary_min_score
        depth_gap = raw_gap.float() if raw_gap is not None else (hidden_depth - visible_depth).clamp_min(0.0)

        if self.hidden_layered_use_raw_support and (raw_confidence is not None or raw_count is not None):
            support_strength = torch.zeros_like(hidden_confidence)
            if raw_confidence is not None:
                conf_scale = max(self.hidden_layered_raw_confidence_scale, 1e-6)
                support_strength = torch.maximum(support_strength, (raw_confidence.float() / conf_scale).clamp(0.0, 1.0))
            if raw_count is not None:
                count_scale = max(self.hidden_layered_raw_count_scale, 1e-6)
                count_strength = 1.0 - torch.exp(-raw_count.float() / count_scale)
                support_strength = torch.maximum(support_strength, count_strength.clamp(0.0, 1.0))
            support_mask = support_strength > 0.0
        else:
            support_strength = hidden_confidence.float()
            support_mask = hidden_confidence > 0.5

        local_mask = support_mask & boundary_mask & (depth_gap <= self.hidden_layered_local_gap_max)
        deep_mask = support_mask & (~local_mask)
        interior_negative = (visible_confidence > 0.5) & (~boundary_mask) & (~support_mask)

        updated = dict(targets)
        updated["hidden_boundary_support"] = boundary_score
        updated["hidden_local_support"] = torch.where(local_mask, support_strength, torch.zeros_like(support_strength))
        updated["hidden_deep_support"] = torch.where(deep_mask, support_strength, torch.zeros_like(support_strength))
        updated["hidden_interior_negative"] = interior_negative.float()
        return updated

    def _apply_hidden_target_mining(self, targets):
        mined = self._mine_hidden_targets(targets)
        if mined is None:
            return targets

        updated = dict(targets)
        updated["hidden_confidence_hard"] = updated["hidden_confidence"]
        updated["hidden_depth_hard"] = updated["hidden_depth"]
        updated["hidden_confidence_dilated"] = mined["new_mask"]
        updated["hidden_confidence_soft"] = mined["soft_target"]
        updated["hidden_depth_dilated"] = mined["depth_target"]
        updated["hidden_confidence_mining_weight"] = mined["boundary_weight"]

        mode = self.hidden_target_mining_mode
        if mode == "replace":
            updated["hidden_confidence"] = mined["soft_target"]
            updated["hidden_depth"] = mined["depth_target"]
        elif mode == "auxiliary":
            updated["hidden_confidence_mined_mask"] = mined["new_mask"]
            updated["hidden_confidence_mined_target"] = mined["mined_target"]
            updated["hidden_depth_mined"] = mined["depth_target"]
        elif mode in {"none", "disabled"}:
            pass
        else:
            raise ValueError(f"Unsupported hidden_target_mining_mode: {mode}")
        return updated

    def __call__(self, batch):
        available = batch.get("teacher_available")
        if available is not None and torch.is_tensor(available):
            if float(available.min().detach().cpu()) < 0.5:
                raise RuntimeError(
                    "Teacher targets are required but at least one batch item has no precomputed packet. "
                    "Set data.teacher_target_root, precompute packets, or use a config that does not require teacher training yet."
                )
        missing = self.REQUIRED_BATCH_KEYS.difference(batch.keys())
        if missing:
            raise KeyError(f"Missing teacher batch keys: {sorted(missing)}")
        targets = {
            "visible_depth": batch["teacher_visible_depth"],
            "visible_confidence": batch["teacher_visible_confidence"],
            "hidden_depth": batch["teacher_hidden_depth"],
            "hidden_confidence": batch["teacher_hidden_confidence"],
        }
        if "teacher_visible_normals" in batch:
            targets["visible_normals"] = batch["teacher_visible_normals"]
        if "teacher_hidden_confidence_raw" in batch:
            targets["hidden_confidence_raw"] = batch["teacher_hidden_confidence_raw"]
        if "teacher_hidden_count" in batch:
            targets["hidden_count"] = batch["teacher_hidden_count"]
        if "teacher_hidden_gap" in batch:
            targets["hidden_gap"] = batch["teacher_hidden_gap"]
        targets = self._apply_hidden_layered_targets(targets)
        return self._apply_hidden_target_mining(targets)


def build_teacher(config):
    if config.teacher.name == "none":
        return NullTeacherAdapter()
    if config.teacher.name == "oracle":
        return OracleTeacherAdapter()
    if config.teacher.name == "vggt_precomputed":
        return VGGTPrecomputedTeacherAdapter(config)
    raise ValueError(f"Unsupported teacher: {config.teacher.name}")
