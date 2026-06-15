#!/usr/bin/env python3
"""Run GRAE inference on TVA pseudo-3D pickle and summarize tracklets."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
import os
import sys
import types
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

if importlib.util.find_spec("nuscenes") is None:
    nuscenes_module = types.ModuleType("nuscenes")
    nuscenes_submodule = types.ModuleType("nuscenes.nuscenes")

    class _DummyNuScenes:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("NuScenes is unavailable; this runner only uses BasePredictor._test().")

    nuscenes_submodule.NuScenes = _DummyNuScenes
    sys.modules["nuscenes"] = nuscenes_module
    sys.modules["nuscenes.nuscenes"] = nuscenes_submodule

import dataset.base as module_dataloader  # noqa: E402
import models as module_arch  # noqa: E402
from dataset.nuscenes_dataset import NusceneseValDataset  # noqa: E402
from trainer.predictor import BasePredictor  # noqa: E402
from utils import read_json  # noqa: E402
from utils.device import resolve_device  # noqa: E402


DEFAULT_CHECKPOINT = (
    "outputs/tva_nyx650_pseudo_smoke/"
    "tva_nyx650_pseudo_smoke/models/checkpoint-epoch0.pth"
)
DEFAULT_ANN_FILE = "data/tva_nyx650/grae_train_tva_nyx650_pseudo3d_smoke.pickle"
DEFAULT_OUTPUT_JSON = (
    "outputs/tva_nyx650_pseudo_smoke/"
    "tva_nyx650_pseudo_smoke/inference/pred_tracklets.json"
)
DEFAULT_SUMMARY_JSON = (
    "outputs/tva_nyx650_pseudo_smoke/"
    "tva_nyx650_pseudo_smoke/inference/tracklet_summary.json"
)
DEFAULT_BBOX3D_JSONL_ENV = "TVA_GRAE_BBOX3D_JSONL"
DEFAULT_TRACKING_SCHEMA = "grae_3dmot.tracking_result/v1"
CLASS_NAMES = [
    "car",
    "pedestrian",
    "bicycle",
    "bus",
    "motorcycle",
    "trailer",
    "truck",
]


def dump_json(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def report_path(path: str | Path | None, external_label: str) -> str | None:
    if path is None:
        return None
    raw_path = Path(path)
    try:
        return str(raw_path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except (OSError, ValueError):
        if raw_path.is_absolute():
            return f"<external:{external_label}>"
        return str(raw_path)


def normalize_tracking_id(raw: Any) -> str:
    """Normalize predictor IDs such as '[12]' into stable string IDs."""
    text = str(raw)
    try:
        value = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text
    arr = np.asarray(value).reshape(-1)
    if arr.size == 1:
        return str(int(arr[0]))
    return text


def _finite_values(values: list[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def bbox_xywh_to_xyxy(bbox: list[float] | tuple[float, ...]) -> list[float]:
    x, y, w, h = [float(value) for value in bbox]
    return [x, y, x + w, y + h]


def load_bbox3d_ground_truth(path: str | Path | None) -> dict[str, list[dict[str, Any]]]:
    if path is None:
        return {}
    path = Path(path)
    if not path.is_file():
        return {}

    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record.get("type") != "bbox3d":
                continue
            by_sample[str(record["sample_token"])].append(record)
    return by_sample


def match_prediction_records_to_bbox3d(
    results: dict[str, list[dict[str, Any]]],
    gt_by_sample: dict[str, list[dict[str, Any]]],
    *,
    max_distance_m: float,
) -> dict[str, list[dict[str, Any]]]:
    assignments: dict[str, list[dict[str, Any]]] = {}

    for sample_token, predictions in results.items():
        gt_records = gt_by_sample.get(sample_token, [])
        available = set(range(len(gt_records)))
        sample_assignments: list[dict[str, Any]] = []

        for pred_index, pred in enumerate(predictions):
            pred_xyz = np.asarray(pred["translation"], dtype=np.float64)
            best_idx = None
            best_distance = float("inf")
            for idx in available:
                gt_xyz = np.asarray(gt_records[idx]["translation"], dtype=np.float64)
                distance = float(np.linalg.norm(pred_xyz - gt_xyz))
                if distance < best_distance:
                    best_idx = idx
                    best_distance = distance

            gt_record = None
            if best_idx is not None and best_distance <= max_distance_m:
                available.remove(best_idx)
                gt_record = gt_records[best_idx]

            sample_assignments.append(
                {
                    "prediction_index": pred_index,
                    "prediction": pred,
                    "bbox3d_record": gt_record,
                    "match_distance_m": best_distance if gt_record is not None else None,
                }
            )

        assignments[sample_token] = sample_assignments

    return assignments


def match_predictions_to_bbox3d(
    results: dict[str, list[dict[str, Any]]],
    gt_by_sample: dict[str, list[dict[str, Any]]],
    *,
    max_distance_m: float,
) -> dict[str, Any]:
    pred_to_gt_counts: dict[str, Counter[int]] = defaultdict(Counter)
    matched = 0
    unmatched = 0
    distances: list[float] = []

    for sample_token, predictions in results.items():
        gt_records = gt_by_sample.get(sample_token, [])
        if not gt_records:
            unmatched += len(predictions)
            continue

        available = set(range(len(gt_records)))
        for pred in predictions:
            pred_xyz = np.asarray(pred["translation"], dtype=np.float64)
            best_idx = None
            best_distance = float("inf")
            for idx in available:
                gt_xyz = np.asarray(gt_records[idx]["translation"], dtype=np.float64)
                distance = float(np.linalg.norm(pred_xyz - gt_xyz))
                if distance < best_distance:
                    best_idx = idx
                    best_distance = distance
            if best_idx is not None and best_distance <= max_distance_m:
                available.remove(best_idx)
                matched += 1
                distances.append(best_distance)
                pred_tid = normalize_tracking_id(pred["tracking_id"])
                pred_to_gt_counts[pred_tid][int(gt_records[best_idx]["track_id"])] += 1
            else:
                unmatched += 1

    purities: list[float] = []
    dominant_gt_by_track: dict[str, dict[str, Any]] = {}
    for pred_tid, counts in pred_to_gt_counts.items():
        total = sum(counts.values())
        if total <= 0:
            continue
        gt_id, dominant_count = counts.most_common(1)[0]
        purity = dominant_count / total
        purities.append(purity)
        dominant_gt_by_track[pred_tid] = {
            "gt_track_id": gt_id,
            "matched_count": total,
            "dominant_count": dominant_count,
            "purity": purity,
        }

    return {
        "matched_predictions": matched,
        "unmatched_predictions": unmatched,
        "matched_fraction": matched / (matched + unmatched) if matched + unmatched else None,
        "distance_m": {
            "max": max(distances) if distances else None,
            "mean": float(np.mean(distances)) if distances else None,
            "median": float(np.median(distances)) if distances else None,
        },
        "dominant_gt_track_purity": {
            "mean": float(np.mean(purities)) if purities else None,
            "median": float(np.median(purities)) if purities else None,
            "min": min(purities) if purities else None,
        },
        "dominant_gt_by_pred_track": dominant_gt_by_track,
    }


def class_id_from_prediction(pred: dict[str, Any], gt_record: dict[str, Any] | None) -> int:
    if gt_record is not None and "class_id" in gt_record:
        return int(gt_record["class_id"])
    tracking_name = pred.get("tracking_name")
    if tracking_name in CLASS_NAMES:
        return CLASS_NAMES.index(tracking_name)
    return 0


def build_tracking_result(
    result: dict[str, Any],
    *,
    bbox3d_jsonl: str | Path | None,
    match_distance_m: float,
    tracker_params: dict[str, Any],
) -> dict[str, Any]:
    results = result.get("results", {})
    gt_by_sample = load_bbox3d_ground_truth(bbox3d_jsonl)
    assignments = match_prediction_records_to_bbox3d(
        results,
        gt_by_sample,
        max_distance_m=match_distance_m,
    )
    track_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "frame_indices": [],
            "scores": [],
            "source_track_ids": Counter(),
        }
    )

    frames: list[dict[str, Any]] = []
    for fallback_frame_index, (sample_token, sample_assignments) in enumerate(assignments.items()):
        matched_records = [
            assignment["bbox3d_record"]
            for assignment in sample_assignments
            if assignment["bbox3d_record"] is not None
        ]
        first_gt = matched_records[0] if matched_records else None
        frame_index = (
            int(first_gt["frame_index"])
            if first_gt is not None and "frame_index" in first_gt
            else fallback_frame_index
        )
        timestamp = (
            float(first_gt["timestamp"])
            if first_gt is not None and "timestamp" in first_gt
            else None
        )
        image = {
            "image_id": int(first_gt["image_id"]) if first_gt is not None and "image_id" in first_gt else None,
            "file_name": str(first_gt["image_file"]) if first_gt is not None and first_gt.get("image_file") else None,
        }

        detections: list[dict[str, Any]] = []
        for assignment in sample_assignments:
            pred = assignment["prediction"]
            gt_record = assignment["bbox3d_record"]
            track_id = normalize_tracking_id(pred["tracking_id"])
            score = float(pred["tracking_score"])
            class_id = class_id_from_prediction(pred, gt_record)
            class_name = pred.get("tracking_name") or f"class_{class_id}"
            bbox_xywh = None
            bbox_xyxy = None
            source_detection = None
            if gt_record is not None:
                bbox_xywh = [float(value) for value in gt_record["bbox_xywh"]]
                bbox_xyxy = bbox_xywh_to_xyxy(bbox_xywh)
                source_detection = {
                    "annotation_id": int(gt_record["annotation_id"]),
                    "input_track_id": int(gt_record["track_id"]),
                    "score": float(gt_record["score"]),
                    "depth_m": float(gt_record["depth_m"]),
                    "depth_source": str(gt_record["depth_source"]),
                    "valid_depth_pixels": int(gt_record["valid_depth_pixels"]),
                    "match_distance_m": float(assignment["match_distance_m"]),
                }
                track_stats[track_id]["source_track_ids"][int(gt_record["track_id"])] += 1

            detections.append(
                {
                    "track_id": track_id,
                    "class_id": class_id,
                    "class_name": class_name,
                    "score": score,
                    "bbox": {
                        "xywh": bbox_xywh,
                        "xyxy": bbox_xyxy,
                        "source": "bbox3d_jsonl" if gt_record is not None else None,
                    },
                    "bbox3d": {
                        "translation": [float(value) for value in pred["translation"]],
                        "size": [float(value) for value in pred["size"]],
                        "rotation": [float(value) for value in pred["rotation"]],
                        "velocity": [float(value) for value in pred["velocity"]],
                    },
                    "source_detection": source_detection,
                }
            )
            track_stats[track_id]["frame_indices"].append(frame_index)
            track_stats[track_id]["scores"].append(score)

        frames.append(
            {
                "frame_index": frame_index,
                "sample_token": sample_token,
                "timestamp": timestamp,
                "image": image,
                "detections": detections,
            }
        )

    tracks: list[dict[str, Any]] = []
    for track_id, stats in sorted(track_stats.items(), key=lambda item: int(item[0]) if item[0].isdigit() else item[0]):
        frame_indices = stats["frame_indices"]
        scores = stats["scores"]
        source_counts: Counter[int] = stats["source_track_ids"]
        dominant_source = source_counts.most_common(1)[0] if source_counts else None
        tracks.append(
            {
                "track_id": track_id,
                "frame_count": len(set(frame_indices)),
                "first_frame_index": min(frame_indices) if frame_indices else None,
                "last_frame_index": max(frame_indices) if frame_indices else None,
                "mean_score": float(np.mean(scores)) if scores else None,
                "dominant_input_track_id": int(dominant_source[0]) if dominant_source else None,
                "dominant_input_track_purity": (
                    dominant_source[1] / sum(source_counts.values()) if dominant_source else None
                ),
            }
        )

    return {
        "schema": DEFAULT_TRACKING_SCHEMA,
        "schema_file": "schemas/grae_tracking_result_v1.schema.json",
        "coordinate_frame": {
            "translation": "[x_camera, z_depth, -y_camera]",
            "size": "[bbox_metric_width, estimated_depth_extent, bbox_metric_height]",
            "rotation": "quaternion [w, x, y, z]",
            "bbox2d": "pixel coordinates in source image",
        },
        "source": {
            "model": "GRAE-3DMOT",
            "bbox3d_jsonl": report_path(bbox3d_jsonl, "bbox3d_jsonl"),
            "match_distance_m": match_distance_m,
        },
        "tracker_params": tracker_params,
        "frames": sorted(frames, key=lambda frame: frame["frame_index"]),
        "tracks": tracks,
    }


def summarize_tracklets(
    result: dict[str, Any],
    *,
    bbox3d_jsonl: str | Path | None,
    match_distance_m: float,
    min_dominant_gt_purity_median: float,
    require_bbox3d_validation: bool,
) -> dict[str, Any]:
    results = result.get("results", {})
    frame_counts: dict[str, int] = {}
    duplicate_frame_track_ids = 0
    invalid_numeric_records = 0
    invalid_size_records = 0
    track_frames: dict[str, set[str]] = defaultdict(set)
    track_scores: dict[str, list[float]] = defaultdict(list)

    for sample_token, predictions in results.items():
        frame_counts[sample_token] = len(predictions)
        ids_this_frame: list[str] = []
        for pred in predictions:
            track_id = normalize_tracking_id(pred["tracking_id"])
            ids_this_frame.append(track_id)
            track_frames[track_id].add(sample_token)
            track_scores[track_id].append(float(pred["tracking_score"]))

            vector_values = (
                list(pred["translation"])
                + list(pred["size"])
                + list(pred["rotation"])
                + list(pred["velocity"])
                + [float(pred["tracking_score"])]
            )
            if not _finite_values(vector_values):
                invalid_numeric_records += 1
            if any(float(value) <= 0 for value in pred["size"]):
                invalid_size_records += 1

        duplicate_frame_track_ids += len(ids_this_frame) - len(set(ids_this_frame))

    track_lengths = {track_id: len(frames) for track_id, frames in track_frames.items()}
    length_values = list(track_lengths.values())
    gt_by_sample = load_bbox3d_ground_truth(bbox3d_jsonl)
    matching = match_predictions_to_bbox3d(
        results, gt_by_sample, max_distance_m=match_distance_m
    ) if gt_by_sample else None

    total_predictions = sum(frame_counts.values())
    checks = {
        "all_frames_have_predictions": bool(frame_counts) and min(frame_counts.values()) > 0,
        "no_duplicate_track_id_within_frame": duplicate_frame_track_ids == 0,
        "all_numeric_fields_finite": invalid_numeric_records == 0,
        "all_sizes_positive": invalid_size_records == 0,
        "has_tracklet_len_ge_10": any(length >= 10 for length in length_values),
        "has_at_least_10_tracklets_len_ge_3": sum(length >= 3 for length in length_values) >= 10,
    }
    if require_bbox3d_validation:
        checks["bbox3d_validation_available"] = matching is not None
    if matching is not None:
        checks["bbox3d_match_fraction_ge_0_95"] = (
            matching["matched_fraction"] is not None
            and matching["matched_fraction"] >= 0.95
        )
        checks["dominant_gt_purity_median_ge_threshold"] = (
            matching["dominant_gt_track_purity"]["median"] is not None
            and matching["dominant_gt_track_purity"]["median"]
            >= min_dominant_gt_purity_median
        )

    return {
        "frames": len(frame_counts),
        "total_predictions": total_predictions,
        "predicted_tracklets": len(track_lengths),
        "tracklet_length": {
            "min": min(length_values) if length_values else 0,
            "median": float(np.median(length_values)) if length_values else 0,
            "max": max(length_values) if length_values else 0,
            "count_ge_3": sum(length >= 3 for length in length_values),
            "count_ge_10": sum(length >= 10 for length in length_values),
            "top10": sorted(track_lengths.items(), key=lambda item: item[1], reverse=True)[:10],
        },
        "frame_detection_count": {
            "min": min(frame_counts.values()) if frame_counts else 0,
            "median": float(np.median(list(frame_counts.values()))) if frame_counts else 0,
            "max": max(frame_counts.values()) if frame_counts else 0,
        },
        "duplicate_frame_track_ids": duplicate_frame_track_ids,
        "invalid_numeric_records": invalid_numeric_records,
        "invalid_size_records": invalid_size_records,
        "bbox3d_matching": matching,
        "min_dominant_gt_purity_median": min_dominant_gt_purity_median,
        "require_bbox3d_validation": require_bbox3d_validation,
        "checks": checks,
        "passed_basic_tracklet_checks": all(checks.values()),
    }


def load_model(config_path: str | Path, checkpoint_path: str | Path, device: torch.device):
    config = read_json(config_path)
    model_type = config["arch"]["type"]
    model_args = dict(config["arch"]["args"])
    model_args["device"] = str(device)
    model = getattr(module_arch, model_type)(**model_args)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        stripped = {
            key.removeprefix("module."): value for key, value in state_dict.items()
        }
        model.load_state_dict(stripped, strict=True)
    return model.to(device).eval(), config


def make_predictor(
    model,
    data_loader,
    device: torch.device,
    *,
    alpha: float,
    conf_th: float,
    age: int,
    high_cost_limit: float,
    low_cost_limit: float,
    emit_fresh_unmatched_tracks: bool,
) -> BasePredictor:
    predictor = BasePredictor.__new__(BasePredictor)
    predictor.device = device
    predictor.data_loader = data_loader
    predictor.model = model
    predictor.iter_time = 0
    predictor.alpha = alpha
    predictor.conf_th = conf_th
    predictor.age = age
    predictor.high_cost_limit = high_cost_limit
    predictor.low_cost_limit = low_cost_limit
    predictor.emit_fresh_unmatched_tracks = emit_fresh_unmatched_tracks
    predictor.outputs = None
    predictor.save_notmatched_track = False
    predictor.class_names = [
        "car",
        "pedestrian",
        "bicycle",
        "bus",
        "motorcycle",
        "trailer",
        "truck",
    ]
    predictor.total_time = 0
    return predictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TVA pseudo-3D GRAE inference and validate tracklet output."
    )
    parser.add_argument("--config", default="config/tva_nyx650_pseudo_smoke.json")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--ann-file", default=DEFAULT_ANN_FILE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--summary-json", default=DEFAULT_SUMMARY_JSON)
    parser.add_argument(
        "--tracking-json",
        default=None,
        help="Optional normalized tracking-result JSON path for downstream visualization.",
    )
    parser.add_argument(
        "--bbox3d-jsonl",
        default=os.environ.get(DEFAULT_BBOX3D_JSONL_ENV),
        help=(
            "Optional pseudo-3D ground-truth JSONL for tracklet validation. "
            f"Defaults to ${DEFAULT_BBOX3D_JSONL_ENV} when set."
        ),
    )
    parser.add_argument(
        "--require-bbox3d-validation",
        action="store_true",
        help="Fail the summary checks when --bbox3d-jsonl is absent or unreadable.",
    )
    parser.add_argument("--match-distance-m", type=float, default=0.15)
    parser.add_argument("--min-dominant-gt-purity-median", type=float, default=0.8)
    parser.add_argument("--alpha", type=float, default=0.24)
    parser.add_argument("--conf-th", type=float, default=0.12)
    parser.add_argument("--age", type=int, default=12)
    parser.add_argument("--high-cost-limit", type=float, default=0.9)
    parser.add_argument("--low-cost-limit", type=float, default=0.8)
    parser.add_argument(
        "--no-emit-fresh-unmatched-tracks",
        action="store_true",
        help="Do not emit aged unmatched tracks in addition to current detections.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device, args.device)
    model, _config = load_model(args.config, args.checkpoint, device)
    dataset = NusceneseValDataset(args.ann_file)
    data_loader = module_dataloader.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    predictor = make_predictor(
        model,
        data_loader,
        device,
        alpha=args.alpha,
        conf_th=args.conf_th,
        age=args.age,
        high_cost_limit=args.high_cost_limit,
        low_cost_limit=args.low_cost_limit,
        emit_fresh_unmatched_tracks=not args.no_emit_fresh_unmatched_tracks,
    )

    with torch.no_grad():
        result = predictor._test()

    summary = summarize_tracklets(
        result,
        bbox3d_jsonl=args.bbox3d_jsonl,
        match_distance_m=args.match_distance_m,
        min_dominant_gt_purity_median=args.min_dominant_gt_purity_median,
        require_bbox3d_validation=args.require_bbox3d_validation,
    )
    tracker_params = {
        "alpha": args.alpha,
        "conf_th": args.conf_th,
        "age": args.age,
        "high_cost_limit": args.high_cost_limit,
        "low_cost_limit": args.low_cost_limit,
        "emit_fresh_unmatched_tracks": not args.no_emit_fresh_unmatched_tracks,
    }
    summary.update(
        {
            "config": report_path(args.config, "config"),
            "checkpoint": report_path(args.checkpoint, "checkpoint"),
            "ann_file": report_path(args.ann_file, "ann_file"),
            "output_json": report_path(args.output_json, "output_json"),
            "tracking_json": report_path(args.tracking_json, "tracking_json"),
            "bbox3d_jsonl": report_path(args.bbox3d_jsonl, "bbox3d_jsonl"),
            "match_distance_m": args.match_distance_m,
            "tracker_params": tracker_params,
        }
    )

    dump_json(result, args.output_json)
    if args.tracking_json:
        tracking_result = build_tracking_result(
            result,
            bbox3d_jsonl=args.bbox3d_jsonl,
            match_distance_m=args.match_distance_m,
            tracker_params=tracker_params,
        )
        dump_json(tracking_result, args.tracking_json)
    dump_json(summary, args.summary_json)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed_basic_tracklet_checks"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
