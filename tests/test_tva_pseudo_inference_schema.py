from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_tva_pseudo_inference.py"
spec = importlib.util.spec_from_file_location("run_tva_pseudo_inference", SCRIPT_PATH)
runner = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


def test_build_tracking_result_includes_2d_bbox_and_track_summary(tmp_path):
    bbox3d_jsonl = tmp_path / "bbox3d.jsonl"
    rows = [
        {"type": "metadata"},
        {
            "type": "bbox3d",
            "frame_index": 0,
            "image_id": 10,
            "image_file": "frame_00010.jpg",
            "sample_token": "frame_00010.jpg",
            "timestamp": 1.0,
            "annotation_id": 100,
            "track_id": 7,
            "class_id": 0,
            "score": 0.9,
            "bbox_xywh": [2.0, 3.0, 4.0, 5.0],
            "depth_m": 1.2,
            "depth_source": "mm",
            "valid_depth_pixels": 20,
            "translation": [0.1, 1.2, -0.2],
            "size": [0.3, 0.02, 0.4],
            "rotation": [1.0, 0.0, 0.0, 0.0],
            "velocity": [0.0, 0.0],
        },
    ]
    bbox3d_jsonl.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )
    result = {
        "results": {
            "frame_00010.jpg": [
                {
                    "sample_token": "frame_00010.jpg",
                    "translation": [0.1, 1.2, -0.2],
                    "size": [0.3, 0.02, 0.4],
                    "rotation": [1.0, 0.0, 0.0, 0.0],
                    "velocity": [0.0, 0.0],
                    "tracking_id": "[42]",
                    "tracking_name": "car",
                    "tracking_score": 0.88,
                }
            ]
        }
    }

    payload = runner.build_tracking_result(
        result,
        bbox3d_jsonl=bbox3d_jsonl,
        match_distance_m=0.15,
        tracker_params={"high_cost_limit": 0.5},
    )

    assert payload["schema"] == "grae_3dmot.tracking_result/v1"
    assert payload["source"]["bbox3d_jsonl"] == "<external:bbox3d_jsonl>"
    assert payload["frames"][0]["image"]["file_name"] == "frame_00010.jpg"
    det = payload["frames"][0]["detections"][0]
    assert det["track_id"] == "42"
    assert det["bbox"]["xywh"] == [2.0, 3.0, 4.0, 5.0]
    assert det["bbox"]["xyxy"] == [2.0, 3.0, 6.0, 8.0]
    assert det["source_detection"]["annotation_id"] == 100
    assert det["source_detection"]["input_track_id"] == 7
    assert det["source_detection"]["match_distance_m"] == 0.0
    assert payload["tracks"][0]["track_id"] == "42"
    assert payload["tracks"][0]["dominant_input_track_id"] == 7
    assert payload["tracks"][0]["dominant_input_track_purity"] == 1.0


def test_bbox3d_validation_required_fails_when_jsonl_missing():
    summary = runner.summarize_tracklets(
        {"results": {"frame": []}},
        bbox3d_jsonl=None,
        match_distance_m=0.15,
        min_dominant_gt_purity_median=0.8,
        require_bbox3d_validation=True,
    )

    assert summary["checks"]["bbox3d_validation_available"] is False
    assert summary["passed_basic_tracklet_checks"] is False
