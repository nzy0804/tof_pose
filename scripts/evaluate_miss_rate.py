from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


COUNT_FIELD_CANDIDATES = (
    "gt_count",
    "truth_count",
    "label_count",
    "expected_count",
    "ai_person_count",
    "pred_count",
    "detected_count",
    "person_count",
    "people_count",
    "count",
)
FRAME_FIELD_CANDIDATES = (
    "log_batch_index",
    "frame_id",
    "source_frame_id",
    "input_index",
    "image_id",
    "capture_timestamp_ms",
    "timestamp_ms",
    "sequence_id",
    "id",
)

MODEL_CONFIDENCE_PATTERN = re.compile(
    r"final_person_counts=\[(?P<counts>[^\]]*)\]"
)
INPUT_COUNT_PATTERN = re.compile(r"\binputs=(?P<inputs>\d+)\b")
LOG_TIMESTAMP_PATTERN = re.compile(r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def _read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]

    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"{path} JSON root must be a list")
        return [dict(item) for item in data]

    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(dict(json.loads(line)))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no} is not valid JSON") from exc
    return records


def _parse_count_list(value: str) -> list[int]:
    counts: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item or item == "-":
            continue
        try:
            counts.append(max(0, int(float(item))))
        except ValueError as exc:
            raise ValueError(f"invalid count {item!r} in final_person_counts=[{value}]") from exc
    return counts


def _normalize_log_time(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip().replace("T", " ")
    if len(normalized) == 16:
        normalized = f"{normalized}:00"
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", normalized):
        raise ValueError(f"invalid time {value!r}; expected 'YYYY-MM-DD HH:MM:SS'")
    return normalized


def _timestamp_from_log_line(line: str) -> str | None:
    match = LOG_TIMESTAMP_PATTERN.search(line)
    if not match:
        return None
    return match.group("timestamp")


def _in_time_range(timestamp: str | None, start_time: str | None, end_time: str | None) -> bool:
    if start_time is None and end_time is None:
        return True
    if timestamp is None:
        return False
    if start_time is not None and timestamp < start_time:
        return False
    if end_time is not None and timestamp > end_time:
        return False
    return True


def _read_prediction_log(
    path: Path,
    *,
    start_time: str | None = None,
    end_time: str | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    log_batch_index = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            if "Infer model confidence:" not in line:
                continue
            timestamp = _timestamp_from_log_line(line)
            if not _in_time_range(timestamp, start_time, end_time):
                continue
            count_match = MODEL_CONFIDENCE_PATTERN.search(line)
            if not count_match:
                continue
            counts = _parse_count_list(count_match.group("counts"))
            log_batch_index += 1
            input_match = INPUT_COUNT_PATTERN.search(line)
            input_count = int(input_match.group("inputs")) if input_match else len(counts)
            for input_index, pred_count in enumerate(counts):
                records.append(
                    {
                        "log_batch_index": log_batch_index,
                        "input_index": input_index,
                        "pred_count": pred_count,
                        "timestamp": timestamp,
                        "log_line_no": line_no,
                        "log_batch_input_count": input_count,
                    }
                )
    return records


def _get_field(record: dict[str, Any], field: str | None, candidates: tuple[str, ...]) -> Any:
    if field:
        return record.get(field)
    for candidate in candidates:
        if candidate in record:
            return record[candidate]
    return None


def _get_count(record: dict[str, Any], field: str | None) -> int:
    value = _get_field(record, field, COUNT_FIELD_CANDIDATES)
    if value is None or value == "":
        return 0
    try:
        return max(0, int(float(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid count value {value!r} in record {record!r}") from exc


def _get_key(record: dict[str, Any], fields: list[str]) -> tuple[str, ...]:
    values = []
    for field in fields:
        value = record.get(field)
        values.append("" if value is None else str(value))
    return tuple(values)


def _auto_key_fields(gt_records: list[dict[str, Any]], pred_records: list[dict[str, Any]]) -> list[str]:
    if not gt_records or not pred_records:
        return ["frame_id"]
    gt_keys = set(gt_records[0].keys())
    pred_keys = set(pred_records[0].keys())
    if {"log_batch_index", "input_index"}.issubset(gt_keys) and {"log_batch_index", "input_index"}.issubset(pred_keys):
        return ["log_batch_index", "input_index"]
    fields: list[str] = []
    if "device_id" in gt_keys and "device_id" in pred_keys:
        fields.append("device_id")
    for candidate in FRAME_FIELD_CANDIDATES:
        if candidate in gt_keys and candidate in pred_keys:
            fields.append(candidate)
            return fields
    return ["frame_id"]


def _parse_key_fields(value: str | None, gt_records: list[dict[str, Any]], pred_records: list[dict[str, Any]]) -> list[str]:
    if value:
        fields = [item.strip() for item in value.split(",") if item.strip()]
        if fields:
            return fields
    return _auto_key_fields(gt_records, pred_records)


def evaluate_by_key(
    gt_records: list[dict[str, Any]],
    pred_records: list[dict[str, Any]],
    *,
    key_fields: list[str],
    gt_count_field: str | None,
    pred_count_field: str | None,
) -> dict[str, Any]:
    pred_by_key: dict[tuple[str, ...], dict[str, Any]] = {}
    for record in pred_records:
        pred_by_key[_get_key(record, key_fields)] = record

    total_gt = 0
    total_missed = 0
    matched_frames = 0
    missing_prediction_frames = 0
    frames_with_miss = 0

    for gt_record in gt_records:
        key = _get_key(gt_record, key_fields)
        gt_count = _get_count(gt_record, gt_count_field)
        pred_record = pred_by_key.get(key)
        if pred_record is None:
            pred_count = 0
            missing_prediction_frames += 1
        else:
            pred_count = _get_count(pred_record, pred_count_field)
            matched_frames += 1
        missed = max(gt_count - pred_count, 0)
        total_gt += gt_count
        total_missed += missed
        if missed > 0:
            frames_with_miss += 1

    return {
        "match_by": "key",
        "key_fields": key_fields,
        "gt_frames": len(gt_records),
        "pred_frames": len(pred_records),
        "matched_frames": matched_frames,
        "missing_prediction_frames": missing_prediction_frames,
        "frames_with_miss": frames_with_miss,
        "gt_person_count": total_gt,
        "missed_person_count": total_missed,
        "miss_rate": (total_missed / total_gt) if total_gt else 0.0,
    }


def evaluate_by_row(
    gt_records: list[dict[str, Any]],
    pred_records: list[dict[str, Any]],
    *,
    gt_count_field: str | None,
    pred_count_field: str | None,
) -> dict[str, Any]:
    total_gt = 0
    total_missed = 0
    matched_frames = 0
    missing_prediction_frames = 0
    frames_with_miss = 0

    for index, gt_record in enumerate(gt_records):
        gt_count = _get_count(gt_record, gt_count_field)
        if index < len(pred_records):
            pred_count = _get_count(pred_records[index], pred_count_field)
            matched_frames += 1
        else:
            pred_count = 0
            missing_prediction_frames += 1
        missed = max(gt_count - pred_count, 0)
        total_gt += gt_count
        total_missed += missed
        if missed > 0:
            frames_with_miss += 1

    return {
        "match_by": "row",
        "gt_frames": len(gt_records),
        "pred_frames": len(pred_records),
        "matched_frames": matched_frames,
        "missing_prediction_frames": missing_prediction_frames,
        "frames_with_miss": frames_with_miss,
        "gt_person_count": total_gt,
        "missed_person_count": total_missed,
        "miss_rate": (total_missed / total_gt) if total_gt else 0.0,
    }


def _build_expected_records(pred_records: list[dict[str, Any]], expected_count: int) -> list[dict[str, Any]]:
    expected = max(0, int(expected_count))
    records: list[dict[str, Any]] = []
    for record in pred_records:
        item = dict(record)
        item["gt_count"] = expected
        records.append(item)
    return records


def _print_summary(result: dict[str, Any]) -> None:
    print(f"match_by: {result['match_by']}")
    if result["match_by"] == "key":
        print(f"key_fields: {','.join(result['key_fields'])}")
    print(f"gt_frames: {result['gt_frames']}")
    print(f"pred_frames: {result['pred_frames']}")
    print(f"matched_frames: {result['matched_frames']}")
    print(f"missing_prediction_frames: {result['missing_prediction_frames']}")
    print(f"frames_with_miss: {result['frames_with_miss']}")
    print(f"gt_person_count: {result['gt_person_count']}")
    print(f"missed_person_count: {result['missed_person_count']}")
    print(f"miss_rate: {result['miss_rate']:.6f}")
    print(f"miss_rate_percent: {result['miss_rate'] * 100:.2f}%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate person miss rate from ground-truth counts and prediction counts/logs.")
    parser.add_argument("--gt", type=Path, help="optional ground-truth CSV/JSONL/JSON file")
    parser.add_argument(
        "--expected-count",
        type=int,
        default=None,
        help="expected person count for every logged frame when --gt is omitted, for example 1 or 2",
    )
    pred_group = parser.add_mutually_exclusive_group(required=True)
    pred_group.add_argument("--pred", type=Path, help="prediction CSV/JSONL/JSON file")
    pred_group.add_argument("--pred-log", type=Path, help="AI service log file; parses final_person_counts from Infer model confidence lines")
    parser.add_argument("--start-time", default=None, help="inclusive log start time, for example '2026-07-09 14:55:00'")
    parser.add_argument("--end-time", default=None, help="inclusive log end time, for example '2026-07-09 15:00:00'")
    parser.add_argument("--match-by", choices=("key", "row"), default="key", help="match frames by key fields or row order")
    parser.add_argument("--key-fields", default=None, help="comma-separated key fields, for example device_id,frame_id")
    parser.add_argument("--gt-count-field", default=None, help="ground-truth count field; auto-detected when omitted")
    parser.add_argument("--pred-count-field", default=None, help="prediction count field; auto-detected when omitted")
    parser.add_argument("--json", action="store_true", help="print result as JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = _normalize_log_time(args.start_time)
    end_time = _normalize_log_time(args.end_time)
    if start_time is not None and end_time is not None and start_time > end_time:
        raise SystemExit("--start-time must be earlier than or equal to --end-time")
    if (start_time is not None or end_time is not None) and not args.pred_log:
        raise SystemExit("--start-time/--end-time are only supported with --pred-log")

    pred_records = (
        _read_prediction_log(args.pred_log, start_time=start_time, end_time=end_time)
        if args.pred_log
        else _read_records(args.pred)
    )
    if args.gt:
        gt_records = _read_records(args.gt)
    elif args.expected_count is not None:
        gt_records = _build_expected_records(pred_records, args.expected_count)
    else:
        raise SystemExit("missing --gt or --expected-count; without either one there is no baseline for miss rate")
    if args.match_by == "row":
        result = evaluate_by_row(
            gt_records,
            pred_records,
            gt_count_field=args.gt_count_field,
            pred_count_field=args.pred_count_field,
        )
    else:
        key_fields = _parse_key_fields(args.key_fields, gt_records, pred_records)
        result = evaluate_by_key(
            gt_records,
            pred_records,
            key_fields=key_fields,
            gt_count_field=args.gt_count_field,
            pred_count_field=args.pred_count_field,
        )
    if args.json:
        if args.pred_log:
            result["start_time"] = start_time
            result["end_time"] = end_time
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        if args.pred_log:
            if start_time is not None:
                print(f"start_time: {start_time}")
            if end_time is not None:
                print(f"end_time: {end_time}")
        _print_summary(result)


if __name__ == "__main__":
    main()
