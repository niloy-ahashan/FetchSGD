#!/usr/bin/env python3
"""
Convert ActionNet raw streamLog HDF5 files to the preprocessed layout expected by
MFedMC ActionSense (per-modality example_matrices_* datasets).

Raw files (device/stream/data time series) cannot be passed directly to main.py.
This script applies the same style of filtering, normalization, resampling (10 Hz),
and 10 s windowing as ActionNet's example_activity_classification pipeline, but:
  - Keeps tactile at 32x32 per timestep (required by Tactile_LSTM's 32*32 input).
  - Writes separate example_matrices_* groups matching dataset.load_and_restructure_hdf5_data.
  - Stores example_label_indexes as 1-based class ids (what MFedMC's loader expects).

Usage (single session):
  python preprocess_actionnet_streamlog.py \\
    --input /path/to/streamLog_....hdf5 \\
    --output ./data/out.hdf5 \\
    --subject-id S00

Usage (merge multiple streamLogs — e.g. several S00 sessions + S01 + S02 + S03):
  python preprocess_actionnet_streamlog.py \\
    --input session_a.hdf5 session_b.hdf5 \\
    --subject-id S00 S00 \\
    --output ./data/merged_clients.hdf5

  python preprocess_actionnet_streamlog.py \\
    --input s00a.hdf5 s00b.hdf5 s01.hdf5 s02a.hdf5 s03a.hdf5 \\
    --subject-id S00 S00 S01 S02 S03 \\
    --output ./data/multi_subject.hdf5

Use every streamLog in a folder (subject id parsed from ..._wearables_SXX.hdf5):
  python preprocess_actionnet_streamlog.py \\
    --input-glob '/path/to/dataset/*streamLog_actionNet-wearables_*.hdf5' \\
    --output ./data/all_subjects.hdf5

Paper-style bundle (5 subjects, canonical filename — pick ids that exist in your dataset):
  python preprocess_actionnet_streamlog.py \\
    --input-glob '/path/to/dataset/*streamLog_actionNet-wearables_*.hdf5' \\
    --only-subjects S00 S01 S02 S03 S04 \\
    --canonical-output \\
    --output ./data

Label ids are remapped once over the union of all examples (MFedMC 20-class cap).

Paper-aligned recipe (same as ActionNet 01_create_examples + MFedMC ActionSense layout):
  - All six streams: gaze position, Myo L/R EMG, tactile L/R (32x32), Xsens joint rotations (22 joints).
  - Resampling 10 Hz, 10 s windows (100 timesteps), 2 s buffers inside each activity.
  - Up to 20 windows per activity class per subject-session, 20 "None" baseline windows.
  - After merging sessions, subsample so each subject has at most 20 examples per class (see
    --max-per-class-per-subject; baseline "None" counts as one class).
  - Full kitchen activity list in `activities_to_classify` (allActs).

Use --only-subjects to match the paper's "5subj" (pick five client ids). Use --canonical-output
to write data_processed_allStreams_10s_10hz_<K>subj_ex20-20_allActs.hdf5 under --output/.
"""

from __future__ import annotations

import argparse
import glob as glob_module
import os
import re
import time
from collections import OrderedDict, defaultdict

import h5py
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.signal import butter, lfilter

# ---------------------------------------------------------------------------
# Configuration (aligned with ActionNet 01_create_examples.py)
# ---------------------------------------------------------------------------

baseline_label = "None"
activities_to_classify = [
    baseline_label,
    "Get/replace items from refrigerator/cabinets/drawers",
    "Peel a cucumber",
    "Clear cutting board",
    "Slice a cucumber",
    "Peel a potato",
    "Slice a potato",
    "Slice bread",
    "Spread almond butter on a bread slice",
    "Spread jelly on a bread slice",
    "Open/close a jar of almond butter",
    "Pour water from a pitcher into a glass",
    "Clean a plate with a sponge",
    "Clean a plate with a towel",
    "Clean a pan with a sponge",
    "Clean a pan with a towel",
    "Get items from cabinets: 3 each large/small plates, bowls, mugs, glasses, sets of utensils",
    "Set table: 3 each large/small plates, bowls, mugs, glasses, sets of utensils",
    "Stack on table: 3 each large/small plates, bowls",
    "Load dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils",
    "Unload dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils",
]
baseline_index = activities_to_classify.index(baseline_label)
activities_renamed = {
    "Open/close a jar of almond butter": ["Open a jar of almond butter"],
    "Get/replace items from refrigerator/cabinets/drawers": [
        "Get items from refrigerator/cabinets/drawers"
    ],
}

resampled_Fs = 10
num_segments_per_subject = 20
num_baseline_segments_per_subject = 20
segment_duration_s = 10
segment_length = int(round(resampled_Fs * segment_duration_s))
buffer_startActivity_s = 2
buffer_endActivity_s = 2

filter_cutoff_emg_Hz = 5
filter_cutoff_tactile_Hz = 2
filter_cutoff_gaze_Hz = 5


def lowpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 5) -> np.ndarray:
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return lfilter(b, a, data.T).T


def load_streamlog(path: str) -> dict:
    """Load one ActionNet HDF5 into nested dict device -> stream -> {time_s, data}."""
    out: dict = {}
    with h5py.File(path, "r") as hdf_file:
        device_name = "experiment-activities"
        stream_name = "activities"
        out.setdefault(device_name, {})
        out[device_name].setdefault(stream_name, {})
        for key in ["time_s", "data"]:
            out[device_name][stream_name][key] = hdf_file[device_name][stream_name][key][:]
        streams = [
            ("eye-tracking-gaze", "position"),
            ("myo-left", "emg"),
            ("myo-right", "emg"),
            ("tactile-glove-left", "tactile_data"),
            ("tactile-glove-right", "tactile_data"),
            ("xsens-joints", "rotation_xzy_deg"),
        ]
        for device_name, stream_name in streams:
            out.setdefault(device_name, {})
            out[device_name].setdefault(stream_name, {})
            for key in ["time_s", "data"]:
                out[device_name][stream_name][key] = hdf_file[device_name][stream_name][key][:]
    return out


def filter_and_normalize(file_data: dict) -> None:
    for myo_key in ["myo-left", "myo-right"]:
        t = file_data[myo_key]["emg"]["time_s"]
        t = np.squeeze(np.array(t))
        fs = (t.size - 1) / (t[-1] - t[0])
        data_stream = file_data[myo_key]["emg"]["data"][:, :]
        y = np.abs(data_stream.astype(float))
        y = lowpass_filter(y, filter_cutoff_emg_Hz, fs)
        file_data[myo_key]["emg"]["data"] = y

    for tactile_key in ["tactile-glove-left", "tactile-glove-right"]:
        t = file_data[tactile_key]["tactile_data"]["time_s"]
        t = np.squeeze(np.array(t))
        fs = (t.size - 1) / (t[-1] - t[0])
        data_stream = file_data[tactile_key]["tactile_data"]["data"][:, :, :]
        y = data_stream.astype(float)
        y = lowpass_filter(y.reshape(y.shape[0], -1), filter_cutoff_tactile_Hz, fs)
        y = y.reshape(data_stream.shape)
        y[0 : int(fs * 30), :, :] = np.mean(y, axis=0)
        y[y.shape[0] - int(fs * 30) : y.shape[0] + 1, :, :] = np.mean(y, axis=0)
        file_data[tactile_key]["tactile_data"]["data"] = y

    if "eye-tracking-gaze" in file_data:
        t = np.squeeze(np.array(file_data["eye-tracking-gaze"]["position"]["time_s"]))
        fs = (t.size - 1) / (t[-1] - t[0])
        y = file_data["eye-tracking-gaze"]["position"]["data"][:, :].astype(float)
        clip_low, clip_high = 0.05, 0.95
        y = np.clip(y, clip_low, clip_high)
        y[y == clip_low] = np.nan
        y[y == clip_high] = np.nan
        y = pd.DataFrame(y).interpolate(method="zero").to_numpy()
        y[np.isnan(y)] = 0.5
        y = lowpass_filter(y, filter_cutoff_gaze_Hz, fs)
        file_data["eye-tracking-gaze"]["position"]["data"] = y

    for myo_key in ["myo-left", "myo-right"]:
        y = file_data[myo_key]["emg"]["data"][:, :]
        y = y / ((np.amax(y) - np.amin(y)) / 2)
        y = y - np.amin(y) - 1
        file_data[myo_key]["emg"]["data"] = y

    for tactile_key in ["tactile-glove-left", "tactile-glove-right"]:
        y = file_data[tactile_key]["tactile_data"]["data"][:, :, :]
        mean_val, std_dev = np.mean(y), np.std(y)
        clip_low = mean_val - 2 * std_dev
        clip_high = mean_val + 3 * std_dev
        y = np.clip(y, clip_low, clip_high)
        file_data[tactile_key]["tactile_data"]["data"] = y
        y = file_data[tactile_key]["tactile_data"]["data"]
        y = y / ((np.amax(y) - np.amin(y)) / 2)
        y = y - np.amin(y) - 1
        file_data[tactile_key]["tactile_data"]["data"] = y

    if "xsens-joints" in file_data:
        y = file_data["xsens-joints"]["rotation_xzy_deg"]["data"][:, :, :].astype(float)
        y = y / ((180.0 - (-180.0)) / 2)
        file_data["xsens-joints"]["rotation_xzy_deg"]["data"] = y

    if "eye-tracking-gaze" in file_data:
        y = file_data["eye-tracking-gaze"]["position"]["data"][:]
        clip_low, clip_high = 0.05, 0.95
        y = np.clip(y, clip_low, clip_high)
        y = (y - np.mean([clip_low, clip_high])) / ((clip_high - clip_low) / 2)
        file_data["eye-tracking-gaze"]["position"]["data"] = y


def resample_stream(file_data: dict, device_name: str, stream_name: str) -> None:
    data = np.squeeze(np.array(file_data[device_name][stream_name]["data"]))
    time_s = np.squeeze(np.array(file_data[device_name][stream_name]["time_s"]))
    target_time_s = np.linspace(
        time_s[0],
        time_s[-1],
        num=int(round(1 + resampled_Fs * (time_s[-1] - time_s[0]))),
        endpoint=True,
    )
    fn = interpolate.interp1d(
        time_s,
        data,
        axis=0,
        kind="linear",
        fill_value="extrapolate",
    )
    data_resampled = fn(target_time_s)
    if np.any(np.isnan(data_resampled)):
        data_resampled[np.isnan(data_resampled)] = 0
    file_data[device_name][stream_name]["time_s"] = target_time_s
    file_data[device_name][stream_name]["data"] = data_resampled


def resample_all(file_data: dict) -> None:
    for device_name, stream_name in [
        ("eye-tracking-gaze", "position"),
        ("myo-left", "emg"),
        ("myo-right", "emg"),
        ("tactile-glove-left", "tactile_data"),
        ("tactile-glove-right", "tactile_data"),
        ("xsens-joints", "rotation_xzy_deg"),
    ]:
        resample_stream(file_data, device_name, stream_name)


def _time_indexes_for_segment(
    time_s: np.ndarray,
    segment_start_time_s: float,
    segment_end_time_s: float,
    device_name: str,
    stream_name: str,
) -> np.ndarray:
    """Pick `segment_length` row indices into a resampled stream (may differ per modality)."""
    time_s = np.squeeze(np.asarray(time_s))
    if time_s.size == 0:
        raise RuntimeError(f"Empty time axis for {device_name}/{stream_name}")

    idx = np.where((time_s >= segment_start_time_s) & (time_s <= segment_end_time_s))[0]
    idx = list(idx)
    # Some modalities start late or have gaps: no samples inside the window.
    if len(idx) == 0:
        mid_t = 0.5 * (segment_start_time_s + segment_end_time_s)
        c = int(np.clip(np.searchsorted(time_s, mid_t), 0, len(time_s) - 1))
        # Prefer samples inside [segment_start, segment_end] if stream only partially overlaps
        lo = int(np.clip(np.searchsorted(time_s, segment_start_time_s), 0, len(time_s) - 1))
        hi = int(np.clip(np.searchsorted(time_s, segment_end_time_s, side="right") - 1, 0, len(time_s) - 1))
        if lo <= hi:
            idx = list(range(lo, hi + 1))
        else:
            idx = [c]

    # Grow toward edges of the recording until we have enough timesteps (ActionNet-style).
    while len(idx) < segment_length:
        if idx[0] > 0:
            idx = [idx[0] - 1] + idx
        elif idx[-1] < len(time_s) - 1:
            idx.append(idx[-1] + 1)
        else:
            # Stream shorter than segment_length, or already at both boundaries: hold edge sample.
            idx.append(idx[-1])

    while len(idx) > segment_length:
        idx.pop()
    return np.array(idx, dtype=np.int64)


def extract_segment(
    file_data: dict,
    segment_start_time_s: float,
    segment_end_time_s: float,
) -> dict[str, np.ndarray]:
    """Return per-modality arrays for one [segment_start, segment_end] window."""
    time_indexes_for = {}
    for device_name, stream_name in [
        ("eye-tracking-gaze", "position"),
        ("myo-left", "emg"),
        ("myo-right", "emg"),
        ("tactile-glove-left", "tactile_data"),
        ("tactile-glove-right", "tactile_data"),
        ("xsens-joints", "rotation_xzy_deg"),
    ]:
        time_s = np.squeeze(np.array(file_data[device_name][stream_name]["time_s"]))
        time_indexes_for[(device_name, stream_name)] = _time_indexes_for_segment(
            time_s,
            segment_start_time_s,
            segment_end_time_s,
            device_name,
            stream_name,
        )

    modalities: dict[str, np.ndarray] = {}

    di, si = "eye-tracking-gaze", "position"
    tix = time_indexes_for[(di, si)]
    data = np.array(file_data[di][si]["data"])[tix, :].astype(np.float32)
    modalities["example_matrices_eye-tracking-gaze_position"] = data

    for arm, key in [("myo-left", "example_matrices_myo-left_emg"), ("myo-right", "example_matrices_myo-right_emg")]:
        tix = time_indexes_for[(arm, "emg")]
        data = np.array(file_data[arm]["emg"]["data"])[tix, :].astype(np.float32)
        modalities[key] = data

    for side, h5k in [
        ("tactile-glove-left", "example_matrices_tactile-glove-left_tactile_data"),
        ("tactile-glove-right", "example_matrices_tactile-glove-right_tactile_data"),
    ]:
        tix = time_indexes_for[(side, "tactile_data")]
        data = np.array(file_data[side]["tactile_data"]["data"])[tix, :, :].astype(np.float32)
        modalities[h5k] = data

    tix = time_indexes_for[("xsens-joints", "rotation_xzy_deg")]
    data = np.array(file_data["xsens-joints"]["rotation_xzy_deg"]["data"])[tix, :, :]
    data = data[:, 0:22, :].astype(np.float32)
    modalities["example_matrices_xsens-joints_rotation_xzy_deg"] = data

    return modalities


def get_feature_examples_for_window(
    file_data: dict,
    label_start_time_s: float,
    label_end_time_s: float,
    count: int,
) -> list[dict[str, np.ndarray]]:
    start_time_s = label_start_time_s + buffer_startActivity_s
    end_time_s = label_end_time_s - buffer_endActivity_s
    if end_time_s - start_time_s < segment_duration_s:
        return []
    segment_start_times_s = np.linspace(
        start_time_s,
        end_time_s - segment_duration_s,
        num=count,
        endpoint=True,
    )
    out: list[dict[str, np.ndarray]] = []
    for sst in segment_start_times_s:
        seg_end = sst + segment_duration_s
        out.append(extract_segment(file_data, sst, seg_end))
    return out


def build_examples_for_subject(
    file_data: dict,
    subject_id: str,
) -> tuple[dict[str, list], list[str], list]:
    example_matrices_by_label: dict = {}
    no_activity: list[dict[str, np.ndarray]] = []

    device_name, stream_name = "experiment-activities", "activities"
    activity_datas = file_data[device_name][stream_name]["data"]
    activity_times_s = np.squeeze(np.array(file_data[device_name][stream_name]["time_s"]))
    activity_datas = [[x.decode("utf-8") for x in datas] for datas in activity_datas]

    activities_labels = []
    activities_start_times_s = []
    activities_end_times_s = []
    exclude_bad = True
    for row_index, ts in enumerate(activity_times_s):
        label = activity_datas[row_index][0]
        is_start = activity_datas[row_index][1] == "Start"
        is_stop = activity_datas[row_index][1] == "Stop"
        rating = activity_datas[row_index][2]
        if exclude_bad and rating in ["Bad", "Maybe"]:
            continue
        if is_start:
            activities_labels.append(label)
            activities_start_times_s.append(ts)
        if is_stop:
            activities_end_times_s.append(ts)

    for label_index, activity_label in enumerate(activities_to_classify):
        if label_index == baseline_index:
            continue
        file_label_indexes = [i for i, lab in enumerate(activities_labels) if lab == activity_label]
        if not file_label_indexes and activity_label in activities_renamed:
            for alt in activities_renamed[activity_label]:
                file_label_indexes = [i for i, lab in enumerate(activities_labels) if lab == alt]
                if file_label_indexes:
                    print(f'  Found renamed activity from "{alt}"')
                    break
        print(f'  Found {len(file_label_indexes)} instances of {activity_label}')
        for fi in file_label_indexes:
            st, en = activities_start_times_s[fi], activities_end_times_s[fi]
            feats = get_feature_examples_for_window(file_data, st, en, num_segments_per_subject)
            example_matrices_by_label.setdefault(activity_label, [])
            example_matrices_by_label[activity_label].extend(feats)

    for li, activity_label in enumerate(activities_labels):
        if li == len(activities_labels) - 1:
            continue
        no_s, no_e = activities_end_times_s[li], activities_start_times_s[li + 1]
        if no_e - no_s < segment_duration_s:
            continue
        print(f'  Baseline pool between "{activity_label}" and next')
        feats = get_feature_examples_for_window(file_data, no_s, no_e, 10)
        no_activity.extend(feats)

    keys_order = [
        "example_matrices_eye-tracking-gaze_position",
        "example_matrices_myo-left_emg",
        "example_matrices_myo-right_emg",
        "example_matrices_tactile-glove-left_tactile_data",
        "example_matrices_tactile-glove-right_tactile_data",
        "example_matrices_xsens-joints_rotation_xzy_deg",
    ]

    example_labels: list[str] = []
    example_subject_ids: list[bytes] = []
    per_key: dict[str, list] = {k: [] for k in keys_order}

    for activity_label_index, activity_label in enumerate(activities_to_classify):
        if activity_label_index == baseline_index:
            continue
        print(
            f" Selecting {num_segments_per_subject} examples for {subject_id} "
            f'of activity "{activity_label}"'
        )
        if activity_label not in example_matrices_by_label:
            print(f"  Warning: no examples for label {activity_label!r}")
            continue
        fms = example_matrices_by_label[activity_label]
        if not fms:
            continue
        idxs = np.round(
            np.linspace(0, len(fms) - 1, num=num_segments_per_subject, endpoint=True)
        ).astype(int)
        for ix in idxs:
            example_labels.append(activity_label)
            example_subject_ids.append(subject_id.encode("utf-8"))
            sample = fms[ix]
            for k in keys_order:
                per_key[k].append(sample[k])

    print(
        f" Selecting {num_baseline_segments_per_subject} baseline examples for {subject_id}"
    )
    if no_activity:
        idxs = np.round(
            np.linspace(
                0,
                len(no_activity) - 1,
                num=num_baseline_segments_per_subject,
                endpoint=True,
            )
        ).astype(int)
        for ix in idxs:
            example_labels.append(baseline_label)
            example_subject_ids.append(subject_id.encode("utf-8"))
            sample = no_activity[ix]
            for k in keys_order:
                per_key[k].append(sample[k])
    else:
        print("  Warning: no inter-activity segments for baseline pool")

    return per_key, example_labels, example_subject_ids


def save_actionsense_hdf5(
    output_path: str,
    per_key: dict[str, list],
    example_labels: list[str],
    example_subject_ids: list,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    # MFedMC models use 20 logits; dataset.py does (index - 1) -> need values in 1..20 only.
    # The ActionNet taxonomy lists 21 entries (None + 20 activities) so raw positions can be 0..20.
    # Remap *used* labels to contiguous 1..K (K<=20) in taxonomy order.
    used_in_order = [a for a in activities_to_classify if a in set(example_labels)]
    if len(used_in_order) > 20:
        raise ValueError(
            f"More than 20 classes appear in the examples ({len(used_in_order)}); "
            "MFedMC ActionSense models expect num_classes=20."
        )
    name_to_compact_1based = {name: i + 1 for i, name in enumerate(used_in_order)}
    compact_indexes = [name_to_compact_1based[lab] for lab in example_labels]

    example_labels_arr = np.array(example_labels, dtype=h5py.string_dtype(encoding="utf-8"))
    example_label_indexes_arr = np.array(compact_indexes, dtype=np.int32)
    example_subject_ids_arr = np.array(example_subject_ids)

    with h5py.File(output_path, "w") as hdf_file:
        hdf_file.create_dataset("example_labels", data=example_labels_arr)
        hdf_file.create_dataset("example_label_indexes", data=example_label_indexes_arr)
        hdf_file.create_dataset("example_subject_ids", data=example_subject_ids_arr)
        for key, samples in per_key.items():
            if not samples:
                raise ValueError(f"No samples for {key}")
            hdf_file.create_dataset(key, data=np.stack(samples, axis=0))
        meta = (
            "example_label_indexes are 1-based contiguous class ids after remapping "
            "to MFedMC's 20-class setup; dataset.py subtracts 1 for training targets."
        )
        hdf_file.attrs["label_encoding_note"] = meta


def process_one_streamlog(path: str, subject_id: str) -> tuple[dict[str, list], list[str], list]:
    """Load one HDF5, run pipeline, return examples for that session."""
    print("Loading", path)
    fd = load_streamlog(path)
    print("Filtering / normalizing...")
    filter_and_normalize(fd)
    print("Resampling to", resampled_Fs, "Hz...")
    resample_all(fd)
    print(f"Building windows (subject {subject_id})...")
    per_key, elabels, sids = build_examples_for_subject(fd, subject_id)
    return per_key, elabels, sids


def merge_per_key(dest: dict[str, list], src: dict[str, list]) -> None:
    for k in dest:
        dest[k].extend(src[k])


def _subject_id_str(s) -> str:
    if isinstance(s, bytes):
        return s.decode("utf-8")
    return str(s)


def subsample_per_subject_per_class(
    merged_per_key: dict[str, list],
    merged_labels: list[str],
    merged_sids: list,
    max_per: int,
    seed: int | None,
) -> tuple[dict[str, list], list[str], list]:
    """Keep at most ``max_per`` examples per (subject id, activity label) after merging sessions."""
    if max_per <= 0:
        return merged_per_key, merged_labels, merged_sids

    n = len(merged_labels)
    rng = np.random.default_rng(seed)
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i in range(n):
        groups[(_subject_id_str(merged_sids[i]), merged_labels[i])].append(i)

    keep: list[int] = []
    dropped = 0
    for _key, idxs in sorted(groups.items()):
        if len(idxs) <= max_per:
            keep.extend(idxs)
        else:
            pick = rng.choice(len(idxs), size=max_per, replace=False)
            keep.extend(idxs[j] for j in pick)
            dropped += len(idxs) - max_per

    keep.sort()
    if dropped:
        print(
            f"Strict cap ({max_per} per subject per class): removed {dropped} example(s) "
            "after merging multiple sessions for the same subject."
        )

    new_labels = [merged_labels[i] for i in keep]
    new_sids = [merged_sids[i] for i in keep]
    new_per_key = {k: [merged_per_key[k][i] for i in keep] for k in merged_per_key}
    return new_per_key, new_labels, new_sids


# Filenames look like ...actionNet-wearables_S00.hdf5 (hyphen before "wearables") or _wearables_S00
_SUBJECT_FROM_NAME = re.compile(r"wearables_(S\d{2})\.hdf5$", re.IGNORECASE)


def paths_and_subjects_from_glob(pattern: str) -> tuple[list[str], list[str]]:
    """Expand a glob; subject id is parsed from each filename (..._wearables_S07.hdf5)."""
    paths = sorted(glob_module.glob(pattern))
    if not paths:
        raise SystemExit(f"No files matched --input-glob {pattern!r}")
    subject_ids: list[str] = []
    resolved: list[str] = []
    for p in paths:
        base = os.path.basename(p)
        m = _SUBJECT_FROM_NAME.search(base)
        if not m:
            raise SystemExit(
                f"Cannot parse subject from filename (expected ...wearables_SXX.hdf5): {base}"
            )
        resolved.append(os.path.abspath(p))
        subject_ids.append(m.group(1).upper())
    return resolved, subject_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input",
        nargs="*",
        default=None,
        help="One or more streamLog_actionNet*.hdf5 paths (use with --subject-id unless a single file, default S00).",
    )
    parser.add_argument(
        "--input-glob",
        default=None,
        metavar="PATTERN",
        help="Glob of HDF5 files (e.g. dataset/*streamLog*wearables*.hdf5). Subject id is read from each filename; "
        "cannot be combined with --input.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output .hdf5 file path, or a directory when using --canonical-output.",
    )
    parser.add_argument(
        "--subject-id",
        nargs="*",
        default=[],
        help="Subject id per --input file. Not used with --input-glob (ids come from filenames).",
    )
    parser.add_argument(
        "--only-subjects",
        nargs="+",
        metavar="S00",
        default=None,
        help="Process only sessions for these subject ids (e.g. five clients for paper-style 5subj).",
    )
    parser.add_argument(
        "--canonical-output",
        action="store_true",
        help=(
            "Write data_processed_allStreams_<10>s_<10>hz_<K>subj_ex20-20_allActs.hdf5 "
            "into directory --output; K = number of distinct subjects in the merged file."
        ),
    )
    parser.add_argument(
        "--max-per-class-per-subject",
        type=int,
        default=20,
        metavar="N",
        help="After merging sessions, keep at most N examples per (subject, activity class). "
        "Baseline 'None' is one class. Use 0 to keep all examples (no cap). Default: 20.",
    )
    parser.add_argument(
        "--subsample-seed",
        type=int,
        default=0,
        help="RNG seed when subsampling excess (subject, class) groups (default: 0).",
    )
    args = parser.parse_args()

    if args.input_glob and args.input:
        parser.error("Use either --input-glob or --input, not both.")
    if args.input_glob:
        if args.subject_id:
            parser.error("--subject-id is not used with --input-glob (subjects parsed from filenames).")
        paths, subject_ids = paths_and_subjects_from_glob(args.input_glob)
        print(f"Discovered {len(paths)} file(s) from glob; subjects: {subject_ids}")
    elif args.input:
        paths = args.input
        if len(paths) == 1:
            if len(args.subject_id) == 0:
                subject_ids = ["S00"]
            elif len(args.subject_id) == 1:
                subject_ids = args.subject_id
            else:
                parser.error("With one --input, supply at most one --subject-id")
        else:
            if len(args.subject_id) != len(paths):
                parser.error(
                    f"Need one --subject-id per --input ({len(paths)} files, got {len(args.subject_id)} subject-id(s))"
                )
            subject_ids = args.subject_id
    else:
        parser.error("Provide --input file(s) or --input-glob PATTERN.")

    if args.only_subjects:
        allow = {s.upper() for s in args.only_subjects}
        filt = [(p, s) for p, s in zip(paths, subject_ids) if s.upper() in allow]
        if not filt:
            parser.error(f"--only-subjects left no sessions (had subjects {sorted(set(subject_ids))}).")
        paths = [x[0] for x in filt]
        subject_ids = [x[1] for x in filt]
        print(
            f"After --only-subjects: {len(paths)} session(s); "
            f"per-file subjects {subject_ids}"
        )

    if args.canonical_output and not os.path.isdir(args.output):
        os.makedirs(args.output, exist_ok=True)

    t0 = time.time()
    merged_per_key: dict[str, list] | None = None
    merged_labels: list[str] = []
    merged_sids: list = []
    total_sessions = 0

    for path, sid in zip(paths, subject_ids):
        print()
        print("=" * 60)
        print(f"Session: {path}")
        print(f"Subject: {sid}")
        print("=" * 60)
        try:
            per_key, elabels, sids = process_one_streamlog(path, sid)
        except Exception as e:
            raise SystemExit(f"Failed on {path}: {e}") from e
        if not elabels:
            print(f"  Warning: no examples from {path}; skipping.")
            continue
        total_sessions += 1
        if merged_per_key is None:
            merged_per_key = {k: [] for k in per_key}
        merge_per_key(merged_per_key, per_key)
        merged_labels.extend(elabels)
        merged_sids.extend(sids)
        print(f"  Added {len(elabels)} examples (running total {len(merged_labels)})")

    if not merged_labels or merged_per_key is None:
        raise SystemExit(
            "No examples produced from any input. Check streams/labels in the HDF5 files."
        )

    if args.max_per_class_per_subject > 0:
        merged_per_key, merged_labels, merged_sids = subsample_per_subject_per_class(
            merged_per_key,
            merged_labels,
            merged_sids,
            args.max_per_class_per_subject,
            args.subsample_seed,
        )

    print()
    print(f"Merging {total_sessions} session(s) -> {len(merged_labels)} total examples")
    unique_subjects = sorted(
        {s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in merged_sids}
    )
    k_subj = len(unique_subjects)
    print(f"Distinct subjects in HDF5: {unique_subjects} (K={k_subj})")
    cap_note = (
        f"then ≤{args.max_per_class_per_subject} per class per subject"
        if args.max_per_class_per_subject > 0
        else "no post-merge per-subject cap (--max-per-class-per-subject 0)"
    )
    print(
        f"Recipe: allStreams, {segment_duration_s}s windows, {resampled_Fs} Hz, "
        f"up to {num_segments_per_subject} examples/class/subject-session + "
        f"{num_baseline_segments_per_subject} baseline; {cap_note}; full activity list."
    )

    if args.canonical_output:
        out_path = os.path.join(
            args.output,
            (
                f"data_processed_allStreams_{segment_duration_s}s_{resampled_Fs}hz_{k_subj}subj_"
                f"ex{num_segments_per_subject}-{num_baseline_segments_per_subject}_allActs.hdf5"
            ),
        )
    else:
        out_path = args.output

    print(f"Saving to {out_path} ...")
    save_actionsense_hdf5(out_path, merged_per_key, merged_labels, merged_sids)
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
