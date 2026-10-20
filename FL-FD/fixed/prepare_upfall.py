"""Build the FL-FD input cache from UP-Fall (fixed version of FL-FD/data_processing.py).

Per (subject, activity, trial), following the released code and the paper (Sec. 3.2, 4.2):
  * sensors: first `--window` rows of each trial; for each of the 5 wearable sensors (ankle,
    right pocket, belt, neck, wrist) the accelerometer and angular-velocity magnitudes are each
    turned into a GAF image (pyts GramianAngularField, method as in the released code)
  * camera:  first `--window` frames, each resized to window x window and converted to gray;
    absolute differences of adjacent frames are summed and rescaled to [0, 255]
  * everything is resized to `--size` x `--size` (paper: inputs resized to 3 x 70 x 70)

Output npz: gaf (N, 5, 2, S, S), cam (N, S, S), subject/activity/trial (N,), sensors (5,).
Trials whose camera folder is missing or incomplete are skipped (and listed), so the cache can be
built while the camera download is partial; re-running rebuilds only if the trial set changed.
"""

import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image
from pyts.image import GramianAngularField

SENSORS = ["Ankle", "RightPocket", "Belt", "Neck", "Wrist"]
# CompleteDataSet.csv column indices (after the 2 header rows): acc x,y,z then gyro x,y,z
SENSOR_COLS = {"Ankle": 1, "RightPocket": 8, "Belt": 15, "Neck": 22, "Wrist": 29}
SUBJECT_COL, ACTIVITY_COL, TRIAL_COL = 43, 44, 45


def args_parser():
    p = argparse.ArgumentParser(description="Prepare the FL-FD UP-Fall cache")
    p.add_argument("--csv", required=True, help="CompleteDataSet.csv")
    p.add_argument("--camera_dir", required=True, help="folder of Subject{s}Activity{a}Trial{t}Camera{c}/")
    p.add_argument("--camera", type=int, default=2, choices=[1, 2])
    p.add_argument("--out", required=True)
    p.add_argument("--window", type=int, default=140, help="samples/frames kept per trial (released code: 140)")
    p.add_argument("--size", type=int, default=70, help="final image side (paper: 70)")
    p.add_argument("--gaf_method", choices=["difference", "summation"], default="difference",
                   help="released code: difference; paper's formula: summation")
    p.add_argument("--drop_subjects", type=str, default="5,9", help="paper drops 5 and 9 (no camera data)")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--rebuild", action="store_true")
    return p.parse_args()


def _resize(img, size):
    return np.asarray(Image.fromarray(img.astype(np.float32), mode="F").resize((size, size), Image.BILINEAR))


def camera_image(job):
    folder, window, size = job
    frames = sorted(f for f in os.listdir(folder) if f.lower().endswith(".png"))
    grays, bad = [], []
    for f in frames:  # first `window` readable frames; a few source PNGs are corrupt
        if len(grays) == window:
            break
        try:
            img = Image.open(os.path.join(folder, f)).convert("RGB")
        except (OSError, SyntaxError):
            bad.append(f)
            continue
        grays.append(np.asarray(img.resize((window, window)).convert("L"), dtype=np.int32))
    delta = sum(np.abs(b - a) for a, b in zip(grays[:-1], grays[1:])).astype(np.float64)
    if delta.max() > delta.min():
        delta = (delta - delta.min()) * (255.0 / (delta.max() - delta.min()))
    return _resize(delta, size), len(grays), bad


def main():
    args = args_parser()
    drop = {int(s) for s in args.drop_subjects.split(",") if s}

    df = pd.read_csv(args.csv, skiprows=2, header=None)
    df = df[~df[SUBJECT_COL].isin(drop)]
    groups = {k: g for k, g in df.groupby([SUBJECT_COL, ACTIVITY_COL, TRIAL_COL], sort=True)}

    keys, missing = [], []
    for s, a, t in groups:
        folder = os.path.join(args.camera_dir, f"Subject{s}Activity{a}Trial{t}Camera{args.camera}")
        (keys if os.path.exists(os.path.join(folder, ".complete")) else missing).append((s, a, t))
    print(f"{len(groups)} sensor trials (subjects {sorted(drop)} dropped); "
          f"{len(keys)} with Camera{args.camera} frames, {len(missing)} without")
    if missing:
        print("  skipped (no camera frames yet): " + ", ".join(f"S{s}A{a}T{t}" for s, a, t in missing))

    if os.path.exists(args.out) and not args.rebuild:
        old = np.load(args.out)
        old_keys = list(zip(old["subject"].tolist(), old["activity"].tolist(), old["trial"].tolist()))
        same = (old_keys == keys and int(old["size"]) == args.size and int(old["window"]) == args.window
                and str(old["gaf_method"]) == args.gaf_method and int(old["camera"]) == args.camera)
        if same:
            print(f"Cache up to date: {args.out}")
            return
        print("Cache is stale (trial set or settings changed) — rebuilding")

    gaf_tf = GramianAngularField(method=args.gaf_method)
    gaf = np.zeros((len(keys), len(SENSORS), 2, args.size, args.size), dtype=np.float32)
    for i, key in enumerate(keys):
        g = groups[key].iloc[: args.window]
        for j, sensor in enumerate(SENSORS):
            c = SENSOR_COLS[sensor]
            acc = np.sqrt((g[[c, c + 1, c + 2]].astype(float) ** 2).sum(axis=1)).to_numpy()
            gyr = np.sqrt((g[[c + 3, c + 4, c + 5]].astype(float) ** 2).sum(axis=1)).to_numpy()
            for ch, series in enumerate((acc, gyr)):
                gaf[i, j, ch] = _resize(gaf_tf.fit_transform(series.reshape(1, -1))[0], args.size)

    jobs = [(os.path.join(args.camera_dir, f"Subject{s}Activity{a}Trial{t}Camera{args.camera}"),
             args.window, args.size) for s, a, t in keys]
    with Pool(args.workers) as pool:
        cam_out = pool.map(camera_image, jobs, chunksize=4)
    cam = np.stack([c for c, _, _ in cam_out]).astype(np.float32)
    for (s, a, t), (_, _, bad) in zip(keys, cam_out):
        if bad:
            print(f"  S{s}A{a}T{t}: skipped {len(bad)} unreadable frame(s): {', '.join(bad)}")
    short = [(k, n) for k, (_, n, _) in zip(keys, cam_out) if n < args.window]
    if short:
        print(f"  {len(short)} trials have < {args.window} frames (used all available): "
              + ", ".join(f"S{s}A{a}T{t}={n}" for (s, a, t), n in short))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(
        args.out, gaf=gaf, cam=cam,
        subject=np.array([k[0] for k in keys]), activity=np.array([k[1] for k in keys]),
        trial=np.array([k[2] for k in keys]), sensors=np.array(SENSORS),
        size=args.size, window=args.window, gaf_method=args.gaf_method, camera=args.camera,
    )
    print(f"Saved {len(keys)} trials to {args.out}")


if __name__ == "__main__":
    main()
