"""Download UP-Fall Camera2 zips and extract each to <out>/Subject{s}Activity{a}Trial{t}Camera2/.

Resumable: a trial is done when its folder holds a `.complete` marker. Zips are verified,
extracted (flat .png members only, basenames only), then deleted. Google Drive's per-file
download quota can trip partway through ("File is not a zip file"); re-run after ~24 h.

Usage: python3 -I FL-FD/fixed/download_camera2.py [links.json] [out_dir] [stage_dir]
upfall_links.json holds the per-trial Drive file IDs scraped from
https://sites.google.com/up.edu.mx/har-up/ (2026-10-15).
"""
import json, os, subprocess, sys, zipfile, time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(HERE, "..", "dataset")
LINKS, OUT, STAGE = (sys.argv[1:] + [os.path.join(HERE, "upfall_links.json"),
                                     os.path.join(DATASET, "camera"),
                                     os.path.join(DATASET, "camera_stage")][len(sys.argv) - 1:])[:3]
WORKERS = 4
URL = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"


def fetch(r):
    name = f"Subject{r['sub']}Activity{r['act']}Trial{r['trial']}Camera2"
    dest = os.path.join(OUT, name)
    if os.path.exists(os.path.join(dest, ".complete")):
        return name, "skip"
    zp = os.path.join(STAGE, name + ".zip")
    for attempt in range(4):
        subprocess.run(["curl", "-sL", "--fail", "--retry", "3", "-o", zp, URL.format(r["id"])])
        try:
            with zipfile.ZipFile(zp) as z:
                if z.testzip() is not None:
                    raise zipfile.BadZipFile("crc")
                os.makedirs(dest, exist_ok=True)
                n = 0
                for info in z.infolist():
                    base = os.path.basename(info.filename)
                    if info.is_dir() or not base.lower().endswith(".png"):
                        continue
                    with z.open(info) as src, open(os.path.join(dest, base), "wb") as dst:
                        dst.write(src.read())
                    n += 1
            open(os.path.join(dest, ".complete"), "w").write(str(n))
            os.remove(zp)
            return name, f"ok {n} frames"
        except (zipfile.BadZipFile, FileNotFoundError, OSError) as e:
            err = repr(e)
            time.sleep(10 * (attempt + 1))
    if os.path.exists(zp):
        os.remove(zp)
    return name, f"FAILED {err}"


def main():
    rows = sorted((r for r in json.load(open(LINKS)) if r["kind"] == "Camera2"),
                  key=lambda r: (r["sub"], r["act"], r["trial"]))
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(STAGE, exist_ok=True)
    done = 0
    with ThreadPoolExecutor(WORKERS) as ex:
        for fut in as_completed([ex.submit(fetch, r) for r in rows]):
            done += 1
            name, status = fut.result()
            print(f"[{done}/{len(rows)}] {name}: {status}", flush=True)


if __name__ == "__main__":
    main()
