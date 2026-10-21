"""HybridSketchMFedMC v2 with FL-FD's low-parameter model -- new entry point, Hybrid files untouched.

The whole algorithm is HybridSketchMFedMC_v2's own code, imported and run as is: its ``main._run``
(data loading, model setup, npz), ``federated.run_federated`` (loss-based client selection, SHAP
modality selection, gradient masking, FetchSGD Count Sketch + server step, LR schedule, shard test
accuracy, communication counting) and ``options.args_parser`` (same flags).  In memory only, two
names inside Hybrid's main module are replaced before ``_run`` is called:

  * ``load_uci_har_mm``: Hybrid's loader, then every feature vector (348-D acc, 213-D gyro) becomes a
    70 x 70 GAF image with FL-FD's own ``to_flfd_images`` (FL-FD/uci_har/main.py), one per modality;
  * ``SketchFusionBNet``: replaced by FL-FD's small CNN (models_flfd.py), --model branch | flfd.

Extra flags (everything else is passed to Hybrid's parser):
  --model {branch,flfd}   branch: FL-FD conv branch per modality + Hybrid sketch fusion + Linear (default)
                          flfd:   FL-FD's input-fusion network unchanged (4,461 params)
  --gaf_method {difference,summation}   (FL-FD's released code: difference)
  --size 70                              GAF image side (FL-FD: 70)

Logs: HybridSketchMFedMC_v2_flfd/result_updated/HybridV2FD_<model>_..._seed<s>.txt
npz:  HybridSketchMFedMC_v2_flfd/results/HybridV2FD_<model>_..._seed<s>.npz
"""

import argparse
import glob
import importlib.util
import io
import os
import sys
import time

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
HYBRID = os.path.join(ROOT, "HybridSketchMFedMC_v2")
sys.path.insert(0, HYBRID)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hybrid = _load("hybrid_v2_main", os.path.join(HYBRID, "main.py"))  # Hybrid v2's main.py, unmodified
flfd_uci = _load("flfd_uci_har_main", os.path.join(ROOT, "FL-FD", "uci_har", "main.py"))  # to_flfd_images
sys.path.insert(0, HERE)
from models import build_param_index_maps  # noqa: E402  (HybridSketchMFedMC_v2)
from models_flfd import FLFDBranchNet, FLFDInputFusionNet  # noqa: E402

PREFIX = "HybridV2FD"


def _parse():
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)  # keep Hybrid's --mode away from --model
    p.add_argument("--model", choices=["branch", "flfd"], default="branch")
    p.add_argument("--gaf_method", choices=["difference", "summation"], default="difference")
    p.add_argument("--size", type=int, default=70)
    extra, rest = p.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    args = hybrid.args_parser()  # Hybrid v2's own flags and defaults
    args.model, args.gaf_method, args.size = extra.model, extra.gaf_method, extra.size
    return args


def _gaf_loader(args, original):
    def load(*a, **kw):
        clients, global_test, meta = original(*a, **kw)
        t0 = time.perf_counter()
        print(f"FL-FD input step: each modality's feature vector -> GAF ({args.gaf_method}) -> "
              f"{args.size} x {args.size} image, one per modality (to_flfd_images)...")
        for part in list(clients) + [global_test]:
            part["xs"] = [flfd_uci.to_flfd_images([x], args.gaf_method, args.size) for x in part["xs"]]
        shapes = [tuple(x.shape[1:]) for x in global_test["xs"]]
        print(f"  done in {time.perf_counter() - t0:.1f} s; per-modality input {shapes}")
        return clients, global_test, meta
    return load


def _model_factory(args):
    def build(mod_dims, num_classes, sketch_r, sketch_c, fusion_mode, **_unused):
        if args.model == "branch":
            model = FLFDBranchNet(len(mod_dims), num_classes, args.size, fusion_mode, sketch_r, sketch_c)
            desc = (f"FLFDBranchNet: per modality FL-FD conv branch (1 -> 3 -> 6 -> 12, {model.feat_dim} features), "
                    f"fusion {fusion_mode}" + (f" {sketch_r} x {sketch_c}" if fusion_mode == "sketch" else "")
                    + f", head Linear({model.head.in_features}, {num_classes})")
        else:
            model = FLFDInputFusionNet(len(mod_dims), num_classes, args.size)
            desc = "FLFDInputFusionNet: FL-FD's network unchanged (input-level fusion of the GAF images)"
        mod_idx, shared_idx, total = build_param_index_maps(model)  # what Hybrid's masks will use
        print(f"Model replaced by FL-FD's low-parameter setup -- {desc}")
        print(f"  learned params {total:,}: per modality {[int(ix.numel()) for ix in mod_idx]}, "
              f"shared {int(shared_idx.numel()):,}; gradient sketch {args.num_rows} x {args.num_cols} "
              f"= {args.num_rows * args.num_cols * 4:,} B per client")
        if args.num_rows * args.num_cols >= total:
            print(f"  note: the sketch ({args.num_rows * args.num_cols:,} cells) is not smaller than the model "
                  f"({total:,} params), so it does not compress")
        return model
    return build


def _stem(args, cum_up_mb):
    return f"{PREFIX}_{args.model}_" + hybrid._txt_stem(args, cum_up_mb)[len("HybridV2_"):]


def _seed_tag_npz(args, since):
    """Hybrid's npz name has no seed: rename this run's file to a seed-tagged, config-tagged name."""
    hits = [f for f in glob.glob(os.path.join(args.results_dir, "HybridV2_*.npz")) if os.path.getmtime(f) >= since]
    if not hits:
        return None
    src = max(hits, key=os.path.getmtime)
    dst = os.path.join(args.results_dir, _stem(args, None).replace("_NAMB_", "_") + ".npz")
    os.replace(src, dst)
    print(f"Results renamed to {dst}")
    return dst


def main():
    args = _parse()
    args.results_dir = os.path.abspath(args.results_dir if os.path.isabs(args.results_dir)
                                       else os.path.join(HERE, args.results_dir))
    hybrid.load_uci_har_mm = _gaf_loader(args, hybrid.load_uci_har_mm)
    hybrid.SketchFusionBNet = _model_factory(args)
    args.feat_dim, args.mm_dropout = 588, 0.2  # what the FL-FD branches really use (printed config only)

    log_buf = io.StringIO()
    stdout_orig = sys.stdout
    sys.stdout = hybrid._Tee(stdout_orig, log_buf)
    cum_up_mb = None
    start = time.time()
    try:
        cum_up_mb = hybrid._run(args)  # Hybrid v2's run, unchanged
        _seed_tag_npz(args, start)
    finally:
        sys.stdout = stdout_orig
        out_dir = os.path.join(HERE, "result_updated")
        os.makedirs(out_dir, exist_ok=True)
        txt_path = os.path.join(out_dir, _stem(args, cum_up_mb) + ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(log_buf.getvalue())
        print(f"Console log saved to {txt_path}")


if __name__ == "__main__":
    main()
