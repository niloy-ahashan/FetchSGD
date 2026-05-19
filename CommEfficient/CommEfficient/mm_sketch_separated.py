"""
Modality-separated sketch fusion for federated multimodal training.

Unlike mm_sketch_fusion.py (which masks the full gradient with d=grad_size
and is mathematically equivalent to a single sketch by linearity), this
variant **extracts compact per-modality gradient vectors** and sketches
each with its own CSVecFed instance (different d → different hash
functions → different collision domains).

Worker side:
    g_img = grad[img_indices]              # compact, d_img elements
    g_txt = grad[txt_indices]              # compact, d_txt elements
    table = S_img(g_img) + S_txt(g_txt)    # same (r, c) shape

Server side recovery:
    Uses per-modality CSVecs to recover heavy hitters from the fused
    table separately, then maps them back to the full gradient vector.

Why this is NOT equivalent to a single sketch:
  - S_img uses d=d_img, S_txt uses d=d_txt → different hash functions
  - Each modality gets a GUARANTEED quota of recovered coordinates,
    preventing one modality's heavy hitters from monopolising the top-k
  - Intra-modality collisions are reduced (fewer coords per hash space)
  - Cross-modality interference appears as unstructured noise that the
    median estimator handles robustly across the r=5 rows

Integration
-----------
Worker (fed_worker.py / forward_grad):
    Replace the sketch_multimodal_fused() call with
    sketch_modality_separated().

Server (fed_aggregator.py / get_server_update):
    Route to server_step_separated() instead of _server_helper_sketched()
    when the flag is active.
"""

from __future__ import annotations

import torch
from CommEfficient.csvecN import CSVecFed


# ------------------------------------------------------------------
# Picklable storage on args  (CUDA tensors can't cross processes)
# ------------------------------------------------------------------

def store_index_maps_on_args(model: torch.nn.Module, args):
    """
    Compute the modality index partition and store plain-Python data on
    *args* so it survives ``multiprocessing.Process`` pickling.

    Call once in the main process before spawning workers.
    """
    maps = build_modality_index_maps(model, torch.device("cpu"))
    args._mm_sep_img_idx = maps[0].tolist()
    args._mm_sep_txt_idx = maps[1].tolist()
    args._mm_sep_d_img = maps[2]
    args._mm_sep_d_txt = maps[3]


def get_index_maps(args, device: torch.device):
    """
    Reconstruct device-local index-map tensors from the plain lists
    stored by ``store_index_maps_on_args``.
    """
    return (
        torch.tensor(args._mm_sep_img_idx, dtype=torch.long, device=device),
        torch.tensor(args._mm_sep_txt_idx, dtype=torch.long, device=device),
        args._mm_sep_d_img,
        args._mm_sep_d_txt,
    )


# ------------------------------------------------------------------
# Index-map builder (run once at model init, cache on the model)
# ------------------------------------------------------------------

def build_modality_index_maps(model: torch.nn.Module, device: torch.device):
    """
    Partition ``model.parameters()`` positions into img-branch vs
    txt-branch and return flat-gradient index tensors for each.

    Follows the same assignment convention as mm_sketch_fusion.py:
      img:  img_extractor, img_refiner, img_to_txt
      txt:  txt_extractor, txt_refiner, txt_to_img, integrator, classifier

    Returns
    -------
    img_indices : LongTensor[d_img]
    txt_indices : LongTensor[d_txt]
    d_img       : int
    d_txt       : int
    """
    img_idx: list[int] = []
    txt_idx: list[int] = []
    offset = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = p.numel()
        positions = list(range(offset, offset + n))
        if (name.startswith("img_extractor")
                or name.startswith("img_refiner")
                or name.startswith("img_to_txt")):
            img_idx.extend(positions)
        else:
            txt_idx.extend(positions)
        offset += n

    return (
        torch.tensor(img_idx, dtype=torch.long, device=device),
        torch.tensor(txt_idx, dtype=torch.long, device=device),
        len(img_idx),
        len(txt_idx),
    )


# ------------------------------------------------------------------
# Worker side — sketch per modality, return summed table
# ------------------------------------------------------------------

def sketch_modality_separated(
    grad: torch.Tensor,
    index_maps: tuple,
    num_cols: int,
    num_rows: int,
    num_blocks: int,
    device: torch.device,
):
    """
    Extract compact per-modality gradient vectors, sketch each with its
    own CSVecFed (different ``d``), and return the element-wise sum of
    the two sketch tables.

    Parameters
    ----------
    grad        : flat gradient vector  [grad_size]
    index_maps  : (img_indices, txt_indices, d_img, d_txt) from
                  ``build_modality_index_maps``
    num_cols, num_rows, num_blocks : sketch geometry (same for both)
    device      : torch device

    Returns
    -------
    fused_table : Tensor[num_rows, num_cols]
    """
    img_indices, txt_indices, d_img, d_txt = index_maps

    g_img = grad[img_indices]
    g_txt = grad[txt_indices]

    sk_img = CSVecFed(d=d_img, c=num_cols, r=num_rows,
                      device=device, numBlocks=num_blocks)
    sk_img.accumulateVec(g_img)

    sk_txt = CSVecFed(d=d_txt, c=num_cols, r=num_rows,
                      device=device, numBlocks=num_blocks)
    sk_txt.accumulateVec(g_txt)

    return sk_img.table + sk_txt.table


# ------------------------------------------------------------------
# Server side — per-modality recovery from the fused table
# ------------------------------------------------------------------

def unsketch_modality_separated(
    table: torch.Tensor,
    index_maps: tuple,
    grad_size: int,
    k: int,
    num_cols: int,
    num_rows: int,
    num_blocks: int,
    device: torch.device,
):
    """
    Recover per-modality heavy hitters from the fused error table,
    then map them back to the full gradient vector.

    ``k`` is split proportionally between modalities by parameter count.

    Returns
    -------
    update : Tensor[grad_size]  — sparse (mostly zero) weight update
    """
    img_indices, txt_indices, d_img, d_txt = index_maps

    k_img = max(1, round(k * d_img / (d_img + d_txt)))
    k_txt = max(1, k - k_img)

    sk_img = CSVecFed(d=d_img, c=num_cols, r=num_rows,
                      device=device, numBlocks=num_blocks)
    sk_img.table[:] = table
    update_img = sk_img.unSketch(k=k_img)

    sk_txt = CSVecFed(d=d_txt, c=num_cols, r=num_rows,
                      device=device, numBlocks=num_blocks)
    sk_txt.table[:] = table
    update_txt = sk_txt.unSketch(k=k_txt)

    full_update = torch.zeros(grad_size, device=device)
    full_update[img_indices] = update_img
    full_update[txt_indices] = update_txt
    return full_update


def resketch_update(
    update: torch.Tensor,
    index_maps: tuple,
    num_cols: int,
    num_rows: int,
    num_blocks: int,
    device: torch.device,
):
    """
    Re-sketch the recovered update using per-modality hash functions.

    Needed for error feedback and momentum factor masking: the server
    must identify which sketch-table cells were "consumed" by the
    recovered coordinates so it can zero them in Verror / Vvelocity.

    Returns
    -------
    sketched_update : Tensor[num_rows, num_cols]
    """
    img_indices, txt_indices, d_img, d_txt = index_maps

    u_img = update[img_indices]
    u_txt = update[txt_indices]

    sk_img = CSVecFed(d=d_img, c=num_cols, r=num_rows,
                      device=device, numBlocks=num_blocks)
    sk_img.accumulateVec(u_img)

    sk_txt = CSVecFed(d=d_txt, c=num_cols, r=num_rows,
                      device=device, numBlocks=num_blocks)
    sk_txt.accumulateVec(u_txt)

    return sk_img.table + sk_txt.table


# ------------------------------------------------------------------
# Server step — drop-in replacement for _server_helper_sketched
# ------------------------------------------------------------------

def server_step_separated(
    aggregated_sketch: torch.Tensor,
    Vvelocity: torch.Tensor,
    Verror: torch.Tensor,
    args,
    lr,
    index_maps: tuple,
):
    """
    Drop-in replacement for ``_server_helper_sketched`` when using
    modality-separated sketching.

    Momentum and error accumulation operate on the fused (r × c) table
    exactly as before.  Only **recovery** and **error feedback** use
    per-modality hash functions.

    Parameters
    ----------
    aggregated_sketch : averaged sketch table from workers  [r, c]
    Vvelocity, Verror : server momentum / error tables      [r, c]
    args              : parsed CLI args (needs .virtual_momentum,
                        .error_type, .k, .grad_size, .num_cols,
                        .num_rows, .num_blocks, .device, and
                        .local_momentum)
    lr                : current learning rate (scalar or vec)
    index_maps        : (img_indices, txt_indices, d_img, d_txt)

    Returns
    -------
    (weight_update, new_Vvelocity, new_Verror)
    """
    rho = args.virtual_momentum

    if args.error_type == "local":
        assert args.virtual_momentum == 0
    elif args.error_type == "virtual":
        assert args.local_momentum == 0

    # --- Momentum (unchanged) ---
    torch.add(aggregated_sketch, Vvelocity, alpha=rho, out=Vvelocity)

    # --- Error accumulation (unchanged) ---
    if args.error_type == "local":
        Verror = Vvelocity
    elif args.error_type == "virtual":
        Verror += Vvelocity

    # --- Recovery: per-modality unSketching ---
    update = unsketch_modality_separated(
        table=Verror,
        index_maps=index_maps,
        grad_size=args.grad_size,
        k=args.k,
        num_cols=args.num_cols,
        num_rows=args.num_rows,
        num_blocks=args.num_blocks,
        device=args.device,
    )

    # --- Error feedback + momentum masking via per-modality re-sketch ---
    sketched_update = resketch_update(
        update=update,
        index_maps=index_maps,
        num_cols=args.num_cols,
        num_rows=args.num_rows,
        num_blocks=args.num_blocks,
        device=args.device,
    )

    if args.error_type == "virtual":
        nz = sketched_update.nonzero()
        Verror[nz[:, 0], nz[:, 1]] = 0

    nz = sketched_update.nonzero()
    Vvelocity[nz[:, 0], nz[:, 1]] = 0

    return update * lr, Vvelocity, Verror
