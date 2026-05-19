"""
Per-modality sketch fusion for federated multimodal training.

Only **two** sketch tables are formed and added (no separate “fusion sketch”):

    S(g) = S(g ⊙ m_img) + S(g ⊙ m_txt)

with m_img + m_txt = 1 everywhere, matching linearity of CSVecFed.

Parameter → bucket assignment (heuristic for shared layers):
  img mask:  img_extractor, img_refiner, img_to_txt
  txt mask:  txt_extractor, txt_refiner, txt_to_img, integrator, classifier

Integrator/classifier sit on the txt side so there is no third additive term;
their gradients still appear exactly once in the full vector, so this equals
sketching the full gradient (same as one accumulateVec(grad)).
"""

from __future__ import annotations

import torch


def build_multimodal_sketch_masks(model: torch.nn.Module, device: torch.device):
    """
    Build float masks matching `get_grad_vec` / `model.parameters()` order
    (requires_grad parameters only). Two-way partition: m_img + m_txt = 1.
    """
    blocks_i, blocks_t = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = p.numel()
        if name.startswith("img_extractor") or name.startswith("img_refiner"):
            blocks_i.append(torch.ones(n))
            blocks_t.append(torch.zeros(n))
        elif name.startswith("txt_extractor") or name.startswith("txt_refiner"):
            blocks_i.append(torch.zeros(n))
            blocks_t.append(torch.ones(n))
        elif name.startswith("img_to_txt"):
            # Cross-modal predictor driven by img latent → img bucket
            blocks_i.append(torch.ones(n))
            blocks_t.append(torch.zeros(n))
        else:
            # txt_to_img, integrator, classifier (and any future head)
            blocks_i.append(torch.zeros(n))
            blocks_t.append(torch.ones(n))

    m_img = torch.cat(blocks_i).to(device=device, dtype=torch.float32)
    m_txt = torch.cat(blocks_t).to(device=device, dtype=torch.float32)
    return m_img, m_txt


def sketch_multimodal_fused(grad: torch.Tensor, sketch_factory, masks):
    """
    Args:
        grad: flattened gradient [grad_size]
        sketch_factory: zero-arg callable returning a fresh CSVecFed with
            table zeroed (same d, c, r as global sketch)
        masks: (m_img, m_txt) same length as grad, binary partition

    Returns:
        Fused sketch table: t_img + t_txt  (same shape as single sketch)
    """
    m_img, m_txt = masks
    sk = sketch_factory()
    sk.accumulateVec(grad * m_img)
    t_img = sk.table.clone()
    sk.zero()
    sk.accumulateVec(grad * m_txt)
    t_txt = sk.table.clone()
    return t_img + t_txt


# --- Three-way: per-modality + concatenation / fusion-head sketches ---------

def build_multimodal_sketch_masks_triple(model: torch.nn.Module, device: torch.device):
    """
    Three disjoint masks (sum to 1):

      img:    img_extractor, img_refiner
      txt:    txt_extractor, txt_refiner
      fusion: img_to_txt, txt_to_img, integrator, classifier

    The "fusion" bucket is the gradient of everything after per-modality
    refinement (cross-modal + concat MLP + classifier), i.e. the head that
    acts on fused / concatenated representations.
    """
    blocks_i, blocks_t, blocks_f = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = p.numel()
        if name.startswith("img_extractor") or name.startswith("img_refiner"):
            blocks_i.append(torch.ones(n))
            blocks_t.append(torch.zeros(n))
            blocks_f.append(torch.zeros(n))
        elif name.startswith("txt_extractor") or name.startswith("txt_refiner"):
            blocks_i.append(torch.zeros(n))
            blocks_t.append(torch.ones(n))
            blocks_f.append(torch.zeros(n))
        else:
            blocks_i.append(torch.zeros(n))
            blocks_t.append(torch.zeros(n))
            blocks_f.append(torch.ones(n))

    m_img = torch.cat(blocks_i).to(device=device, dtype=torch.float32)
    m_txt = torch.cat(blocks_t).to(device=device, dtype=torch.float32)
    m_fus = torch.cat(blocks_f).to(device=device, dtype=torch.float32)
    return m_img, m_txt, m_fus


def sketch_multimodal_fused_triple(grad: torch.Tensor, sketch_factory, masks):
    """
    S(g) = S(g ⊙ m_img) + S(g ⊙ m_txt) + S(g ⊙ m_fus)

    Same CSVecFed geometry each time; linearity ⇒ equals one accumulateVec(grad).
    """
    m_img, m_txt, m_fus = masks
    sk = sketch_factory()
    sk.accumulateVec(grad * m_img)
    t_img = sk.table.clone()
    sk.zero()
    sk.accumulateVec(grad * m_txt)
    t_txt = sk.table.clone()
    sk.zero()
    sk.accumulateVec(grad * m_fus)
    t_fus = sk.table.clone()
    return t_img + t_txt + t_fus
