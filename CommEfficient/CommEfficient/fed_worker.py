from CommEfficient.cmvec import CMVec
from CommEfficient.cmvec1 import CMVec1
from CommEfficient.cmvec2 import CMVec2
from CommEfficient.csvecN import CSVecFed
from CommEfficient.mn import MN
from mm_sketch_fusion import (
    build_multimodal_sketch_masks,
    sketch_multimodal_fused,
    build_multimodal_sketch_masks_triple,
    sketch_multimodal_fused_triple,
)
from mm_sketch_separated import (
    get_index_maps,
    sketch_modality_separated,
)
import torch
import numpy as np
import ctypes
# from CommEfficient.cmvec1 import CMVec
from utils import get_param_vec, set_param_vec, get_grad, _topk, clip_grad
import copy
import os
import time
import math
import torch.multiprocessing as multiprocessing
from csvec import CSVec
import torch.distributed as dist
import queue

_grad_debug_counter = [0]

def _print_grad_stats(grad, call_idx):
    """Print gradient statistics before sketch compression."""
    g = grad.detach()
    n = g.numel()
    n_pos = (g > 0).sum().item()
    n_neg = (g < 0).sum().item()
    n_zero = (g == 0).sum().item()

    # print(f"\n{'='*60}")
    # print(f"GRADIENT BEFORE SKETCH  (call #{call_idx})")
    # print(f"{'='*60}")
    # print(f"  Shape:    {tuple(g.shape)}   ({n:,} elements)")
    # print(f"  Range:    [{g.min().item():.6e},  {g.max().item():.6e}]")
    # print(f"  Mean:     {g.mean().item():.6e}")
    # print(f"  Std:      {g.std().item():.6e}")
    # print(f"  L2 norm:  {torch.norm(g).item():.6e}")
    # print(f"  Positive: {n_pos:>10,}  ({100*n_pos/n:.2f}%)")
    # print(f"  Negative: {n_neg:>10,}  ({100*n_neg/n:.2f}%)")
    # print(f"  Zero:     {n_zero:>10,}  ({100*n_zero/n:.2f}%)")

    abs_g = g.abs()
    # print(f"  |g| percentiles:  "
    #       f"50%={torch.quantile(abs_g.float(), 0.5).item():.6e}  "
    #       f"90%={torch.quantile(abs_g.float(), 0.9).item():.6e}  "
    #       f"99%={torch.quantile(abs_g.float(), 0.99).item():.6e}  "
    #       f"max={abs_g.max().item():.6e}")

    # print(f"  First 20 values:  {g[:20].cpu().tolist()}")
    # print(f"  Last  20 values:  {g[-20:].cpu().tolist()}")

    topk_vals, topk_idx = g.abs().topk(10)
    # print(f"  Top-10 by |value|:")
    # for rank, (idx, val) in enumerate(zip(topk_idx.tolist(),
    #                                       topk_vals.tolist())):
    #     print(f"    #{rank}: index={idx:>8,}  value={g[idx].item():+.6e}")
    # print(f"{'='*60}\n")


def worker_loop(input_model, ps_weights, client_weights, client_errors,
                client_velocities, batches_queue, results_queue, fedavg_lr,
                rank, world_size, compute_loss_train, compute_loss_val,
                args):
    torch.cuda.set_device(rank - 1)

    model = input_model.to(args.device)

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(args.port)
    # torch.distributed.init_process_group("nccl", rank=rank,
    #                                      world_size=world_size)
    if args.device == "cpu" or args.num_devices == 1:
        backend = "gloo"
    else:
        backend = "nccl"

    torch.distributed.init_process_group(
        backend=backend,
        init_method="tcp://127.0.0.1:29500",  # ✅ same rendezvous address
        rank=rank,
        world_size=world_size
    )


    print_counter = 0
    max_debug_prints = 5



    while True:
        try:
            # batches is a list of batches that we should process
            # as if we were different workers for each batch
            # each batch in batches will have data belonging to a
            # single client (asserted in process_batch)
            batches = batches_queue.get(timeout=250)
        except queue.Empty:
            print("batch queue was empty")
            return
        if batches is None:
            # reached the end of training
            break

        # get the latest weights from the parameter server
        local_ps_weights = ps_weights.clone().to(args.device)

        # sum the gradient over all batches
        if args.mode in ["uncompressed", "true_topk",
                         "local_topk", "fedavg"]:
            shape = (args.grad_size,)
        elif args.mode == "sketch":
            shape = (args.num_rows, args.num_cols)
        sum_g = torch.zeros(shape).to(args.device).float()

        # first batch, first tensor (client_indices), first datum
        is_train = batches[0][0][0] != -1

        # this is the starting learning rate (which possibly decays) when
        # carrying out fedavg
        lr = fedavg_lr.to(args.device)

        all_results = []
        # loop over workers we have to process (see comment above)
        for batch in batches:
            if args.mode == "fedavg" and is_train:
                assert args.error_type == "none"
                assert args.local_momentum == 0

                original_ps_weights = local_ps_weights.clone()
                # split "batch", which is this client's entire dataset,
                # into smaller batches to run local SGD on
                if args.fedavg_batch_size == -1:
                    local_batches = [batch]
                    n_batches = 1
                else:
                    local_batches = [torch.split(t, args.fedavg_batch_size)
                                     for t in batch]
                    n_batches = len(local_batches[0])
                    local_batches = [tuple(split[i]
                                           for split in local_batches)
                                     for i in range(n_batches)]

                n_steps = n_batches * args.num_fedavg_epochs
                step = 0
                accum_results = None
                for epoch in range(args.num_fedavg_epochs):
                    for local_batch in local_batches:
                        g, results = process_batch(
                                local_batch, model, local_ps_weights,
                                client_weights,
                                client_errors, client_velocities,
                                compute_loss_train, compute_loss_val, args
                            )
                        if accum_results is None:
                            accum_results = results
                        else:
                            # accumulate results
                            for i in range(len(accum_results)):
                                accum_results[i] += results[i]
                        # g is the sum of gradients over examples, but
                        # we need to update the model with the avg grad
                        g /= local_batch[0].size()[0]
                        decay = args.fedavg_lr_decay ** step
                        local_ps_weights -= g * lr * decay
                        step += 1
                # compute average results from accum_results
                results = [r / n_steps for r in accum_results]
                g = original_ps_weights - local_ps_weights
                # weight by the batch size (which in the case of fedavg
                # is the client's dataset size) so that clients without
                # much data are downweighted
                g *= batch[0].size()[0]

                # reset local_ps_weights so that if this process has
                # to process another worker batch, the next worker
                # starts from the correct weights
                local_ps_weights[:] = original_ps_weights[:]

            else:
                # for all non-fedavg modes, we just do a single step
                if args.do_test:
                    # daniel says don't commit debugging code but i don't want to type this out everytime 
                    if is_train:
                        g, results = torch.ones(args.grad_size).to(args.device), tuple(1.0 for _ in range(args.num_results_train))
                    else:
                        g, results = torch.ones(args.grad_size).to(args.device), tuple(1.0 for _ in range(args.num_results_val))
                else:
                    g, results = process_batch(
                            batch, model, local_ps_weights, client_weights,
                            client_errors, client_velocities,
                            compute_loss_train, compute_loss_val, args, 
                        )

            if is_train:
                sum_g += g
            all_results.append(results)

        results_queue.put(all_results)

        if is_train:
            # reduce the locally summed g across devices
            torch.distributed.reduce(sum_g, 0)

def process_batch(batch, model, ps_weights, client_weights,
                  client_errors, client_velocities,
                  compute_loss_train, compute_loss_val, args):
        client_indices = batch[0]
        is_train = client_indices[0] != -1
        batch = batch[1:]
        batch = [t.to(args.device) for t in batch]
        assert (client_indices - client_indices[0]).abs().sum() == 0
        client_id = client_indices[0]

        # figure out what model weights this worker should use
        new_worker_weights = None
        if args.do_topk_down:
            worker_weights = client_weights[client_id].to(args.device)
            new_worker_weights = get_new_worker_weights(ps_weights,
                                                        worker_weights,
                                                        args)
            new_worker_weights = new_worker_weights.to(args.device)
        else:
            new_worker_weights = ps_weights

        # get model ready
        set_param_vec(model, new_worker_weights)

        transmit = None
        if is_train:
            model.train()
            model.zero_grad()
            # get our client's local velocity & local error vectors
            velocity = None
            error = None
            if client_velocities is not None:
                velocity = client_velocities[client_id].to(args.device)
            if client_errors is not None:
                error = client_errors[client_id].to(args.device)

            results, transmit = local_step(model, velocity, error, batch,
                                           compute_loss_train, args)
        else:
            model.eval()
            results = forward_grad(model, batch, compute_loss_val, args,
                                   compute_grad=False)
        return transmit, results

def local_step(model, velocity, error, batch, compute_loss, args):
    # g is a (possibly compressed) gradient
    g, results = forward_grad(model, batch, compute_loss, args)

    # locally, we need to deal with the sum of gradients across
    # examples, since we will torch.distributed.reduce the to_transmits,
    g *= batch[0].size(0)

    # if needed, do local momentum
    if args.local_momentum > 0:
        # this does velocity[:] = m * velocity + g, but twice as fast
        torch.add(g, velocity, alpha=args.local_momentum, out=velocity)

    # if needed, do local error correction
    if args.error_type == "local":
        error += velocity if velocity is not None else g
        to_transmit = error
    else:
        to_transmit = velocity if velocity is not None else g

    if args.mode == "local_topk":
        assert args.error_type in ["local", "none"]
        # topk is impossibly slow on CPU, very fast on GPU
        to_transmit = _topk(to_transmit.to(args.device), k=args.k)

        nz = to_transmit.nonzero()
        if error is not None:
            # error feedback
            error[nz] = 0

        # if we're doing local momentum, do momentum factor masking
        if args.local_momentum > 0:
            velocity[nz] = 0

    # sketched sgd with local error accumulation doesn't really make
    # sense, since when we send a sketch we don't know what portion
    # of the sketch is the "error"
    if error is not None:
        assert args.mode not in ["sketch", "uncompressed"]

    # we want to do momentum factor masking for all the compression
    # methods, but that's not possible to do for sketching, since
    # it's unknown which coordinates to mask out
    if velocity is not None:
        assert args.mode != "sketch"

    return results, to_transmit

def get_new_worker_weights(ps_weights, worker_weights, args):
    device = args.device

    ps_weights = ps_weights.to(device)
    worker_weights = worker_weights.to(device)

    # we'll update the old worker_weights with a possibly compressed
    # version of diff_vec
    diff_vec = ps_weights - worker_weights
    if args.do_topk_down:
        weight_update = _topk(diff_vec, k=args.k)
    else:
        weight_update = diff_vec

    new_worker_weights = worker_weights + weight_update
    return new_worker_weights



def forward_grad(model, batch, compute_loss, args, compute_grad=True):

    print_counter = 0

    device = args.device

    # divide up batch (for gradient accumulation when memory constrained)
    #num_shards = args.num_train_batch_shards
    # need the max(1, ...) since the last batch in an epoch might be small
    #microbatch_size = max(1, batch[0].size()[0] // num_shards)
    if args.microbatch_size > 0:
        microbatch_size = min(batch[0].size()[0], args.microbatch_size)
    else:
        microbatch_size = batch[0].size()[0]

    # accumulators for the loss & metric values
    accum_loss = 0
    accum_metrics = None

    num_iters = math.ceil(batch[0].size()[0] / microbatch_size)
    for i in range(num_iters):
        # extract current microbatch
        start = i * microbatch_size
        end = (i+1) * microbatch_size
        microbatch = [t[start:end] for t in batch]

        # forward pass
        loss, *metrics = compute_loss(model, microbatch, args)

        # if first time through, we find out how many metrics there are
        if accum_metrics is None:
            accum_metrics = [0 for _ in metrics]

        # accumulate loss & metrics, weighted by how many data points
        # were actually used
        accum_loss += loss.item() * microbatch[0].size()[0]
        for i, m in enumerate(metrics):
            accum_metrics[i] += m.item() * microbatch[0].size()[0]

        # backward pass
        if compute_grad:
            loss.backward()

    # gradient clipping
    if compute_grad and args.max_grad_norm is not None and args.mode not in ["sketch"]:
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       args.max_grad_norm * num_iters)

    # "average" here is over the data in the batch
    average_loss = accum_loss / batch[0].size()[0]
    average_metrics = [m / batch[0].size()[0] for m in accum_metrics]

    results = [average_loss] + average_metrics

    if not compute_grad:
        return results

    grad = get_grad(model, args)

    # Shift gradients upward so all are >= 0 (For CM sketching)
    # grad -= grad.min()

    if args.do_dp:
        grad = clip_grad(args.l2_norm_clip, grad)
        if args.dp_mode == "worker":
            noise = torch.normal(mean=0, std=args.noise_multiplier, size=grad.size()).to(args.device)
            noise *= np.sqrt(args.num_workers)
            grad += noise

    # print("Gradient stats - min: {:.6f}, max: {:.6f}, mean: {:.6f}, std: {:.6f}".format(
    #     grad.min().item(), grad.max().item(), grad.mean().item(), grad.std().item()
    # ))
    
    # print("Gradient shape:", grad.shape)
    # print("First 10 gradient values:", grad[:10].cpu().numpy())
    # print("Last 10 gradient values:", grad[-10:].cpu().numpy())
    # print("----")
    
    # compress the gradient if needed
    if args.mode == "sketch":
        is_mm = type(model).__name__ == "MultiModalNet"
        use_sep = getattr(args, "mm_sketch_separated", False) and is_mm
        use_tri = getattr(args, "mm_sketch_fusion_tri", False) and is_mm and not use_sep
        use_pair = (getattr(args, "mm_sketch_fusion", False) and is_mm
                    and not use_tri and not use_sep)

        if use_sep:
            if not hasattr(model, "_fed_mm_sep_index_maps"):
                model._fed_mm_sep_index_maps = get_index_maps(
                    args, args.device
                )

            if _grad_debug_counter[0] < 5:
                _print_grad_stats(grad, _grad_debug_counter[0])
                _grad_debug_counter[0] += 1

            g = sketch_modality_separated(
                grad,
                model._fed_mm_sep_index_maps,
                num_cols=args.num_cols,
                num_rows=args.num_rows,
                num_blocks=args.num_blocks,
                device=args.device,
            )

            if compute_grad and args.max_grad_norm is not None:
                g = clip_grad(args.max_grad_norm, g)
        elif use_tri:
            if not hasattr(model, "_fed_mm_sketch_masks_tri"):
                model._fed_mm_sketch_masks_tri = (
                    build_multimodal_sketch_masks_triple(model, args.device)
                )
            masks = model._fed_mm_sketch_masks_tri

            if _grad_debug_counter[0] < 5:
                _print_grad_stats(grad, _grad_debug_counter[0])
                _grad_debug_counter[0] += 1

            def _sketch_factory_tri():
                s = CSVecFed(
                    d=args.grad_size,
                    c=args.num_cols,
                    r=args.num_rows,
                    device=args.device,
                    numBlocks=args.num_blocks,
                )
                return s

            g = sketch_multimodal_fused_triple(grad, _sketch_factory_tri, masks)

            if compute_grad and args.max_grad_norm is not None:
                g = clip_grad(args.max_grad_norm, g)
        elif use_pair:
            if not hasattr(model, "_fed_mm_sketch_masks"):
                model._fed_mm_sketch_masks = build_multimodal_sketch_masks(
                    model, args.device
                )
            masks = model._fed_mm_sketch_masks

            if _grad_debug_counter[0] < 5:
                _print_grad_stats(grad, _grad_debug_counter[0])
                _grad_debug_counter[0] += 1

            def _sketch_factory():
                s = CSVecFed(
                    d=args.grad_size,
                    c=args.num_cols,
                    r=args.num_rows,
                    device=args.device,
                    numBlocks=args.num_blocks,
                )
                return s

            g = sketch_multimodal_fused(grad, _sketch_factory, masks)

            if compute_grad and args.max_grad_norm is not None:
                g = clip_grad(args.max_grad_norm, g)
        else:
            sketch = CSVecFed(d=args.grad_size, c=args.num_cols,
                r=args.num_rows, device=args.device,
                numBlocks=args.num_blocks)
            
            #Sketch for MN:
            # sketch = MN(d=args.grad_size, c=args.num_cols,
            #         r=args.num_rows, device=args.device,
            #         numBlocks=args.num_blocks,
            #         use_mn=args.use_mn,
            #         m=args.mn_num_fake_items)

            if _grad_debug_counter[0] < 5:
                _print_grad_stats(grad, _grad_debug_counter[0])
                _grad_debug_counter[0] += 1

            sketch.accumulateVec(grad)


            # # Measure noise after accumulating gradients
            # if args.use_mn:
            #     measured_noise = sketch.measure_noise()

            # --- Debug: Compare CM sketch table vs actual gradient ---

            # if print_counter < 1:

            #     # Actual flattened gradient (the input to sketch)
            #     actual_grad = grad.detach().cpu()

            #     # Estimated gradient recovered from sketch
            #     estimated_grad = sketch.unSketch(k=args.k).detach().cpu()

            #     # For safe indexing
            #     n = actual_grad.numel()

            #     # Print first 10
            #     print("\n--- First 10 gradients ---")
            #     for i in range(10):
            #         print(f"Index {i:6d}: Actual={actual_grad[i]:.6e}, Estimated={estimated_grad[i]:.6e}")

            #     # Print last 10
            #     print("\n--- Last 10 gradients ---")
            #     for i in range(n-10, n):
            #         print(f"Index {i:6d}: Actual={actual_grad[i]:.6e}, Estimated={estimated_grad[i]:.6e}")

            #     # Optional: check global errors
            #     abs_error = torch.abs(actual_grad - estimated_grad)
            #     print(f"\nMean abs error: {abs_error.mean().item():.6e}")
            #     print(f"Max abs error: {abs_error.max().item():.6e}")

            #     print_counter += 1



            # gradient clipping
            if compute_grad and args.max_grad_norm is not None:
                sketch = clip_grad(args.max_grad_norm, sketch)
            g = sketch.table
    elif args.mode == "true_topk":
        g = grad
    elif args.mode == "local_topk":
        # ideally we'd return the compressed version of the gradient,
        # i.e. _topk(grad, k=args.k). However, for sketching we do momentum
        # in the sketch, whereas for topk we do momentum before taking topk
        # so we have to return an inconsistent quantity here
        g = grad
    elif args.mode == "fedavg":
        # logic for doing fedavg happens in process_batch
        g = grad
    elif args.mode == "uncompressed":
        g = grad

    return g, results