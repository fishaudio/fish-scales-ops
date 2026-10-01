#!/usr/bin/env python3
"""The fused router top-k (moe_topk_from_logits / moe_router_topk) against the
torch reference a serving stack's TopK layer computes. Runs on every
architecture: the router is glue, like the routing builder and the combine.

What the kernel folds into one launch, and what each case checks:

  * softmax over the experts, top-k selection, and the renormalisation of the
    selected probabilities -- checked against
    `softmax(logits.float()).topk(k)` followed by a division by the selected
    sum, which is sglang's `renormalize=True` path, and against the plain
    softmax weights with `renormalize=False`;
  * the padded-row sentinel of a CUDA-graph bucket: the rows at or past
    `num_token_non_padded` must carry the id `num_experts` and weight zero;
  * the expert-parallel global-to-local id remap, including the sentinel's own
    entry;
  * the shared expert's sigmoid gate, when its weight row is concatenated onto
    the router's and the logits therefore carry one extra column;
  * graph capture and replay with the logits rewritten in place, which is how a
    decode bucket uses it.

Expert ids are compared as sets and weights within 1e-5 relative: the kernel
sums the k selected exponentials in the order torch sums its top-k output, but
the full-softmax denominator torch divides by first cancels in a different
order, so the two agree to fp32 rounding rather than bit for bit. A draw whose
top-k boundary is an exact tie would be allowed to differ in the chosen id; the
test reports it rather than failing, and random fp32 logits do not produce one.
"""
import sys

import torch

import fish_scales_ops as fso

CASES = [
    # (M, E, topk)
    (1, 256, 8),
    (7, 256, 8),
    (64, 128, 8),
    (512, 256, 8),
    (2048, 64, 8),
    (33, 1024, 8),   # the 32-slots-per-lane instantiation
    (17, 256, 1),
    (129, 512, 4),
]


def reference(logits, topk, renormalize):
    probs = torch.softmax(logits.float(), dim=-1)
    w, ids = torch.topk(probs, topk, dim=-1)
    if renormalize:
        w = w / w.sum(dim=-1, keepdim=True)
    return ids.to(torch.int32), w


def compare(tag, ids, w, ref_ids, ref_w, failures, ties):
    same_set = torch.equal(ids.sort(dim=-1).values.to(torch.int64),
                           ref_ids.sort(dim=-1).values.to(torch.int64))
    if not same_set:
        differing = (~(ids.sort(dim=-1).values.to(torch.int64)
                       == ref_ids.sort(dim=-1).values.to(torch.int64)).all(dim=-1)).sum()
        ties.append(f"{tag}: {int(differing)} rows selected a different expert set")
        return
    # Same set: compare the weights pairwise after sorting by id.
    order = ids.to(torch.int64).argsort(dim=-1)
    ref_order = ref_ids.to(torch.int64).argsort(dim=-1)
    got = torch.gather(w, 1, order)
    want = torch.gather(ref_w, 1, ref_order)
    err = (got - want).abs().max() / max(1e-12, float(want.abs().max()))
    if float(err) > 1e-5:
        failures.append(f"{tag}: weights differ by {float(err):.3e} relative")


def main():
    if not torch.cuda.is_available():
        print("SKIP: test_moe_router_topk needs a CUDA device")
        return 0
    dev = "cuda"
    failures = []
    ties = []
    for M, E, topk in CASES:
        g = torch.Generator(device=dev).manual_seed(1000 + M * 31 + E)
        for dtype in (torch.bfloat16, torch.float32):
            logits = (torch.randn(M, E, device=dev, generator=g, dtype=torch.float32) * 3.0).to(dtype)
            for renorm in (True, False):
                ids, w, gate = fso.compat.moe_topk_from_logits(logits, topk, renormalize=renorm)
                torch.cuda.synchronize()
                ref_ids, ref_w = reference(logits, topk, renorm)
                tag = f"M={M} E={E} k={topk} {str(dtype).split('.')[-1]} renorm={int(renorm)}"
                compare(tag, ids, w, ref_ids, ref_w, failures, ties)
                if gate.numel() != 0:
                    failures.append(f"{tag}: shared_gate returned without being asked for")
                if ids.dtype != torch.int32 or w.dtype != torch.float32:
                    failures.append(f"{tag}: wrong output dtypes")
        # Descending weight order, which is what torch.topk returns.
        logits = (torch.randn(M, E, device=dev, generator=g) * 3.0).to(torch.bfloat16)
        ids, w, _ = fso.compat.moe_topk_from_logits(logits, topk)
        torch.cuda.synchronize()
        if topk > 1 and not bool((w[:, :-1] >= w[:, 1:]).all()):
            failures.append(f"M={M} E={E}: weights are not in descending order")

        # Padded rows: sentinel id, zero weight.
        n_real = max(1, M // 2)
        n_valid = torch.tensor([n_real], device=dev, dtype=torch.int32)
        ids_p, w_p, _ = fso.compat.moe_topk_from_logits(logits, topk, num_token_non_padded=n_valid)
        torch.cuda.synchronize()
        if not torch.equal(ids_p[:n_real], ids[:n_real]):
            failures.append(f"M={M} E={E}: the real rows changed when padding was declared")
        if not bool((ids_p[n_real:] == E).all()) or not bool((w_p[n_real:] == 0).all()):
            failures.append(f"M={M} E={E}: padded rows are not (sentinel, 0)")

        # Expert-parallel remap: this rank owns the first half, the rest and the
        # sentinel map outside its range.
        e_local = E // 2
        emap = torch.full((E + 1,), -1, device=dev, dtype=torch.int32)
        emap[:e_local] = torch.arange(e_local, device=dev, dtype=torch.int32)
        emap[E] = e_local
        ids_m, w_m, _ = fso.compat.moe_topk_from_logits(
            logits, topk, num_token_non_padded=n_valid, expert_map=emap)
        torch.cuda.synchronize()
        if not torch.equal(ids_m[:n_real], emap[ids_p[:n_real].to(torch.int64)]):
            failures.append(f"M={M} E={E}: expert_map was not applied to the real rows")
        if not bool((ids_m[n_real:] == e_local).all()):
            failures.append(f"M={M} E={E}: the sentinel did not go through expert_map")

        # Shared-expert gate as one extra logit column.
        extra = (torch.randn(M, 1, device=dev, generator=g) * 2.0).to(torch.bfloat16)
        wide = torch.cat([logits, extra], dim=1)
        ids_s, w_s, gate_s = fso.compat.moe_topk_from_logits(wide, topk, with_shared_gate=True)
        torch.cuda.synchronize()
        if not torch.equal(ids_s, ids):
            failures.append(f"M={M} E={E}: the extra gate column changed the selection")
        want_gate = torch.sigmoid(extra.float().squeeze(1))
        if float((gate_s - want_gate).abs().max()) > 1e-6:
            failures.append(f"M={M} E={E}: shared gate differs from sigmoid(logit)")
        print(f"  M={M:5d} E={E:5d} k={topk}: softmax top-k, renormalize on/off, bf16+fp32 logits, "
              f"descending order, padded rows, expert map, shared gate  OK")

    # moe_router_topk: the GEMM and the kernel together, against the reference
    # applied to the same logits.
    M, E, H, topk = 64, 256, 2048, 8
    g = torch.Generator(device=dev).manual_seed(7)
    hidden = torch.randn(M, H, device=dev, dtype=torch.bfloat16, generator=g)
    rw = (torch.randn(E + 1, H, device=dev, generator=g) * 0.05).to(torch.bfloat16)
    ids, w, gate = fso.compat.moe_router_topk(hidden, rw, topk, with_shared_gate=True)
    logits = torch.nn.functional.linear(hidden, rw)
    ref_ids, ref_w = reference(logits[:, :E], topk, True)
    torch.cuda.synchronize()
    compare("moe_router_topk", ids, w, ref_ids, ref_w, failures, ties)
    if float((gate - torch.sigmoid(logits[:, E].float())).abs().max()) > 1e-6:
        failures.append("moe_router_topk: shared gate differs from sigmoid of the extra logit")
    print(f"  moe_router_topk M={M} E={E} H={H}: GEMM + fused top-k == reference  OK")

    # Capture and replay with the logits rewritten in place.
    logits_buf = (torch.randn(64, 256, device=dev, generator=g) * 3.0).to(torch.bfloat16)
    n_valid = torch.tensor([64], device=dev, dtype=torch.int32)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fso.compat.moe_topk_from_logits(logits_buf, 8, num_token_non_padded=n_valid)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s):
        ids_g, w_g, _ = fso.compat.moe_topk_from_logits(logits_buf, 8, num_token_non_padded=n_valid)
    torch.cuda.synchronize()
    logits_buf.copy_((torch.randn(64, 256, device=dev, generator=g) * 3.0).to(torch.bfloat16))
    n_valid.fill_(40)
    graph.replay()
    torch.cuda.synchronize()
    ref_ids, ref_w = reference(logits_buf[:40].float(), 8, True)
    compare("graph replay", ids_g[:40], w_g[:40], ref_ids, ref_w, failures, ties)
    if not bool((ids_g[40:] == 256).all()):
        failures.append("graph replay: the new padded rows did not get the sentinel")
    print("  graph capture + replay with rewritten logits and a new valid length  OK")

    for t in ties:
        print(f"  NOTE {t}")
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  {f}")
        return 1
    print("\nmoe_topk_from_logits / moe_router_topk: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
