"""FlashAttention-4 SM100 (Blackwell tcgen05) with fused MXFP8 1x32 output epilogue and PDL.

Eliminates:
  - the BF16 attention output buffer write (halves O store bandwidth: 1B/elem vs 2B/elem),
  - the separate activation quantization kernel launch before out_proj (32 launches/tower),
  - the non-PDL stream barrier between QKV -> Attention -> out_proj.

Outputs (attn_xq [M, H*64] float8_e4m3fn, attn_sx [align(M, 128) * (H*64/128)] int32)
conforming to fso/activation_contract.h (SM100 1-D atom layout, UE8M0 1x32 scales,
zeroed padded rows [M, align(M, 128))).

Constraints: head_dim == 64 (exact 2 MXFP8 blocks per head), nheads even (pair of heads =
1 int32 scale word = 2 disjoint aligned uint16_t stores, zero atomics).
"""
from __future__ import annotations

import inspect
import math
import os
from functools import partial
from typing import Callable, Optional

import cuda.bindings.driver as cuda_drv
import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Float32, Int32, Uint16, Uint32, const_expr
from cutlass._mlir.dialects import arith
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack
import cutlass.utils.blackwell_helpers as sm100_utils_basic
import cutlass.pipeline as pipeline
import torch

from flash_attn.cute import config as fa_cfg
from quack import copy_utils
from flash_attn.cute import utils as fa_utils
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute.cute_dsl_utils import to_cute_tensor
from flash_attn.cute.flash_fwd_sm100 import (
    DescaleTensors,
    FlashAttentionForwardSm100,
    NamedBarrierFwdSm100,
)
from flash_attn.cute.named_barrier import NamedBarrierFwdSm100 as _NB  # noqa: F401
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.softmax import apply_learnable_sink

from .._env import env_flag


def _sel(cond, a, b, *, loc=None, ip=None):
    return type(a)(
        arith.select(
            cond.ir_value(loc=loc, ip=ip),
            a.ir_value(loc=loc, ip=ip),
            b.ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    )


def _f32_to_u32(x, *, loc=None, ip=None):
    return Uint32(arith.bitcast(Uint32.mlir_type, x.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


def _u32_to_f32(x, *, loc=None, ip=None):
    return Float32(arith.bitcast(Float32.mlir_type, x.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


def _fso_e8m0_from_amax(amax: Float32):
    """Exact bit-identical UE8M0 byte + quant_scale = 2^(127 - byte) (dense_epi_math.cuh)."""
    amax_f = _sel(amax > Float32(1e-10), amax, Float32(1e-10))
    bits = _f32_to_u32(amax_f)
    e = (bits >> Uint32(23)) & Uint32(0xFF)
    f = bits & Uint32(0x7FFFFF)
    sub = _sel(f <= Uint32(0x600000), Int32(8), Int32(7))
    b = _sel(e == Uint32(255), Int32(254), Int32(e) - sub)
    b = _sel(b < Int32(0), Int32(0), _sel(b > Int32(254), Int32(254), b))
    qs_bits = _sel(
        b == Int32(0),
        Uint32(0x3F800000),
        _sel(b == Int32(254), Uint32(0x00400000), Uint32(Int32(254) - b) << Uint32(23)),
    )
    qs = _u32_to_f32(qs_bits)
    return b, qs


class FlashAttentionForwardSm100MxFP8(FlashAttentionForwardSm100):
    """Subclass of FA4 SM100 that quantizes O (D=64) to MXFP8 1x32 in correction_epilogue
    and writes SM100 atom UE8M0 scale halfwords directly into mLSE (repurposed as mSFO_u16)."""

    def __init__(self, *args, nheads_total: int = 20, enable_pdl: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.head_dim_v_padded == 64, "MXFP8 fused FA4 epilogue requires head_dim_v == 64"
        assert nheads_total % 2 == 0, "nheads must be even"
        self.nheads_total = nheads_total
        self.num_kp = nheads_total // 2
        self.enable_pdl = enable_pdl

    @cute.jit
    def load(self, *args, **kwargs):
        if const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()
        res = super().load(*args, **kwargs)
        # All Q/K/V TMA loads are now issued; release downstream weights-first out_proj
        # so it prefetches static W_o onto idle SMs while MMA/softmax/epilogue drain.
        # out_proj still waits at griddepcontrol.wait before reading attn_xq / attn_sx.
        if const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()
        return res

    @cute.jit
    def epilogue_s2g(self, *args, **kwargs):
        return super().epilogue_s2g(*args, **kwargs)

    @cute.jit
    def correction_loop(
        self,
        thr_mma_qk: cute.ThrMma,
        thr_mma_pv: cute.ThrMma,
        tStS: cute.Tensor,
        tOtO: cute.Tensor,
        sScale: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,  # Repurposed as mSFO_u16: shape (M_pad * num_kp * 2, 1, 1)
        sO: cute.Tensor,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_o_acc: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        pipeline_o_epi: pipeline.PipelineAsync,
        pipeline_load_epi: Optional[pipeline.PipelineAsync],
        learnable_sink: Optional[cute.Tensor],
        descale_tensors: Optional[DescaleTensors],
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: cute.CopyAtom,
        softmax_scale_log2: Float32,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        blocksparse_tensors=None,
        tile_scheduler=None,
    ):
        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.correction_warp_ids))
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        mma_tile_coord_v = thr_mma_qk.thr_idx

        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tStScale_layout = cute.composition(tStS.layout, cute.make_layout((self.m_block_size, 1)))
        tStScales = tuple(
            cute.make_tensor(tStS.iterator + self.tmem_vec_offset[stage], tStScale_layout)
            for stage in range(self.q_stage)
        )
        tScScale = cute.composition(tScS, cute.make_layout((self.m_block_size, 1)))
        tmem_load_v_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(1)), self.qk_acc_dtype
        )
        thr_tmem_load_vec = tcgen05.make_tmem_copy(tmem_load_v_atom, tStScales[0]).get_slice(tidx)
        tStScales_t2r = [thr_tmem_load_vec.partition_S(tStScales[stage]) for stage in range(self.q_stage)]
        tSrScale_t2r_shape = thr_tmem_load_vec.partition_D(tScScale).shape

        for stage in cutlass.range(self.q_stage):
            pipeline_s_p_o.consumer_release_w_index(stage)

        sm_stats_consumer_phase = Int32(0)
        o_corr_consumer_phase = Int32(0)
        corr_epi_producer_phase = Int32(1)
        load_epi_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
        o_acc_consumer_count = Int32(0)

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            work_m, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, work_m, split_idx=split_idx, num_splits=num_splits
            )

            mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, head_idx]
            gO = None
            if const_expr(not self.use_tma_O):
                tiler_gO = ((self.mma_tiler_pv[0] * self.q_stage), self.head_dim_v_padded)
                gO = cute.local_tile(mO_cur, tiler_gO, (work_m, 0))
                gO = fa_utils.select(cute.flat_divide(gO, (self.mma_tiler_pv[0],)), mode=[0, 2, 1])
                gO = cute.flat_divide(gO, (self.mma_tiler_pv[0] // self.cta_group_size,))[
                    None, mma_tile_coord_v, None, None
                ]

            if const_expr(self.use_correction_warps_for_epi and self.overlap_sO_sQ and self.is_persistent):
                pipeline_load_epi.producer_acquire(load_epi_producer_state)

            # Ignore first signal from softmax as no correction is required (matches flash_fwd_sm100.py)
            if const_expr(not self.use_s_ping_pong):
                for stage in cutlass.range_constexpr(self.q_stage):
                    sm_stats_barrier.arrive_and_wait_w_index(index=stage * 4 + warp_idx)
                    if const_expr(stage == 0 or not self.sm_stats_cross_stage):
                        pipeline_sm_stats.consumer_release_w_index(stage)
            sm_stats_consumer_phase ^= 1

            seqlen_corr_loop_steps = n_block_max - n_block_min - 1
            for _ in cutlass.range(seqlen_corr_loop_steps, unroll=1):
                for stage in cutlass.range_constexpr(self.q_stage):
                    sm_stats_index = (o_acc_consumer_count & 1) if const_expr(self.use_s_ping_pong) else stage
                    sm_stats_barrier.arrive_and_wait_w_index(index=sm_stats_index * 4 + warp_idx)
                    scale = sScale[tidx + sm_stats_index * self.m_block_size]
                    if const_expr(not self.use_s_ping_pong and not self.sm_stats_cross_stage):
                        pipeline_sm_stats.consumer_release_w_index(stage)
                    should_rescale = cute.arch.vote_ballot_sync(scale < Float32(1.0)) != 0
                    if should_rescale:
                        if const_expr(self.use_s_ping_pong):
                            pipeline_o_acc.consumer_wait_w_index_phase(
                                o_acc_consumer_count & 1, (o_acc_consumer_count >> 1) & 1
                            )
                        o_tmem = tOtO[None, None, None, (o_acc_consumer_count & 1) if const_expr(self.use_s_ping_pong) else stage]
                        self.correction_rescale(thr_mma_pv, o_tmem, tidx, scale)
                    if const_expr(self.use_s_ping_pong):
                        pipeline_s_p_o.consumer_release_w_index((o_acc_consumer_count + 1) & 1)
                        pipeline_sm_stats.consumer_release_w_index(sm_stats_index)
                        o_acc_consumer_count += 1
                    else:
                        pipeline_s_p_o.consumer_release_w_index(stage)
                        if const_expr(self.sm_stats_cross_stage):
                            pipeline_sm_stats.consumer_release_w_index(self.q_stage - 1 - stage)
                sm_stats_consumer_phase ^= 1
            if const_expr(self.q_stage == 2 and self.sm_stats_cross_stage):
                pipeline_sm_stats.consumer_release_w_index(1)

            for stage in cutlass.range_constexpr(self.q_stage):
                sm_stats_index = (o_acc_consumer_count & 1) if const_expr(self.use_s_ping_pong) else stage
                sm_stats_barrier.arrive_and_wait_w_index(index=sm_stats_index * 4 + warp_idx)
                row_sum = sScale[tidx + sm_stats_index * self.m_block_size]
                pipeline_sm_stats.consumer_release_w_index(sm_stats_index)
                acc_zero = row_sum == Float32(0.0) or row_sum != row_sum
                scale = cute.arch.rcp_approx(row_sum if not acc_zero else Float32(1.0))
                if const_expr(self.use_s_ping_pong):
                    pipeline_o_acc.consumer_wait_w_index_phase(
                        o_acc_consumer_count & 1, (o_acc_consumer_count >> 1) & 1
                    )
                else:
                    pipeline_o_acc.consumer_wait_w_index_phase(stage, o_corr_consumer_phase)
                if const_expr(not self.use_correction_warps_for_epi):
                    pipeline_o_epi.producer_acquire_w_index_phase(stage, corr_epi_producer_phase)
                gO_stage = gO[None, None, stage] if const_expr(gO is not None) else None
                o_tmem = tOtO[None, None, None, (o_acc_consumer_count & 1) if const_expr(self.use_s_ping_pong) else stage]
                self.correction_epilogue_mxfp8(
                    thr_mma_pv,
                    o_tmem,
                    tidx,
                    stage,
                    work_m,
                    head_idx,
                    seqlen.seqlen_q,
                    scale,
                    sO[None, None, stage],
                    mLSE,
                    mO_cur,
                    gO_stage,
                    gmem_tiled_copy_O,
                )
                if const_expr(self.use_s_ping_pong):
                    pipeline_s_p_o.consumer_release_w_index((o_acc_consumer_count + 1) & 1)
                    o_acc_consumer_count += 1
                else:
                    pipeline_s_p_o.consumer_release_w_index(stage)
                if const_expr(not self.use_correction_warps_for_epi):
                    pipeline_o_epi.producer_commit_w_index(stage)

            o_corr_consumer_phase ^= 1
            sm_stats_consumer_phase ^= 1
            corr_epi_producer_phase ^= 1

            if const_expr(pipeline_load_epi is not None and self.use_correction_warps_for_epi):
                pipeline_load_epi.producer_acquire(load_epi_producer_state)
                with cute.arch.elect_one():
                    pipeline_load_epi.producer_commit(load_epi_producer_state)
                load_epi_producer_state.advance()

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

        if const_expr(self.use_correction_warps_for_epi and self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()

    @cute.jit
    def correction_epilogue_mxfp8(
        self,
        thr_mma: cute.ThrMma,
        tOtO: cute.Tensor,
        tidx: Int32,
        stage: Int32,
        work_m: Int32,
        head_idx: Int32,
        seqlen_q: Int32,
        scale: Float32,
        sO: cute.Tensor,
        mSFO_u16: cute.Tensor,
        mO_cur: Optional[cute.Tensor] = None,
        gO: Optional[cute.Tensor] = None,
        gmem_tiled_copy_O: Optional[cute.TiledCopy] = None,
    ):
        # For o_dtype = Float8E4M3FN (width=8), corr_tile_size = 32 = exact 1 MXFP8 block!
        corr_tile_size = 32
        tOsO = thr_mma.get_slice(0).partition_C(sO)
        tOcO = thr_mma.partition_C(cute.make_identity_tensor(self.mma_tiler_pv[:2]))

        tOtO_i = cute.logical_divide(tOtO, cute.make_layout((self.m_block_size, corr_tile_size)))
        tOcO_i = cute.logical_divide(tOcO, cute.make_layout((self.m_block_size, corr_tile_size)))
        tOsO_i = cute.logical_divide(tOsO, cute.make_layout((self.m_block_size, corr_tile_size)))

        epi_subtile = (self.epi_tile[0], corr_tile_size)
        tmem_copy_atom = sm100_utils_basic.get_tmem_load_op(
            self.mma_tiler_pv,
            self.o_layout,
            self.o_dtype,
            self.pv_acc_dtype,
            epi_subtile,
            use_2cta_instrs=self.use_2cta_instrs,
        )
        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_copy_atom, tOtO_i[(None, None), 0])
        thr_tmem_load = tiled_tmem_load.get_slice(tidx)
        smem_copy_atom = sm100_utils_basic.get_smem_store_op(
            self.o_layout, self.o_dtype, self.pv_acc_dtype, tiled_tmem_load
        )
        tiled_smem_store = cute.make_tiled_copy_D(smem_copy_atom, tiled_tmem_load)

        tOtO_t2r = thr_tmem_load.partition_S(tOtO_i[(None, None), None])
        tOsO_s2r = copy_utils.partition_D_position_independent(thr_tmem_load, tOsO_i[(None, None), None])
        tOcO_t2r = thr_tmem_load.partition_D(tOcO_i[(None, None), None])

        # D = 64 -> exactly 2 blocks of 32 elements (i = 0, i = 1) per row!
        b0 = Int32(0)
        b1 = Int32(0)
        for i in cutlass.range_constexpr(2):
            tOtO_t2r_i = tOtO_t2r[None, 0, 0, i]
            tOsO_r2s_i = tOsO_s2r[None, 0, 0, i]
            tOrO_frg = cute.make_rmem_tensor(tOcO_t2r[None, 0, 0, i].shape, self.pv_acc_dtype)
            cute.copy(tiled_tmem_load, tOtO_t2r_i, tOrO_frg)
            amax = Float32(0.0)
            for j in cutlass.range_constexpr(0, 32, 2):
                v0, v1 = cute.arch.mul_packed_f32x2((tOrO_frg[j], tOrO_frg[j + 1]), (scale, scale))
                tOrO_frg[j] = v0
                tOrO_frg[j + 1] = v1
                a0 = cute.math.abs(v0)
                a1 = cute.math.abs(v1)
                amax = _sel(a0 > amax, a0, amax)
                amax = _sel(a1 > amax, a1, amax)
            byte_i, qs = _fso_e8m0_from_amax(amax)
            if const_expr(i == 0):
                b0 = byte_i
            else:
                b1 = byte_i
            for j in cutlass.range_constexpr(0, 32, 2):
                q0, q1 = cute.arch.mul_packed_f32x2((tOrO_frg[j], tOrO_frg[j + 1]), (qs, qs))
                tOrO_frg[j] = q0
                tOrO_frg[j + 1] = q1
            copy_utils.cvt_copy(tiled_smem_store, tOrO_frg, tOsO_r2s_i)
        cute.arch.fence_view_async_shared()

        # Store 2 UE8M0 bytes (aligned uint16_t) into SM100 atom layout for this (q_row, head_idx).
        # Padded rows [seqlen_q, align(seqlen_q, 128)) receive 0 so sx obeys fso/activation_contract.h.
        mma_tile_coord_v = thr_mma.thr_idx
        q_block = self._q_block(work_m, stage, mma_tile_coord_v)
        q_row = q_block * self.m_block_size + tidx
        m_pad = ((seqlen_q + Int32(127)) // Int32(128)) * Int32(128)
        if q_row < m_pad:
            kp = head_idx >> Int32(1)
            r = q_row & Int32(127)
            word_idx = ((q_row >> Int32(7)) * Int32(self.num_kp) + kp) * Int32(128) + (r & Int32(31)) * Int32(4) + (r >> Int32(5))
            u16_idx = (word_idx << Int32(1)) + (head_idx & Int32(1))
            val16 = Uint16(b0 | (b1 << Int32(8))) if q_row < seqlen_q else Uint16(0)
            mSFO_u16[(u16_idx, Int32(0), Int32(0))] = val16

        if const_expr(self.use_correction_warps_for_epi):
            assert not self.use_tma_O
            assert gmem_tiled_copy_O is not None
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwdSm100.Epilogue),
                number_of_threads=self.num_epilogue_threads,
            )
            self._store_O_to_gmem(sO, gO, mO_cur, gmem_tiled_copy_O, tidx, seqlen_q, q_block)


# Compile cache keyed by (M_bucket_config_key, nheads, pdl)
_COMPILE_CACHE: dict = {}


def _patch_launch_pdl(fa_fwd: FlashAttentionForwardSm100MxFP8):
    """Enable use_pdl on the kernel launch without duplicating FlashAttentionForwardSm100.__call__."""
    if not fa_fwd.enable_pdl:
        return
    orig_kernel = fa_fwd.kernel

    def kernel_with_pdl(*args, **kwargs):
        launcher = orig_kernel(*args, **kwargs)
        orig_launch = launcher.launch

        def launch_pdl(*largs, **lkwargs):
            lkwargs["use_pdl"] = True
            return orig_launch(*largs, **lkwargs)

        launcher.launch = launch_pdl
        return launcher

    fa_fwd.kernel = kernel_with_pdl


_SM_COUNT_BY_DEV: dict[int, int] = {}


def _get_sm_count(device: Optional[torch.device] = None) -> int:
    idx = torch.cuda.current_device() if device is None else (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    sms = _SM_COUNT_BY_DEV.get(idx)
    if sms is None:
        sms = torch.cuda.get_device_properties(idx).multi_processor_count
        _SM_COUNT_BY_DEV[idx] = sms
    return sms


def _make_config(M: int, nheads: int = 20, q_stage_override: Optional[int] = None,
                 device: Optional[torch.device] = None):
    # Pipeline-optimal SM100/SM103 config for H=20, D=64:
    # - Use q_stage=1 whenever ceil(M/128)*H fits in one SM wave (doubles CTA parallelism
    #   and cuts padded rows at M=182/364/533: saves -28 to -54 us across 32 layers!).
    # - Disable static persistent scheduler when grid fits in one wave (-5 us at M=78/117).
    # - Match stock FA4 kv_stage (12 for q_stage=1 + ping-pong, 10 for q_stage=2),
    #   Sm100Registers(softmax=200, correction=64, other=48), sm_stats_cross_stage=False.
    sms = _get_sm_count(device)
    ctas_q1 = ((M + 127) // 128) * nheads
    q_stage = (1 if ctas_q1 <= sms else 2) if q_stage_override is None else q_stage_override
    ctas = ((M + 128 * q_stage - 1) // (128 * q_stage)) * nheads
    is_pers = ctas > sms
    tile_m = 128
    tile_n = 128
    regs = fa_cfg.Sm100Registers(softmax=200, correction=64, other=48)
    kv_stage = 12 if q_stage == 1 else 10
    use_pp = (q_stage == 1)
    cfg = fa_cfg.FwdConfig(
        device_capacity=10,
        tile_m=tile_m,
        tile_n=tile_n,
        mma_pv_is_rs=False,
        intra_wg_overlap=False,
        q_stage=q_stage,
        pack_gqa=False,
        use_clc_scheduler=False,
        is_static_persistent=is_pers,
        use_tma_o=True,
        use_s_ping_pong=use_pp,
        num_splits=1,
        use_2cta_instrs=False,
        registers=regs,
        ex2_emu_freq=0,
        ex2_emu_start_frg=0,
        overlap_sO_sQ=False,
        kv_stage=kv_stage,
        sm_stats_cross_stage=False,
    )
    return cfg


def _to_cute(t: torch.Tensor, leading_dim: int, align: int = 16):
    ct = from_dlpack(t, assumed_align=align)
    ct = ct.mark_layout_dynamic(leading_dim=leading_dim)
    return ct


def attn_fa4_mxfp8(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   softmax_scale: Optional[float] = None,
                   out_q: Optional[torch.Tensor] = None,
                   out_scale: Optional[torch.Tensor] = None):
    """Run SM100 FA4 bidirectional attention directly producing MXFP8 (xq [M, H*64], sx).

    q, k, v: BF16 [1, M, H, 64] (may be strided views of fused [M, 3, H, 64] QKV!).
    Returns: (xq [M, H*64] float8_e4m3fn, sx [align(M, 128) * (H*64/128)] int32).
    """
    if not q.is_cuda or torch.cuda.get_device_capability(q.device)[0] != 10:
        raise NotImplementedError("attn_fa4_mxfp8: requires sm_100 / sm_103 CUDA device")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("attn_fa4_mxfp8: q, k, v must be 4-D tensors [1, M, H, 64]")
    if q.shape != k.shape or q.shape != v.shape or q.shape[0] != 1 or q.shape[3] != 64:
        raise ValueError(
            f"attn_fa4_mxfp8: expected matching [1, M, H, 64] shapes, got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}"
        )
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise ValueError("attn_fa4_mxfp8: q, k, v must be bfloat16")
    if k.device != q.device or v.device != q.device:
        raise ValueError("attn_fa4_mxfp8: q, k, v must reside on the same CUDA device")
    if q.stride(3) != 1 or k.stride(3) != 1 or v.stride(3) != 1:
        raise ValueError("attn_fa4_mxfp8: q, k, v must be contiguous along head_dim (stride(3) == 1)")
    if (q.data_ptr() & 15) or (k.data_ptr() & 15) or (v.data_ptr() & 15):
        raise ValueError("attn_fa4_mxfp8: q, k, v must be 16-byte aligned")
    M = int(q.shape[1])
    H = int(q.shape[2])
    if H <= 0 or (H % 2) != 0:
        raise ValueError(f"attn_fa4_mxfp8: nheads must be a positive even integer, got {H}")
    D = H * 64
    M_pad = (M + 127) // 128 * 128
    num_kp = D // 128

    if out_q is None:
        out_q = torch.empty((1, M, H, 64), dtype=torch.float8_e4m3fn, device=q.device)
    else:
        if (not out_q.is_cuda or out_q.device != q.device or out_q.dtype != torch.float8_e4m3fn
                or not out_q.is_contiguous() or out_q.numel() != M * D or (out_q.data_ptr() & 15)):
            raise ValueError("attn_fa4_mxfp8: out_q must be 16-byte aligned contiguous CUDA float8_e4m3fn [M, H*64]")
        out_q = out_q.view(1, M, H, 64)
    if out_scale is None:
        out_scale = torch.empty((M_pad * num_kp,), dtype=torch.int32, device=q.device)
    else:
        if (not out_scale.is_cuda or out_scale.device != q.device or out_scale.dtype != torch.int32
                or not out_scale.is_contiguous() or out_scale.numel() != M_pad * num_kp
                or (out_scale.data_ptr() & 15)):
            raise ValueError("attn_fa4_mxfp8: out_scale must be 16-byte aligned contiguous CUDA int32 [M_pad*(D/128)]")
    if M == 0:
        return out_q.view(0, D), out_scale

    sx_u16 = out_scale.view(torch.uint16).view(1, 1, M_pad * num_kp * 2)
    scale = Float32((1.0 / math.sqrt(64.0)) if softmax_scale is None else float(softmax_scale))
    pdl = not env_flag("FSO_DISABLE_PDL")
    sms = _get_sm_count(q.device)
    ctas_q1 = ((M + 127) // 128) * H
    q_stage_key = 1 if ctas_q1 <= sms else 2
    ctas = ((M + 128 * q_stage_key - 1) // (128 * q_stage_key)) * H
    pers_key = ctas > sms
    ckey = (q_stage_key, pers_key, H, q.stride(1), q.stride(2), pdl)

    fn = _COMPILE_CACHE.get(ckey)
    q_ct = _to_cute(q, leading_dim=3, align=16)
    k_ct = _to_cute(k, leading_dim=3, align=16)
    v_ct = _to_cute(v, leading_dim=3, align=16)
    o_ct = _to_cute(out_q, leading_dim=3, align=16)
    sf_ct = _to_cute(sx_u16, leading_dim=2, align=16)
    stream = cuda_drv.CUstream(torch.cuda.current_stream(q.device).cuda_stream)

    if fn is None:
        cfg = _make_config(M, H, device=q.device)
        fa_fwd = FlashAttentionForwardSm100MxFP8(
            64,
            64,
            config=cfg,
            qhead_per_kvhead=1,
            is_causal=False,
            is_local=False,
            nheads_total=H,
            enable_pdl=pdl,
        )
        _patch_launch_pdl(fa_fwd)
        fn = cute.compile(
            fa_fwd,
            q_ct,
            k_ct,
            v_ct,
            o_ct,
            sf_ct,
            scale,
            stream=stream,
        )
        _COMPILE_CACHE[ckey] = fn

    fn(
        q_ct,
        k_ct,
        v_ct,
        o_ct,
        sf_ct,
        scale,
        stream=stream,
    )
    return out_q.view(M, D), out_scale
