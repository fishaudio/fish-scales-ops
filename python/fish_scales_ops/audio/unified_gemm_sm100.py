"""Single Unified Blackwell (sm_100 / sm_103) Cluster-DSMEM MXFP8 GEMM Kernel (v2).

Eliminates ALL host-side shape/size kernel routing (no `if M <= 128 / 256 / 384 / 576`,
no host `split_k = 4 / 2 / 1` dispatch, no host `WF_MIN_K_BLOCKS` split, ZERO global
memory reduction workspace, ZERO atomics, ZERO extra SMEM overhead!):

  - Host ALWAYS launches ONE single compiled kernel (per epilogue flavor) with fixed
    hardware cluster grid `grid = (max_clusters, 1, 4)` (`(37, 1, 4)` = 148 CTAs = 100%
    of B300 SMs), `cluster = (1, 1, 4)`.
  - Key micro-architectural innovations:
      1. **100% In-Kernel Cluster-DSMEM Adaptive Routing (`S in {4, 2, 1}`)**:
         Reads runtime `(M, N, K)` inside the GPU kernel and dynamically selects:
           - `S = 4` (full-cluster 4-way DSMEM Split-K): `base_tiles <= 37` and `k_blocks >= 16`.
           - `S = 2` (dual independent DSMEM Split-K pairs `(0,1)` & `(2,3)` per cluster):
             (`base_tiles <= 37` and `k_blocks >= 8`) or (`base_tiles <= 74` and `k_blocks >= 20`).
           - `S = 1` (4 independent persistent data-parallel CTAs per cluster): otherwise.
      2. **Zero-SMEM-Cost `sA` Union Overlay for `reduce_mailbox`**:
         Because `S > 1` executes at most 1 tile per CTA, `sA` is 100% dead when the
         epilogue reduction begins. After a 20 ns cluster handshake across all warps,
         `reduce_mailbox` overlays directly onto `sA`'s 1024B-aligned SMEM buffer—reclaiming
         +2 full mainloop `ab_stage`s (`128x64`: 8 stages, `128x128`: 6 stages) for FREE!
      3. **GPC-Local Weight-Strip Rasterization (`n_idx` inner)**:
         Adjacent CTAs in the same 4-CTA GPC cluster process consecutive token tiles of the
         SAME 128-row weight strip `W[m_idx]`, maximizing L2/GPC weight reuse in `S = 1`.
      4. **In-Kernel Dynamic Weights-First (`fso_pre`) + Early PDL + Fast Idle-CTA Exit**:
         Idle CTAs (`my_group_id >= base_tiles`) skip TMEM alloc/dealloc completely and exit
         immediately after PDL handshakes, freeing SMs for the next PDL-overlapped kernel.
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple, Union

import torch
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass import Float32, Int32
from cutlass.cute.arch import griddepcontrol_launch_dependents, griddepcontrol_wait
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import PipelineTmaUmma, PipelineUmmaAsync
from cutlass.cute.runtime import from_dlpack, make_ptr
import cuda.bindings.driver as cuda_drv

from ..gemm._vendor.flashinfer_sm100_blockscaled.dense_blockscaled_gemm_sm100_splitk import (
    Sm100BlockScaledSplitKGemmKernel,
    _store_shared_remote_v4,
)


class Sm100AudioClusterUnifiedGemmKernel(Sm100BlockScaledSplitKGemmKernel):
    """Single 1-CTA cluster=(1,1,4) SM100 MXFP8 GEMM with zero-SMEM-overlay DSMEM S in {1,2,4}."""

    def __init__(
        self,
        mma_tiler_mn: Tuple[int, int] = (128, 64),
        enable_pdl: bool = True,
        fso_epilogue: Optional[Tuple[bool, int, bool]] = None,
        fso_wf_cap: int = 3,
        fso_early_pdl: bool = True,
    ):
        super().__init__(
            sf_vec_size=32,
            mma_tiler_mn=mma_tiler_mn,
            split_k_slices=4,
            enable_pdl=enable_pdl,
            fso_epilogue=fso_epilogue,
            fso_weights_first=True,
            fso_wf_cap=fso_wf_cap,
            fso_early_pdl=fso_early_pdl,
        )

    def _setup_attributes(self):
        """Same as SplitK setup, except reduce_mailbox overlays sA (0 extra mailbox SMEM bytes!)."""
        mma_inst_bits_k = self.MMA_INST_BITS_K
        self.mma_inst_shape_mnk = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_bits_k // self.a_dtype.width,
        )
        self.mma_inst_shape_mnk_sfb = (
            self.mma_inst_shape_mnk[0],
            cute.round_up(self.mma_inst_shape_mnk[1], 128),
            self.mma_inst_shape_mnk[2],
        )
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype, self.b_dtype, self.a_major_mode, self.b_major_mode,
            self.sf_dtype, self.sf_vec_size, self.cta_group, self.mma_inst_shape_mnk[:2],
        )
        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype, self.b_dtype, self.a_major_mode, self.b_major_mode,
            self.sf_dtype, self.sf_vec_size, cute.nvgpu.tcgen05.CtaGroup.ONE, self.mma_inst_shape_mnk_sfb[:2],
        )
        mma_inst_tile_k = self.MMA_INST_TILE_K
        self.mma_tiler = (
            self.mma_inst_shape_mnk[0],
            self.mma_inst_shape_mnk[1],
            self.mma_inst_shape_mnk[2] * mma_inst_tile_k,
        )
        self.mma_tiler_sfb = (
            self.mma_inst_shape_mnk_sfb[0],
            self.mma_inst_shape_mnk_sfb[1],
            self.mma_inst_shape_mnk_sfb[2] * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cta_tile_shape_mnk_sfb = (
            self.mma_tiler_sfb[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_sfb[1],
            self.mma_tiler_sfb[2],
        )
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma.thr_id.shape,),
        )
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma_sfb.thr_id.shape,),
        )
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.num_mcast_ctas_sfb = cute.size(self.cluster_layout_sfb_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.is_sfb_mcast = self.num_mcast_ctas_sfb > 1

        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk, False, self.c_layout, self.c_dtype,
        )
        self.epi_tile_n = cute.size(self.epi_tile[1])
        self.num_epilogue_subtiles = (
            self.cta_tile_shape_mnk[0] * self.cta_tile_shape_mnk[1]
            // (cute.size(self.epi_tile[0]) * cute.size(self.epi_tile[1]))
        )
        self.num_epilogue_threads = 32 * len(self.epilog_warp_id)
        self.reduce_values_per_thread = (
            cute.size(self.epi_tile[0]) * cute.size(self.epi_tile[1])
        ) // self.num_epilogue_threads
        epi_tile_elements = cute.size(self.epi_tile[0]) * cute.size(self.epi_tile[1])
        self.num_mailbox_slots = (
            self.num_epilogue_subtiles + self.split_k_slices - 1
        ) // self.split_k_slices
        self.reduce_mailbox_elements = (
            (self.split_k_slices - 1) * self.num_mailbox_slots * epi_tile_elements
        )
        # Only reserve the tiny mbarrier array (num_epilogue_subtiles * 8 bytes):
        # reduce_mailbox overlays sA directly in SMEM -> +2 extra ab_stages for FREE!
        self.reduce_smem_bytes = self.num_epilogue_subtiles * 8

        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = self._compute_stages(
            tiled_mma, self.mma_tiler, self.a_dtype, self.a_major_mode,
            self.b_dtype, self.b_major_mode, self.epi_tile, self.c_dtype,
            self.c_layout, self.sf_dtype, self.sf_vec_size, self.smem_capacity,
            self.occupancy, 1024 + self.reduce_smem_bytes,
        )

        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage,
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, self.mma_tiler, self.sf_vec_size, self.num_ab_stage,
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, self.mma_tiler, self.sf_vec_size, self.num_ab_stage,
        )
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype, self.c_layout, self.epi_tile, self.num_c_stage,
        )
        # Verify sA buffer is strictly larger than the overlaid FP32 reduce_mailbox
        assert cute.cosize(self.a_smem_layout_staged.outer) >= self.reduce_mailbox_elements * 4

        self.overlapping_accum = self.num_acc_stage == 1
        if self.overlapping_accum:
            raise ValueError("Cluster-unified kernel needs non-overlapping accumulator")
        sf_atom_mn = 32
        self.num_sfa_tmem_cols = (self.cta_tile_shape_mnk[0] // sf_atom_mn) * mma_inst_tile_k
        self.num_sfb_tmem_cols = (self.cta_tile_shape_mnk_sfb[1] // sf_atom_mn) * mma_inst_tile_k
        self.num_sf_tmem_cols = self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        self.num_accumulator_tmem_cols = self.cta_tile_shape_mnk[1] * self.num_acc_stage

    @cute.jit
    def wrapper_unified(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        sf_m: cutlass.Int64,
        sf_n: cutlass.Int64,
        sf_k: cutlass.Int64,
        l: cutlass.Constexpr,
        a_sf_ptr: cute.Pointer,
        b_sf_ptr: cute.Pointer,
        alpha_tensor: cute.Tensor,
        bias_ptr: Optional[cute.Pointer],
        res_ptr: Optional[cute.Pointer],
        max_active_clusters: cutlass.Constexpr,
        current_stream,
        swap_ab: cutlass.Constexpr = True,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        m = cute.size(mA, mode=[0])
        k = cute.size(mA, mode=[1])
        n = cute.size(mB, mode=[0])

        a_tensor = cute.make_tensor(
            mA.iterator,
            layout=cute.make_ordered_layout((m, k, l), order=(1, 0, 2)),
        )
        b_tensor = cute.make_tensor(
            mB.iterator,
            layout=cute.make_ordered_layout((n, k, l), order=(1, 0, 2)),
        )
        c_order = (0, 1, 2) if cutlass.const_expr(swap_ab) else (1, 0, 2)
        c_tensor = cute.make_tensor(
            mC.iterator,
            layout=cute.make_ordered_layout((m, n, l), order=c_order),
        )
        bias_tensor = None
        if cutlass.const_expr(bias_ptr is not None):
            bias_tensor = cute.make_tensor(
                bias_ptr,
                layout=cute.make_layout((m, n, l), stride=(1, 0, 0)),
            )
        res_tensor = None
        if cutlass.const_expr(res_ptr is not None):
            res_tensor = cute.make_tensor(
                res_ptr,
                layout=cute.make_ordered_layout((m, n, l), order=c_order),
            )
        sfa_tensor = cute.make_tensor(
            a_sf_ptr,
            layout=cute.make_ordered_layout(
                (32, 4, sf_m, 4, sf_k, l),
                order=(2, 1, 4, 0, 3, 5),
            ),
        )
        sfb_tensor = cute.make_tensor(
            b_sf_ptr,
            layout=cute.make_ordered_layout(
                (32, 4, sf_n, 4, sf_k, l),
                order=(2, 1, 4, 0, 3, 5),
            ),
        )
        self._launch_unified(
            a_tensor, b_tensor, sfa_tensor, sfb_tensor, c_tensor,
            alpha_tensor, max_active_clusters, current_stream,
            epilogue_op, bias_tensor, res_tensor,
        )

    @cute.jit
    def _launch_unified(
        self,
        a_tensor: cute.Tensor,
        b_tensor: cute.Tensor,
        sfa_tensor: cute.Tensor,
        sfb_tensor: cute.Tensor,
        c_tensor: cute.Tensor,
        alpha: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream,
        epilogue_op: cutlass.Constexpr,
        fso_bias: Optional[cute.Tensor],
        fso_res: Optional[cute.Tensor],
    ):
        (
            tiled_mma, tiled_mma_sfb,
            tma_atom_a, tma_tensor_a,
            tma_atom_b, tma_tensor_b,
            tma_atom_sfa, tma_tensor_sfa,
            tma_atom_sfb, tma_tensor_sfb,
            tma_atom_c, tma_tensor_c,
        ) = self._prepare_tma(
            a_tensor, b_tensor, sfa_tensor, sfb_tensor, c_tensor
        )

        # Constant full-machine cluster grid: ZERO host-side M/N/K routing!
        grid = (max_active_clusters, 1, 4)
        self.buffer_align_bytes = 1024

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_holding_buf: cutlass.Int32
            reduce_full_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_epilogue_subtiles
            ]
            # NOTE: NO separate reduce_mailbox! reduce_mailbox overlays sA in SMEM (0 extra bytes!).
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.c_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.kernel_unified(
            tiled_mma, tiled_mma_sfb,
            tma_atom_a, tma_tensor_a,
            tma_atom_b, tma_tensor_b,
            tma_atom_sfa, tma_tensor_sfa,
            tma_atom_sfb, tma_tensor_sfb,
            tma_atom_c, tma_tensor_c,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            epilogue_op,
            alpha,
            fso_bias,
            fso_res,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            smem=self.shared_storage.size_in_bytes(),  # type: ignore[attr-defined]
            min_blocks_per_mp=1,
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel_unified(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        epilogue_op: cutlass.Constexpr,
        alpha: cute.Tensor,
        fso_bias: Optional[cute.Tensor],
        fso_res: Optional[cute.Tensor],
    ):
        alpha_value = alpha[0].to(cutlass.Float32)
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        bidx, bidy, bidz = cute.arch.block_idx()
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        tidx, _, _ = cute.arch.thread_idx()

        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_block_cnt = Int32(cute.size(gA_mkl, mode=[3]))
        num_m_tiles = Int32(cute.size(gC_mnl, mode=[2]))
        num_n_tiles = Int32(cute.size(gC_mnl, mode=[3]))
        base_tiles = num_m_tiles * num_n_tiles

        # -------------------------------------------------------------------
        # 100% IN-KERNEL CLUSTER-DSMEM ADAPTIVE ROUTING (S in {4, 2, 1})
        # -------------------------------------------------------------------
        num_clusters = Int32(cute.arch.grid_dim()[0])
        cluster_id = Int32(bidx)
        rank = Int32(cta_rank_in_cluster)

        split_k = Int32(1)
        if base_tiles <= num_clusters and k_block_cnt >= Int32(16):
            split_k = Int32(4)
        elif base_tiles <= num_clusters * Int32(2) and k_block_cnt >= Int32(8):
            split_k = Int32(2)

        groups_per_cluster = Int32(4) // split_k
        group_in_cluster = rank // split_k
        sub_rank = rank % split_k
        group_base_rank = group_in_cluster * split_k
        total_groups = num_clusters * groups_per_cluster
        my_group_id = cluster_id * groups_per_cluster + group_in_cluster

        k_base = k_block_cnt // split_k
        k_rem = k_block_cnt % split_k
        k_start = sub_rank * k_base + (sub_rank if sub_rank < k_rem else k_rem)
        k_cnt = k_base + (Int32(1) if sub_rank < k_rem else Int32(0))

        # Fast-exit for completely idle CTAs (CTA-uniform: skips TMEM alloc/dealloc!)
        # When split_k > 1, we still participate in the two cluster barriers cleanly.
        is_active_cta = my_group_id < base_tiles
        # In-kernel adaptive rasterization: n_idx inner when >= 8 activation tiles share each weight strip
        use_n_inner = num_n_tiles >= Int32(8)

        if warp_idx == self.tma_warp_id and is_active_cta:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)
            cpasync.prefetch_descriptor(tma_atom_c)

        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        tmem_holding_buf = storage.tmem_holding_buf.ptr
        reduce_full_mbar_ptr = storage.reduce_full_mbar_ptr.data_ptr()

        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_pipeline = PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilog_warp_id)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        if split_k > Int32(1):
            if warp_idx == self.tma_warp_id:
                with cute.arch.elect_one():
                    for subtile_idx in cutlass.range_constexpr(self.num_epilogue_subtiles):
                        cute.arch.mbarrier_init(reduce_full_mbar_ptr + subtile_idx, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.cluster_arrive_relaxed(aligned=True)
            cute.arch.cluster_wait()
        else:
            cute.arch.mbarrier_init_fence()
            cute.arch.barrier(
                barrier_id=self.cta_sync_bar_id, number_of_threads=self.threads_per_cta
            )


        sC = storage.sC.get_tensor(
            c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner
        )
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)

        # Zero-SMEM overlay: reduce_mailbox shares sA's 1024B-aligned SMEM after S>1 mainloop!
        reduce_mailbox = cute.make_tensor(
            cute.recast_ptr(sA.iterator, dtype=cutlass.Float32),
            cute.make_layout((self.reduce_mailbox_elements,)),
        )

        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfa_full_mcast_mask = None
        sfb_full_mcast_mask = None

        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gSFB_nkl = cute.local_tile(
            mSFB_nkl,
            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None),
        )

        thr_mma = tiled_mma.get_slice(0)
        thr_mma_sfb = tiled_mma_sfb.get_slice(0)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)

        fso_tCcC = None
        if cutlass.const_expr(self.fso_epi):
            fso_tCcC = self._fso_epi_partition_mma(thr_mma, mC_mnl)

        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a, block_in_cluster_coord_vmnk[2], a_cta_layout,
            cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3),
        )
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b, block_in_cluster_coord_vmnk[1], b_cta_layout,
            cute.group_modes(sB, 0, 3), cute.group_modes(tCgB, 0, 3),
        )
        tAsSFA, tAgSFA = cpasync.tma_partition(
            tma_atom_sfa, block_in_cluster_coord_vmnk[2], a_cta_layout,
            cute.group_modes(sSFA, 0, 3), cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)

        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        tBsSFB, tBgSFB = cpasync.tma_partition(
            tma_atom_sfb, block_in_cluster_coord_sfb_vmnk[1], sfb_cta_layout,
            cute.group_modes(sSFB, 0, 3), cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )

        if not is_active_cta:
            griddepcontrol_wait()
            if split_k > Int32(1):
                cute.arch.cluster_arrive_relaxed(aligned=True)
                cute.arch.cluster_wait()
            griddepcontrol_launch_dependents()
        else:
            if warp_idx != self.tma_warp_id:
                griddepcontrol_wait()

            # ===================================================================
            # Specialized TMA load warp (n_idx inner -> GPC weight-strip locality!)
            # ===================================================================
            if warp_idx == self.tma_warp_id:
                ab_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_ab_stage
                )
                tile_id = my_group_id

                # First tile peeled for IN-KERNEL dynamic weights-first prefetch
                n_idx = (tile_id % num_n_tiles) if use_n_inner else (tile_id // num_m_tiles)
                m_idx = (tile_id // num_n_tiles) if use_n_inner else (tile_id % num_m_tiles)
                sfb_n_idx = n_idx // Int32(2) if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64) else n_idx

                fso_tAgA = tAgA[(None, m_idx, None, Int32(0))]
                fso_tBgB = tBgB[(None, n_idx, None, Int32(0))]
                fso_tAgSFA = tAgSFA[(None, m_idx, None, Int32(0))]
                fso_tBgSFB = tBgSFB[(None, sfb_n_idx, None, Int32(0))]

                wf_max = Int32(self._fso_wf_stages())
                fso_pre = Int32(0)
                if k_cnt >= Int32(5):
                    lim = k_cnt - Int32(1)
                    fso_pre = lim if lim < wf_max else wf_max

                ab_producer_state.reset_count()
                fso_act_state = ab_producer_state.clone()
                fso_peek = cutlass.Boolean(1)
                if ab_producer_state.count < fso_pre:
                    fso_peek = ab_pipeline.producer_try_acquire(ab_producer_state)
                for _fso_k in cutlass.range(0, fso_pre, 1, unroll=1):
                    ab_pipeline.producer_acquire(ab_producer_state, fso_peek)
                    kb = k_start + ab_producer_state.count
                    cute.copy(
                        tma_atom_a, fso_tAgA[(None, kb)], tAsA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sfa, fso_tAgSFA[(None, kb)], tAsSFA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=sfa_full_mcast_mask,
                    )
                    ab_producer_state.advance()
                    fso_peek = cutlass.Boolean(1)
                    if ab_producer_state.count < fso_pre:
                        fso_peek = ab_pipeline.producer_try_acquire(ab_producer_state)

                griddepcontrol_wait()

                for _fso_k in cutlass.range(0, fso_pre, 1, unroll=1):
                    kb = k_start + fso_act_state.count
                    cute.copy(
                        tma_atom_b, fso_tBgB[(None, kb)], tBsB[(None, fso_act_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(fso_act_state),
                        mcast_mask=b_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sfb, fso_tBgSFB[(None, kb)], tBsSFB[(None, fso_act_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(fso_act_state),
                        mcast_mask=sfb_full_mcast_mask,
                    )
                    fso_act_state.advance()

                fso_peek = cutlass.Boolean(1)
                if ab_producer_state.count < k_cnt:
                    fso_peek = ab_pipeline.producer_try_acquire(ab_producer_state)
                for _fso_k in cutlass.range(fso_pre, k_cnt, 1, unroll=1):
                    ab_pipeline.producer_acquire(ab_producer_state, fso_peek)
                    kb = k_start + ab_producer_state.count
                    cute.copy(
                        tma_atom_a, fso_tAgA[(None, kb)], tAsA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b, fso_tBgB[(None, kb)], tBsB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=b_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sfa, fso_tAgSFA[(None, kb)], tAsSFA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=sfa_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sfb, fso_tBgSFB[(None, kb)], tBsSFB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=sfb_full_mcast_mask,
                    )
                    ab_producer_state.advance()
                    fso_peek = cutlass.Boolean(1)
                    if ab_producer_state.count < k_cnt:
                        fso_peek = ab_pipeline.producer_try_acquire(ab_producer_state)

                tile_id = tile_id + total_groups

                while tile_id < base_tiles:
                    n_idx = (tile_id % num_n_tiles) if use_n_inner else (tile_id // num_m_tiles)
                    m_idx = (tile_id // num_n_tiles) if use_n_inner else (tile_id % num_m_tiles)
                    sfb_n_idx = n_idx // Int32(2) if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64) else n_idx

                    tAgA_slice = tAgA[(None, m_idx, None, Int32(0))]
                    tBgB_slice = tBgB[(None, n_idx, None, Int32(0))]
                    tAgSFA_slice = tAgSFA[(None, m_idx, None, Int32(0))]
                    tBgSFB_slice = tBgSFB[(None, sfb_n_idx, None, Int32(0))]

                    ab_producer_state.reset_count()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < k_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
                    for _k in cutlass.range(0, k_cnt, 1, unroll=1):
                        ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status)
                        kb = k_start + ab_producer_state.count
                        cute.copy(
                            tma_atom_a, tAgA_slice[(None, kb)], tAsA[(None, ab_producer_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                            mcast_mask=a_full_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_b, tBgB_slice[(None, kb)], tBsB[(None, ab_producer_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                            mcast_mask=b_full_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_sfa, tAgSFA_slice[(None, kb)], tAsSFA[(None, ab_producer_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                            mcast_mask=sfa_full_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_sfb, tBgSFB_slice[(None, kb)], tBsSFB[(None, ab_producer_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                            mcast_mask=sfb_full_mcast_mask,
                        )
                        ab_producer_state.advance()
                        peek_ab_empty_status = cutlass.Boolean(1)
                        if ab_producer_state.count < k_cnt:
                            peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
                    tile_id = tile_id + total_groups

                if cutlass.const_expr(self.fso_early_pdl):
                    griddepcontrol_launch_dependents()
                ab_pipeline.producer_tail(ab_producer_state)
                if split_k > Int32(1):
                    cute.arch.cluster_arrive_relaxed(aligned=True)
                    cute.arch.cluster_wait()

            # ===================================================================
            # Specialized MMA warp
            # ===================================================================
            if warp_idx == self.mma_warp_id:
                tmem_ptr_read_threads = 32 * len((self.mma_warp_id, *self.epilog_warp_id))
                cute.arch.barrier(
                    barrier_id=self.tmem_ptr_sync_bar_id,
                    number_of_threads=tmem_ptr_read_threads,
                )

                acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
                    self.acc_dtype, alignment=16, ptr_to_buffer_holding_addr=tmem_holding_buf,
                )
                tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

                sfa_tmem_ptr = cute.recast_ptr(
                    acc_tmem_ptr + self.num_accumulator_tmem_cols, dtype=self.sf_dtype
                )
                tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                    tiled_mma, self.mma_tiler, self.sf_vec_size,
                    cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
                )
                tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

                sfb_tmem_ptr = cute.recast_ptr(
                    acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols,
                    dtype=self.sf_dtype,
                )
                tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                    tiled_mma, self.mma_tiler, self.sf_vec_size,
                    cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
                )
                tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

                tiled_copy_s2t_sfa, tCsSFA_compact_s2t, tCtSFA_compact_s2t = (
                    self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
                )
                tiled_copy_s2t_sfb, tCsSFB_compact_s2t, tCtSFB_compact_s2t = (
                    self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB)
                )

                ab_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_ab_stage
                )
                acc_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_acc_stage
                )

                tile_id = my_group_id
                while tile_id < base_tiles:
                    n_idx = (tile_id % num_n_tiles) if use_n_inner else (tile_id // num_m_tiles)
                    acc_stage_index = acc_producer_state.index
                    tCtAcc = tCtAcc_base[(None, None, None, acc_stage_index)]

                    ab_consumer_state.reset_count()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_cnt:
                        peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)
                    acc_pipeline.producer_acquire(acc_producer_state)

                    tCtSFB_mma = tCtSFB
                    if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                        offset = Int32((n_idx % Int32(2)) * Int32(2))
                        shifted_ptr = cute.recast_ptr(
                            acc_tmem_ptr
                            + self.num_accumulator_tmem_cols
                            + self.num_sfa_tmem_cols
                            + offset,
                            dtype=self.sf_dtype,
                        )
                        tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)

                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for _k_block in cutlass.range(0, k_cnt, 1, unroll=1):
                        ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)
                        s2t_stage_coord = (None, None, None, None, ab_consumer_state.index)
                        cute.copy(
                            tiled_copy_s2t_sfa, tCsSFA_compact_s2t[s2t_stage_coord], tCtSFA_compact_s2t,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfb, tCsSFB_compact_s2t[s2t_stage_coord], tCtSFB_compact_s2t,
                        )
                        num_kphases = cute.size(tCrA, mode=[2])
                        for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                            kphase_coord = (None, None, kphase_idx, ab_consumer_state.index)
                            sf_kphase_coord = (None, None, kphase_idx)
                            tiled_mma.set(tcgen05.Field.SFA, tCtSFA[sf_kphase_coord].iterator)
                            tiled_mma.set(tcgen05.Field.SFB, tCtSFB_mma[sf_kphase_coord].iterator)
                            cute.gemm(tiled_mma, tCtAcc, tCrA[kphase_coord], tCrB[kphase_coord], tCtAcc)
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        ab_pipeline.consumer_release(ab_consumer_state)
                        ab_consumer_state.advance()
                        peek_ab_full_status = cutlass.Boolean(1)
                        if ab_consumer_state.count < k_cnt:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)

                    acc_pipeline.producer_commit(acc_producer_state)
                    acc_producer_state.advance()
                    tile_id = tile_id + total_groups

                if split_k > Int32(1):
                    cute.arch.cluster_arrive_relaxed(aligned=True)
                    cute.arch.cluster_wait()
                acc_pipeline.producer_tail(acc_producer_state)

            # ===================================================================
            # Specialized Epilogue warps
            # ===================================================================
            if warp_idx < self.mma_warp_id:
                if warp_idx == self.epilog_warp_id[0]:
                    cute.arch.alloc_tmem(
                        self.num_tmem_alloc_cols, tmem_holding_buf, is_two_cta=False,
                    )
                tmem_ptr_read_threads = 32 * len((self.mma_warp_id, *self.epilog_warp_id))
                cute.arch.barrier(
                    barrier_id=self.tmem_ptr_sync_bar_id,
                    number_of_threads=tmem_ptr_read_threads,
                )

                acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
                    self.acc_dtype, alignment=16, ptr_to_buffer_holding_addr=tmem_holding_buf,
                )
                tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

                epi_tidx = tidx
                tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = (
                    self.epilog_tmem_copy_and_partition(
                        epi_tidx, tCtAcc_base, tCgC, epi_tile, False
                    )
                )
                tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
                tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(
                    tiled_copy_t2r, tTR_rC, epi_tidx, sC
                )
                tma_atom_c, bSG_sC, bSG_gC_partitioned = (
                    self.epilog_gmem_copy_and_partition(
                        epi_tidx, tma_atom_c, tCgC, epi_tile, sC
                    )
                )
                fso_epi = None
                if cutlass.const_expr(self.fso_epi):
                    fso_epi = self._fso_epi_setup(
                        tiled_copy_t2r, tiled_copy_r2s, epi_tidx, epi_tile, tTR_rAcc, mC_mnl, fso_tCcC
                    )

                acc_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_acc_stage
                )
                epilog_threads = 32 * len(self.epilog_warp_id)
                c_producer_group = pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, epilog_threads
                )
                c_pipeline = pipeline.PipelineTmaStore.create(
                    num_stages=self.num_c_stage, producer_group=c_producer_group,
                )

                values_per_thread = self.reduce_values_per_thread
                values_per_peer = self.num_epilogue_threads * values_per_thread
                epi_subtile_bytes = (cute.size(epi_tile[0]) * cute.size(epi_tile[1])) * 4
                num_prev_subtiles = Int32(0)

                tile_id = my_group_id
                while tile_id < base_tiles:
                    n_idx = (tile_id % num_n_tiles) if use_n_inner else (tile_id // num_m_tiles)
                    m_idx = (tile_id // num_n_tiles) if use_n_inner else (tile_id % num_m_tiles)
                    mma_tile_coord_mnl = (m_idx, n_idx, Int32(0))

                    bSG_gC = bSG_gC_partitioned[(None, None, None, *mma_tile_coord_mnl)]
                    fso_tile = None
                    if cutlass.const_expr(self.fso_epi):
                        fso_tile = self._fso_epi_tile(fso_epi, mma_tile_coord_mnl)

                    acc_stage_index = acc_consumer_state.index
                    tTR_tAcc = tTR_tAcc_base[
                        (None, None, None, None, None, acc_stage_index)
                    ]
                    acc_pipeline.consumer_wait(acc_consumer_state)

                    tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                    bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))
                    subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])

                    if split_k == Int32(1):
                        # Mode S = 1: Pure data-parallel native epilogue (zero DSMEM!)
                        for subtile_idx in cutlass.range(subtile_cnt):
                            tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                            cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                            if cutlass.const_expr(self.fso_epi):
                                acc_vec = self._fso_epi_apply(
                                    fso_epi, fso_tile, subtile_idx, tTR_rAcc,
                                    alpha_value, epilogue_op, fso_bias, fso_res,
                                )
                            else:
                                acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                                acc_vec = epilogue_op((alpha_value * acc_vec).to(self.c_dtype))
                            tRS_rC.store(acc_vec)

                            c_buffer = (num_prev_subtiles + subtile_idx) % self.num_c_stage
                            cute.copy(
                                tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buffer)],
                            )
                            cute.arch.fence_proxy("async.shared", space="cta")
                            cute.arch.barrier(
                                barrier_id=self.epilog_sync_bar_id,
                                number_of_threads=epilog_threads,
                            )
                            if warp_idx == self.epilog_warp_id[0]:
                                cute.copy(
                                    tma_atom_c, bSG_sC[(None, c_buffer)], bSG_gC[(None, subtile_idx)],
                                )
                                c_pipeline.producer_commit()
                                c_pipeline.producer_acquire()
                            cute.arch.barrier(
                                barrier_id=self.epilog_sync_bar_id,
                                number_of_threads=epilog_threads,
                            )
                        num_prev_subtiles = num_prev_subtiles + subtile_cnt
                    else:
                        # Wait until all 4 CTAs in cluster finished mainloop so sA -> reduce_mailbox is safe!
                        cute.arch.cluster_arrive_relaxed(aligned=True)
                        cute.arch.cluster_wait()

                        peer_count = split_k - Int32(1)
                        for epi_phase in cutlass.range_constexpr(2):
                            for subtile_idx in cutlass.range(subtile_cnt):
                                owner_sub = Int32(subtile_idx) % split_k
                                owner_cluster_rank = group_base_rank + owner_sub
                                owned_ordinal = Int32(subtile_idx) // split_k
                                subtile_mailbox_base = owned_ordinal * peer_count * values_per_peer
                                reduce_mbar_ptr = reduce_full_mbar_ptr + subtile_idx
                                is_owner = sub_rank == owner_sub
                                take_subtile = is_owner
                                if cutlass.const_expr(epi_phase == 0):
                                    take_subtile = sub_rank != owner_sub
                                if take_subtile:
                                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)
                                    tRS_rAcc = tiled_copy_r2s.retile(tTR_rAcc)

                                    if cutlass.const_expr(epi_phase == 0):
                                        peer_slot = sub_rank
                                        if sub_rank > owner_sub:
                                            peer_slot = sub_rank - Int32(1)
                                        remote_peer_base = subtile_mailbox_base + peer_slot * values_per_peer
                                        for value_idx in cutlass.range_constexpr(0, values_per_thread, 4):
                                            remote_off = (
                                                remote_peer_base
                                                + (value_idx // 4) * (self.num_epilogue_threads * 4)
                                                + epi_tidx * 4
                                            )
                                            _store_shared_remote_v4(
                                                tRS_rAcc[value_idx],
                                                tRS_rAcc[value_idx + 1],
                                                tRS_rAcc[value_idx + 2],
                                                tRS_rAcc[value_idx + 3],
                                                reduce_mailbox.iterator + remote_off,
                                                reduce_mbar_ptr,
                                                owner_cluster_rank,
                                            )
                                    else:
                                        if warp_idx == self.epilog_warp_id[0]:
                                            with cute.arch.elect_one():
                                                cute.arch.mbarrier_arrive_and_expect_tx(
                                                    reduce_mbar_ptr,
                                                    peer_count * epi_subtile_bytes,
                                                )
                                        cute.arch.mbarrier_wait(reduce_mbar_ptr, phase=0)

                                        for peer_slot in cutlass.range_constexpr(3):
                                            if Int32(peer_slot) < peer_count:
                                                peer_base = subtile_mailbox_base + Int32(peer_slot) * values_per_peer
                                                for value_idx in cutlass.range_constexpr(values_per_thread):
                                                    mailbox_idx = (
                                                        peer_base
                                                        + (value_idx // 4) * (self.num_epilogue_threads * 4)
                                                        + epi_tidx * 4
                                                        + (value_idx % 4)
                                                    )
                                                    tRS_rAcc[value_idx] = (
                                                        tRS_rAcc[value_idx] + reduce_mailbox[mailbox_idx]
                                                    )

                                        if cutlass.const_expr(self.fso_epi):
                                            acc_vec = self._fso_epi_apply(
                                                fso_epi, fso_tile, subtile_idx, tTR_rAcc,
                                                alpha_value, epilogue_op, fso_bias, fso_res,
                                            )
                                        else:
                                            acc_vec = tRS_rAcc.load()
                                            acc_vec = epilogue_op((alpha_value * acc_vec).to(self.c_dtype))
                                        tRS_rC.store(acc_vec)

                                        c_buffer = owned_ordinal % self.num_c_stage
                                        cute.copy(
                                            tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buffer)],
                                        )
                                        cute.arch.fence_proxy("async.shared", space="cta")
                                        cute.arch.barrier(
                                            barrier_id=self.epilog_sync_bar_id,
                                            number_of_threads=epilog_threads,
                                        )
                                        if warp_idx == self.epilog_warp_id[0]:
                                            cute.copy(
                                                tma_atom_c, bSG_sC[(None, c_buffer)], bSG_gC[(None, subtile_idx)],
                                            )
                                            c_pipeline.producer_commit()
                                            c_pipeline.producer_acquire()
                                        cute.arch.barrier(
                                            barrier_id=self.epilog_sync_bar_id,
                                            number_of_threads=epilog_threads,
                                        )

                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()
                    tile_id = tile_id + total_groups

                if warp_idx == self.epilog_warp_id[0]:
                    cute.arch.relinquish_tmem_alloc_permit(is_two_cta=False)
                cute.arch.barrier(
                    barrier_id=self.epilog_sync_bar_id, number_of_threads=epilog_threads
                )
                if warp_idx == self.epilog_warp_id[0]:
                    cute.arch.dealloc_tmem(
                        acc_tmem_ptr, self.num_tmem_alloc_cols, is_two_cta=False
                    )
                c_pipeline.producer_tail()

            if cutlass.const_expr(not self.fso_early_pdl):
                griddepcontrol_launch_dependents()


# ---------------------------------------------------------------------------
# Runtime compile cache (ZERO host shape routing, ZERO global workspace!)
# ---------------------------------------------------------------------------

_COMPILED: Dict[Tuple, object] = {}
_ALPHA: Dict[torch.device, Tuple[torch.Tensor, object]] = {}
_MAC4: Optional[int] = None


def _get_mac4() -> int:
    global _MAC4
    if _MAC4 is None:
        _MAC4 = int(utils.HardwareInfo().get_max_active_clusters(4))
    return _MAC4


def _get_alpha(device: torch.device):
    hit = _ALPHA.get(device)
    if hit is not None:
        return hit
    t = torch.ones(1, dtype=torch.float32, device=device)
    ct = from_dlpack(t, assumed_align=4)
    ct.element_type = cutlass.Float32
    _ALPHA[device] = (t, ct)
    return t, ct


def _get_compiled(
    mma_tiler_mn: Tuple[int, int],
    epi_flags: Optional[Tuple[bool, int, bool]],
    pdl: bool,
    early_pdl: bool,
    sf_m: int,
    sf_n: int,
    sf_k: int,
):
    key = (mma_tiler_mn, epi_flags, pdl, early_pdl)
    fn = _COMPILED.get(key)
    if fn is not None:
        return fn

    mac4 = _get_mac4()
    wf_cap = 5 if mma_tiler_mn[1] >= 128 else 3
    gemm = Sm100AudioClusterUnifiedGemmKernel(
        mma_tiler_mn=mma_tiler_mn,
        enable_pdl=pdl,
        fso_epilogue=epi_flags,
        fso_wf_cap=wf_cap,
        fso_early_pdl=early_pdl,
    )
    ms = cute.AddressSpace.gmem
    sym_m, sym_n, sym_k = cute.sym_int(), cute.sym_int(), cute.sym_int()
    a_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float8E4M3FN, (sym_m, sym_k), stride_order=(1, 0), assumed_align=16, memspace=ms
    )
    b_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float8E4M3FN, (sym_n, sym_k), stride_order=(1, 0), assumed_align=16, memspace=ms
    )
    c_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (sym_n, sym_m), stride_order=(0, 1), assumed_align=16, memspace=ms
    )
    a_sf_fake = make_ptr(cutlass.Float8E8M0FNU, 16, ms, 16)
    b_sf_fake = make_ptr(cutlass.Float8E8M0FNU, 16, ms, 16)
    alpha_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (1,), assumed_align=4, memspace=ms
    )
    has_bias = bool(epi_flags and epi_flags[0])
    has_res = bool(epi_flags and epi_flags[2])
    bias_fake = make_ptr(cutlass.BFloat16, 16, ms, 2) if has_bias else None
    res_fake = make_ptr(cutlass.BFloat16, 16, ms, 2) if has_res else None
    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)

    fn = cute.compile(
        gemm.wrapper_unified,
        a_fake, b_fake, c_fake,
        sf_m, sf_n, sf_k, 1,
        a_sf_fake, b_sf_fake, alpha_fake,
        bias_fake, res_fake,
        mac4, stream, True,
        options="--opt-level 2",
    )
    _COMPILED[key] = fn
    return fn


def linear_mxfp8_unified(
    x_fp8: torch.Tensor,
    w_fp8: torch.Tensor,
    sx: torch.Tensor,
    sw: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    early_pdl: bool = True,
    mma_tiler_mn: Tuple[int, int] = (128, 64),
) -> torch.Tensor:
    """Run the single unified cluster-DSMEM SM100 MXFP8 GEMM (zero host shape routing!)."""
    m, k = int(x_fp8.shape[0]), int(x_fp8.shape[1])
    n = int(w_fp8.shape[0])
    pdl = os.environ.get("FSO_DISABLE_PDL", "0") != "1"
    epi_flags = None
    if bias is not None or residual is not None:
        epi_flags = (bias is not None, 0, residual is not None)

    out = torch.empty((m, n), dtype=torch.bfloat16, device=x_fp8.device)
    launch_out = out.as_strided(out.shape, (1, out.shape[0]))
    sf_m = (n + 127) // 128
    sf_n = (m + 127) // 128
    sf_k = (k // 32 + 3) // 4

    fn = _get_compiled(mma_tiler_mn, epi_flags, pdl, bool(early_pdl and pdl), sf_m, sf_n, sf_k)

    gmem = cute.AddressSpace.gmem
    ct_a = from_dlpack(w_fp8, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    ct_a.element_type = cutlass.Float8E4M3FN
    ct_b = from_dlpack(x_fp8, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    ct_b.element_type = cutlass.Float8E4M3FN
    ct_c = from_dlpack(launch_out, assumed_align=16).mark_layout_dynamic(leading_dim=0)
    ct_c.element_type = cutlass.BFloat16

    p_sfa = make_ptr(cutlass.Float8E8M0FNU, sw.data_ptr(), gmem, assumed_align=16)
    p_sfb = make_ptr(cutlass.Float8E8M0FNU, sx.data_ptr(), gmem, assumed_align=16)
    _, ct_alpha = _get_alpha(x_fp8.device)
    p_bias = make_ptr(cutlass.BFloat16, bias.data_ptr(), gmem, assumed_align=2) if bias is not None else None
    p_res = make_ptr(cutlass.BFloat16, residual.data_ptr(), gmem, assumed_align=2) if residual is not None else None
    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)

    fn(
        ct_a, ct_b, ct_c,
        sf_m, sf_n, sf_k,
        p_sfa, p_sfb, ct_alpha,
        p_bias, p_res,
        stream,
    )
    return out
