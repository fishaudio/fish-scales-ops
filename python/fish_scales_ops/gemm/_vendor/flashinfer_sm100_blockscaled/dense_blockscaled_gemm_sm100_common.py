# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from typing import Any, Optional, Tuple, Type, Union

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass import Float32, Int32
from cutlass._mlir.dialects import arith
from cutlass.cute.nvgpu import cpasync, tcgen05


# fso-epilogue (fish-scales-ops issue #12, S1b1): the dense epilogue of fso's interface v1 for the two kernels of
# this package, `y = RN_bf16(act(acc + bias_term(bias[n])) + residual[m, n])`, applied in fp32 after the fp32
# reduction and before the one bf16 rounding. The device helpers below are the DSL spelling of
# csrc/gemm/include/blockscale_gemm/common/epilogue_device.cuh (the canonical text for every architecture): every
# fp32 operation is an explicit round-to-nearest NVVM intrinsic (`add.rn.f32` / `mul.rn.f32` in the PTX), which
# ptxas never contracts into an fma, so act(v) is rounded to fp32 before the residual is added, exactly as the C++
# `__fadd_rn` / `__fmul_rn` form. The erf is the CUDA 13.0 toolkit's erff written out (`_fso_erff` below), not the
# DSL's `math.erf`, whose libdevice differs in the last bit. A kernel built without `fso_epilogue` (the default) generates exactly the
# code it generated before these additions: everything here is behind `cutlass.const_expr(self.fso_epi)`, and the
# compiled wrapper entry is the unchanged `wrapper` below. The rounding-mode surface (`cute.math.RoundingMode`,
# `cute.math.add(..., rounding=)`) is looked up when an epilogue kernel is built, not at import, so that a DSL
# release without it still imports this package and runs the identity kernels (fso_epilogue_supported() says
# whether the epilogue variants can be built on the installed DSL).


def fso_epilogue_supported() -> bool:
    """fso-epilogue: True when the installed DSL has the explicit-rounding scalar ops the epilogue uses."""
    return all(hasattr(cute.math, name) for name in ("RoundingMode", "add", "mul", "fma", "abs", "exp2"))


def _fso_rn():
    return cute.math.RoundingMode.NEAREST_EVEN


def _fso_add_rn(a, b):
    """fp32 a + b, rounded to nearest once (`add.rn.f32`)."""
    return cute.math.add(a, b, rounding=_fso_rn())


def _fso_mul_rn(a, b):
    """fp32 a * b, rounded to nearest once (`mul.rn.f32`)."""
    return cute.math.mul(a, b, rounding=_fso_rn())


def _fso_select(cond, a, b, *, loc=None, ip=None):
    """`cond ? a : b` on two scalars of one DSL numeric type (an `arith.select`)."""
    return type(a)(
        arith.select(
            cond.ir_value(loc=loc, ip=ip),
            a.ir_value(loc=loc, ip=ip),
            b.ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    )


def _fso_bias_term(b):
    """The bias term of the contract: float(bias[n]) with +0.0 replaced by -0.0 (a select, so that bias = 0 is
    the identity on every accumulator value, -0.0 included)."""
    return _fso_select(b == Float32(0.0), Float32(-0.0), b)


def _fso_clamp_index(i, extent):
    """min(i, extent - 1) on Int32 scalars: the coordinate of an output element past the tensor's edge is folded
    onto the edge, so the loads below never leave the tensor (the TMA store discards those elements)."""
    i = Int32(i)
    last = Int32(extent) - Int32(1)
    return _fso_select(i < last, i, last)


def _fso_f32_bits(bits: int):
    """The fp32 constant with these IEEE-754 bits (the hex spelling of nvcc's PTX, `0f3F8060FE` etc.)."""
    import struct
    return Float32(struct.unpack("<f", struct.pack("<I", bits))[0])


def _fso_fma_rn(a, b, c):
    """fp32 a * b + c, one rounding (`fma.rn.f32`)."""
    return cute.math.fma(a, b, c, rounding=_fso_rn())


# erff of the CUDA 13.0 toolkit (libdevice `__nv_erff` as nvcc 13.0, V13.0.48, emits it: the PTX of a kernel calling
# erff, kept in fso_issue12_20261028/runs/S1b1/probes/erf/gelu_ref.ptx), spelled out operation for operation. The
# DSL's own `cute.math.erf` lowers to the erff of the libdevice bundled with nvidia-cutlass-dsl, which is another
# revision of the same algorithm (threshold 0x3F807B28, first coefficients 0x388E49B7 / 0x388D8E09 against the
# toolkit's 0x3F8060FE, 0x38EB4C3A / 0x38B1E96A) and differs from the toolkit's erff in the last fp32 bit on a few
# per cent of inputs; the dense epilogue must give the bytes of the C++ routes, which nvcc compiles against the
# toolkit's libdevice, so the toolkit's coefficients are written here. Two minimax polynomials selected
# branchlessly: in t * t for |t| < 1.00296 (erf(t) = t + t * p(t * t)), in |t| above it (erf(t) = copysign(1 -
# 2^(q(|t|) * |t| - |t|), t), the exponential as ex2.approx.ftz). Every multiply-add is an fma.rn, as in the PTX;
# the two plain products of the PTX (t * t, and nothing else) have no addend they could be contracted with there,
# so mul.rn gives the same bits.
_FSO_ERFF_THRESHOLD = 0x3F8060FE
_FSO_ERFF_COEFFS = (  # (|t| >= threshold branch, |t| < threshold branch), highest order first
    (0x38EB4C3A, 0x38B1E96A),
    (0xBAAE005B, 0xBA574D20),
    (0x3C09919F, 0x3BAAD5EA),
    (0xBD24D99A, 0xBCDC1BE7),
    (0x3E235519, 0x3DE718AF),
    (0x3F69B4F9, 0xBEC093AC),
    (0x3F210A14, 0x3E0375D3),
)


def _fso_erff(t):
    """erff(t) in fp32 with the bits of the CUDA 13.0 toolkit's erff (see above). `t` is a Float32 scalar."""
    a = cute.math.abs(t)
    big = a >= _fso_f32_bits(_FSO_ERFF_THRESHOLD)
    s = _fso_select(big, a, _fso_mul_rn(t, t))
    r = _fso_fma_rn(_fso_select(big, _fso_f32_bits(_FSO_ERFF_COEFFS[0][0]), _fso_f32_bits(_FSO_ERFF_COEFFS[0][1])),
                    s, _fso_select(big, _fso_f32_bits(_FSO_ERFF_COEFFS[1][0]), _fso_f32_bits(_FSO_ERFF_COEFFS[1][1])))
    for hi, lo in _FSO_ERFF_COEFFS[2:]:
        r = _fso_fma_rn(r, s, _fso_select(big, _fso_f32_bits(hi), _fso_f32_bits(lo)))
    q = _fso_select(big, _fso_mul_rn(s, Float32(-1.0)), t)
    r = _fso_fma_rn(r, q, q)
    e = cute.math.exp2(r, approx=True, ftz=True)
    u = _fso_add_rn(Float32(1.0), _fso_mul_rn(e, Float32(-1.0)))  # 1 - e, the subtraction of the PTX
    v = _fso_select(t < Float32(0.0), _fso_mul_rn(u, Float32(-1.0)), u)  # copysign(u, t): |t| >= 1 here
    return _fso_select(big, v, r)


def _fso_gelu_erf(v):
    """0.5f * v * (1.f + erff(v * 0.70710678f)) with every fp32 operation rounded on its own, in the operation
    order of `fso::dense_gelu_erf`, and erff as the CUDA 13.0 toolkit computes it (`_fso_erff`)."""
    half_v = _fso_mul_rn(Float32(0.5), v)
    e = _fso_erff(_fso_mul_rn(v, Float32(0.70710678118654752440)))
    return _fso_mul_rn(half_v, _fso_add_rn(Float32(1.0), e))


class _FsoEpilogueFrags:
    """The per-thread state of the fso epilogue inside the epilogue warps: the coordinate fragment of the output
    partitioned like the accumulator fragment, the output extents and the fp32 scratch in both register orders."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Sm100BlockScaledGemmCommon:
    """Shared launch and copy-partition helpers for SM100 block-scaled GEMM."""

    # fso-epilogue: set by _fso_set_epilogue from the kernels' __init__.
    fso_epi: bool = False
    fso_epi_bias: bool = False
    fso_epi_act: int = 0
    fso_epi_res: bool = False

    def _fso_set_epilogue(self, fso_epilogue: Optional[Tuple[bool, int, bool]]) -> None:
        """fso-epilogue: `fso_epilogue` is None (today's kernel, byte for byte) or `(has_bias, act, has_residual)`
        with `act` 0 (none) or 1 (erf-GELU), the fields of fso's `DenseEpilogue` this instance applies."""
        if fso_epilogue is None:
            self.fso_epi = False
            self.fso_epi_bias, self.fso_epi_act, self.fso_epi_res = False, 0, False
            return
        has_bias, act, has_res = fso_epilogue
        if act not in (0, 1):
            raise ValueError(f"fso_epilogue: act must be 0 (none) or 1 (gelu_erf), got {act}")
        if not (has_bias or act or has_res):
            raise ValueError("fso_epilogue: pass None for the identity epilogue")
        self.fso_epi = True
        self.fso_epi_bias, self.fso_epi_act, self.fso_epi_res = bool(has_bias), int(act), bool(has_res)

    def _fso_epi_partition_mma(self, thr_mma, mC_mnl):
        """fso-epilogue, kernel prologue: the coordinate tensor of C, tiled and MMA-partitioned exactly as `tCgC`
        is built from `mC_mnl`, so its fragments carry the (m, n, l) coordinate of every accumulator element.
        Layout arithmetic only; nothing is loaded here."""
        tiler_mn = cute.slice_(self.mma_tiler, (None, None, 0))
        cC_mnl = cute.make_identity_tensor(mC_mnl.shape)
        return thr_mma.partition_C(cute.local_tile(cC_mnl, tiler_mn, (None, None, None)))

    def _fso_epi_setup(self, tiled_copy_t2r, tiled_copy_r2s, epi_tidx, epi_tile, tTR_rAcc, mC_mnl, tCcC):
        """fso-epilogue, epilogue warps, once per CTA: partition the coordinate tensor with the TMEM-load thread
        slice (the accumulator fragment's own partition) and allocate the fp32 scratch, as a (T2R, T2R_M, T2R_N)
        tensor and as its retiled view in the order of the smem-store fragment `tRS_rC`."""
        thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, RestM, RestN, RestL)
        tTR_cC = thr_copy_t2r.partition_D(
            cute.flat_divide(tCcC[((None, None), 0, 0, None, None, None)], epi_tile)
        )
        tTR_rV = cute.make_rmem_tensor(tTR_rAcc.shape, self.acc_dtype)
        return _FsoEpilogueFrags(
            tTR_cC=tTR_cC,
            m_total=cute.size(mC_mnl, mode=[0]),
            n_total=cute.size(mC_mnl, mode=[1]),
            tTR_rV=tTR_rV,
            tRS_rV=tiled_copy_r2s.retile(tTR_rV),
            n_values=cute.size(tTR_rAcc.shape),
        )

    def _fso_epi_tile(self, epi, tile_coord_mnl):
        """fso-epilogue, per output tile: this tile's coordinate fragment, with (EPI_M, EPI_N) grouped into one
        subtile mode the way the kernels index `tTR_tAcc` and `bSG_gC`: (T2R, T2R_M, T2R_N, SUBTILE)."""
        t = epi.tTR_cC[(None, None, None, None, None, *tile_coord_mnl)]
        return cute.group_modes(t, 3, cute.rank(t))

    def _fso_epi_apply(self, epi, tTR_cC_t, subtile_idx, tTR_rAcc, alpha_value, epilogue_op, fso_bias, fso_res):
        """fso-epilogue, per subtile, in the thread that owns the final fp32 accumulator values `tTR_rAcc` (after
        the cluster sum in the split-K kernel): per value, load its bias (index: the output row m, under swap_ab
        the weight's N) and its residual element from the kernel's tensors at the clamped coordinate (plain
        coherent `ld.global`, after the kernel's griddepcontrol.wait), then
        v = alpha * acc; v += bias_term; v = act(v); v += residual; and return the bf16 vector for `tRS_rC`."""
        tTR_cC_s = tTR_cC_t[(None, None, None, subtile_idx)]
        for i in range(epi.n_values):
            c = tTR_cC_s[i]
            v = _fso_mul_rn(alpha_value, tTR_rAcc[i])
            if cutlass.const_expr(self.fso_epi_bias or self.fso_epi_res):
                m_i = _fso_clamp_index(c[0], epi.m_total)
                n_i = _fso_clamp_index(c[1], epi.n_total)
            if cutlass.const_expr(self.fso_epi_bias):
                v = _fso_add_rn(v, _fso_bias_term(fso_bias[(m_i, n_i, 0)].to(Float32)))
            if cutlass.const_expr(self.fso_epi_act == 1):
                v = _fso_gelu_erf(v)
            if cutlass.const_expr(self.fso_epi_res):
                v = _fso_add_rn(v, fso_res[(m_i, n_i, 0)].to(Float32))
            epi.tTR_rV[i] = v
        return epilogue_op(epi.tRS_rV.load().to(self.c_dtype))

    def _setup_attributes(self) -> None:
        raise NotImplementedError

    def __call__(
        self,
        a_tensor: cute.Tensor,
        b_tensor: cute.Tensor,
        sfa_tensor: cute.Tensor,
        sfb_tensor: cute.Tensor,
        c_tensor: cute.Tensor,
        alpha: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: Any,
        epilogue_op: cutlass.Constexpr = lambda x: x,
        fso_bias: Optional[cute.Tensor] = None,  # fso-epilogue
        fso_res: Optional[cute.Tensor] = None,  # fso-epilogue
    ) -> Any:
        raise NotImplementedError

    def _prepare_tma(
        self,
        a_tensor: cute.Tensor,
        b_tensor: cute.Tensor,
        sfa_tensor: cute.Tensor,
        sfb_tensor: cute.Tensor,
        c_tensor: cute.Tensor,
    ):
        """Build the shared block-scaled MMA and TMA descriptors."""
        # Setup static attributes before smem/grid/tma computation
        self.a_dtype: Type[cutlass.Numeric] = a_tensor.element_type
        self.b_dtype: Type[cutlass.Numeric] = b_tensor.element_type
        self.sf_dtype: Type[cutlass.Numeric] = sfa_tensor.element_type
        self.c_dtype: Type[cutlass.Numeric] = c_tensor.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a_tensor).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b_tensor).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c_tensor)

        # Check if input data types are compatible with MMA instruction
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

        # Setup attributes that dependent on gemm inputs
        self._setup_attributes()

        # Setup sfa/sfb tensor by filling A/B tensor to scale factor atom layout
        # ((Atom_M, Rest_M),(Atom_K, Rest_K),RestL)
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
            a_tensor.shape, self.sf_vec_size
        )
        sfa_tensor = cute.make_tensor(sfa_tensor.iterator, sfa_layout)

        # ((Atom_N, Rest_N),(Atom_K, Rest_K),RestL)
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
            b_tensor.shape, self.sf_vec_size
        )
        sfb_tensor = cute.make_tensor(sfb_tensor.iterator, sfb_layout)

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mnk[:2],
        )

        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mnk_sfb[:2],
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # Setup TMA load for A
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a_tensor,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # Setup TMA load for B
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b_tensor,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # Setup TMA load for SFA
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfa_smem_layout = cute.slice_(
            self.sfa_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            sfa_tensor,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        # Setup TMA load for SFB
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfb_smem_layout = cute.slice_(
            self.sfb_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb_tensor,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
            x = tma_tensor_sfb.stride[0][1]
            y = cute.ceil_div(tma_tensor_sfb.shape[0][1], 4)

            new_shape = (
                (tma_tensor_sfb.shape[0][0], ((2, 2), y)),
                tma_tensor_sfb.shape[1],
                tma_tensor_sfb.shape[2],
            )
            # Use right multiplication for ScaledBasis (3 * x instead of x * 3)
            x_times_3 = 3 * x
            new_stride = (
                (tma_tensor_sfb.stride[0][0], ((x, x), x_times_3)),
                tma_tensor_sfb.stride[1],
                tma_tensor_sfb.stride[2],
            )
            tma_tensor_sfb_new_layout = cute.make_layout(new_shape, stride=new_stride)
            tma_tensor_sfb = cute.make_tensor(
                tma_tensor_sfb.iterator, tma_tensor_sfb_new_layout
            )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (
            a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size
        ) * atom_thr_size

        # Setup TMA store for C
        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            c_tensor,
            epi_smem_layout,
            self.epi_tile,
        )

        return (
            tiled_mma,
            tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            tma_atom_c,
            tma_tensor_c,
        )

    @staticmethod
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        a_major_mode: cute.nvgpu.OperandMajorMode,
        b_dtype: Type[cutlass.Numeric],
        b_major_mode: cute.nvgpu.OperandMajorMode,
        epi_tile: cute.Tile,
        c_dtype: Type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        smem_capacity: int,
        occupancy: int,
        reserved_smem_bytes: int,
    ) -> Tuple[int, int, int]:
        """Computes the optimal number of pipeline stages for operands.

        This method uses heuristics to determine the number of stages for the
        accumulator (ACC), A/B operands, and C operand based on the kernel
        configuration and available shared memory.

        Args:
            tiled_mma (cute.TiledMma): The tiled MMA object.
            mma_tiler_mnk (Tuple[int, int, int]): The shape (M, N, K) of the
                MMA tiler.
            a_dtype (Type[cutlass.Numeric]): Data type of operand A.
            a_major_mode (cute.nvgpu.OperandMajorMode): Layout of operand A.
            b_dtype (Type[cutlass.Numeric]): Data type of operand B.
            b_major_mode (cute.nvgpu.OperandMajorMode): Layout of operand B.
            epi_tile (cute.Tile): The epilogue tile shape.
            c_dtype (Type[cutlass.Numeric]): Data type of operand C.
            c_layout (utils.LayoutEnum): Layout of operand C.
            sf_dtype (Type[cutlass.Numeric]): Data type of the scale factors.
            sf_vec_size (int): Vector size of the scale factors.
            smem_capacity (int): Total available shared memory in bytes.
            occupancy (int): Target number of CTAs per SM.
            reserved_smem_bytes (int): Per-CTA bytes reserved outside the staged operands and output.

        Returns:
            Tuple[int, int, int]: A tuple containing the number of stages for
                (ACC, A/B, C).
        """
        # ACC stages
        num_acc_stage = 1 if mma_tiler_mnk[1] == 256 else 2

        # Default C stages
        num_c_stage = 2

        # Calculate smem layout and size for one stage of A, B, SFA, SFB and C
        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mma_tiler_mnk,
            a_dtype,
            1,  # a tmp 1 stage is provided
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mma_tiler_mnk,
            b_dtype,
            1,  # a tmp 1 stage is provided
        )
        sfa_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,  # a tmp 1 stage is provided
        )
        sfb_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,  # a tmp 1 stage is provided
        )

        c_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            1,
        )

        ab_bytes_per_stage = (
            cute.size_in_bytes(a_dtype, a_smem_layout_stage_one)
            + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfa_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)
        )
        c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
        c_bytes = c_bytes_per_stage * num_c_stage

        # Calculate A/B/SFA/SFB stages:
        # Start with total smem per CTA (capacity / occupancy)
        # Subtract reserved bytes and initial C stages bytes
        # Divide remaining by bytes needed per A/B/SFA/SFB stage
        num_ab_stage = (
            smem_capacity // occupancy - (reserved_smem_bytes + c_bytes)
        ) // ab_bytes_per_stage

        # Refine epilogue stages:
        # Calculate remaining smem after allocating for A/B/SFA/SFB stages and reserved bytes
        # Add remaining unused smem to epilogue
        num_c_stage += (
            smem_capacity
            - occupancy * ab_bytes_per_stage * num_ab_stage
            - occupancy * (reserved_smem_bytes + c_bytes)
        ) // (occupancy * c_bytes_per_stage)

        return num_acc_stage, num_ab_stage, num_c_stage

    # fully dynamic shape
    @cute.jit
    def wrapper(
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
        max_active_clusters: cutlass.Constexpr,
        current_stream,
        swap_ab: cutlass.Constexpr = False,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Executes the wrapped GEMM kernel with dynamically shaped tensors.

        Uses TVM-FFI for efficient tensor passing: A, B, C, and alpha are passed
        as cute.Tensor directly (torch tensors at runtime via TVM-FFI's C-level
        dlpack, with negligible conversion cost). Scale factor tensors remain as
        pointers because their 6D BlockScaledBasicChunk layout cannot be expressed
        as a torch tensor.

        Args:
            mA (cute.Tensor): Input tensor A, shape (m, k), dtype Float8.
            mB (cute.Tensor): Input tensor B, shape (n, k), dtype Float8.
            mC (cute.Tensor): Output tensor C, shape (m, n).
            sf_m (cutlass.Int64): Scale factor M dim (ceil(m/128)).
            sf_n (cutlass.Int64): Scale factor N dim (ceil(n/128)).
            sf_k (cutlass.Int64): Scale factor K dim (ceil(k/sf_vec_size/4)).
            l (cutlass.Constexpr): Batch dimension (L).
            a_sf_ptr (cute.Pointer): Pointer to scale factor tensor for A.
            b_sf_ptr (cute.Pointer): Pointer to scale factor tensor for B.
            alpha_tensor (cute.Tensor): Alpha scaling factor, shape (1,), float32.
            max_active_clusters (cutlass.Constexpr): Max active clusters.
            current_stream: CUDA stream (managed by TVM-FFI fake stream).
            swap_ab (cutlass.Constexpr): Whether A/B are swapped (controls C layout).
            epilogue_op (cutlass.Constexpr): Elementwise epilogue function.
        """
        # A, B and C are passed as cute.Tensor.
        # fso-trim: the FP4 encoding (Uint8 operands recast to Float4E2M1FN
        # with k = k_packed * 2) was removed; only the MXFP8 encoding, whose
        # operands are already FP8, is kept.
        m = cute.size(mA, mode=[0])
        k_raw = cute.size(mA, mode=[1])
        n = cute.size(mB, mode=[0])

        if cutlass.const_expr(mA.element_type != mB.element_type):
            raise TypeError(
                "Unsupported mixed input dtypes for block-scaled GEMM: "
                "mA and mB must have matching element_type (both FP8)."
            )
        # MXFP8 path: input tensors are already FP8.
        k = k_raw
        a_ptr = mA.iterator
        b_ptr = mB.iterator

        a_tensor = cute.make_tensor(
            a_ptr,
            layout=cute.make_ordered_layout((m, k, l), order=(1, 0, 2)),
        )
        b_tensor = cute.make_tensor(
            b_ptr,
            layout=cute.make_ordered_layout((n, k, l), order=(1, 0, 2)),
        )
        # Reshape C to (m, n, l) -- swap_ab is constexpr, determines layout at compile time
        if cutlass.const_expr(swap_ab):
            c_tensor = cute.make_tensor(
                mC.iterator,
                layout=cute.make_ordered_layout((m, n, l), order=(0, 1, 2)),
            )
        else:
            c_tensor = cute.make_tensor(
                mC.iterator,
                layout=cute.make_ordered_layout((m, n, l), order=(1, 0, 2)),
            )
        # Scale factor tensors: 6D BlockScaledBasicChunk layout from pointers
        # (32, 4, sf_m, 4, sf_k, l) with order (2, 1, 4, 0, 3, 5)
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

        self(
            a_tensor,
            b_tensor,
            sfa_tensor,
            sfb_tensor,
            c_tensor,
            alpha_tensor,
            max_active_clusters,
            current_stream,
            epilogue_op,
        )

    # fso-epilogue: the entry of a kernel built with `fso_epilogue`. It is `wrapper` with two more operands, the
    # bias and the residual as raw pointers (None for a field the instance does not apply), so that `wrapper` and
    # the identity compiles through it stay exactly as they were. The bias is seen as a (m, n, l) tensor with stride
    # 0 along n (the kernel's m is the output column, which is the bias index: under swap_ab the weight's N), the
    # residual with the output tensor's own layout.
    @cute.jit
    def wrapper_epilogue(
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
        swap_ab: cutlass.Constexpr = False,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        m = cute.size(mA, mode=[0])
        k = cute.size(mA, mode=[1])
        n = cute.size(mB, mode=[0])

        if cutlass.const_expr(mA.element_type != mB.element_type):
            raise TypeError(
                "Unsupported mixed input dtypes for block-scaled GEMM: "
                "mA and mB must have matching element_type (both FP8)."
            )
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
        self(
            a_tensor,
            b_tensor,
            sfa_tensor,
            sfb_tensor,
            c_tensor,
            alpha_tensor,
            max_active_clusters,
            current_stream,
            epilogue_op,
            bias_tensor,
            res_tensor,
        )

    def mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for smem to tmem load for scale factor tensor, then use it to partition smem memory (source) and tensor memory (destination).

        Args:
            sSF (cute.Tensor): The scale factor tensor in smem
            tSF (cute.Tensor): The scale factor tensor in tmem

        Returns:
            A tuple containing (tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t) where:
                - tiled_copy_s2t: The tiled copy operation for smem to tmem load for scale factor tensor(s2t)
                - tCsSF_compact_s2t: The partitioned scale factor tensor in smem
                - tSF_compact_s2t: The partitioned scale factor tensor in tmem
        """
        # (MMA, MMA_MN, MMA_K, STAGE)
        tCsSF_compact = cute.filter_zeros(sSF)
        # (MMA, MMA_MN, MMA_K)
        tCtSF_compact = cute.filter_zeros(tSF)

        # Make S2T CopyAtom and tiledCopy
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)

        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t, tCsSF_compact_s2t_
        )
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)

        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for tensor memory load, then use it to partition tensor memory (source) and register array (destination).

        Args:
            tidx (cutlass.Int32): The thread index in epilogue warp groups
            tAcc (cute.Tensor): The accumulator tensor to be copied and partitioned
            gC_mnl (cute.Tensor): The global tensor C
            epi_tile (cute.Tile): The epilogue tiler
            use_2cta_instrs (bool): Whether use_2cta_instrs is enabled

        Returns:
            A tuple containing (tiled_copy_t2r, tTR_tAcc, tTR_rAcc) where:
                - tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
                - tTR_tAcc: The partitioned accumulator tensor
                - tTR_rAcc: The accumulated tensor in register used to hold t2r results
        """
        # Make tiledCopy for tensor memory load
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.c_layout,
            self.c_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, STAGE)
        tAcc_epi = cute.flat_divide(
            tAcc[((None, None), 0, 0, None)],
            epi_tile,
        )
        # (EPI_TILE_M, EPI_TILE_N)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
        )

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_M, STAGE)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
        gC_mnl_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, RestM, RestN, RestL)
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
        # (T2R, T2R_M, T2R_N)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    def epilog_smem_copy_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rC: cute.Tensor,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for shared memory store, then use it to partition register array (source) and shared memory (destination).

        Args:
            tiled_copy_t2r (cute.TiledCopy): The tiled copy operation for tmem to register copy(t2r)
            tTR_rC (cute.Tensor): The partitioned accumulator tensor
            tidx (cutlass.Int32): The thread index in epilogue warp groups
            sC (cute.Tensor): The shared memory tensor to be copied and partitioned
            sepi (cute.Tensor):

        Returns:
            A tuple containing (tiled_copy_r2s, tRS_rC, tRS_sC) where:
                - tiled_copy_r2s: The tiled copy operation for register to smem copy(r2s)
                - tRS_rC: The partitioned tensor C (register source)
                - tRS_sC: The partitioned tensor C (smem destination)
        """
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        # (R2S, R2S_M, R2S_N)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        return tiled_copy_r2s, tRS_rC, tRS_sC

    def epilog_gmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        atom: Union[cute.CopyAtom, cute.TiledCopy],
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        sC: cute.Tensor,
    ) -> Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for global memory store, then use it to:
        partition shared memory (source) and global memory (destination) for TMA store version.

        Args:
            tidx (cutlass.Int32): The thread index in epilogue warp groups
            atom (Union[cute.CopyAtom, cute.TiledCopy]): The copy_atom_c to be used for TMA store version, or tiled_copy_t2r for none TMA store version
            gC_mnl (cute.Tensor): The global tensor C
            epi_tile (cute.Tile): The epilogue tiler
            sC (cute.Tensor): The shared memory tensor to be copied and partitioned

        Returns:
            A tuple containing (tma_atom_c, bSG_sC, bSG_gC) where:
                - tma_atom_c: The TMA copy atom
                - bSG_sC: The partitioned shared memory tensor C
                - bSG_gC: The partitioned global tensor C
        """
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
        gC_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )

        tma_atom_c = atom
        sC_for_tma_partition = cute.group_modes(sC, 0, 2)
        gC_for_tma_partition = cute.group_modes(gC_epi, 0, 2)
        # ((ATOM_V, REST_V), EPI_M, EPI_N)
        # ((ATOM_V, REST_V), EPI_M, EPI_N, RestM, RestN, RestL)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            sC_for_tma_partition,
            gC_for_tma_partition,
        )
        return tma_atom_c, bSG_sC, bSG_gC
