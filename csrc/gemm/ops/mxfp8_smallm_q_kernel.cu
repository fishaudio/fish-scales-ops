/*
 * sm_120 small-M dense MXFP8 route (issue #12 decision 3): the kernels of form G1 (pre-quantized x), identity epilogue.
 * One translation unit per (form, epilogue) so the four sets compile in parallel; see mxfp8_smallm_kernels.inc.
 */
#define FSO_SMALLM_FORM 0
#define FSO_SMALLM_EPI 0
#include "mxfp8_smallm_kernels.inc"
