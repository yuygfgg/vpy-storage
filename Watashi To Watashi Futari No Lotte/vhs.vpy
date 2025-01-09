from __future__ import annotations

import vapoursynth as vs
from vapoursynth import core
import mvsfunc as mvf
import havsfunc as haf
import vsTAAmbk as taa
import rksfunc, vsmlrt, vsutil, y5gfunc, muvsfunc, functools, fvsfunc, lvsfunc
from typing import Tuple

from functools import partial
from typing import Any, Optional, Sequence, SupportsFloat, Tuple

import vapoursynth as vs

from lvsfunc.kernels import Catrom, Kernel
file_path = "/Volumes/untitled/lotte/01.mkv"

class cfg:

    width = 0
    height = 0

configure = cfg()

@vsutil.disallow_variable_format
@vsutil.disallow_variable_resolution
def autodb_dpir(clip: vs.VideoNode, edgevalue: int = 24,
                strs: Sequence[float] = [30, 50, 75],
                thrs: Sequence[Tuple[float, float, float]] = [(1.5, 2.0, 2.0), (3.0, 4.5, 4.5), (5.5, 7.0, 7.0)],
                matrix: Optional[lvsfunc.types.Matrix | int] = None,
                cuda: bool = True,
                write_props: bool = False,
                **vsdpir_args: Any) -> vs.VideoNode:
    
    assert clip.format

    def _eval_db(n: int, f: Sequence[vs.VideoFrame],
                 clip: vs.VideoNode, db_clips: Sequence[vs.VideoNode],
                 nthrs: Sequence[Tuple[float, float, float]]) -> vs.VideoNode:

        evref_diff, y_next_diff, y_prev_diff = [
            lvsfunc.util.get_prop(f[i], prop, float)
            for i, prop in zip(range(3), ['EdgeValRefDiff', 'YNextDiff', 'YPrevDiff'])
        ]
        # f_type = get_prop(f[0], '_PictType', bytes).decode('utf-8')

        # if f_type == 'I':
        #     y_next_diff = (y_next_diff + evref_diff) / 2

        out = clip
        nthr_used = (-1., ) * 3
        for dblk, nthr in zip(db_clips, nthrs):
            if all(p > t for p, t in zip([evref_diff, y_next_diff, y_prev_diff], nthr)):
                out = dblk
                nthr_used = nthr

        if write_props:
            for prop_name, prop_val in zip(
                ['Adb_EdgeValRefDiff', 'Adb_YNextDiff', 'Adb_YPrevDiff',
                 'Adb_EdgeValRefDiffThreshold', 'Adb_YNextDiffThreshold', 'Adb_YPrevDiffThreshold'],
                [evref_diff, y_next_diff, y_prev_diff] + list(nthr_used)
            ):
                out = out.std.SetFrameProp(prop_name, floatval=max(prop_val * 255, -1))

        return out

    if len(strs) != len(thrs):
        raise ValueError('autodb_dpir: You must pass an equal amount of values to '
                         f'strenght {len(strs)} and thrs {len(thrs)}!')

    nthrs = [tuple(x / 255 for x in thr) for thr in thrs]

    is_rgb = clip.format.color_family is vs.RGB

    if not matrix and not is_rgb:
        matrix = lvsfunc.util.get_prop(clip.get_frame(0), "_Matrix", int)

    rgb = core.resize.Bicubic(clip, format=vs.RGBS, matrix_in=matrix) if not is_rgb else clip

    assert rgb.format

    maxvalue = (1 << rgb.format.bits_per_sample) - 1
    evref = core.std.Prewitt(rgb)
    evref = core.akarin.Expr(evref, f"x {edgevalue} >= {maxvalue} x ?")
    evref_rm = evref.std.Median().std.Convolution(matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])

    diffevref = core.std.PlaneStats(evref, evref_rm, prop='EdgeValRef')
    diffnext = core.std.PlaneStats(rgb, rgb.std.DeleteFrames([0]), prop='YNext')
    diffprev = core.std.PlaneStats(rgb, rgb[0] + rgb, prop='YPrev')

    db_clips = [
        vsdpir(rgb, strength=st, mode='deblock', cuda=cuda, **vsdpir_args)
        .std.SetFrameProp('Adb_DeblockStrength', intval=int(st)) for st in strs
    ]

    debl = core.std.FrameEval(
        rgb, partial(_eval_db, clip=rgb, db_clips=db_clips, nthrs=nthrs),
        prop_src=[diffevref, diffnext, diffprev]
    )

    return core.resize.Bicubic(debl, format=clip.format.id, matrix=matrix if not is_rgb else None)


@vsutil.disallow_variable_format
@vsutil.disallow_variable_resolution
def vsdpir(clip: vs.VideoNode, strength: SupportsFloat | vs.VideoNode | None = 25, mode: str = 'deblock',
           matrix: lvsfunc.types.Matrix | int | None = None, tiles: int | Tuple[int] | None = None,
           cuda: bool = True, i444: bool = False, kernel: Kernel = Catrom(), **dpir_args: Any) -> vs.VideoNode:

    try:
        from vsmlrt import DPIR, Backend, DPIRModel
    except ModuleNotFoundError:
        raise ModuleNotFoundError("deblock: 'vsmlrt is required to use deblocking function.'")

    assert clip.format

    bit_depth = vsutil.get_depth(clip)
    is_rgb, is_gray = (clip.format.color_family is f for f in (vs.RGB, vs.GRAY))

    clip_32 = vsutil.depth(clip, 32, dither_type=vsutil.Dither.ERROR_DIFFUSION).std.Limiter()

    # TODO: Replace with a match-case?
    if mode.lower() == 'deblock':
        model = DPIRModel.drunet_deblocking_color if not is_gray else DPIRModel.drunet_deblocking_grayscale
    elif mode.lower() == 'denoise':
        model = DPIRModel.drunet_color if not is_gray else DPIRModel.drunet_gray
    else:
        raise ValueError(f"""vsdpir: '"{mode}" is not a valid mode!'""")

    dpir_args |= dict(strength=strength, tiles=tiles, model=model)

    if "backend" not in dpir_args:
        dpir_args |= dict(backend=Backend.TRT() if cuda else Backend.ORT_COREML())

    if is_rgb or is_gray:
        return vsutil.depth(DPIR(clip_32, **dpir_args), bit_depth)

    if matrix is None:
        matrix = lvsfunc.util.get_prop(clip.get_frame(0), "_Matrix", int)

    targ_matrix = lvsfunc.types.Matrix(matrix)
    targ_format = clip.format.replace(subsampling_w=0, subsampling_h=0) if i444 else clip.format

    clip_rgb = kernel.resample(clip_32, vs.RGBS, matrix_in=targ_matrix)  # type:ignore[arg-type]

    run_dpir = DPIR(clip_rgb, **dpir_args)

    return kernel.resample(run_dpir, targ_format, targ_matrix)  # type:ignore[arg-type]


clip = core.bs.VideoSource(file_path, rff=True, threads=1)
crgb = clip.resize.Bicubic(format=vs.RGB48, matrix_in=5)
src = crgb.resize.Spline36(format=vs.YUV420P16, matrix=1)
deblock = autodb_dpir(src, cuda=False, strs=[35, 55, 75])
# deblock = autodb_dpir(src, cuda=True, strs=[35, 55, 75])

res = haf.QTGMC(deblock, TFF=True, Preset="Slower", FPSDivisor=2, Precise=True, SourceMatch=3, NNeurons=2)

dering = haf.FineDehalo(res)
dering = haf.FineDehalo(dering)
dering = haf.FineDehalo(dering)

mask = muvsfunc.AnimeMask(input=res, shift=1.5, mode=1).std.Binarize(threshold=1)
dering = core.std.MaskedMerge(clipa=res, clipb=dering, mask=mask, first_plane=True)

stable = taa.temporal_stabilize(clip=dering, src=res)

denoised = y5gfunc.Fast_BM3DWrapper(clip=stable, sigma_Y=5, chroma=False)
# denoised = rksfunc.BM3DWrapper(c420p16=stable, chroma=False, sy=5)
cleaned = haf.EdgeCleaner(denoised, 10)
w2x = rksfunc.w2xtrt(clip=rksfunc.uvsr(cleaned, mode="nnedi3"), noise=3, ofmt=True) # model=Waifu2xModel.anime_style_art_rgb
chroma_imp = rksfunc.mergeuv(clipy=cleaned, clipuv=w2x)

# aaed = aa(chroma_imp)

# diff = vsutil.get_y(core.akarin.Expr([aaed, chroma_imp], 'x y - abs 100 *'))
outputs = [clip, src, deblock, res, dering, stable, denoised, cleaned, chroma_imp]

y5gfunc.screen_shot(src, y5gfunc.ranger(2, src.num_frames, 719), '/Users/a1/vhs_cmp', 'vhs_%d.png', True)
y5gfunc.screen_shot(chroma_imp, y5gfunc.ranger(2, chroma_imp.num_frames, 719), '/Users/a1/vhs_cmp', 'filtered_%d.png', True)
y5gfunc.output(outputs, debug=True)
y5gfunc.reset_output_index()
import subprocess; subprocess.run('rm -r /private/var/folders/gb/m6gky8t10yx68drv06ncbj_h0000gn/T/onnxruntime*', shell=True, check=True)