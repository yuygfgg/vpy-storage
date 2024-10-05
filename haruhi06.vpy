# For haruhi 06

import vapoursynth as vs
from vapoursynth import core
import mvsfunc as mvf
import havsfunc as haf
import vsTAAmbk as taa
import rksfunc, ksksfunc, vsmlrt, vsutil

class cfg:

    width = 0
    height = 0
    episode = "02"
    clean = True

configure = cfg()

def load_source(filepath):
    clip = core.lsmas.LWLibavSource(filepath)
    if configure.width == 0 or configure.height == 0:
        configure.width = clip.width
        configure.height = clip.height
    clip = core.resize.Bicubic(clip=clip, width=configure.width, height=configure.height, format=vs.YUV420P16, matrix_s="709", matrix_in_s="709")
    return clip

def dehalo(clip):
    luma = vsutil.get_y(clip)

    l_mask = haf.FineDehalo(luma, showmask=4)
    mask_outer = l_mask.std.Maximum().std.Maximum().std.Inflate()
    mask_outer2 = core.std.Maximum(mask_outer).std.Maximum().std.Inflate()

    mask_inner1 = vsutil.iterate(mask_outer, core.std.Minimum, 4)
    mask_inner = core.akarin.Expr([mask_inner1, luma], "y 20000 < x 0 > and x 0 ?") # ((y < 20000) and (x > 0)) ? x : 0
    r_mask1 = core.akarin.Expr([mask_outer2, mask_inner], 'x y -')
    r_mask = core.akarin.Expr([r_mask1, luma], "y 20000 < x 0 > and 0 x ?") # ((y < 20000) and (x > 0)) ? 0 : x
    r_mask = core.std.Minimum(r_mask).std.Inflate()
    dering = haf.DeHalo_alpha(clp=clip, rx=2.5, ry=2.5, darkstr=0.4)
    dering_merged = core.std.MaskedMerge(clip, dering, r_mask)
    dering_merged = core.rgvs.RemoveGrain(clip=dering_merged, mode=24)

    dehalod = core.resize.Bicubic(clip=dering_merged, width=configure.width, height=configure.height, format=vs.YUV420P16, matrix_s="709")

    return dehalod

def denoise(clip):
    denoised = mvf.BM3D(input=clip, sigma=[4, 0, 0], radius1=1)
    denoised = mvf.BM3D(input=denoised, sigma=[0, 5, 5], radius1=0)
    return denoised

def deband(clip):
    debanded = rksfunc.SynDeband(cyuv16=clip)
    return debanded

def aa(clip):
    aa_clip = taa.TAAmbk(clip, aatype=2, aatypeu=0, aatypev=0, preaa=-1, strength=0.45, cycle=1, mtype=3, mclip=None,
           mthr=None, mthr2=None, mlthresh=None, mpand=(1, 0), txtmask=0, txtfade=0, thin=0, dark=0.0, sharp=-1,
           aarepair=0, postaa=None, src=None, stabilize=1, down8=False, showmask=0, opencl=True)
    return aa_clip

def aa_oped(clip):
    aa_clip = taa.TAAmbk(clip, aatype=1, aatypeu=0, aatypev=0, preaa=-1, strength=0, cycle=0, mtype=3, mclip=None,
           mthr=None, mthr2=None, mlthresh=None, mpand=(1, 0), txtmask=1, txtfade=0, thin=0, dark=0.0, sharp=-1,
           aarepair=0, postaa=None, src=None, stabilize=1, down8=False, showmask=0, opencl=True)
    return aa_clip

# def dnchroma_stabv(toduv: vs.VideoNode) -> vs.VideoNode:
#     from havsfunc import QTGMC
#     from vsutil import split, join
#     half = rksfunc.half444(toduv)
#     y, u, v = split(half)
#     mc_args = dict(InputType=1, SourceMatch=3, Sharpness=0)
#     v_stab = QTGMC(v, **mc_args)
#     v_stab = QTGMC(v_stab.std.Transpose(), **mc_args).std.Transpose()
#     half_stab = join([y, u, v]).fmtc.bitdepth(bits=32)
#     half_w2x = rksfunc.w2xtrt(half_stab, 3, ofmt=True)
#     duv = half_stab.bm3dcpu.BM3Dv2(half_w2x, 2, 2, 8, 0, 2, 8)
#     duv_swap_v = core.std.ShufflePlanes([duv, duv, half_w2x], [0, 1, 2], vs.YUV)
#     return rksfunc.mergeuv(toduv, duv_swap_v.fmtc.bitdepth(bits=16))

def chroma_aa(clip):
    srcY, srcU, srcV = vsutil.split(clip)
    line_mask_1080P = taa.TAAmbk(clip=clip, aatype=1, aatypeu=0, aatypev=0, opencl=True, showmask=1, mthr=56, strength=0.5)
    line_mask_540P = core.resize.Bicubic(clip=line_mask_1080P, width=960, height=540, src_left=-0.5)
    clip_RGBS = core.resize.Bicubic(clip=clip, width=960, height=540, format=vs.RGBS)
    chroma_aaed_RGBS = vsmlrt.CUGAN(clip=clip_RGBS, noise=0, scale=2, tiles=1, backend=vsmlrt.Backend.ORT_COREML(num_streams=2))
    chroma_aaed_YUV = mvf.ToYUV(chroma_aaed_RGBS)
    chroma_aaed_YUV = mvf.Depth(chroma_aaed_YUV, 16)
    chroma_aaed_YUV = core.resize.Bicubic(clip=chroma_aaed_YUV, width=1920, height=1080, format=vs.YUV420P16)
    chroma_aaedU = core.std.ShufflePlanes(chroma_aaed_YUV, 1, vs.GRAY)
    chroma_aaedV = core.std.ShufflePlanes(chroma_aaed_YUV, 2, vs.GRAY)
    mergedU = core.std.MaskedMerge(srcU, chroma_aaedU, line_mask_540P, first_plane=True)
    mergedV = core.std.MaskedMerge(srcV, chroma_aaedV, line_mask_540P, first_plane=True)
    merged_YUV = core.std.ShufflePlanes([srcY, mergedU, mergedV], [0, 0, 0], vs.YUV)
    return merged_YUV

def cleanup():
    print("Cleaning onnx coreml cache...")
    import subprocess; subprocess.run('rm -r /private/var/folders/gb/m6gky8t10yx68drv06ncbj_h0000gn/T/onnxruntime*', shell=True, check=True)

def p_oped(clip):
    denoised1080P = denoise(clip)
    nrmask = ksksfunc.DBMask(denoised1080P)
    dbed1080P = deband(denoised1080P)
    aaed1080P = aa_oped(denoised1080P)
    merged1080P = core.std.MaskedMerge(dbed1080P,aaed1080P,nrmask,first_plane=True) # type: ignore
    return merged1080P

def p_body(clip):
    merged_YUV = chroma_aa(clip)
    denoised1080P = denoise(merged_YUV)
    nrmask = ksksfunc.DBMask(denoised1080P)
    # nrmask.set_output(3)
    dbed1080P = deband(denoised1080P)
    aaed1080P = aa(denoised1080P)
    merged1080P = core.std.MaskedMerge(dbed1080P,aaed1080P,nrmask,first_plane=True) # type: ignore
    dehalod = dehalo(merged1080P)
    sharp = core.warp.AWarpSharp2(dehalod,thresh=128,blur=3,type=1,depth=[4, 2, 2],chroma=1,cplace="mpeg2", opt=True)
    return sharp

def main():

    # eg = core.lsmas.LWLibavSource(f'/Volumes/untitled/S01E{configure.episode}_vcb.mkv')
    src16 = load_source(f'/Volumes/untitled/S01E{configure.episode}.mkv')

    body_part = p_body(src16)
    oped = p_oped(src16)

    op = oped.std.Trim(first=2682, last=4844)
    ed = oped.std.Trim(first=32120)
    before_op = body_part.std.Trim(first=0, last=2681)
    body = body_part.std.Trim(first=4845, last=32119)
    fin = before_op + op + body + ed

    fin.set_output(0)
    # must align frames with vcb!
    # eg.set_output(1)
    # src16.set_output(2)

    if configure.clean:
        cleanup()

main()
