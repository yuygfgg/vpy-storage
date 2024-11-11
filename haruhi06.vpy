# For haruhi 06

import vapoursynth as vs
from vapoursynth import core
import mvsfunc as mvf
import havsfunc as haf
import vsTAAmbk as taa
import rksfunc, ksksfunc, vsmlrt, vsutil, adptvgrnMod, nnedi3_resample, Vine, LimitedSharpen2, CSMOD, adjust

class cfg:

    width = 0
    height = 0
    episode = "01"
    clean = False

configure = cfg()

def load_source(filepath):
    clip = core.lsmas.LWLibavSource(filepath)
    if configure.width == 0 or configure.height == 0:
        configure.width = clip.width
        configure.height = clip.height
    clip = core.fmtc.bitdepth(clip=clip, bits=16)
    return clip

def dehalo_mask_dark(clip: vs.VideoNode): # without dark protection
    luma = vsutil.get_y(clip)
    l_mask = haf.FineDehalo(luma, showmask=4)
    mask_outer = l_mask.std.Maximum().std.Maximum().std.Inflate()
    mask_outer2 = core.std.Maximum(mask_outer).std.Maximum().std.Inflate()
    mask_inner1 = vsutil.iterate(mask_outer, core.std.Minimum, 4)
    mask_inner = mask_inner1
    r_mask1 = core.akarin.Expr([mask_outer2, mask_inner], 'x y -')
    r_mask = r_mask1
    r_mask_highres = nnedi3_resample.nnedi3_resample(input=r_mask, target_width=clip.width*2, target_height=clip.height*2, mode="nnedi3cl", nns=4, qual=2)
    r_mask_min = core.std.Minimum(clip=r_mask_highres)
    r_mask_back = core.resize.Spline36(clip=r_mask_min, width=clip.width ,height=clip.height)
    return r_mask_back

def dehalo_mask_small(clip: vs.VideoNode): # thinner
    luma = vsutil.get_y(clip)
    l_mask = haf.FineDehalo(luma, showmask=4)
    mask_outer = l_mask.std.Maximum().std.Inflate()
    mask_outer2 = core.std.Inflate(mask_outer)
    mask_inner1 = vsutil.iterate(mask_outer, core.std.Minimum, 3)
    mask_inner = core.akarin.Expr([mask_inner1, luma], "y 20000 < x 0 > and x 0 ?")
    r_mask1 = core.akarin.Expr([mask_outer2, mask_inner], 'x y -')
    r_mask = core.akarin.Expr([r_mask1, luma], "y 20000 < x 0 > and 0 x ?")
    return r_mask

def dehalo_mask_main(clip: vs.VideoNode):
    luma = vsutil.get_y(clip)
    l_mask = haf.FineDehalo(luma, showmask=4)
    mask_outer = l_mask.std.Maximum().std.Maximum().std.Inflate()
    mask_outer2 = core.std.Maximum(mask_outer).std.Maximum().std.Inflate()
    mask_inner1 = vsutil.iterate(mask_outer, core.std.Minimum, 4)
    mask_inner = core.akarin.Expr([mask_inner1, luma], "y 20000 < x 0 > and x 0 ?") # ((y < 20000) and (x > 0)) ? x : 0
    r_mask1 = core.akarin.Expr([mask_outer2, mask_inner], 'x y -')
    r_mask = core.akarin.Expr([r_mask1, luma], "y 20000 < x 0 > and 0 x ?") # ((y < 20000) and (x > 0)) ? 0 : x
    r_mask_highres = nnedi3_resample.nnedi3_resample(input=r_mask, target_width=clip.width*2, target_height=clip.height*2, mode="nnedi3cl", nns=4, qual=2)
    r_mask_min = core.std.Minimum(clip=r_mask_highres)
    r_mask_back = core.resize.Spline36(clip=r_mask_min, width=clip.width ,height=clip.height)
    return r_mask_back

def dehalo(clip):
    clipY = vsutil.get_y(clip)
    clipY720P = core.resize.Spline36(clip=clipY, width=1280, height=720, format=vs.GRAY16)
    main_mask720 = dehalo_mask_main(clipY720P)
    dark_mask720 = dehalo_mask_dark(clipY720P)
    dering1 = rksfunc.USMDering(cyuv16=clipY720P, mrad=1, mthr=75)
    dering1 = core.std.MaskedMerge(clipY720P, dering1, dark_mask720, first_plane=True)  # type: ignore
    dering1.set_output(6)
    dering2 = haf.FineDehalo(src=dering1, contra=1, edgeproc=1, darkstr=0.5)
    dering2 = core.std.MaskedMerge(clipY720P, dering2, dark_mask720, first_plane=True)
    dering2.set_output(7)
    dering3 = haf.EdgeCleaner(c=dering2, strength=10)
    dering3 = core.std.MaskedMerge(dering2, dering3, main_mask720, first_plane=True)
    dering3.set_output(8)
    backY = nnedi3_resample.nnedi3_resample(input=dering3, target_width=1920, target_height=1080, mode="nnedi3cl", nns=4, qual=2)
    ret = rksfunc.mergeuv(clipy=backY, clipuv=clip)
    return ret

def denoise(clip):
    denoised = rksfunc.Fast_BM3DWrapper(c420p16=clip)
    return denoised

def deband(clip):
    debanded = rksfunc.SynDeband(cyuv16=clip)
    return debanded

def aa(clip):
    aa_clip = taa.TAAmbk(clip, aatype=2, aatypeu=1, aatypev=1, preaa=-1, strength=0.4, cycle=1, mtype=3, mclip=None,
           mthr=None, mthr2=None, mlthresh=None, mpand=(1, 0), txtmask=0, txtfade=0, thin=1, dark=0.0, sharp=-0.25,
           aarepair=0, postaa=None, src=None, stabilize=1, down8=False, showmask=0, opencl=True)
    return aa_clip

def aa_oped(clip):
    aa_clip = taa.TAAmbk(clip, aatype=1, aatypeu=0, aatypev=0, preaa=-1, strength=0, cycle=0, mtype=3, mclip=None,
           mthr=None, mthr2=None, mlthresh=None, mpand=(1, 0), txtmask=1, txtfade=0, thin=0, dark=0.0, sharp=-1,
           aarepair=0, postaa=None, src=None, stabilize=3, down8=False, showmask=0, opencl=True)
    return aa_clip

def chroma_aa(clip: vs.VideoNode) -> vs.VideoNode: 
    configure.clean = True
    
    base = taa.TAAmbk(clip=clip, aatype=1, aatypeu=0, aatypev=0, opencl=True, showmask=1, mthr=56, strength=0)
    line_mask = base.std.Maximum().std.Minimum()
    
    low_sat_mask = base.std.Maximum().std.Minimum()
    dark_mask = core.std.Binarize(vsutil.get_y(clip), 13000)
    low_sat_mask = core.akarin.Expr([low_sat_mask, dark_mask], "x y -").std.Maximum().std.Binarize(8500)
    low_sat_mask.set_output(9)
    low_sat_mask = low_sat_mask.resize.Spline36(width=960, height=540)
    half = rksfunc.half444(clip)
    y, u, v = vsutil.split(half)
    v_stab = haf.QTGMC(v, InputType=1, SourceMatch=3, Sharpness=0, opencl=True)
    v_stab = haf.QTGMC(v_stab.std.Transpose(), InputType=1, SourceMatch=3, Sharpness=0, opencl=True).std.Transpose()
    u_stab = haf.QTGMC(u, InputType=1, SourceMatch=3, Sharpness=0, opencl=True)
    u_stab = haf.QTGMC(u_stab.std.Transpose(), InputType=1, SourceMatch=3, Sharpness=0, opencl=True).std.Transpose()
    half_stab = vsutil.join([y, u_stab, v_stab])
    half_w2x = rksfunc.w2xtrt(half_stab, 3, ofmt=True)
    half_w2x_low_sat = adjust.Tweak(clip=half_w2x, sat=0.85)
    half_w2x = core.std.MaskedMerge(half_w2x, half_w2x_low_sat, low_sat_mask, first_plane=True)
    ret = rksfunc.mergeuv(clip, half_w2x.resize.Spline36(format=vs.YUV444P16))
    merge = core.std.MaskedMerge(clip, ret, line_mask, first_plane=True)
    return merge

def smooth_line(clip):
    blur = haf.MinBlur(clp=clip, r=1)
    around_line_mask = dehalo_mask_small(clip)
    ret = core.std.MaskedMerge(clip, blur, around_line_mask, first_plane=True)
    return keep_darker(ret, clip)

def keep_darker(clipa, clipb): 
    return core.akarin.Expr(clips=[clipa,clipb], expr=["x y > y x ?", "x", "x"])

def cleanup():
    print("Cleaning onnx coreml cache...")
    import subprocess; subprocess.run('rm -r /private/var/folders/gb/m6gky8t10yx68drv06ncbj_h0000gn/T/onnxruntime*', shell=True, check=True)

def p_oped(clip):
    denoised1080P = denoise(clip)
    nrmask = ksksfunc.DBMask(denoised1080P)
    dbed1080P = deband(denoised1080P)
    aaed1080P = aa_oped(denoised1080P)
    merged1080P = core.std.MaskedMerge(dbed1080P,aaed1080P,nrmask,first_plane=True) # type: ignore
    noise = adptvgrnMod.adptvgrnMod(clip_in=merged1080P, strength=[1,0])
    return noise

def p_body(clip):
    merged_YUV = chroma_aa(clip)
    merged_YUV = haf.FineDehalo(merged_YUV, contra=1, edgeproc=1)
    # merged_YUV = dehalo(merged_YUV)
    denoised1080P = denoise(merged_YUV)
    nrmask = ksksfunc.DBMask(denoised1080P)
    dbed1080P = deband(denoised1080P)
    aaed1080P = aa(denoised1080P)
    merged1080P = core.std.MaskedMerge(dbed1080P,aaed1080P,nrmask,first_plane=True) # type: ignore
    
    merged1080P.set_output(3)
    dehalod = dehalo(merged1080P)
    
    dehalod.set_output(4)
    around_line_mask = dehalo_mask_small(dehalod)
    around_line_mask.set_output(5)
    # line_mask.set_output(4)
    # ret.set_output(5)
    # sharp = core.warp.AWarpSharp2(ret, thresh=128, blur=3, type=1, depth=[1, 1, 1], chroma=1, cplace="mpeg2", opt=True)
    # dark = haf.FastLineDarkenMOD(c=sharp)
    dark = keep_darker(dehalod,merged1080P)
    # dark = mvf.LimitFilter(flt=ret, src=merged1080P, brighten_thr=0.01)

    # sharp = LimitedSharpen2.LimitedSharpen2(clp=clean, strength=50, edgemode=1)
    # sharp = core.cas.CAS(clip=clean, sharpness=0.1, opt=2)
    # sharp = core.std.MaskedMerge(clean, sharp, line_mask, first_plane=True)
    sharp = CSMOD.CSMOD(dark, preset='detail', ss_hq=1)
    finn = smooth_line(sharp)
    noise = adptvgrnMod.adptvgrnMod(clip_in=finn, strength=[1,0], size=1.5)
    return noise

def main():

    eg = core.lsmas.LWLibavSource(f'/Volumes/untitled/S01E{configure.episode}_vcb.mkv')
    eg = mvf.Depth(eg,16)
    src16 = load_source(f'/Volumes/untitled/S01E{configure.episode}.mkv').std.Trim(first=24)
    # src16 = load_source('/Users/a1/00000.m2ts')
    body_part = p_body(src16)
    oped = p_oped(src16)

    op = oped.std.Trim(first=3592, last=5755)
    ed = oped.std.Trim(first=33976)
    before_op = body_part.std.Trim(first=0, last=3591)
    body = body_part.std.Trim(first=5756, last=33975)
    fin = before_op + op + body + ed
    
    # fin = adptvgrnMod.adptvgrnMod(clip_in=body_part, strength=1)
    
    if configure.clean:
        cleanup()
    fin.set_output(0)
    eg.set_output(1)
    src16.set_output(2)

main()
