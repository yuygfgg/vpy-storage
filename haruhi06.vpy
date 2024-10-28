# For haruhi 06

import vapoursynth as vs
from vapoursynth import core
import mvsfunc as mvf
import havsfunc as haf
import vsTAAmbk as taa
import rksfunc, ksksfunc, vsmlrt, vsutil
from nnedi3_resample import nnedi3_resample

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
    clip = core.resize.Bicubic(clip=clip, width=configure.width, height=configure.height, format=vs.YUV420P16, matrix_s="709", matrix_in_s="709")
    return clip

def dehalo(clipo):
    clip = vsutil.get_y(clipo)
    clip720P = core.resize.Spline16(clip=clip, width=1280, height=720)
    dering1 = rksfunc.USMDering(cyuv16=clip720P, mrad=2, mthr=75)
    dering2 = haf.FineDehalo(src=dering1, contra=1)
    back = nnedi3_resample(input=dering2, target_height=1080, target_width=1920, mode="nnedi3cl")
    line = taa.TAAmbk(clip=clip, aatype=1, aatypeu=0, aatypev=0, opencl=True, showmask=1, mthr=56, strength=0.5).std.Maximum().std.Maximum()
    merge_back = core.std.MaskedMerge(clip, back, line, first_plane=True)
    yuv = core.std.ShufflePlanes(clips=[merge_back, clipo], planes=[0,1,2], colorfamily=vs.YUV)
    return yuv

def denoise(clip):
    denoised = rksfunc.Fast_BM3DWrapper(c420p16=clip)
    return denoised

def deband(clip):
    debanded = rksfunc.SynDeband(cyuv16=clip)
    return debanded

def aa(clip):
    aa_clip = taa.TAAmbk(clip, aatype=2, aatypeu=0, aatypev=0, preaa=-1, strength=0.45, cycle=1, mtype=3, mclip=None,
           mthr=None, mthr2=None, mlthresh=None, mpand=(1, 0), txtmask=0, txtfade=0, thin=0, dark=0.0, sharp=-1,
           aarepair=0, postaa=None, src=None, stabilize=3, down8=False, showmask=0, opencl=True)
    return aa_clip

def aa_oped(clip):
    aa_clip = taa.TAAmbk(clip, aatype=1, aatypeu=0, aatypev=0, preaa=-1, strength=0, cycle=0, mtype=3, mclip=None,
           mthr=None, mthr2=None, mlthresh=None, mpand=(1, 0), txtmask=1, txtfade=0, thin=0, dark=0.0, sharp=-1,
           aarepair=0, postaa=None, src=None, stabilize=3, down8=False, showmask=0, opencl=True)
    return aa_clip

def dnchroma_stabv(toduv):
    configure.clean = True
    half = rksfunc.half444(toduv)
    y, u, v = vsutil.split(half)
    v_stab = haf.QTGMC(v, InputType=1, SourceMatch=3, Sharpness=0, opencl=True) # type: ignore
    v_stab = haf.QTGMC(v_stab.std.Transpose(), InputType=1, SourceMatch=3, Sharpness=0, opencl=True).std.Transpose() # type: ignore
    half_stab = vsutil.join([y, u, v]).fmtc.bitdepth(bits=32)
    half_w2x = rksfunc.w2xtrt(half_stab, 3, ofmt=True)
    # duv = half_stab.bm3dcpu.BM3Dv2(half_w2x, 2, 2, 8, 0, 2, 8)
    duv = half_stab.bm3dcpu.BM3Dv2(ref=half_w2x, sigma=2, block_step=6, bm_range=8, radius=0, ps_num=2, ps_range=4)
    duv_swap_v = core.std.ShufflePlanes([duv, duv, half_w2x], [0, 1, 2], vs.YUV)
    return rksfunc.mergeuv(toduv, duv_swap_v.fmtc.bitdepth(bits=16))

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
    merged_YUV = dnchroma_stabv(clip)
    denoised1080P = denoise(merged_YUV)
    nrmask = ksksfunc.DBMask(denoised1080P)
    dbed1080P = deband(denoised1080P)
    aaed1080P = aa(denoised1080P)
    merged1080P = core.std.MaskedMerge(dbed1080P,aaed1080P,nrmask,first_plane=True) # type: ignore
    dehalod = dehalo(merged1080P)
    ret = haf.EdgeCleaner(c=dehalod, strength=10, rep=True, rmode=1, smode=1)
    sharp = core.warp.AWarpSharp2(ret, thresh=128,blur=3,type=1,depth=[2, 0, 0],chroma=1,cplace="mpeg2", opt=True)
    return sharp

def main():

    eg = core.lsmas.LWLibavSource(f'/Volumes/untitled/S01E{configure.episode}_vcb.mkv')
    src16 = load_source(f'/Volumes/untitled/S01E{configure.episode}.mkv').std.Trim(first=24)

    body_part = p_body(src16)
    oped = p_oped(src16)

    op = oped.std.Trim(first=3592, last=5755)
    ed = oped.std.Trim(first=33976)
    before_op = body_part.std.Trim(first=0, last=3591)
    body = body_part.std.Trim(first=5756, last=33975)
    fin = before_op + op + body + ed

    fin.set_output(0)
    # must align frames with vcb!
    # eg.set_output(1)
    # src16.set_output(2)

    if configure.clean:
        cleanup()

main()
