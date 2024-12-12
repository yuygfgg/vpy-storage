
import vapoursynth as vs
from vapoursynth import core
import mvsfunc as mvf
import havsfunc as haf
import vsTAAmbk as taa
import rksfunc, ksksfunc, vsdpir, adptvgrnMod, muvsfunc, CSMOD, vsmlrt, vsutil, nnedi3_resample
import kagefunc as kgf

file_path = "/Volumes/untitled/00000.m2ts"

class cfg:
    width = 0
    height = 0

def keep_darker(clipa, clipb): 
    return core.akarin.Expr(clips=[clipa,clipb], expr=["x y min", "x", "x"])

def aa(clip):
    Y = vsutil.get_y(clip)
    aaY = taa.TAAmbk(Y, aatype=2, aatypeu=0, aatypev=0, preaa=0, strength=0, cycle=0, mtype=3, mclip=None,
           mthr=None, mthr2=None, mlthresh=None, mpand=(1, 0), txtmask=1, txtfade=0, thin=0, dark=0, sharp=0,
           aarepair=0, postaa=None, src=None, stabilize=3, down8=False, showmask=0, opencl=True)
    ret = rksfunc.mergeuv(aaY, clip)
    return ret

configure = cfg()


def load_source(filepath):
    clip = core.lsmas.LWLibavSource(filepath)
    if configure.width == 0 or configure.height == 0:
        configure.width = clip.width
        configure.height = clip.height
    # clip = core.fmtc.bitdepth(clip=clip, bits=16)
    clip = core.resize.Bicubic(clip=clip, width=configure.width, height=configure.height, format=vs.YUV420P16, matrix_s="709", matrix_in_s="709")
    return clip

def output(output_list):
    for index, clip in enumerate(output_list):
        clip.set_output(index)

src = load_source(filepath=file_path)
src = rksfunc.uvsr(c420p16=src, mode='nnedi3', opencl=True)

rescale = kgf.inverse_scale(source=src, width=960, height=540, kernel="bilinear", mask_detail=True)
yuv = core.resize.Spline36(clip=rescale, width=960, height=540, format=vs.YUV444P16, matrix_s="709")

backe = nnedi3_resample.nnedi3_resample(input=yuv, target_width=1920, target_height=1080, mode="nnedi3cl", nns=4, qual=2)
mask = muvsfunc.AnimeMask(input=backe, shift=0.7, mode=1)
backe = mvf.LimitFilter(flt=backe, src=src, thr=7, elast=5)
backe = keep_darker(backe, src)
dehalod = haf.FineDehalo(backe, edgeproc=1, contra=1)
# dehalod = core.std.MaskedMerge(clipa=backe, clipb=dehalod, mask=mask, first_plane=True)
dehalod = haf.EdgeCleaner(c=dehalod, strength=15)

rescaled = mvf.ToYUV(dehalod)
rescaled = core.resize.Spline36(clip=rescaled, width=1920, height=1080, format=vs.YUV420P16)

denoised = rksfunc.Fast_BM3DWrapper(rescaled, sigma_Y=2, sigma_chroma=3, radius_Y=1)

nrmask = ksksfunc.DBMask(denoised)
aaedY1080P = aa(denoised)
dbed1080P = rksfunc.SynDeband(cyuv16=aaedY1080P, limit=True, limit_elast=1.75)
merged1080P = core.std.MaskedMerge(dbed1080P,aaedY1080P,nrmask,first_plane=True) # type: ignore
# sharp = CSMOD.CSMOD(merged1080P, preset="detail", opencl=True)
sharp = core.warp.AWarpSharp2(clip=merged1080P, thresh=128, blur=3, type=1, depth=[1, 0, 0], chroma=1, cplace="mpeg2", opt=True)
outputs = [sharp, merged1080P, src, rescale, yuv, backe, dehalod, merged1080P, mask]

output(outputs)
# import subprocess; subprocess.run('rm -r /private/var/folders/gb/m6gky8t10yx68drv06ncbj_h0000gn/T/onnxruntime*', shell=True, check=True)
