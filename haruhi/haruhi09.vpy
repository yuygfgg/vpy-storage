# For haruhi 09

import vapoursynth as vs
from vapoursynth import core
import mvsfunc as mvf
import havsfunc as haf
import vsTAAmbk as taa
import rksfunc, ksksfunc, vsmlrt, vsutil, adptvgrnMod, nnedi3_resample, Vine, kagefunc, y5gfunc, muvsfunc, lvsfunc, getfnative

file_path = "/Volumes/untitled/haruhi/raw_video/00003.m2ts"

class cfg:

    width = 0
    height = 0

def aa(clip):
    clipY = core.std.ShufflePlanes(clip, 0, vs.GRAY)
    aa_clipY = taa.TAAmbk(clipY, aatype=1, aatypeu=0, aatypev=0, preaa=0, strength=0, cycle=0, mtype=3, mclip=None,
           mthr=None, mthr2=None, mlthresh=None, mpand=(1, 0), txtmask=1, txtfade=0, thin=0, dark=0, sharp=-1,
           aarepair=2, postaa=None, src=None, stabilize=3, down8=False, showmask=0, opencl=True)
    aa_clip = rksfunc.mergeuv(clipy=aa_clipY, clipuv=clip)
    return aa_clip

configure = cfg()


def load_source(filepath):
    clip = core.lsmas.LWLibavSource(filepath)
    if configure.width == 0 or configure.height == 0:
        configure.width = clip.width
        configure.height = clip.height
    clip = core.fmtc.bitdepth(clip=clip, bits=16)
    clip = rksfunc.KrigBilateral(c420p16=clip)
    clip = core.resize.Spline36(clip=clip, width=configure.width, height=configure.height, format=vs.YUV420P16, matrix_s="709", matrix_in_s="709")
    return clip

def denoise(clip):
    denoised = rksfunc.Fast_BM3DWrapper(c420p16=clip)
    return denoised

def deband(clip):
    debanded = rksfunc.SynDeband(cyuv16=clip)
    return debanded

srcorg = load_source(filepath=file_path)

src_fft = core.fftspectrum.FFTSpectrum(clip=mvf.Depth(srcorg,8), grid=True)

srcorg = mvf.Depth(srcorg,32)

rescaled1, cmask1, osd1 = y5gfunc.rescale(clip=srcorg, src_height=y5gfunc.ranger(714, 715, 0.1)+[713, 714, 716, 717, 718, 719], bw=[1920], bh=[1080], descale_kernel=["Debicubic"], b=[1/3], c=[1/3], show_detail_mask=True, show_fft=False) # type: ignore
rescaled2, cmask2, osd2 = y5gfunc.rescale(clip=srcorg, src_height=y5gfunc.ranger(955, 957,0.1)+[953, 954, 958, 959], bw=[1920], bh=[1080], descale_kernel=["Delanczos"], taps=2, show_detail_mask=True, show_fft=False) # type: ignore

select_expr = "src0.MaxDelta src0.Descaled * src1.MaxDelta src1.Descaled * argmax2"

osd = core.akarin.Select([osd1, osd2], [rescaled1, rescaled2], select_expr)
rescaled = core.akarin.Select([rescaled1, rescaled2], [rescaled1, rescaled2], select_expr)
cmask = core.akarin.Select([cmask1, cmask2], [rescaled1, rescaled2], select_expr)

rescaled_fft = core.fftspectrum.FFTSpectrum(clip=mvf.Depth(rescaled,8), grid=True)

src = rescaled.resize.Spline36(format=vs.YUV420P16)


denoised = denoise(src)

aaed = aa(denoised)

nrmask = ksksfunc.DBMask(denoised)

debanded = deband(denoised)

merged1080P = core.std.MaskedMerge(debanded,aaed,nrmask,first_plane=True) # type: ignore

fin = adptvgrnMod.adptvgrnMod(clip_in=merged1080P, strength=[1,0], size=1.5)

rescaled = mvf.Depth(rescaled, 10)

outputs = [srcorg, rescaled, (fin,0), osd, cmask, src_fft, rescaled_fft]

y5gfunc.reset_output_index()
y5gfunc.output(outputs)
