# For haruhi 09

import vapoursynth as vs
from vapoursynth import core
import mvsfunc as mvf
import havsfunc as haf
import vsTAAmbk as taa
import rksfunc, ksksfunc, vsmlrt, vsutil, adptvgrnMod, nnedi3_resample, Vine, kagefunc


file_path = "/Volumes/untitled/00003.m2ts"

class cfg:

    width = 0
    height = 0

def aa(clip):
    clipY = core.std.ShufflePlanes(clip, 0, vs.GRAY)
    aa_clipY = taa.TAAmbk(clipY, aatype=2, aatypeu=0, aatypev=0, preaa=0, strength=0.25, cycle=1, mtype=3, mclip=None,
           mthr=None, mthr2=None, mlthresh=None, mpand=(1, 0), txtmask=1, txtfade=0, thin=0, dark=0.25, sharp=-1,
           aarepair=0, postaa=None, src=None, stabilize=3, down8=False, showmask=0, opencl=True)
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

def output(output_list):
    for index, clip in enumerate(output_list):
        clip.set_output(index)

src = load_source(filepath=file_path)

src = kagefunc.inverse_scale(source=src, width=1268, height=713, kernel="bicubic", b=1/3, c=1/3, mask_detail=True)

src = nnedi3_resample.nnedi3_resample(input=src, target_width=1920, target_height=1080, mode="nnedi3cl")
src = core.resize.Spline36(clip=src, width=1920, height=1080, format=vs.YUV420P16)

eg = core.lsmas.LWLibavSource('/Volumes/untitled/S01E08_vcb.mkv')

denoised = denoise(src)

aaed = aa(denoised)

nrmask = ksksfunc.DBMask(denoised)

debanded = deband(denoised)

merged1080P = core.std.MaskedMerge(debanded,aaed,nrmask,first_plane=True) # type: ignore

# dehalo = haf.FineDehalo(src=merged1080P, contra=1, edgeproc=True)

# dehalo = rksfunc.USMDering(merged1080P, mrad=1, mthr=45)
dehalo = rksfunc.USMDering(cyuv16=merged1080P)
dehalo = haf.FineDehalo(src=dehalo, contra=1, edgeproc=1, darkstr=0.5)
# dehalo = haf.EdgeCleaner(c=dehalo, strength=10)

fin = adptvgrnMod.adptvgrnMod(clip_in=dehalo, strength=[1,0], size=1.5)
fin = rksfunc.KrigBilateral(c420p16=fin).resize.Spline36(format=vs.YUV420P16)
# fin = rksfunc.KrigBilateral(c420p16=fin)

outputs = [fin, dehalo, merged1080P, eg, src, denoised, aaed]

output(outputs)
# import subprocess; subprocess.run('rm -r /private/var/folders/gb/m6gky8t10yx68drv06ncbj_h0000gn/T/onnxruntime*', shell=True, check=True)
