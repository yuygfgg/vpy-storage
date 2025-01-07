import vapoursynth as vs
from vapoursynth import core
import mvsfunc as mvf
import havsfunc as haf
import vsTAAmbk as taa
import rksfunc, vsmlrt, vsutil, y5gfunc
from typing import Tuple

file_path = "/Volumes/untitled/gimai/raw_video/03.m2ts"

class cfg:

    width = 0
    height = 0

def aa(clip):
    clipY = vsutil.get_y(clip)
    aa_clipY = taa.TAAmbk(clipY, aatype=1, aatypeu=0, aatypev=0, preaa=0, strength=0, cycle=0, mtype=5, mclip=None,
            mthr=None, mthr2=None, mlthresh=None, mpand=(1, 0), txtmask=1, txtfade=0, thin=0, dark=0, sharp=-1,
            aarepair=2, postaa=True, src=None, stabilize=3, down8=False, showmask=0, opencl=True)
    aa_clip = rksfunc.mergeuv(clipy=aa_clipY, clipuv=clip)
    return aa_clip

configure = cfg()


def load_source(filepath):
    clip = core.lsmas.LWLibavSource(filepath)
    if configure.width == 0 or configure.height == 0:
        configure.width = clip.width
        configure.height = clip.height
    clip = core.resize.Spline36(clip=clip, width=configure.width, height=configure.height, format=vs.YUV420P16, matrix_s="709", matrix_in_s="709")
    return clip

def denoise(clip):
    # denoised = rksfunc.BM3DWrapper(c420p16=clip, sy=1, dsy=1)
    denoised = y5gfunc.Fast_BM3DWrapper(clip=clip, sigma_Y=1, delta_sigma_Y=1)
    return denoised

def deband(clip):
    debanded = y5gfunc.SynDeband(clip=clip, limit=False)
    return debanded

def selective_denoise(clip: vs.VideoNode, debug: bool = False) -> vs.VideoNode:
    cut = core.std.StackHorizontal([clip.std.Crop(right=clip.width-240), clip.std.Crop(left=clip.width-240)])
    
    cut = cut.std.PlaneStats()
    
    light_denoise = rksfunc.BM3DWrapper(c420p16=clip, sy=1, dsy=1)
    heavy_denoise = rksfunc.BM3DWrapper(c420p16=clip, sy=5, dsy=2)
    
    if debug:
        light_denoise = core.akarin.Text(light_denoise, "\n\nlight")
        heavy_denoise = core.akarin.Text(heavy_denoise, "\n\nheavy")
    
    ret = core.akarin.Select(clip_src=[light_denoise, heavy_denoise], prop_src=cut, expr="x.PlaneStatsAverage 0.12 < 1 0 ?")
    return ret

srcorg = load_source(filepath=file_path)

srcorg = mvf.Depth(srcorg, 32)

rescaled, cmask, osd = y5gfunc.rescale(clip=srcorg, src_height=915.25, bw=[1920], bh=[1080], descale_kernel=["Delanczos"], taps=[6], show_detail_mask=True, show_fft=False, threshold_min=1)  # type: ignore

denoised= selective_denoise(rescaled.resize.Spline36(format=vs.YUV420P16))

aaed = aa(denoised)

nrmask = y5gfunc.DBMask(denoised)

debanded = deband(denoised)

merged1080P = core.std.MaskedMerge(debanded,aaed,nrmask,first_plane=True) # type: ignore

uvsr = core.placebo.Shader(clip=merged1080P, shader_s=y5gfunc.cfl_shader).resize.Spline36(format=vs.YUV420P10)

fin = uvsr

outputs = [srcorg, rescaled, (fin,0), osd, cmask]


y5gfunc.output(outputs, debug=True)
y5gfunc.reset_output_index()