
import vapoursynth as vs
from vapoursynth import core
import mvsfunc as mvf
import havsfunc as haf
import vsutil
import vsTAAmbk as taa
import vsmlrt
from nnedi3_resample import nnedi3_resample
import rksfunc, ksksfunc
import kagefunc as kgf
import vsdpir



class cfg:

    width = 0
    height = 0


configure = cfg()

def load_source(filepath):
    clip = core.lsmas.LWLibavSource(filepath)
    if configure.width == 0 or configure.height == 0:
        configure.width = clip.width
        configure.height = clip.height
    # clip = core.fmtc.bitdepth(clip=clip, bits=16)
    clip = core.resize.Bicubic(clip=clip, width=configure.width, height=configure.height, format=vs.YUV420P16, matrix_s="709", matrix_in_s="709")
    return clip


def denoiseY(clip):
    clipY = vsutil.get_y(clip)
    denoisedY = vsmlrt.DPIR(clip=clipY.resize.Bicubic(format=vs.GRAYS), strength=1, model=vsmlrt.DPIRModel.drunet_gray, tiles=2, backend=vsmlrt.Backend.ORT_COREML(num_streams=4)).resize.Bicubic(format=vs.GRAY16)
    # denoisedY = mvf.LimitFilter(clipY, denoisedY, thr=13, elast=3)
    # denoisedY = vsdpir.dpir(clip=clipY, task="denoise", strength=1.5).resize.Bicubic(format=vs.GRAY16)
    return denoisedY

def denoiseUV(clip):
    return core.nlm_ispc.NLMeans(clip=clip, d=0, h=2, channels="UV", wmode=3)

def deband(clip, ref_clip):
    debanded = rksfunc.SynDeband(cyuv16=clip, limit=True, limit_elast=1.75)
    return debanded

def aa(clip):
    # clipY = core.std.ShufflePlanes(clip, 0, vs.GRAY)
    aa_clip = taa.TAAmbk(clip, aatype=1, aatypeu=0, aatypev=0, preaa=0, strength=0.22, cycle=0, mtype=3, mclip=None,
           mthr=None, mthr2=None, mlthresh=None, mpand=(1, 0), txtmask=1, txtfade=0, thin=0, dark=0.25, sharp=-1,
           aarepair=0, postaa=None, src=None, stabilize=3, down8=False, showmask=0, opencl=True)
    # aa_clipY = core.std.ShufflePlanes(aa_clip, 0, vs.GRAY)
    return aa_clip

def merge_Y_UV(aaedY, debanded):
    dbedY = core.std.ShufflePlanes(debanded, 0, vs.GRAY)
    mergedY = mvf.LimitFilter(dbedY, aaedY, thr=1.0, elast=1.5)
    merged = core.std.ShufflePlanes([mergedY, debanded], [0, 1, 2], vs.YUV)
    return core.resize.Bicubic(clip=merged, width=1920, height=1080, format=vs.YUV420P16, matrix_s="709")

def Sharp(src_nos,strength=1,sigma=1,t_h=12,t_l=2.5,mode=1,op=1):
    edge=core.tcanny.TCanny(vsutil.get_y(src_nos),sigma=sigma,t_h=t_h,t_l=t_l,mode=mode,op=op).std.Maximum().std.Minimum().std.Maximum()
    edge=haf.MinBlur(edge,1)
    mmd_s=core.std.MergeDiff(src_nos,core.std.MakeDiff(src_nos,haf.MinBlur(src_nos,strength)))
    sharp=core.std.MaskedMerge(mmd_s,src_nos,edge)
    return sharp

def show_diff(clipA, clipB):
    clipA = core.resize.Bicubic(clip=clipA, width=configure.width, height=configure.height, format=vs.YUV420P16, matrix_s="709", matrix_in_s="709")
    clipB = core.resize.Bicubic(clip=clipB, width=configure.width, height=configure.height, format=vs.YUV420P16, matrix_s="709", matrix_in_s="709")
    return core.std.MakeDiff(clipA, clipB) 
    
def cleanup():
    print("Cleaning onnx coreml cache...")
    import subprocess; subprocess.run('rm -r /private/var/folders/gb/m6gky8t10yx68drv06ncbj_h0000gn/T/onnxruntime*', shell=True, check=True)

def output(output_list):
    for index, clip in enumerate(output_list):
        clip.set_output(index)

src1080P = load_source('/Users/a1/00002.m2ts')

srcY = vsutil.get_y(src1080P)
srcY = core.resize.Bicubic(clip=srcY, width=1920, height=1080, format=vs.GRAYS)

src_descaledY = kgf.inverse_scale(source=srcY, width=1500, height=844, kernel="lanczos", taps=3, mask_detail=True, use_gpu=False).resize.Point(format=vs.GRAY16)

denoisedY = denoiseY(src_descaledY)
# denoisedY = core.bm3dcpu.BM3Dv2(clip=src_descaledY, radius=1)
# denoisedY = core.bm3dcpu.BM3Dv2(clip=src_descaledY,ref=denoisedY, radius=1).resize.Point(format=vs.GRAY16)

denoised1080PY = nnedi3_resample(input=denoisedY, target_width=1920, target_height=1080, mode="nnedi3cl")

denoised1080P = core.std.ShufflePlanes([denoised1080PY, src1080P], [0, 1, 2], vs.YUV)

denoised_merged = denoiseUV(denoised1080P)

dbed1080P = deband(denoised_merged, denoised_merged)

aaedY1080P = aa(denoised_merged)

nrmask = ksksfunc.DBMask(denoised_merged)

merged1080P = core.std.MaskedMerge(dbed1080P,aaedY1080P,nrmask,first_plane=True) # type: ignore

dehalo=haf.FineDehalo(merged1080P,thlimi=80,thlima=160,brightstr=0.5,darkstr=0)
dering=haf.HQDeringmod(dehalo,mrad=1,msmooth=1,mthr=58,minp=1,sharp=1,thr=2.8,elast=2.0)
dering=haf.EdgeCleaner(dering,strength=10,rmode=20)
dark = haf.FastLineDarkenMOD(c=dering, strength=20)
sharp = core.warp.AWarpSharp2(dark,thresh=128,blur=3,type=1,depth=[2, 1, 1],chroma=1,cplace="mpeg2", opt=True)

ret = core.resize.Bicubic(clip=sharp, width=1920, height=1080, format=vs.YUV420P10, matrix_s="709", matrix_in_s="709")

eg = core.lsmas.LWLibavSource('/Volumes/untitled/S01E01_magic.mkv')

diff = show_diff(ret, src1080P)

outputs = [ret, dering, merged1080P, denoised_merged, src_descaledY, srcY, src1080P, diff, eg]

output(outputs)

cleanup()
