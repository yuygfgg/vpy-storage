import vapoursynth as vs
from vapoursynth import core
import mvsfunc as mvf
import havsfunc as haf
import vsTAAmbk as taa
import rksfunc, vsutil, y5gfunc, muvsfunc, functools, CSMOD, fvsfunc, random

file_path = "/Volumes/untitled/00007.m2ts"

class cfg:

    width = 0
    height = 0

configure = cfg()

def load_source(filepath):
    clip = core.bs.VideoSource(filepath)
    if configure.width == 0 or configure.height == 0:
        configure.width = clip.width
        configure.height = clip.height
    clip = core.resize.Spline36(clip=clip, width=configure.width, height=configure.height, format=vs.YUV420P16, matrix_s="709", matrix_in_s="709")
    return clip

def deband(clip):
    debanded = y5gfunc.SynDeband(clip=clip, limit=False)
    return debanded

nnedi3_args: dict = {'field': 1, 'nsize': 4, 'nns': 4, 'qual': 2}
nnedi3 = functools.partial(core.nnedi3cl.NNEDI3CL, **nnedi3_args)

src = load_source(filepath=file_path)

aaed2 = nnedi3(nnedi3(src, dh=True), dw=True)
aaed2Y = vsutil.get_y(aaed2)
aaed2Y = muvsfunc.SSIM_downsample(aaed2, 1920, 1080, src_left=-0.5, src_top=-0.5).fmtc.bitdepth(bits=16)
aaed2 = rksfunc.mergeuv(aaed2Y, src)
aa_mask: vs.VideoNode = taa.TAAmbk(src, aatype=1, aatypeu=0, aatypev=0, preaa=0, strength=0, cycle=2, mtype=3, mclip=None,
            mthr=None, mthr2=None, mlthresh=None, mpand=(1, 0), txtmask=0, txtfade=0, thin=0, dark=0.1, sharp=0,
            aarepair=2, postaa=False, src=None, stabilize=0, down8=False, showmask=1, opencl=True)
aaed2 = core.std.MaskedMerge(src, aaed2, aa_mask, first_plane=True)
aaed2 = taa.temporal_stabilize(aaed2, src)

nrmask = y5gfunc.DBMask(src)
debanded = y5gfunc.SynDeband(clip=src, limit=True)
merged = core.std.MaskedMerge(debanded, aaed2, nrmask, first_plane=True) # type: ignore

sharp = CSMOD.CSMOD(merged, preset="detail")
ringing_mask = muvsfunc.AnimeMask(sharp, shift=0.75, mode=1)
sharp = core.std.MaskedMerge(sharp, merged, ringing_mask, first_plane=True)

dering_mask = muvsfunc.AnimeMask(sharp, shift=0.75, mode=1).std.Binarize(1)
dering = haf.FineDehalo(sharp, edgeproc=1, contra=1)
# dering = rksfunc.USMDering(sharp)
dering = core.akarin.Expr([dering, sharp], "x y min")
dering = rksfunc.mergeuv(dering, sharp)
dering = core.std.MaskedMerge(clipa=sharp, clipb=dering, mask=dering_mask, first_plane=True)


fin = fvsfunc.rfs(dering, src, "[0, 1451] [3647, 3755] [3911, 4031] [5705, 5790] [7352, 7772]")

# encoded = core.lsmas.LWLibavSource('/Volumes/untitled/ll.mkv')

outputs = [src, aaed2Y, aa_mask, merged, sharp, ringing_mask, dering, (fin,0)]

# rand = random.sample(range(fin.num_frames), k=50)
# y5gfunc.screen_shot(clip=fin, frames=rand, path='/Users/a1/output/', file_name="filtered_%d", overwrite=True)
# y5gfunc.screen_shot(clip=src, frames=rand, path='/Users/a1/output/', file_name="src_%d", overwrite=True)


y5gfunc.reset_output_index()
y5gfunc.output(outputs, debug=False)
