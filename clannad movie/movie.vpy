import random
import vapoursynth as vs
from vapoursynth import core
import mvsfunc as mvf
import havsfunc as haf
import vsTAAmbk as taa
import vsutil
import y5gfunc, kagefunc, muvsfunc, rksfunc, nnedi3_resample, CSMOD, vsmlrt, kagefunc, lvsfunc, functools, nnedi3_rpow2, zvs

from typing import List


def not_descale_oped(n: int, f: List[vs.VideoFrame], not_descale_list: List[List[int]]) -> vs.VideoFrame:
    def _is_in_range(ranges: List[List[int]], num) -> bool:
        for start, end in ranges:
            if start <= num <= end:
                return True
        return False
    
    fout = f[0].copy()
    
    if _is_in_range(not_descale_list, n):
        fout = f[1].copy()
        fout.props['Descaled'] = 0
    else:
        fout = f[0].copy()
        try:
            fout.props['Descaled'] = fout.props['Descaled']
        except:
            pass
    
    return fout


def not_descale_main(n: int, f: List[vs.VideoFrame], oped: List[List[int]], threshold: float = 0) -> vs.VideoFrame:
    def _is_not_in_range(ranges: List[List[int]], num) -> bool:
        for start, end in ranges:
            if start <= num <= end:
                return False
        return True
    
    fout = f[0].copy()
    
    if _is_not_in_range(oped, n):
        if f[2].props['PlaneStatsAverage'] > threshold: # type: ignore
            # print(f"frame {n}: Not descaling because cmask average {f[2].props['PlaneStatsAverage']} > threshold {threshold}!")
            fout = f[1].copy()
            fout.props['Descaled'] = 0
        else:
            fout = f[0].copy()
            try:
                fout.props['Descaled'] = fout.props['Descaled']
            except:
                pass
    else:
        fout = f[0].copy()
        try:
            fout.props['Descaled'] = fout.props['Descaled']
        except:
            pass
    
    return fout

def aa(clip):
    aa_clip = taa.TAAmbk(clip, aatype=3, aatypeu=0, aatypev=0, preaa=0, strength=0, cycle=0, mtype=3, mclip=None,
            mthr=None, mthr2=None, mlthresh=None, mpand=(1, 0), txtmask=1, txtfade=0, thin=0, dark=0, sharp=-1,
            aarepair=2, postaa=True, src=None, stabilize=1, down8=False, showmask=0, opencl=True)
    line_mask = taa.TAAmbk(clip, showmask=1, mtype=3)
    txt_mask = taa.TAAmbk(clip, showmask=-1, txtmask=1)
    aa_clip = rksfunc.mergeuv(clipy=aa_clip, clipuv=clip)
    return aa_clip, line_mask, txt_mask
    
# bd source
clip = core.lsmas.LWLibavSource('/Volumes/untitled/00012.m2ts')

# webrip reference
xyx98 = core.lsmas.LWLibavSource('/Users/a1/Downloads/剧场版 clannad.mkv').std.Trim(20)
xyx98 = nnedi3_resample.nnedi3_resample(xyx98, 1920, 1080, nns=4, qual=2, nsize=4, mode="nnedi3cl")
xyx98 = core.resize.Spline36(xyx98, format=vs.YUV420P16, matrix_s="709", matrix_in_s="709")

# dvdrip reference
fch = core.lsmas.LWLibavSource('/Users/a1/Downloads/劇場版 CLANNAD [DVD 720x480 23.976fps AVC-yuv420p10 FLAC] - fch1993.mkv')
fch = core.std.BlankClip(fch).std.Trim(first=0, last=192) + fch
fch = nnedi3_resample.nnedi3_resample(fch, 1920, 1080, nns=4, qual=2, nsize=4, mode="nnedi3cl")
fch = core.resize.Spline36(fch, format=vs.YUV420P16, matrix_s="709", matrix_in_s="709")

# encoded
enc = core.lsmas.LWLibavSource('/Users/a1/clannad_movie_10bit.mkv')
enc = core.resize.Spline36(enc, format=vs.YUV420P16, matrix_s="709", matrix_in_s="709")

# edge fix
edge_fix = core.edgefixer.Continuity(clip, left=0, top=2, right=2, bottom=0)
edge_fix = core.resize.Spline36(edge_fix, format=vs.YUV444PS)

# rescale
sharp = core.cas.CAS(edge_fix, 0.5)

oped_not_descale_list = [[2390, 2454], [2567, 2634], [2850, 2917], [2980, 3048], [3049, 3556], [3762, 3833], [3890, 3959], [4100, 4194], [4347, 4489], [4632, 4869], [129371, 135131]]
oped_list = [[2389, 4870], [129371, 135131]]

rescaled_raw, cmask, osd  = y5gfunc.rescale(clip=edge_fix, descale_kernel="Debilinear", src_height=720, bw=1920, bh=1080, show_detail_mask=True, exclude_common_mask=True) # type: ignore

## dering
rescaled_raw = mvf.Depth(rescaled_raw, 16)
cleaned = haf.FineDehalo(rescaled_raw, edgeproc=1, contra=1)
mask = muvsfunc.AnimeMask(input=rescaled_raw, shift=0.75, mode=1).std.Binarize(threshold=1)
cleaned = core.akarin.Expr([cleaned, rescaled_raw], ["x y min", "x", ""])
cleaned = rksfunc.mergeuv(cleaned, rescaled_raw)
rescaled_dering = core.std.MaskedMerge(clipa=rescaled_raw, clipb=cleaned, mask=mask, first_plane=True)
rescaled_dering = mvf.Depth(rescaled_dering, 32)

## selective rescale
rescaled_dering = core.std.ModifyFrame(rescaled_dering, [rescaled_dering, sharp], functools.partial(not_descale_oped, not_descale_list=oped_not_descale_list))
cmask = core.std.ModifyFrame(cmask, [cmask, core.std.BlankClip(cmask)], functools.partial(not_descale_oped, not_descale_list=oped_not_descale_list))
osd = core.std.ModifyFrame(osd, [osd, osd], functools.partial(not_descale_oped, not_descale_list=oped_not_descale_list))

planestats = core.std.PlaneStats(cmask)
planestats = core.akarin.Text(planestats, "\n{PlaneStatsAverage}")

rescaled_dering = core.std.ModifyFrame(rescaled_dering, [rescaled_dering, sharp, planestats], functools.partial(not_descale_main, oped=oped_list))
cmask = core.std.ModifyFrame(cmask, [cmask, core.std.BlankClip(cmask), planestats], functools.partial(not_descale_main, oped=oped_list))
osd = core.std.ModifyFrame(osd, [osd, osd, planestats], functools.partial(not_descale_main, oped=oped_list))

osd = core.std.CopyFrameProps(osd, rescaled_dering, "PlaneStatsAverage")
osd = core.akarin.Text(clips=osd, text="\n\n\nDescaled: {Descaled}")

#####

rescaled_dering = core.resize.Spline36(rescaled_dering, format=vs.YUV420P16)

aaed, line_mask, txt_mask = aa(rescaled_dering)

nrmask = y5gfunc.DBMask(rescaled_dering)
debanded = y5gfunc.SynDeband(clip=rescaled_dering, limit=True)
merged = core.std.MaskedMerge(debanded, aaed, nrmask, first_plane=True) # type: ignore

#####

fin = clip.std.Trim(first=0, last=202).resize.Spline36(format=vs.YUV420P16) + merged.std.Trim(first=203)
fin = mvf.Depth(fin, 10, useZ=True)

outputs = [clip, enc, fch, rescaled_raw, rescaled_dering, cmask, osd, planestats, debanded, (fin,0), line_mask, txt_mask]
# txt_mask.set_output(9)


rand = random.sample(range(clip.num_frames), k=50)
y5gfunc.screen_shot(clip=clip, frames=rand, path='/Users/a1/output/', file_name="src_%d", overwrite=True)
y5gfunc.screen_shot(clip=merged, frames=rand, path='/Users/a1/output/', file_name="filtered_%d", overwrite=True)
y5gfunc.screen_shot(clip=enc, frames=rand, path='/Users/a1/output/', file_name="encoded_%d", overwrite=True)

y5gfunc.output(outputs)
y5gfunc.reset_output_index()