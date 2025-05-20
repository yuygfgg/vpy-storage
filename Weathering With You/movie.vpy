from vstools import core, vs
import vstools
import vssource
import y5gfunc
import vsTAAmbk as taa
import vsrgtools
import vsdehalo

core.set_affinity(24)

def aa(clip) -> vs.VideoNode:
    Y = vstools.get_y(clip)
    aaY = taa.TAAmbk(Y, aatype=2, aatypeu=0, aatypev=0, preaa=0, strength=0, cycle=0, mtype=3, mclip=None,
                mthr=None, mthr2=None, mlthresh=None, mpand=(1, 0), txtmask=1, txtfade=0, thin=0, dark=0, sharp=0,
                aarepair=2, postaa=None, src=None, stabilize=3, down8=False, showmask=0, opencl=True, cuda=True)
    ret = vstools.join(aaY, clip)
    return ret

src_path = R"/home/v100/Downloads/Weathering With You.Limited Collectors Edition.USA.UHD.BluRay.DV.[2019]/BDMV/STREAM/00011.m2ts"

bl = vssource.FFMS2.source(src_path, track=0, chroma_location=vs.CHROMA_TOP_LEFT).resize2.Spline36(format=vs.YUV420P16)
el = vssource.FFMS2.source(src_path, track=1).resize2.Point(width=bl.width, height=bl.height, format=vs.YUV420P10)
bl = bl.std.CopyFrameProps(el, 'DolbyVisionRPU').placebo.Tonemap(src_csp=3, dst_csp=1).resize2.Spline36(format=vs.YUV420P16, chromaloc_in_s="top_left", chromaloc_s="top_left")
hdr = core.vsnlq.MapNLQ(bl, el).std.SetFrameProps(_Matrix=vs.MATRIX_BT2020_NCL, _Primaries=vs.PRIMARIES_BT2020, _Transfer=vs.TRANSFER_ST2084)
hdr = vstools.depth(hdr, 16)
denoised = y5gfunc.Fast_BM3DWrapper(hdr.resize2.Spline36(format=vs.YUV420P16), core.bm3dcuda_rtc, sigma_Y=2, sigma_chroma=2, preset_Y_basic=y5gfunc.BM3DPreset.LC, preset_chroma_basic=y5gfunc.BM3DPreset.LC, preset_Y_final=y5gfunc.BM3DPreset.LC, preset_chroma_final=y5gfunc.BM3DPreset.LC, fast=True)
nrmask = y5gfunc.DBMask(denoised)
aaed = aa(denoised)
dbed = y5gfunc.SynDeband(aaed, limit=True, limit_elast=1.75)
merged = core.std.MaskedMerge(dbed,aaed,nrmask,first_plane=True) # type: ignore
sharpen = vsrgtools.contrasharpening(merged, hdr)
ringing_mask = y5gfunc.AnimeMask(sharpen, shift=0.75, mode=1)
dering_mask = ringing_mask.std.Binarize(1)
dering = vsdehalo.fine_dehalo(sharpen)
dering = core.akarin.Expr([dering, sharpen], ["x y min", "y", "y"])
dering = core.std.MaskedMerge(clipa=sharpen, clipb=dering, mask=dering_mask, first_plane=True)
dering = vsdehalo.edge_cleaner(dering)
fin = vstools.replace_ranges(dering, hdr, sorted(set(range(0, 931)) | set(range(5174, 5557)) | set(range(151878, 152055))))
# hdr.set_output()
fin = core.resize2.Bicubic(fin, format=vs.YUV420P12)
fin.set_output(0)

y5gfunc.encode_video(fin)
