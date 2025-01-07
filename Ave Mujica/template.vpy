import vapoursynth as vs
from vapoursynth import core
import mvsfunc as mvf
import havsfunc as haf
import vsTAAmbk as taa
import rksfunc, vsmlrt, vsutil, random, y5gfunc, muvsfunc, CSMOD
class cfg:

    width = 0
    height = 0

def aa(clip):
    clipY = vsutil.get_y(clip)
    aa_clipY = taa.TAAmbk(clipY, aatype=2, aatypeu=0, aatypev=0, preaa=0, strength=0, cycle=0, mtype=5, mclip=None,
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

def source_choose(cr: vs.VideoNode, amzn: vs.VideoNode, debug=False) -> vs.VideoNode:
    # am_dn = y5gfunc.Fast_BM3DWrapper(clip=amzn, sigma_Y=7, sigma_chroma=0)
    am_dn = rksfunc.BM3DWrapper(c420p16=amzn, sy=7, sc=0)
    am_diff = core.std.PlaneStats(clipa=amzn, clipb=am_dn)
    
    if debug: 
        cr = core.akarin.Text(cr, "\ncr")
        amzn = core.akarin.Text(amzn, "\namzn")
    
    ret = core.akarin.Select([cr, amzn], [am_diff], "src0.PlaneStatsDiff 0.0025 >")

    if debug: 
        ret = core.std.CopyFrameProps(clip=ret, prop_src=am_diff, props='PlaneStatsDiff')
        ret = core.akarin.Text(ret, "\n\nPlaneStatsDiff_amzn: {PlaneStatsDiff}")
    
    return ret

cr = load_source(filepath='/Users/a1/Downloads/mujica01_cr.mkv')
amzn = load_source(filepath='/Users/a1/Downloads/mujica01_amzn.mkv')

src = source_choose(cr, amzn, debug=False)

src = mvf.Depth(src, 32)

rescaled, cmask, osd = y5gfunc.rescale(clip=src, src_height=806, bw=1440, bh=810, descale_kernel="Debicubic", b=0, c=0.33, show_detail_mask=True)  # type: ignore

# denoised = y5gfunc.Fast_BM3DWrapper(clip=rescaled.resize.Spline36(format=vs.YUV420P16), sigma_Y=3.5) 
denoised = rksfunc.BM3DWrapper(c420p16=rescaled.resize.Spline36(format=vs.YUV420P16), sy=3.5)

aaed = aa(denoised)

nrmask = y5gfunc.DBMask(denoised)

debanded = y5gfunc.SynDeband(clip=denoised, limit=False)

merged1080P = core.std.MaskedMerge(debanded, aaed, nrmask, first_plane=True) # type: ignore

cleaned = haf.EdgeCleaner(merged1080P, rmode=17, strength=10)
mask = muvsfunc.AnimeMask(input=denoised, shift=0.7, mode=1).std.Binarize(threshold=1)
mask = core.akarin.Expr([mask, vsutil.get_y(cmask).resize.Point(format=vs.GRAY16)], "x y -").std.Binarize(threshold=10000)
cleaned = core.std.MaskedMerge(clipa=denoised, clipb=cleaned, mask=mask, first_plane=True)
cleaned = rksfunc.mergeuv(cleaned, denoised)

sharp = CSMOD.CSMOD(cleaned, preset="detail")
sharp = core.resize.Spline36(clip=sharp, format=vs.YUV420P10)

outputs = [amzn, cr, src, rescaled, (sharp,0), osd, cmask, mask]

y5gfunc.reset_output_index()
y5gfunc.output(outputs, debug=False)

# rand = random.sample(range(sharp.num_frames), k=10)
# y5gfunc.screen_shot(clip=srcorg, frames=rand, path='/Users/a1/output/', file_name="src_%d", overwrite=True)
# y5gfunc.screen_shot(clip=sharp, frames=y5gfunc.ranger(0,sharp.num_frames,720), path='/Users/a1/tofns/', file_name="y5gfiltered_%d", overwrite=True)
