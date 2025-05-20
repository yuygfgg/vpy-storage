import vapoursynth as vs
from vapoursynth import core
import havsfunc as haf
import vsTAAmbk as taa
import muvsfunc, vstools, y5gfunc, vsdehalo, subprocess, os, vsrgtools

episode_num = "04"

if "posix" in os.name:
    SRC = {
        "01": "/Volumes/smb/[BDMV] 16bitセンセーション ANOTHER LAYER [JP]/Vol.1/BD_VIDEO/BDMV/STREAM/00003.m2ts",
        "02": "/Volumes/smb/[BDMV] 16bitセンセーション ANOTHER LAYER [JP]/Vol.1/BD_VIDEO/BDMV/STREAM/00004.m2ts",
        "03": "/Volumes/smb/[BDMV] 16bitセンセーション ANOTHER LAYER [JP]/Vol.1/BD_VIDEO/BDMV/STREAM/00005.m2ts",
        "04": "/Volumes/smb/[BDMV] 16bitセンセーション ANOTHER LAYER [JP]/Vol.2/BD_VIDEO/BDMV/STREAM/00003.m2ts",
        "05": "/Volumes/smb/[BDMV] 16bitセンセーション ANOTHER LAYER [JP]/Vol.2/BD_VIDEO/BDMV/STREAM/00004.m2ts",
        "06": "/Volumes/smb/[BDMV] 16bitセンセーション ANOTHER LAYER [JP]/Vol.3/BD_VIDEO/BDMV/STREAM/00003.m2ts",
        "07": "/Volumes/smb/[BDMV] 16bitセンセーション ANOTHER LAYER [JP]/Vol.3/BD_VIDEO/BDMV/STREAM/00004.m2ts",
        "08": "/Volumes/smb/[BDMV] 16bitセンセーション ANOTHER LAYER [JP]/Vol.4/BD_VIDEO/BDMV/STREAM/00003.m2ts",
        "09": "/Volumes/smb/[BDMV] 16bitセンセーション ANOTHER LAYER [JP]/Vol.4/BD_VIDEO/BDMV/STREAM/00004.m2ts",
        "10": "/Volumes/smb/[BDMV] 16bitセンセーション ANOTHER LAYER [JP]/Vol.5/BD_VIDEO/BDMV/STREAM/00003.m2ts",
        "11": "/Volumes/smb/[BDMV] 16bitセンセーション ANOTHER LAYER [JP]/Vol.5/BD_VIDEO/BDMV/STREAM/00004.m2ts",
        "12": "/Volumes/smb/[BDMV] 16bitセンセーション ANOTHER LAYER [JP]/Vol.6/BD_VIDEO/BDMV/STREAM/00003.m2ts",
        "13": "/Volumes/smb/[BDMV] 16bitセンセーション ANOTHER LAYER [JP]/Vol.6/BD_VIDEO/BDMV/STREAM/00004.m2ts",
    }
else:
    SRC = {
        "01": R"C:\Users\CZ\Downloads\[BDMV] 16bitセンセーション ANOTHER LAYER [JP]\Vol.1\BD_VIDEO\BDMV\STREAM\00003.m2ts",
        "02": R"C:\Users\CZ\Downloads\[BDMV] 16bitセンセーション ANOTHER LAYER [JP]\Vol.1\BD_VIDEO\BDMV\STREAM\00004.m2ts",
        "03": R"C:\Users\CZ\Downloads\[BDMV] 16bitセンセーション ANOTHER LAYER [JP]\Vol.1\BD_VIDEO\BDMV\STREAM\00005.m2ts",
        "04": R"C:\Users\CZ\Downloads\[BDMV] 16bitセンセーション ANOTHER LAYER [JP]\Vol.2\BD_VIDEO\BDMV\STREAM\00003.m2ts",
        "05": R"C:\Users\CZ\Downloads\[BDMV] 16bitセンセーション ANOTHER LAYER [JP]\Vol.2\BD_VIDEO\BDMV\STREAM\00004.m2ts",
        "06": R"C:\Users\CZ\Downloads\[BDMV] 16bitセンセーション ANOTHER LAYER [JP]\Vol.3\BD_VIDEO\BDMV\STREAM\00003.m2ts",
        "07": R"C:\Users\CZ\Downloads\[BDMV] 16bitセンセーション ANOTHER LAYER [JP]\Vol.3\BD_VIDEO\BDMV\STREAM\00004.m2ts",
        "08": R"C:\Users\CZ\Downloads\[BDMV] 16bitセンセーション ANOTHER LAYER [JP]\Vol.4\BD_VIDEO\BDMV\STREAM\00003.m2ts",
        "09": R"C:\Users\CZ\Downloads\[BDMV] 16bitセンセーション ANOTHER LAYER [JP]\Vol.4\BD_VIDEO\BDMV\STREAM\00004.m2ts",
        "10": R"C:\Users\CZ\Downloads\[BDMV] 16bitセンセーション ANOTHER LAYER [JP]\Vol.5\BD_VIDEO\BDMV\STREAM\00003.m2ts",
        "11": R"C:\Users\CZ\Downloads\[BDMV] 16bitセンセーション ANOTHER LAYER [JP]\Vol.5\BD_VIDEO\BDMV\STREAM\00004.m2ts",
        "12": R"C:\Users\CZ\Downloads\[BDMV] 16bitセンセーション ANOTHER LAYER [JP]\Vol.6\BD_VIDEO\BDMV\STREAM\00003.m2ts",
        "13": R"C:\Users\CZ\Downloads\[BDMV] 16bitセンセーション ANOTHER LAYER [JP]\Vol.6\BD_VIDEO\BDMV\STREAM\00004.m2ts",
    }


file_path = SRC[episode_num]

class cfg:
    width = 0
    height = 0

configure = cfg()

def aa(clip) -> vs.VideoNode:
    Y = vstools.get_y(clip)
    aaY = taa.TAAmbk(Y, aatype=1, aatypeu=0, aatypev=0, preaa=0, strength=0, cycle=0, mtype=3, mclip=None,
                mthr=None, mthr2=None, mlthresh=None, mpand=(1, 0), txtmask=1, txtfade=0, thin=0, dark=0, sharp=-1,
                aarepair=2, postaa=None, src=None, stabilize=3, down8=False, showmask=0, opencl=True, cuda=True)
    ret = vstools.join(aaY, clip)
    return ret

def stablize(clip) -> vs.VideoNode:
    superclip = core.mv.Super(clip, pel=2, sharp=1)
    bv = core.mv.Analyse(superclip, isb=True, delta=1, overlap=2, blksize=4, search=5, searchparam=4)
    fv = core.mv.Analyse(superclip, isb=False, delta=1, overlap=2, blksize=4, search=5, searchparam=4)
    compensated_forward = core.mv.Flow(clip, superclip, fv)
    compensated_backward = core.mv.Flow(clip, superclip, bv)
    temporal_smooth = core.std.Merge(compensated_forward, compensated_backward, weight=0.5)
    final_clip = core.std.Merge(clip, temporal_smooth, weight=0.6)
    ret = vsrgtools.contrasharpening(final_clip, clip)
    return ret

def skip(org: vs.VideoNode, flt: vs.VideoNode, debug: bool = False, thr: float = 0.1) -> vs.VideoNode:
    if debug:
        org = org.akarin.Text("\norg")
        flt = flt.akarin.Text("\nflt")
    b = core.std.StackHorizontal([core.std.Crop(vstools.depth(org, 8), left=1680), core.std.Crop(vstools.depth(org, 8), right=1680)]).std.PlaneStats()
    b.set_output(2)
    return core.akarin.Select([org, flt], b, f"x.PlaneStatsAverage {thr} >")

src = y5gfunc.load_source(file_path=file_path)
src.set_output(1)
if 'posix' not in os.name:
    _, cmask, osd = y5gfunc.rescale(clip=src, src_height=720.1, descale_kernel="Delanczos", taps=5, show_detail_mask=True) # type: ignore
    # rescaled.set_output()
    denoised = y5gfunc.Fast_BM3DWrapper(src.resize2.Spline36(format=vs.YUV420P16), core.bm3dcuda_rtc, sigma_Y=1, sigma_chroma=2, preset_Y_basic=y5gfunc.BM3DPreset.MAGIC, preset_chroma_basic=y5gfunc.BM3DPreset.MAGIC, preset_Y_final=y5gfunc.BM3DPreset.MAGIC, preset_chroma_final=y5gfunc.BM3DPreset.MAGIC)
    nrmask = y5gfunc.DBMask(denoised)
    aaedY1080P = aa(denoised)
    aaedY1080P = core.std.MaskedMerge(clipa=aaedY1080P, clipb=denoised.resize2.Spline36(format=vs.YUV420P16), mask=cmask.resize2.Spline36(format=vs.YUV420P16))
    dbed1080P = y5gfunc.SynDeband(aaedY1080P, limit=True, limit_elast=1.75)
    merged1080P = core.std.MaskedMerge(dbed1080P,aaedY1080P,nrmask,first_plane=True) # type: ignore
    merged1080P.set_output(3)
    mask = muvsfunc.AnimeMask(input=merged1080P, shift=0.7, mode=1).std.Binarize(1)
    mask = core.akarin.Expr([mask.fmtc.bitdepth(bits=16), vstools.get_y(cmask).fmtc.bitdepth(bits=16)], "x y -").std.Binarize(threshold=10000)
    mask.set_output(4)
    dehalod = vsdehalo.fine_dehalo(merged1080P)
    dehalod = core.std.MaskedMerge(clipa=merged1080P, clipb=dehalod, mask=mask, first_plane=True)

    stable = stablize(dehalod)
    stable.set_output(6)

    finn = skip(src, dehalod)
    finn.set_output(0)

if "posix" not in os.name:

    encoder = subprocess.Popen(
        [
            R"C:\Users\CZ\Downloads\x265-Yuuki-4.1+97-lol\x265.exe",
            "--y4m",
            "--input",
            "-",
            "--rc-lookahead",
            "130",
            "--crf",
            "15",
            "--hist-scenecut",
            "--preset",
            "slowxx",
            "--tune",
            "vq3",
            "--rect",
            "--deblock",
            "-1:-1",
            "-D",
            "10",
            "--aq-mode",
            "3",
            "--aq-strength",
            "0.7",
            "--qcomp",
            "0.7",
            "--transfer",
            "bt709",
            "--colorprim",
            "bt709",
            "--colormatrix",
            "bt709",
            "--range",
            "limited",
            "-o",
            Rf"C:\Users\CZ\Desktop\E{episode_num}.mkv",             
        ],
        stdin=subprocess.PIPE,
    )
    y5gfunc.encode_video(vstools.depth(finn, 10), encoder)
    audios = y5gfunc.extract_audio_tracks(file_path, Rf"C:\Users\CZ\Desktop\{episode_num}")
    chapter = y5gfunc.get_bd_chapter(file_path, Rf"C:\Users\CZ\Desktop\{episode_num}.txt")
    y5gfunc.mux_mkv(Rf"C:\Users\CZ\Desktop\16bit{episode_num}fin.mkv", videos={'path': Rf"C:\Users\CZ\Desktop\E{episode_num}.mkv", 'language': 'jpn'}, audios=audios, chapters=chapter)

