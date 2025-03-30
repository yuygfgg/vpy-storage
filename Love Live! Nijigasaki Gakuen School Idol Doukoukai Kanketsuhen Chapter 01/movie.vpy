import random
from vstools import vs, core
import vstools
import mvsfunc as mvf
import havsfunc as haf
import vsTAAmbk as taa
import rksfunc, vsutil, muvsfunc, functools, CSMOD, fvsfunc
from vsdehalo import fine_dehalo
import vsaa
from y5gfunc import *
import subprocess

file_path = "/Volumes/smb/[BDMV][250129][Love Live! Nijigasaki Gakuen School Idol Doukoukai Kanketsuhen][Vol.01]/DISC 1/BDMV/STREAM/00007.m2ts"

nnedi3_args: dict = {'field': 1, 'nsize': 4, 'nns': 4, 'qual': 2}
nnedi3 = functools.partial(core.nnedi3cl.NNEDI3CL, **nnedi3_args)

src = load_source(file_path=file_path)
src = vstools.depth(src, 16)

denoised = Fast_BM3DWrapper(src, sigma_Y=1.2, sigma_chroma=2, preset_Y_basic="lc", preset_Y_final="lc", preset_chroma_basic="lc", preset_chroma_final="lc")
# denoised = haf.SMDegrain(src)

aaed2 = nnedi3(nnedi3(denoised, dh=True), dw=True)
aaed2Y = vstools.get_y(aaed2)
aaed2Y = vstools.depth(muvsfunc.SSIM_downsample(aaed2, 1920, 1080, src_left=-0.5, src_top=-0.5), 16)
aaed2 = rksfunc.mergeuv(aaed2Y, denoised)
aa_mask: vs.VideoNode = taa.TAAmbk(denoised, mtype=3, showmask=1)
aaed2 = core.std.MaskedMerge(denoised, aaed2, aa_mask, first_plane=True)
aaed2 = taa.temporal_stabilize(aaed2, denoised)

debanded, nrmask = SynDeband(clip=denoised, limit=True, include_mask=True)
merged = core.std.MaskedMerge(debanded, aaed2, nrmask, first_plane=True) # type: ignore

sharp = CSMOD.CSMOD(merged, preset="detail")
ringing_mask = muvsfunc.AnimeMask(sharp, shift=0.75, mode=1)
sharp = core.std.MaskedMerge(sharp, merged, ringing_mask, first_plane=True)

dering_mask = ringing_mask.std.Binarize(1)
dering = haf.FineDehalo(sharp, contra=1)
dering = core.akarin.Expr([dering, sharp], ["x y min", "y", "y"])
dering = core.std.MaskedMerge(clipa=sharp, clipb=dering, mask=dering_mask, first_plane=True)

fin = fvsfunc.rfs(dering, denoised, "[0, 1451] [3647, 3755] [3911, 4031] [5705, 5790] [7352, 7772]")

# encoded = core.lsmas.LWLibavSource('/Volumes/untitled/ll.mkv')

outputs = [(fin,0), src, merged, sharp, ringing_mask, dering]

if __name__ != "__main__":
    output(outputs, debug=False)
    reset_output_index()
else:
    # print('a')
    # rand = random.sample(range(fin.num_frames), k=50)
    # screen_shot(clip=src, frames=rand, path='/Users/a1/output/', file_name="src_%d", overwrite=True)
    # encoder = subprocess.Popen(["/usr/local/bin/x265", "--y4m", "--input", "-", "--rc-lookahead", "130", "--crf", "14.5", "--preset", "slowxx", "--tune", "vq3", "--no-amp", "--hist-scenecut", "--no-rect", "--no-sao", "--deblock", "-1:-1", "-D", "10", "--transfer", "bt709", "--colorprim", "bt709", "--colormatrix", "bt709", "--range", "limited", '/Volumes/untitled/llrel.mkv'], stdin=subprocess.PIPE)
    audios_lossless = extract_audio_tracks(file_path, "/Users/a1/", config=AudioConfig(main_lossless_2ch=TrackConfig(mode=ProcessMode.COMPRESS, format='flac'), main_lossless_multi=TrackConfig(mode=ProcessMode.COMPRESS, format='flac')))
    chapter = get_bd_chapter(file_path, '/Users/a1/ll.txt')
    
    encode_video(fin, encoder)

    mux_mkv('/Volumes/untitled/ll_re_no_sub_plz.mkv', videos={'path': '/Volumes/untitled/llrel.mkv', 'language': 'jpn'}, audios=audios_lossless, chapters=chapter)
