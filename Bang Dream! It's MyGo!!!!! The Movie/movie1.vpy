from itertools import chain
import vapoursynth as vs
from vapoursynth import core
import havsfunc as haf
import vstools, y5gfunc, muvsfunc, adptvgrnMod, subprocess

src_path = "/Volumes/untitled/[BDMV][251105][Bang Dream! It's MyGo!!!!! The Movie][Vol.01-02]/BRMM_10982 (Part 1)/BDMV/STREAM/00000.m2ts"
output_path = "/Users/a1/mygo_01fin.mkv"

src = y5gfunc.load_source(src_path)
src = vstools.depth(src, 32)

rescaled, cmask, osd = y5gfunc.rescale(src, descale_kernel=["Debicubic"], b=0, c=0.33, src_height=806, bw=[1920], bh=[1080], show_detail_mask=True)  # type: ignore
assert isinstance(cmask, vs.VideoNode)

rescaled = vstools.replace_ranges(
    rescaled, src, list(chain(range(0, 572), range(164400, src.num_frames - 1)))
)
cmask = vstools.replace_ranges(
    cmask,
    core.std.BlankClip(cmask, color=0),
    list(chain(range(0, 572), range(164400, src.num_frames - 1))),
)

rescaled16 = vstools.depth(rescaled, 16)
denoised = y5gfunc.Fast_BM3DWrapper(
    rescaled16, sigma_Y=3.25, preset=y5gfunc.BM3DPreset.LC
)

merged, dbmask = y5gfunc.SynDeband(clip=denoised, include_mask=True)

cleaned = haf.EdgeCleaner(merged, rmode=17, strength=10)
mask = muvsfunc.AnimeMask(input=merged, shift=0.7, mode=1).std.Binarize(threshold=1)
mask = core.akarin.Expr(
    [mask, vstools.get_y(cmask).resize.Point(format=vs.GRAY16)], "x y - 10000 > 65535 *"
)

cleaned = core.std.MaskedMerge(clipa=merged, clipb=cleaned, mask=mask, first_plane=True)
cleaned = vstools.join(cleaned, merged)

filtered = adptvgrnMod.adptvgrnMod(cleaned, strength=0.25)

if __name__ != "__main__":
    outputs = [src, cmask, filtered]
    y5gfunc.output(outputs, debug=False)
    y5gfunc.reset_output_index()
else:
    audios = y5gfunc.extract_audio_tracks(src_path, "/Users/a1")
    audios = audios[0:1]  # keep only first audio track
    audios[0]["default"] = True

    try:
        subs = y5gfunc.extract_pgs_subtitles(src_path, "/Users/a1")
    except:
        print("no subs!")
        subs = None

    chapter = y5gfunc.get_bd_chapter(src_path, "/Users/a1/mygo_01.txt")

    encoder = subprocess.Popen(
        [
            "/usr/local/bin/x265",
            "--y4m",
            "--input",
            "-",
            "--rc-lookahead",
            "130",
            "--crf",
            "14.5",
            "--hist-scenecut",
            "--preset",
            "slowxx",
            "--tune",
            "vq3",
            "--no-amp",
            "--no-rect",
            "--no-sao",
            "--deblock",
            "-1:-1",
            "-D",
            "10",
            "--transfer",
            "bt709",
            "--colorprim",
            "bt709",
            "--colormatrix",
            "bt709",
            "--range",
            "limited",
            "/Volumes/untitled/mygo_1.mkv",
        ],
        stdin=subprocess.PIPE,
    )
    y5gfunc.encode_video(vstools.depth(filtered, 10), encoder)
    y5gfunc.mux_mkv(
        output_path,
        videos={"path": "/Volumes/untitled/mygo_1.mkv", "language": "jpn"},
        audios=audios,
        subtitles=subs,
        chapters=chapter,
    )
