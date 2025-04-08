from pathlib import Path
from vstools import core, vs
import vstools
import subprocess
import CSMOD
from y5gfunc import *  # type: ignore

episode_num = "01"

workdir_path = "/Users/a1/encoding/Zatsu_Tabi/"
episodedir_path = workdir_path + f"{episode_num}/"

src_path = f"/Users/a1/encoding/Zatsu_Tabi/[SubsPlease] Zatsu Tabi - That's Journey - {episode_num} (1080p).mkv"
chs_raw_path = (
    f"{workdir_path}subs/[Haruhana&STYHSub] Zatsu Tabi - {episode_num}.chs_jpn.ass"
)
cht_raw_path = (
    f"{workdir_path}subs/[Haruhana&STYHSub] Zatsu Tabi - {episode_num}.cht_jpn.ass"
)
fontdir = f"{workdir_path}fonts"
chapter_path = f"{workdir_path}chapter/Zatsu Tabi - {episode_num} [Chapter].txt"
chs_path = f"{episodedir_path}chs.mkv"
cht_path = f"{episodedir_path}cht.mkv"
chi_path = f"{episodedir_path}chi.mkv"


# ------ filter chain start
src = load_source(src_path)

rescaled, upscaled, cmask, osd = rescale(src, ["Debicubic"], src_height=900, bw=1920, bh=1080, b=[-0.5, -0.33, 0], c=[1], show_upscaled=True, show_detail_mask=True, scene_stable=True)  # type: ignore
denoised = Fast_BM3DWrapper(
    rescaled.resize2.Bicubic(format=vs.YUV420P16), sigma_Y=2, sigma_chroma=2.5
)

debanded, nrmask = SynDeband(clip=denoised, limit=True, include_mask=True)
merged = core.std.MaskedMerge(debanded, denoised, nrmask, first_plane=True)  # type: ignore

sharp = CSMOD.CSMOD(merged, preset="detail")

filtered = vstools.depth(sharp, 10)
# ------ filter chain end

chs = core.assrender.TextSub(filtered, chs_raw_path, fontdir=fontdir)
cht = core.assrender.TextSub(filtered, cht_raw_path, fontdir=fontdir)

if __name__ != "__main__":
    outputs = [src, chs, cht, rescaled, osd, filtered]
    output(outputs, debug=False)
    reset_output_index()
else:
    audios_lossy = extract_audio_tracks(
        src_path,
        workdir_path + f"{episode_num}/",
        config=AudioConfig(main_lossy=TrackConfig(mode=ProcessMode.COPY)),
    )
    audios_lossy = [audios_lossy[0]]

    subset_fonts([chs_raw_path, cht_raw_path], fontdir, episodedir_path)
    chs_subset = {
        "path": f"{episodedir_path}[Haruhana&STYHSub] Zatsu Tabi - {episode_num}.chs_jpn.rename.ass",
        "language": "zh-ch",
        "track_name": "简日双语",
        "default": True,
    }
    cht_subset = {
        "path": f"{episodedir_path}[Haruhana&STYHSub] Zatsu Tabi - {episode_num}.cht_jpn.rename.ass",
        "language": "zh-tw",
        "track_name": "繁日双语",
        "default": False,
    }
    fontdir = f"{episodedir_path}subsetted_fonts/"

    chapter = Path(chapter_path) if chapter_path else None

    chi_encoder = subprocess.Popen(
        [
            "/usr/local/bin/x265",
            "--y4m",
            "--input",
            "-",
            "--rc-lookahead",
            "130",
            "--crf",
            "18",
            "--preset",
            "slowxx",
            "--tune",
            "vq3",
            "--no-amp",
            "--hist-scenecut",
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
            chi_path,
        ],
        stdin=subprocess.PIPE,
    )
    chs_encoder = subprocess.Popen(
        [
            "/usr/local/bin/x265",
            "--y4m",
            "--input",
            "-",
            "--rc-lookahead",
            "130",
            "--crf",
            "19",
            "--preset",
            "slowxx",
            "--tune",
            "vq3",
            "--sao",
            "--limit-sao",
            "--no-amp",
            "--hist-scenecut",
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
            chs_path,
        ],
        stdin=subprocess.PIPE,
    )
    cht_encoder = subprocess.Popen(
        [
            "/usr/local/bin/x265",
            "--y4m",
            "--input",
            "-",
            "--rc-lookahead",
            "130",
            "--crf",
            "19",
            "--preset",
            "slowxx",
            "--tune",
            "vq3",
            "--sao",
            "--limit-sao",
            "--no-amp",
            "--hist-scenecut",
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
            cht_path,
        ],
        stdin=subprocess.PIPE,
    )
    encode_video(
        [filtered, chs, cht], [chi_encoder, chs_encoder, cht_encoder], multi=True
    )

    mux_mkv(
        f"{episodedir_path}chifin.mkv",
        videos={"path": chi_path, "language": "jpn"},
        audios=audios_lossy,
        subtitles=[chs_subset, cht_subset],
        chapters=chapter,
        fonts_dir=fontdir,
    )
    mux_mkv(
        f"{episodedir_path}chsfin.mkv",
        videos={"path": chs_path, "language": "jpn"},
        audios=audios_lossy,
        subtitles=None,
        chapters=chapter,
        fonts_dir=None,
    )
    mux_mkv(
        f"{episodedir_path}chtfin.mkv",
        videos={"path": cht_path, "language": "jpn"},
        audios=audios_lossy,
        subtitles=None,
        chapters=chapter,
        fonts_dir=None,
    )
