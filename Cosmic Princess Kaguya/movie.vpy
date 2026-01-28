from vstools import core, vs
import vstools, y5gfunc, subprocess
import CSMOD, fvsfunc, adptvgrnMod
import vsmlrt

core.set_affinity(16)

mappings = (
    "[770 1405] [34172 34262] [36859 37003] "
    "[37086 37205] [39644 39654] [39902 39943] [40169 40184] [40218 40253] "
    "[41395 41445] [43344 44108] [67104 67114] [73480 74353] [102729 103284] [117205 122400] [131505 135982] [163440 164705] [174809 177450] [179009 179137]"
)

src_path = R"C:\Users\Administrator\Downloads\Cosmic.Princess.Kaguya.2026.1080p.NF.WEB-DL.MULTi.DDP5.1.H.264-VARYG.mkv"
av1_src_path = R'C:\Users\Administrator\Downloads\Cosmic Princess Kaguya! (2026) MULTi 1080p WEB AV1 E-AC-3 -Tsundere-Raws (NF).mkv'

h264_src = core.bs.VideoSource(src_path).resize.Bicubic(format=vs.YUV420P16)
av1_src = core.bs.VideoSource(av1_src_path).resize.Bicubic(format=vs.YUV420P16)

src = fvsfunc.rfs(h264_src, av1_src, mappings)

src = vstools.depth(src, 16)
denoised = y5gfunc.Fast_BM3DWrapper(
    src, sigma_Y=3, sigma_chroma=4, preset=y5gfunc.BM3DPreset.MAGIC, bm3d=core.bm3dcuda_rtc
)
aaed = y5gfunc.double_aa(denoised)

debanded, mask = y5gfunc.sakiko_deband_v2(denoised, preset="low", include_mask=True)
merged = core.std.MaskedMerge(debanded, aaed, mask)

sharpen = CSMOD.CSMOD(merged.resize.Spline36(format=vs.YUV420P16), preset="detail")
dehalod = y5gfunc.simple_dehalo(sharpen, mode="fine_dehalo", maximum=1)
chroma_fixed = vsmlrt.ArtCNN(
    dehalod.resize2.Bilinear(format=vs.YUV444PS),
    model=vsmlrt.ArtCNNModel.ArtCNN_R8F64_Chroma,
    backend=vsmlrt.Backend.TRT(num_streams=8, fp16=True),
).resize2.Spline36(format=vs.YUV420P16)
filtered = adptvgrnMod.adptvgrnMod(chroma_fixed, strength=0.25)

if __name__ != "__main__":
    outputs = [h264_src, av1_src, filtered, dehalod, mask]
    y5gfunc.output(outputs, debug=False)
    y5gfunc.reset_output_index()
else:
    encoder = subprocess.Popen(
        [
            R"C:\Users\Administrator\Downloads\vapoursynth-portable\x265.exe",
            "--y4m",
            "--input",
            "-",
            "--rc-lookahead",
            "140",
            "--crf",
            "16",
            "--ref",
            "5",
            "--max-merge",
            "5",
            "--subme",
            "5",
            "--cbqpoffs",
            "-2",
            "--crqpoffs",
            "-2",
            "--aq-mode",
            "3",
            "--aq-strength",
            "0.7",
            "--min-keyint",
            "1",
            "--tu-inter-depth",
            "4",
            "--tu-intra-depth",
            "4",
            "--hist-scenecut",
            "--preset",
            "slowxx",
            "--tune",
            "vq3",
            "--no-amp",
            "--rect",
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
            R"C:\Users\Administrator\Downloads\cosmic_princess_v3.mkv",
        ],
        stdin=subprocess.PIPE,
    )
    y5gfunc.encode_video(vstools.depth(filtered, 10), encoder)
