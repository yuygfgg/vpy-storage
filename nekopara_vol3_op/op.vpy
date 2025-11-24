import vapoursynth as vs
from vapoursynth import core
import havsfunc as haf
import vstools, subprocess
import CSMOD, vsmlrt
from y5gfunc import *

src_path = "./src.mkv"
output_path = "./encode.mkv"

src = load_source(src_path)

denoised = vsmlrt.DPIR(
    core.resize.Bicubic(src, format=vs.RGBS),
    strength=7,
    model=vsmlrt.DPIRModel.drunet_color,
    backend=vsmlrt.Backend.ORT_COREML(num_streams=3),
).resize2.Bicubic(format=vs.YUV420P16, matrix_in_s="709", matrix_s="709")

debanded, dbmask = SynDeband(clip=denoised, include_mask=True)

merged = core.std.MaskedMerge(debanded, denoised, dbmask, first_plane=True)

filtered = CSMOD.CSMOD(merged, source=src, preset="very slow")

y_upscaled = SSIM_downsample(
    vsmlrt.ArtCNN(
        vstools.depth(vstools.get_y(filtered), 32),
        model=vsmlrt.ArtCNNModel.ArtCNN_R8F64,
        backend=vsmlrt.Backend.ORT_COREML(num_streams=2),
    ),
    1920,
    1080,
)
upscaled = vstools.join(
    vstools.depth(y_upscaled, 16), filtered.resize2.Bicubic(1920, 1080)
)

if __name__ != "__main__":
    outputs = [src, upscaled]
    output(outputs, debug=False)
    reset_output_index()
    subprocess.run(
        "rm -r /private/var/folders/gb/m6gky8t10yx68drv06ncbj_h0000gn/T/onnxruntime*",
        shell=True,
        check=True,
    )
else:
    # audios = extract_audio_tracks(src_path, "/Users/a1")
    # audios = audios[0:1]  # keep only first audio track
    # audios[0]["default"] = True

    encoder = subprocess.Popen(
        [
            "/usr/local/bin/x265",
            "--y4m",
            "--input",
            "-",
            "--rc-lookahead",
            "130",
            "--crf",
            "21.5",
            "--hist-scenecut",
            "--preset",
            "slowxx",
            "--tune",
            "vq3",
            "--no-amp",
            "--no-rect",
            "--sao",
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
            "./tmp.mkv",
        ],
        stdin=subprocess.PIPE,
    )
    encode_video(vstools.depth(upscaled, 10), encoder)
    subprocess.run(
        "rm -r /private/var/folders/gb/m6gky8t10yx68drv06ncbj_h0000gn/T/onnxruntime*",
        shell=True,
        check=True,
    )
