cfl_shader = R'''
//!PARAM chroma_offset_x
//!TYPE float
0.0

//!PARAM chroma_offset_y
//!TYPE float
0.0

//!HOOK CHROMA
//!BIND LUMA
//!BIND CHROMA
//!SAVE LUMA_LR
//!WIDTH CHROMA.w
//!HEIGHT LUMA.h
//!WHEN CHROMA.w LUMA.w <
//!DESC Chroma From Luma Prediction (Hermite 1st step, Downscaling Luma)

float comp_wd(vec2 v) {
    float x = min(length(v), 1.0);
    return smoothstep(0.0, 1.0, 1.0 - x);
}

vec4 hook() {
    vec2 luma_pos = LUMA_pos;
    luma_pos.x += chroma_offset_x / LUMA_size.x;
    float start  = ceil((luma_pos.x - (1.0 / CHROMA_size.x)) * LUMA_size.x - 0.5);
    float end = floor((luma_pos.x + (1.0 / CHROMA_size.x)) * LUMA_size.x - 0.5);

    float wt = 0.0;
    float luma_sum = 0.0;
    vec2 pos = luma_pos;

    for (float dx = start.x; dx <= end.x; dx++) {
        pos.x = LUMA_pt.x * (dx + 0.5);
        vec2 dist = (pos - luma_pos) * CHROMA_size;
        float wd = comp_wd(dist);
        float luma_pix = LUMA_tex(pos).x;
        luma_sum += wd * luma_pix;
        wt += wd;
    }

    vec4 output_pix = vec4(luma_sum /= wt, 0.0, 0.0, 1.0);
    return clamp(output_pix, 0.0, 1.0);
}

//!HOOK CHROMA
//!BIND LUMA_LR
//!BIND CHROMA
//!BIND LUMA
//!SAVE LUMA_LR
//!WIDTH CHROMA.w
//!HEIGHT CHROMA.h
//!WHEN CHROMA.w LUMA.w <
//!DESC Chroma From Luma Prediction (Hermite 2nd step, Downscaling Luma)

float comp_wd(vec2 v) {
    float x = min(length(v), 1.0);
    return smoothstep(0.0, 1.0, 1.0 - x);
}

vec4 hook() {
    vec2 luma_pos = LUMA_LR_pos;
    luma_pos.y += chroma_offset_y / LUMA_LR_size.y;
    float start  = ceil((luma_pos.y - (1.0 / CHROMA_size.y)) * LUMA_LR_size.y - 0.5);
    float end = floor((luma_pos.y + (1.0 / CHROMA_size.y)) * LUMA_LR_size.y - 0.5);

    float wt = 0.0;
    float luma_sum = 0.0;
    vec2 pos = luma_pos;

    for (float dy = start; dy <= end; dy++) {
        pos.y = LUMA_LR_pt.y * (dy + 0.5);
        vec2 dist = (pos - luma_pos) * CHROMA_size;
        float wd = comp_wd(dist);
        float luma_pix = LUMA_LR_tex(pos).x;
        luma_sum += wd * luma_pix;
        wt += wd;
    }

    vec4 output_pix = vec4(luma_sum /= wt, 0.0, 0.0, 1.0);
    return clamp(output_pix, 0.0, 1.0);
}

//!HOOK CHROMA
//!BIND HOOKED
//!BIND LUMA
//!BIND LUMA_LR
//!WHEN CHROMA.w LUMA.w <
//!WIDTH LUMA.w
//!HEIGHT LUMA.h
//!OFFSET ALIGN
//!DESC Chroma From Luma Prediction (Upscaling Chroma)

#define USE_12_TAP_REGRESSION 1
#define USE_8_TAP_REGRESSIONS 1
#define DEBUG 0

float comp_wd(vec2 v) {
    float d = min(length(v), 2.0);
    float d2 = d * d;
    float d3 = d2 * d;

    if (d < 1.0) {
        return 1.25 * d3 - 2.25 * d2 + 1.0;
    } else {
        return -0.75 * d3 + 3.75 * d2 - 6.0 * d + 3.0;
    }
}

vec4 hook() {
    float ar_strength = 0.8;
    vec2 mix_coeff = vec2(0.8);
    vec2 corr_exponent = vec2(4.0);

    vec4 output_pix = vec4(0.0, 0.0, 0.0, 1.0);
    float luma_zero = LUMA_texOff(0.0).x;

    vec2 pp = HOOKED_pos * HOOKED_size - vec2(0.5);
    vec2 fp = floor(pp);
    pp -= fp;

#ifdef HOOKED_gather
    vec2 quad_idx[4] = {{0.0, 0.0}, {2.0, 0.0}, {0.0, 2.0}, {2.0, 2.0}};

    vec4 luma_quads[4];
    vec4 chroma_quads[4][2];

    for (int i = 0; i < 4; i++) {
        luma_quads[i] = LUMA_LR_gather(vec2((fp + quad_idx[i]) * HOOKED_pt), 0);
        chroma_quads[i][0] = HOOKED_gather(vec2((fp + quad_idx[i]) * HOOKED_pt), 0);
        chroma_quads[i][1] = HOOKED_gather(vec2((fp + quad_idx[i]) * HOOKED_pt), 1);
    }

    vec2 chroma_pixels[16];
    chroma_pixels[0]  = vec2(chroma_quads[0][0].w, chroma_quads[0][1].w);
    chroma_pixels[1]  = vec2(chroma_quads[0][0].z, chroma_quads[0][1].z);
    chroma_pixels[2]  = vec2(chroma_quads[1][0].w, chroma_quads[1][1].w);
    chroma_pixels[3]  = vec2(chroma_quads[1][0].z, chroma_quads[1][1].z);
    chroma_pixels[4]  = vec2(chroma_quads[0][0].x, chroma_quads[0][1].x);
    chroma_pixels[5]  = vec2(chroma_quads[0][0].y, chroma_quads[0][1].y);
    chroma_pixels[6]  = vec2(chroma_quads[1][0].x, chroma_quads[1][1].x);
    chroma_pixels[7]  = vec2(chroma_quads[1][0].y, chroma_quads[1][1].y);
    chroma_pixels[8]  = vec2(chroma_quads[2][0].w, chroma_quads[2][1].w);
    chroma_pixels[9]  = vec2(chroma_quads[2][0].z, chroma_quads[2][1].z);
    chroma_pixels[10] = vec2(chroma_quads[3][0].w, chroma_quads[3][1].w);
    chroma_pixels[11] = vec2(chroma_quads[3][0].z, chroma_quads[3][1].z);
    chroma_pixels[12] = vec2(chroma_quads[2][0].x, chroma_quads[2][1].x);
    chroma_pixels[13] = vec2(chroma_quads[2][0].y, chroma_quads[2][1].y);
    chroma_pixels[14] = vec2(chroma_quads[3][0].x, chroma_quads[3][1].x);
    chroma_pixels[15] = vec2(chroma_quads[3][0].y, chroma_quads[3][1].y);

    float luma_pixels[16];
    luma_pixels[0]  = luma_quads[0].w;
    luma_pixels[1]  = luma_quads[0].z;
    luma_pixels[2]  = luma_quads[1].w;
    luma_pixels[3]  = luma_quads[1].z;
    luma_pixels[4]  = luma_quads[0].x;
    luma_pixels[5]  = luma_quads[0].y;
    luma_pixels[6]  = luma_quads[1].x;
    luma_pixels[7]  = luma_quads[1].y;
    luma_pixels[8]  = luma_quads[2].w;
    luma_pixels[9]  = luma_quads[2].z;
    luma_pixels[10] = luma_quads[3].w;
    luma_pixels[11] = luma_quads[3].z;
    luma_pixels[12] = luma_quads[2].x;
    luma_pixels[13] = luma_quads[2].y;
    luma_pixels[14] = luma_quads[3].x;
    luma_pixels[15] = luma_quads[3].y;
#else
    vec2 pix_idx[16] = {{-0.5,-0.5}, {0.5,-0.5}, {1.5,-0.5}, {2.5,-0.5},
                        {-0.5, 0.5}, {0.5, 0.5}, {1.5, 0.5}, {2.5, 0.5},
                        {-0.5, 1.5}, {0.5, 1.5}, {1.5, 1.5}, {2.5, 1.5},
                        {-0.5, 2.5}, {0.5, 2.5}, {1.5, 2.5}, {2.5, 2.5}};

    float luma_pixels[16];
    vec2 chroma_pixels[16];

    for (int i = 0; i < 16; i++) {
        luma_pixels[i] = LUMA_LR_tex(vec2((fp + pix_idx[i]) * HOOKED_pt)).x;
        chroma_pixels[i] = HOOKED_tex(vec2((fp + pix_idx[i]) * HOOKED_pt)).xy;
    }
#endif

#if (DEBUG == 1)
    vec2 chroma_spatial = vec2(0.5);
    mix_coeff = vec2(1.0);
#else
    float wd[16];
    float wt = 0.0;
    vec2 ct = vec2(0.0);

    vec2 chroma_min = min(min(min(chroma_pixels[5], chroma_pixels[6]), chroma_pixels[9]), chroma_pixels[10]);
    vec2 chroma_max = max(max(max(chroma_pixels[5], chroma_pixels[6]), chroma_pixels[9]), chroma_pixels[10]);

    const int dx[16] = {-1, 0, 1, 2, -1, 0, 1, 2, -1, 0, 1, 2, -1, 0, 1, 2};
    const int dy[16] = {-1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};

    for (int i = 0; i < 16; i++) {
        wd[i] = comp_wd(vec2(dx[i], dy[i]) - pp);
        wt += wd[i];
        ct += wd[i] * chroma_pixels[i];
    }

    vec2 chroma_spatial = ct / wt;
    chroma_spatial = clamp(mix(chroma_spatial, clamp(chroma_spatial, chroma_min, chroma_max), ar_strength), 0.0, 1.0);
#endif

#if (USE_12_TAP_REGRESSION == 1 || USE_8_TAP_REGRESSIONS == 1)
    const int i12[12] = {1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14};
    const int i4y[4] = {1, 2, 13, 14};
    const int i4x[4] = {4, 7, 8, 11};
    const int i4[4] = {5, 6, 9, 10};

    float luma_sum_4 = 0.0;
    float luma_sum_4y = 0.0;
    float luma_sum_4x = 0.0;
    vec2 chroma_sum_4 = vec2(0.0);
    vec2 chroma_sum_4y = vec2(0.0);
    vec2 chroma_sum_4x = vec2(0.0);

    for (int i = 0; i < 4; i++) {
        luma_sum_4 += luma_pixels[i4[i]];
        luma_sum_4y += luma_pixels[i4y[i]];
        luma_sum_4x += luma_pixels[i4x[i]];
        chroma_sum_4 += chroma_pixels[i4[i]];
        chroma_sum_4y += chroma_pixels[i4y[i]];
        chroma_sum_4x += chroma_pixels[i4x[i]];
    }

    float luma_avg_12 = (luma_sum_4 + luma_sum_4y + luma_sum_4x) / 12.0;
    float luma_var_12 = 0.0;
    vec2 chroma_avg_12 = (chroma_sum_4 + chroma_sum_4y + chroma_sum_4x) / 12.0;
    vec2 chroma_var_12 = vec2(0.0);
    vec2 luma_chroma_cov_12 = vec2(0.0);

    for (int i = 0; i < 12; i++) {
        luma_var_12 += pow(luma_pixels[i12[i]] - luma_avg_12, 2.0);
        chroma_var_12 += pow(chroma_pixels[i12[i]] - chroma_avg_12, vec2(2.0));
        luma_chroma_cov_12 += (luma_pixels[i12[i]] - luma_avg_12) * (chroma_pixels[i12[i]] - chroma_avg_12);
    }

    vec2 corr = clamp(abs(luma_chroma_cov_12 / max(sqrt(luma_var_12 * chroma_var_12), 1e-6)), 0.0, 1.0);
    mix_coeff = pow(corr, corr_exponent) * mix_coeff;
#endif

#if (USE_12_TAP_REGRESSION == 1)
    vec2 alpha_12 = luma_chroma_cov_12 / max(luma_var_12, 1e-6);
    vec2 beta_12 = chroma_avg_12 - alpha_12 * luma_avg_12;
    vec2 chroma_pred_12 = clamp(alpha_12 * luma_zero + beta_12, 0.0, 1.0);
#endif

#if (USE_8_TAP_REGRESSIONS == 1)
    const int i8y[8] = {1, 2, 5, 6, 9, 10, 13, 14};
    const int i8x[8] = {4, 5, 6, 7, 8, 9, 10, 11};

    float luma_avg_8y = (luma_sum_4 + luma_sum_4y) / 8.0;
    float luma_avg_8x = (luma_sum_4 + luma_sum_4x) / 8.0;
    float luma_var_8y = 0.0;
    float luma_var_8x = 0.0;
    vec2 chroma_avg_8y = (chroma_sum_4 + chroma_sum_4y) / 8.0;
    vec2 chroma_avg_8x = (chroma_sum_4 + chroma_sum_4x) / 8.0;
    vec2 luma_chroma_cov_8y = vec2(0.0);
    vec2 luma_chroma_cov_8x = vec2(0.0);

    for (int i = 0; i < 8; i++) {
        luma_var_8y += pow(luma_pixels[i8y[i]] - luma_avg_8y, 2.0);
        luma_var_8x += pow(luma_pixels[i8x[i]] - luma_avg_8x, 2.0);
        luma_chroma_cov_8y += (luma_pixels[i8y[i]] - luma_avg_8y) * (chroma_pixels[i8y[i]] - chroma_avg_8y);
        luma_chroma_cov_8x += (luma_pixels[i8x[i]] - luma_avg_8x) * (chroma_pixels[i8x[i]] - chroma_avg_8x);
    }

    vec2 alpha_8y = luma_chroma_cov_8y / max(luma_var_8y, 1e-6);
    vec2 alpha_8x = luma_chroma_cov_8x / max(luma_var_8x, 1e-6);
    vec2 beta_8y = chroma_avg_8y - alpha_8y * luma_avg_8y;
    vec2 beta_8x = chroma_avg_8x - alpha_8x * luma_avg_8x;
    vec2 chroma_pred_8y = clamp(alpha_8y * luma_zero + beta_8y, 0.0, 1.0);
    vec2 chroma_pred_8x = clamp(alpha_8x * luma_zero + beta_8x, 0.0, 1.0);
    vec2 chroma_pred_8 = mix(chroma_pred_8y, chroma_pred_8x, 0.5);
#endif

#if (USE_12_TAP_REGRESSION == 1 && USE_8_TAP_REGRESSIONS == 1)
    output_pix.xy = mix(chroma_spatial, mix(chroma_pred_12, chroma_pred_8, 0.5), mix_coeff);
#elif (USE_12_TAP_REGRESSION == 1 && USE_8_TAP_REGRESSIONS == 0)
    output_pix.xy = mix(chroma_spatial, chroma_pred_12, mix_coeff);
#elif (USE_12_TAP_REGRESSION == 0 && USE_8_TAP_REGRESSIONS == 1)
    output_pix.xy = mix(chroma_spatial, chroma_pred_8, mix_coeff);
#else
    output_pix.xy = chroma_spatial;
#endif

    output_pix.xy = clamp(output_pix.xy, 0.0, 1.0);
    return output_pix;
}

//!PARAM distance_coeff
//!TYPE float
//!MINIMUM 0.0
2.0

//!PARAM intensity_coeff
//!TYPE float
//!MINIMUM 0.0
128.0

//!HOOK CHROMA
//!BIND CHROMA
//!BIND LUMA
//!DESC Chroma From Luma Prediction (Smoothing Chroma)

float comp_w(vec2 spatial_distance, float intensity_distance) {
    return max(100.0 * exp(-distance_coeff * pow(length(spatial_distance), 2.0) - intensity_coeff * pow(intensity_distance, 2.0)), 1e-32);
}

vec4 hook() {
    vec4 output_pix = vec4(0.0, 0.0, 0.0, 1.0);
    float luma_zero = LUMA_texOff(0).x;
    float wt = 0.0;
    vec2 ct = vec2(0.0);

    for (int i = -1; i < 2; i++) {
        for (int j = -1; j < 2; j++) {
            vec2 chroma_pixels = CHROMA_texOff(vec2(i, j)).xy;
            float luma_pixels = LUMA_texOff(vec2(i, j)).x;
            float w = comp_w(vec2(i, j), luma_zero - luma_pixels);
            wt += w;
            ct += w * chroma_pixels;
        }
    }

    output_pix.xy = clamp(ct / wt, 0.0, 1.0);
    return output_pix;
}
'''

import logging, functools
import lvsfunc
import vapoursynth as vs
from vapoursynth import core
import mvsfunc as mvf
import havsfunc as haf
import vsTAAmbk as taa
import rksfunc, ksksfunc, vsdpir, adptvgrnMod, muvsfunc, CSMOD, vsmlrt, vsutil, nnedi3_resample, xvs
import numpy as np
from getfnative import descale_cropping_args
import kagefunc as kgf

VERBOSE_LOGGING = True
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def auto_descale_nnedi3(clip, possible_heights, loss_threshold=1, bh=[], bw=[]):
    """
    :param clip: 输入的clip
    :param possible_heights: 可能的原生宽度列表
    :param loss_threshold: 判断loss显著增加的阈值
    """
    
    best_params = [None]
    best_loss = [float('inf')]
    prev_params = [None]
    
    if bh == []:
        bh = [clip.height, clip.height-1]
    if bw == []:
        bw = [clip.width, clip.width-1]

    kernel_list = [
        # {'name': 'Debicubic',  'dparams': {'b': 1/3,  'c': 1/3},     'rparams': {'filter_param_a': 1/3,  'filter_param_b': 1/3}},
        # {'name': 'Debicubic',  'dparams': {'b': 0,    'c': 0.5},     'rparams': {'filter_param_a': 0,    'filter_param_b': 0.5}},
        # {'name': 'Debicubic',  'dparams': {'b': 0,    'c': 0},       'rparams': {'filter_param_a': 0,    'filter_param_b': 0}},
        # {'name': 'Debicubic',  'dparams': {'b': -0.5, 'c': 0.25},    'rparams': {'filter_param_a': -0.5, 'filter_param_b': 0.25}},
        # {'name': 'Debicubic',  'dparams': {'b': -1,   'c': 0},       'rparams': {'filter_param_a': -1,   'filter_param_b': 0}},
        # {'name': 'Debicubic',  'dparams': {'b': -2,   'c': 1},       'rparams': {'filter_param_a': -2,   'filter_param_b': 1}},

        # {'name': 'Delanczos',  'dparams': {'taps': 2},               'rparams': {'filter_param_a': 2}},
        # {'name': 'Delanczos',  'dparams': {'taps': 3},               'rparams': {'filter_param_a': 3}},
        {'name': 'Delanczos',  'dparams': {'taps': 4},               'rparams': {'filter_param_a': 4}},
        # {'name': 'Delanczos',  'dparams': {'taps': 5},               'rparams': {'filter_param_a': 5}},
        # {'name': 'Delanczos',  'dparams': {'taps': 6},               'rparams': {'filter_param_a': 6}},
        # {'name': 'Delanczos',  'dparams': {'taps': 8},               'rparams': {'filter_param_a': 8}},

        # Spline kernels always get the lowest loss, even they are clearly not the correct ones
        # {'name': 'Despline16', 'dparams': {}, 'rparams': {}},
        # {'name': 'Despline36', 'dparams': {}, 'rparams': {}},
        {'name': 'Despline64', 'dparams': {}, 'rparams': {}},

        # {'name': 'Debilinear', 'dparams': {}, 'rparams': {}},
    ]

    def iterate(clip, func, count):
        for _ in range(count):
            clip = func(clip)
        return clip
    
    def get_resize_name(descale_name):
        if descale_name.startswith('De'):
            return descale_name[2:].capitalize()
        return descale_name

    def generate_descale_mask(source, upscaled, threshold=0.05):
        mask = core.akarin.Expr([source, upscaled], 'x y - abs').std.Binarize(threshold)
        mask = iterate(mask, core.std.Maximum, 2)
        mask = iterate(mask, core.std.Inflate, 2)
        return mask

    def process_frame(n, f):
        nonlocal best_params, best_loss, prev_params

        if VERBOSE_LOGGING:
            logger.info(f"Processing frame {n}")

        frame = f.copy()

        src_luma = core.std.ShufflePlanes([clip], planes=0, colorfamily=vs.GRAY)
        src_luma_frame = src_luma.get_frame(n)
        src_luma_arr = np.asarray(src_luma_frame[0]).astype(np.float32)

        if prev_params[0]:
            if VERBOSE_LOGGING:
                logger.info(f"Frame {n}: Using previous parameters")
            params = prev_params[0]
            descaled = getattr(core.descale, params['kernel'])(
                src_luma, 
                width=params.get('width', 0), 
                height=params.get('height', 0), 
                src_left=params.get('src_left', 0),
                src_top=params.get('src_top', 0),
                src_width=params.get('src_width', src_luma.width),
                src_height=params.get('src_height', src_luma.height),
                **params.get('dkernel_params', {})
            )
            resize_name = get_resize_name(params['kernel'])
            upscaled = getattr(core.resize, resize_name)(
                descaled, 
                width=clip.width, 
                height=clip.height,
                src_left=params.get('src_left', 0),
                src_top=params.get('src_top', 0),
                src_width=params.get('src_width', src_luma.width),
                src_height=params.get('src_height', src_luma.height),
                **params.get('rkernel_params', {})
            )
            upscaled_frame = upscaled.get_frame(n)
            upscaled_arr = np.asarray(upscaled_frame[0]).astype(np.float32)
            diff = src_luma_arr - upscaled_arr
            current_loss = np.mean(np.abs(diff))
            if VERBOSE_LOGGING:
                logger.info(f"Frame {n}: current_loss = {current_loss}")
            if current_loss < best_loss[0] * (1 + loss_threshold):
                best_params[0] = params
                best_loss[0] = current_loss # type: ignore
            else:
                best_params[0] = None
                best_loss[0] = float('inf')

        if best_params[0] is None:
            if VERBOSE_LOGGING:
                logger.info(f"Frame {n}: Performing full scan")
            
            for kernel_info in kernel_list:
                kernel = kernel_info['name']
                dkernel_params = kernel_info['dparams']
                rkernel_params = kernel_info['rparams']
                for src_height in possible_heights:
                    for base_height in bh:
                        for base_width in bw:
                            dargs = descale_cropping_args(src_luma, src_height, base_height, base_width)
                            if VERBOSE_LOGGING:
                                logger.info(f"Frame {n}: trying dargs = {dargs}, base_height={base_height}, base_width={base_width}")
                            descaled = getattr(core.descale, kernel)(
                                src_luma,
                                **dargs,
                                **dkernel_params
                            )
                            resize_name = get_resize_name(kernel)
                            upscaled = getattr(core.resize, resize_name)(
                                descaled, 
                                width=clip.width, 
                                height=clip.height,
                                src_left=dargs['src_left'],
                                src_top=dargs['src_top'],
                                src_width=dargs['src_width'],
                                src_height=dargs['src_height'],
                                **rkernel_params
                            )
                            upscaled_frame = upscaled.get_frame(n)
                            upscaled_arr = np.asarray(upscaled_frame[0]).astype(np.float32)
                            diff = src_luma_arr - upscaled_arr
                            loss = np.mean(np.abs(diff))
                            if loss < best_loss[0]:
                                best_loss[0] = loss # type: ignore
                                best_params[0] = { # type: ignore
                                    'kernel': kernel,
                                    'dkernel_params': dkernel_params,
                                    'rkernel_params': rkernel_params,
                                    **dargs
                                }

        if best_params[0] is None:
            if VERBOSE_LOGGING:
                logger.info(f"Frame {n}: No good parameters found, returning original frame")
            return frame

        params = best_params[0]
        if VERBOSE_LOGGING:
            logger.info(f"Frame {n}: Best parameters - {params}")
        descaled = getattr(core.descale, params['kernel'])(
            src_luma, 
            width=params.get('width', 0), 
            height=params.get('height', 0), 
            src_left=params.get('src_left', 0),
            src_top=params.get('src_top', 0),
            src_width=params.get('src_width', src_luma.width),
            src_height=params.get('src_height', src_luma.height),
            **params.get('dkernel_params', {})
        )
        rescaled = nnedi3_resample.nnedi3_resample(
            descaled, clip.width, clip.height, mode="nnedi3cl", qual=2, nns=4,
                src_left=params.get('src_left', 0),
                src_top=params.get('src_top', 0),
                src_width=params.get('src_width', src_luma.width),
                src_height=params.get('src_height', src_luma.height)
        )
        
        resize_name = get_resize_name(params['kernel'])
        upscaled = getattr(core.resize, resize_name)(
            descaled, 
            width=clip.width, 
            height=clip.height,
            src_left=params.get('src_left', 0),
            src_top=params.get('src_top', 0),
            src_width=params.get('src_width', src_luma.width),
            src_height=params.get('src_height', src_luma.height),
            **params.get('rkernel_params', {})
        )
        mask = generate_descale_mask(src_luma, upscaled)
        luma = core.std.MaskedMerge(rescaled, src_luma, mask)
        luma_frame = luma.get_frame(n)
        luma_arr = np.asanyarray(luma_frame[0]).astype(np.float32)
        
        np.copyto(np.asarray(frame[0]), luma_arr)

        prev_params[0] = best_params[0] # type: ignore

        return frame

    result = clip.std.ModifyFrame(clips=clip, selector=process_frame)
    
    return result

file_path = "/Volumes/untitled/00003.m2ts"

class cfg:
    width = 0
    height = 0

configure = cfg()

def load_source(filepath):
    clip = core.lsmas.LWLibavSource(filepath)
    if configure.width == 0 or configure.height == 0:
        configure.width = clip.width
        configure.height = clip.height
    clip = core.resize.Bicubic(clip=clip, width=configure.width, height=configure.height, format=vs.YUV420P16, matrix_s="709", matrix_in_s="709")
    return clip

def aa(clip):
    Y = vsutil.get_y(clip)
    aaY = taa.TAAmbk(Y, aatype=1, aatypeu=0, aatypev=0, preaa=0, strength=0, cycle=0, mtype=3, mclip=None,
           mthr=None, mthr2=None, mlthresh=None, mpand=(1, 0), txtmask=1, txtfade=0, thin=0, dark=0, sharp=-1,
           aarepair=0, postaa=None, src=None, stabilize=3, down8=False, showmask=0, opencl=True)
    ret = rksfunc.mergeuv(aaY, clip)
    return ret

def output(output_list):
    for index, clip in enumerate(output_list):
        clip.set_output(index)
        
# EP01: first 2433 frams don't need filtering
src = load_source(filepath=file_path)
src.set_output(1)
src = core.placebo.Shader(clip=src, shader_s=cfl_shader)
src = mvf.Depth(src,32)
src_luma_f = vsutil.get_y(src)
dargs = descale_cropping_args(src, 720.1, 1080, 1920)
descaled = src_luma_f.descale.Delanczos(**dargs, taps=5) # type: ignore
nnedi3 = functools.partial(core.nnedi3cl.NNEDI3CL, field=1, nsize=4, nns=4, qual=2)
n2x = nnedi3(nnedi3(descaled, dh=True), dw=True)
rescaled = muvsfunc.SSIM_downsample(clip=n2x, w=1920, h=1080, sigmoid=True, src_left=dargs['src_left']*2-0.5, src_top=dargs['src_top']*2-0.5, src_width=dargs['src_width']*2, src_height=dargs['src_height']*2)
upscaled = descaled.resize.Lanczos(width=1920, height=1080, src_left=dargs['src_left'], src_top=dargs['src_top'], src_width=dargs['src_width'], src_height=dargs['src_height'], filter_param_a=5)
dmask = core.akarin.Expr([src_luma_f, upscaled], 'x y - abs')
dmask = dmask.std.Maximum().std.Maximum().std.Maximum()
dmask = dmask.std.Inflate().std.Inflate().std.Inflate()
cmask = dmask.akarin.Expr(f'x {10/255} >= 1 0 ?')
cmask = cmask.std.Maximum().std.Maximum().std.Minimum().std.Inflate()
rescaled_finY = core.std.MaskedMerge(clipa=rescaled, clipb=src_luma_f, mask=cmask)

rescaled = rksfunc.mergeuv(clipy=rescaled_finY, clipuv=src)
# fin_descale = rescaled.std.Trim(last=526) + src.std.Trim(first=527, last=736) + rescaled.std.Trim(first=737)

# rescaled.set_output()
denoised = rksfunc.Fast_BM3DWrapper(rescaled.resize.Spline36(format=vs.YUV420P16), radius_Y=1, sigma_Y=0.8, sigma_chroma=1.6)
nrmask = ksksfunc.DBMask(denoised)
aaedY1080P = aa(denoised)
aaedY1080P = core.std.MaskedMerge(clipa=aaedY1080P, clipb=rescaled.resize.Spline36(format=vs.YUV420P16), mask=cmask.resize.Spline36(format=vs.YUV420P16))
dbed1080P = rksfunc.SynDeband(cyuv16=aaedY1080P, limit=True, limit_elast=1.75)
merged1080P = core.std.MaskedMerge(dbed1080P,aaedY1080P,nrmask,first_plane=True) # type: ignore
merged1080P.set_output(3)
mask = muvsfunc.AnimeMask(input=merged1080P, shift=0.7, mode=1).std.Binarize(10)
mask = core.akarin.Expr([mask.fmtc.bitdepth(bits=16), vsutil.get_y(cmask).fmtc.bitdepth(bits=16)], "x y -").std.Binarize(threshold=10000)
mask.set_output(4)
dehalod = haf.FineDehalo(merged1080P, edgeproc=1, contra=1)
dehalod = core.std.MaskedMerge(clipa=merged1080P, clipb=dehalod, mask=mask, first_plane=True)
# dehalod = haf.EdgeCleaner(dehalod)
# dehalod = haf.EdgeCleaner(c=dehalod, strength=15)
dehalod.set_output(5)

finn = src.std.Trim(last=2433).resize.Spline36(format=vs.YUV420P16) + dehalod.std.Trim(first=2434) if "00003" in file_path else dehalod

finn.set_output(0)

# descale_mask = kgf.inverse_scale_mask(source=src, width=1280, height=720, kernel="lanczos", taps=4, mask_detail=True)
# descale_mask = nnedi3_resample.nnedi3_resample(descale_mask, 1920, 1080, mode="nnedi3cl").resize.Point(format=vs.YUV444PS)
# descaled720 = kgf.inverse_scale(source=src, width=1280, height=720, kernel="lanczos", taps=5, mask_detail=True)
# yuv = core.resize.Spline36(clip=descaled720, width=1280, height=720, format=vs.YUV444P16, matrix_s="709")

# backe = nnedi3_resample.nnedi3_resample(input=yuv, target_width=1920, target_height=1080, mode="nnedi3cl", nns=4, qual=2)
# backe = mvf.Depth(backe,32)
# mergeback = core.std.MaskedMerge(clipa=backe, clipb=src, mask=descale_mask)
# mergeback.set_output(2)