from __future__ import annotations
from typing import Dict, List, Tuple, Callable, Optional
from dataclasses import dataclass, field
import io, math, random, wave
import numpy as np

try:
    from scipy.signal import lfilter as _scipy_lfilter, fftconvolve as _scipy_fftconvolve
    _HAS_SCIPY = True
except ImportError: _HAS_SCIPY = False

try:
    import numba
    from numba import njit
    _HAS_NUMBA = True
except ImportError: _HAS_NUMBA = False

if _HAS_NUMBA:
    @njit(cache=True)
    def _nb_k_weight(x, sr):
        n = len(x); y = np.empty(n, dtype=np.float64)
        if sr == 48000: s1_b0, s1_b1, s1_b2, s1_a1, s1_a2 = 1.53512485958624, -2.61187896900947, 1.23925952524423, -1.69019851598276, 0.732626261909496; s2_b0, s2_b1, s2_b2, s2_a1, s2_a2 = 1.0, -2.0, 1.0, -1.99004745481143, 0.990072500365666
        elif sr == 44100: s1_b0, s1_b1, s1_b2, s1_a1, s1_a2 = 1.53058976, -2.60968407, 1.24087861, -1.68589681, 0.73233165; s2_b0, s2_b1, s2_b2, s2_a1, s2_a2 = 1.0, -2.0, 1.0, -1.98993544, 0.98996959
        else:
            for i in range(n): y[i] = x[i]
            return y
        z1 = z2 = 0.0
        for i in range(n): xn=x[i]; yn=s1_b0*xn+z1; z1=s1_b1*xn-s1_a1*yn+z2; z2=s1_b2*xn-s1_a2*yn; y[i]=yn
        z1 = z2 = 0.0
        for i in range(n): xn=y[i]; yn=s2_b0*xn+z1; z1=s2_b1*xn-s2_a1*yn+z2; z2=s2_b2*xn-s2_a2*yn; y[i]=yn
        return y

    @njit(cache=True)
    def _nb_compute_loudness(x_k):
        n = len(x_k)
        if n == 0: return -70.0
        s = 0.0
        for i in range(n): v=x_k[i]; s += v*v
        ms = s / n
        if ms < 1e-20: return -70.0
        return -0.691 + 10.0 * np.log10(ms)

    @njit(cache=True)
    def _nb_apply_gain(x, gain):
        n = len(x); out = np.empty(n, dtype=np.float64)
        for i in range(n): out[i] = x[i] * gain
        return out

    @njit(cache=True)
    def _nb_master_limiter(x, threshold):
        n = len(x); out = np.empty(n, dtype=np.float64); gain = 1.0; rs = 1.0 / 2400.0
        for i in range(n):
            xn = x[i]; ax = xn
            if ax < 0.0: ax = -ax
            if ax * gain > threshold: tg = threshold / max(ax, 1e-10); gain = gain * 0.9 + tg * 0.1
            else: gain = min(1.0, gain + rs)
            out[i] = xn * gain
        return out

    @njit(cache=True)
    def _nb_karplus_strong(f, n, sr, decay, brightness, exciter):
        if f <= 0: return np.zeros(n, dtype=np.float32)
        delay_len = max(1, int(sr / f))
        buf = np.zeros(delay_len, dtype=np.float32)
        for i in range(delay_len):
            if i < len(exciter): buf[i] = exciter[i]
        out = np.zeros(n, dtype=np.float32)
        filt_coef = 1.0 - brightness
        for i in range(n):
            idx = i % delay_len
            next_idx = (i + 1) % delay_len
            out[i] = buf[idx]
            buf[idx] = decay * (buf[idx] * filt_coef + buf[next_idx] * (1 - filt_coef))
        p = 0.0
        for i in range(n):
            v = abs(out[i])
            if v > p: p = v
        if p > 0:
            for i in range(n): out[i] /= p
        return out

    @njit(cache=True)
    def _nb_fm_fb(c, mr, mi, fb, n, sr):
        out = np.zeros(n, dtype=np.float32)
        phase_c = 0.0
        phase_m = 0.0
        last_c = 0.0
        incr_c = 2 * np.pi * c / sr
        incr_m = 2 * np.pi * c * mr / sr
        for i in range(n):
            mod = np.sin(phase_m) * mi + last_c * fb
            val = np.sin(phase_c + mod)
            out[i] = val
            last_c = val
            phase_c += incr_c
            phase_m += incr_m
            if phase_c > 2 * np.pi: phase_c -= 2 * np.pi
            if phase_m > 2 * np.pi: phase_m -= 2 * np.pi
        p = 0.0
        for i in range(n):
            v = abs(out[i])
            if v > p: p = v
        if p > 0:
            for i in range(n): out[i] /= p
        return out

    @njit(cache=True)
    def _nb_fm_dx7(c, m1, m2, i1, i2, fb, n, sr):
        out = np.zeros(n, dtype=np.float32)
        ph_c, ph_m1, ph_m2 = 0.0, 0.0, 0.0
        last_m1 = 0.0
        incr_c = 2 * np.pi * c / sr
        incr_m1 = 2 * np.pi * m1 / sr
        incr_m2 = 2 * np.pi * m2 / sr
        for i in range(n):
            op2 = np.sin(ph_m2)
            op1 = np.sin(ph_m1 + i2 * op2 + fb * last_m1)
            carrier = np.sin(ph_c + i1 * op1)
            out[i] = carrier
            last_m1 = op1
            ph_c += incr_c
            ph_m1 += incr_m1
            ph_m2 += incr_m2
            if ph_c > 2 * np.pi: ph_c -= 2 * np.pi
            if ph_m1 > 2 * np.pi: ph_m1 -= 2 * np.pi
            if ph_m2 > 2 * np.pi: ph_m2 -= 2 * np.pi
        p = 0.0
        for i in range(n):
            v = abs(out[i])
            if v > p: p = v
        if p > 0:
            for i in range(n): out[i] /= p
        return out

    @njit(cache=True)
    def _nb_mix(mix, flat, offsets, lengths, starts):
        n_mix, n_flat = mix.shape[0], flat.shape[0]
        for i in range(len(starts)):
            s, ln, o = starts[i], lengths[i], offsets[i]
            for j in range(ln):
                if s + j < n_mix and o + j < n_flat: mix[s + j] += flat[o + j]

def _k_weight_pure(x, sr):
    if sr == 48000: s1b=[1.53512485958624,-2.61187896900947,1.23925952524423]; s1a=[1.0,-1.69019851598276,0.732626261909496]; s2b=[1.0,-2.0,1.0]; s2a=[1.0,-1.99004745481143,0.990072500365666]
    elif sr == 44100: s1b=[1.53058976,-2.60968407,1.24087861]; s1a=[1.0,-1.68589681,0.73233165]; s2b=[1.0,-2.0,1.0]; s2a=[1.0,-1.98993544,0.98996959]
    else: return x.copy()
    if _HAS_SCIPY:
        from scipy.signal import lfilter
        y = lfilter(s1b, s1a, x); y = lfilter(s2b, s2a, y); return y
    y = np.zeros_like(x); z1 = z2 = 0.0; b0, b1, b2 = s1b; a1, a2 = s1a[1], s1a[2]
    for i in range(len(x)): xn=x[i]; yn=b0*xn+z1; z1=b1*xn-a1*yn+z2; z2=b2*xn-a2*yn; y[i]=yn
    z1 = z2 = 0.0; b0, b1, b2 = s2b; a1, a2 = s2a[1], s2a[2]
    for i in range(len(y)): xn=y[i]; yn=b0*xn+z1; z1=b1*xn-a1*yn+z2; z2=b2*xn-a2*yn; y[i]=yn
    return y

def _compute_lufs(x, sr):
    if len(x) < sr//2: return -70.0
    xf = x.astype(np.float64)
    if _HAS_NUMBA: xk=_nb_k_weight(xf,sr); return float(_nb_compute_loudness(xk))
    xk = _k_weight_pure(xf, sr); ms = np.mean(xk**2)
    if ms < 1e-20: return -70.0
    return -0.691 + 10.0 * np.log10(ms)

def _normalize_to_lufs(x, sr, target_lufs=-14.0, tp_db=-1.0):
    if len(x) < sr//2: return x
    xf = x.astype(np.float64)
    cl = _compute_lufs(xf, sr)
    if not np.isfinite(cl) or cl <= -70.0: return x
    gdb = target_lufs - cl; gl = 10.0 ** (gdb / 20.0)
    if _HAS_NUMBA: xn = _nb_apply_gain(xf, gl)
    else: xn = xf * gl
    tp = 10.0 ** (tp_db / 20.0)
    if _HAS_NUMBA: xn = _nb_master_limiter(xn, tp)
    else:
        p = np.max(np.abs(xn))
        if p > tp: xn = xn * (tp / p)
    return xn

class DSP:
    @staticmethod
    def env_exp(n: int, sr: int, attack: float = 0.001, decay_rate: float = 30.0, attack_curve: float = 2.0) -> np.ndarray:
        t = np.arange(n, dtype=np.float64) / sr; env = np.exp(-t * decay_rate); a_samp = max(1, int(attack * sr))
        if a_samp < n:
            a_env = np.linspace(0, 1, a_samp, dtype=np.float64) ** attack_curve
            env[:a_samp] = a_env * np.exp(-np.arange(a_samp, dtype=np.float64) / sr * decay_rate)
        return env

    @staticmethod
    def fir_lowpass(cutoff: float, sr: int = 44100, n_taps: int = 127, window: str = "blackman") -> np.ndarray:
        if n_taps % 2 == 0: n_taps += 1
        cutoff = min(max(1.0, cutoff), sr / 2.0 - 1.0)
        if cutoff <= 0: return np.zeros(n_taps)
        fc = cutoff / sr; n = np.arange(n_taps, dtype=np.float64); center = (n_taps - 1) / 2.0; h = np.sinc(2 * fc * (n - center))
        match window:
            case "blackman": w = 0.42 - 0.5*np.cos(2*np.pi*n/(n_taps-1)) + 0.08*np.cos(4*np.pi*n/(n_taps-1))
            case "hamming": w = 0.54 - 0.46*np.cos(2*np.pi*n/(n_taps-1))
            case "hann": w = 0.5 - 0.5*np.cos(2*np.pi*n/(n_taps-1))
            case _: w = np.ones(n_taps)
        h *= w; s = np.sum(h)
        if s > 0: h /= s
        return h

    @staticmethod
    def fir_highpass(cutoff: float, sr: int = 44100, n_taps: int = 127, window: str = "blackman") -> np.ndarray:
        hlp = DSP.fir_lowpass(cutoff, sr, n_taps, window); h = -hlp; h[len(h)//2] += 1.0; return h

    @staticmethod
    def fir_bandpass(low: float, high: float, sr: int = 44100, n_taps: int = 255, window: str = "blackman") -> np.ndarray:
        return DSP.fir_lowpass(high, sr, n_taps, window) - DSP.fir_lowpass(low, sr, n_taps, window)

    @staticmethod
    def fir_peak(center: float, bandwidth: float, sr: int = 44100, n_taps: int = 255, gain: float = 2.0) -> np.ndarray:
        if center <= 0 or center >= sr / 2: return np.zeros(n_taps)
        n = np.arange(n_taps, dtype=np.float64); center_t = (n_taps - 1) / 2.0
        t = (n - center_t) / sr
        sinc_env = np.sinc(bandwidth * t)
        peak_env = np.cos(2 * np.pi * center * t)
        h = sinc_env * peak_env * gain
        w = 0.42 - 0.5*np.cos(2*np.pi*n/(n_taps-1)) + 0.08*np.cos(4*np.pi*n/(n_taps-1))
        h *= w
        return h

    @staticmethod
    def convolve(x: np.ndarray, h: np.ndarray) -> np.ndarray:
        n, nh = len(x), len(h)
        if nh <= 1: return x * h[0] if nh == 1 else x
        if _HAS_SCIPY and nh > 64: return _scipy_fftconvolve(x, h, mode='full')[:n].astype(np.float32)
        if nh < 64 and n < 2048: return np.convolve(x, h, mode='full')[:n].astype(np.float32)
        MAX_FFT_BITS = 20; total = n + nh - 1
        if total.bit_length() <= MAX_FFT_BITS:
            n_fft = 1 << total.bit_length()
            return np.fft.irfft(np.fft.rfft(x, n_fft) * np.fft.rfft(h, n_fft), n_fft)[:n].astype(np.float32)
        n_fft = 1 << MAX_FFT_BITS; block = n_fft - nh + 1; H = np.fft.rfft(h, n_fft); out = np.zeros(total, dtype=np.float32)
        for i in range(0, n, block):
            end = min(i + block, n); yb = np.fft.irfft(np.fft.rfft(x[i:end], n_fft) * H, n_fft)
            out[i:i + min(len(yb), total - i)] += yb[:min(len(yb), total - i)]
        return out[:n].astype(np.float32)

    @staticmethod
    def filt_lp(x, c, sr=44100, n_taps=127): return DSP.convolve(x, DSP.fir_lowpass(c, sr, n_taps))
    @staticmethod
    def filt_hp(x, c, sr=44100, n_taps=127): return DSP.convolve(x, DSP.fir_highpass(c, sr, n_taps))
    @staticmethod
    def filt_bp(x, l, h, sr=44100, n_taps=255): return DSP.convolve(x, DSP.fir_bandpass(l, h, sr, n_taps))
    @staticmethod
    def filt_peak(x, c, bw, sr=44100, n_taps=255, g=2.0): return DSP.convolve(x, DSP.fir_peak(c, bw, sr, n_taps, g))

    @staticmethod
    def filt_resonator(x, f, q, sr=44100, gain=1.0):
        w0 = 2 * np.pi * f / sr; r = np.exp(-w0 / (2 * max(0.5, q))); wd = w0 * np.sqrt(max(0.0, 1.0 - 1.0 / (4 * q * q))) if q > 0.5 else w0
        n_ir = min(len(x), max(16, int(sr * max(0.005, 6.0 * q / max(1.0, f))))); t = np.arange(n_ir, dtype=np.float64); ir = (r ** t) * np.sin(wd * t)
        p = np.max(np.abs(ir))
        if p > 1e-10: ir = ir / p * gain
        return DSP.convolve(x, ir)

    @staticmethod
    def filt_formant(x, sr, vowel='a'):
        formants = {
            'a': [(700, 1200, 1.0), (1220, 1200, 0.6), (2600, 1200, 0.3)],
            'e': [(530, 800, 1.0), (1840, 800, 0.6), (2480, 800, 0.3)],
            'i': [(270, 400, 1.0), (2290, 400, 0.6), (3010, 400, 0.3)],
            'o': [(570, 600, 1.0), (840, 600, 0.6), (2410, 600, 0.3)],
            'u': [(300, 400, 1.0), (870, 400, 0.6), (2240, 400, 0.3)],
        }
        out = np.zeros_like(x)
        for f, bw, gain in formants.get(vowel, formants['a']):
            out += DSP.filt_bp(x, f - bw/2, f + bw/2, sr, n_taps=255) * gain
        p = np.max(np.abs(out))
        if p > 0: out /= p * 1.5
        return out

    @staticmethod
    def filt_flanger(x, sr, depth=0.002, rate=0.5, feedback=0.3, mix=0.5):
        if len(x) == 0: return x
        n = len(x)
        out = np.zeros(n, dtype=np.float32)
        max_delay = int(sr * depth * 2) + 2
        buf = np.zeros(max_delay, dtype=np.float32)
        idx = 0
        for i in range(n):
            t = i / sr
            delay_samp = int(sr * depth * (1 + np.sin(2 * np.pi * rate * t)))
            delay_samp = min(delay_samp, max_delay - 1)
            delayed = buf[(idx - delay_samp) % max_delay]
            out[i] = (1.0 - mix) * x[i] + mix * delayed
            buf[idx] = x[i] + delayed * feedback
            idx = (idx + 1) % max_delay
        return out

    @staticmethod
    def filt_spring_reverb(x, sr, decay=0.7, damp=3000.0):
        if len(x) == 0: return x
        out = np.copy(x)
        ap_delays = [int(sr * d) for d in [0.005, 0.007, 0.011, 0.013]]
        ap_gains = [0.6, 0.5, 0.4, 0.3]
        for d_len, g in zip(ap_delays, ap_gains):
            if d_len <= 1: continue
            buf = np.zeros(d_len, dtype=np.float32)
            idx = 0
            for i in range(len(x)):
                delayed = buf[idx]
                out[i] = x[i] + g * delayed
                buf[idx] = delayed * decay + x[i]
                idx = (idx + 1) % d_len
        return DSP.filt_lp(out, damp, sr, n_taps=127)

    @staticmethod
    def feedback_delay(x, sr, delay_ms=100.0, feedback=0.4, cutoff=2000.0, mix=0.5):
        if len(x) == 0: return x
        delay_samp = max(1, int(sr * delay_ms / 1000.0))
        out = np.copy(x)
        buf = np.zeros(delay_samp, dtype=np.float32)
        idx = 0
        lp_state = 0.0
        alpha = np.exp(-2 * np.pi * cutoff / sr)
        for i in range(len(x)):
            delayed = buf[idx]
            out[i] = (1.0 - mix) * x[i] + mix * delayed
            fb_val = delayed * feedback
            lp_state = (1 - alpha) * fb_val + alpha * lp_state
            buf[idx] = x[i] + lp_state
            idx = (idx + 1) % delay_samp
        return out

    @staticmethod
    def fx_bitcrush(x, bits=8.0, downsample=2.0):
        if len(x) == 0: return x
        max_ds = max(1.0, downsample)
        idx = np.arange(0, len(x), max_ds).astype(np.int64)
        if len(idx) == 0: return x.copy()
        idx[-1] = min(idx[-1], len(x) - 1)  # prevent out-of-bounds
        ds_x = x[idx]
        levels = max(2, int(2 ** bits))
        crushed = np.round(ds_x * levels) / levels
        out = np.interp(np.arange(len(x), dtype=np.float64), idx, crushed).astype(np.float32)
        return out

    @staticmethod
    def fx_wow_flutter(x, sr, rate=0.5, depth=0.001):
        if len(x) == 0: return x
        n = len(x)
        t = np.arange(n, dtype=np.float64) / sr
        delay_samp = int(sr * depth * 4)
        out = np.zeros(n, dtype=np.float32)
        buf = np.zeros(delay_samp + 2, dtype=np.float32)
        for i in range(n):
            d = depth * (1 + np.sin(2 * np.pi * rate * t[i]) + 0.5 * np.sin(2 * np.pi * rate * 2.3 * t[i]))
            d_samp = min(delay_samp, int(sr * d))
            frac = (sr * d) - d_samp
            idx = i % (delay_samp + 2)
            buf[idx] = x[i]
            out[i] = buf[(idx - d_samp) % (delay_samp + 2)] * (1 - frac) + buf[(idx - d_samp - 1) % (delay_samp + 2)] * frac
        return out

    @staticmethod
    def fx_chorus(x, sr, rate=0.5, depth=0.002, mix=0.5):
        if len(x) == 0: return x
        n = len(x)
        out = np.zeros(n, dtype=np.float32)
        max_delay = int(sr * depth * 2) + 2
        buf = np.zeros(max_delay, dtype=np.float32)
        idx = 0
        for i in range(n):
            t = i / sr
            d1 = int(sr * depth * (1 + np.sin(2 * np.pi * rate * t)))
            d2 = int(sr * depth * (1 + np.sin(2 * np.pi * rate * 1.13 * t + 0.5)))
            d3 = int(sr * depth * (1 + np.sin(2 * np.pi * rate * 0.87 * t + 1.5)))
            v1 = buf[(idx - d1) % max_delay]
            v2 = buf[(idx - d2) % max_delay]
            v3 = buf[(idx - d3) % max_delay]
            out[i] = (1.0 - mix) * x[i] + mix * 0.33 * (v1 + v2 + v3)
            buf[idx] = x[i]
            idx = (idx + 1) % max_delay
        return out

    @staticmethod
    def fx_phaser(x, sr, rate=0.5, depth=0.5, mix=0.5):
        if len(x) == 0: return x
        n = len(x)
        t = np.arange(n) / sr
        lfo = 200.0 + depth * 800.0 * (0.5 + 0.5 * np.sin(2 * np.pi * rate * t))
        out = np.zeros(n, dtype=np.float32)
        y_prev = 0.0
        x_prev = 0.0
        for i in range(n):
            f = lfo[i]
            tan_f = np.tan(np.pi * f / sr)
            a = (tan_f - 1) / (tan_f + 1)
            apf_out = a * x[i] + x_prev - a * y_prev
            x_prev = x[i]
            y_prev = apf_out
            out[i] = (1.0 - mix) * x[i] + mix * apf_out
        return out

    @staticmethod
    def fx_tape_sat(x, drive=1.5, sr=44100):
        sat = np.tanh(x * drive) / max(0.001, np.tanh(drive))
        sat = DSP.filt_lp(sat, 8000, sr, n_taps=63)
        return sat.astype(np.float32)

    @staticmethod
    def fx_foldback(x, threshold=0.8):
        if len(x) == 0: return x
        thresh = max(0.1, threshold)
        out = np.copy(x)
        mask = np.abs(out) > thresh
        out[mask] = thresh - (np.abs(out[mask]) - thresh)
        out[mask] = np.sign(x[mask]) * out[mask]
        p = np.max(np.abs(out))
        if p > 0: out /= p
        return out.astype(np.float32)

    @staticmethod
    def fx_granular(x, sr, grain_size_ms=50.0, overlap=0.5, pitch_var=0.1, pos_var=0.2, rng=None):
        if len(x) == 0: return x
        if rng is None: rng = np.random.RandomState(42)
        n = len(x)
        grain_samp = max(1, int(sr * grain_size_ms / 1000.0))
        hop_samp = max(1, int(grain_samp * (1.0 - overlap)))
        out = np.zeros(n, dtype=np.float32)
        window = np.hanning(grain_samp).astype(np.float32)
        
        num_grains = max(1, n // hop_samp + 1)
        for i in range(num_grains):
            pos = int(i * hop_samp + rng.uniform(-pos_var, pos_var) * hop_samp)
            pos = max(0, min(n - grain_samp, pos))
            pitch_mult = 2 ** rng.uniform(-pitch_var, pitch_var)
            
            indices = np.linspace(0, grain_samp - 1, int(grain_samp / pitch_mult))
            indices = np.clip(indices, 0, grain_samp - 1)
            left = np.floor(indices).astype(int)
            right = np.minimum(left + 1, grain_samp - 1)
            frac = indices - left
            grain = x[pos:pos+grain_samp] * window[:grain_samp]
            stretched = grain[left] * (1 - frac) + grain[right] * frac
            
            out_pos = i * hop_samp
            end_pos = min(n, out_pos + len(stretched))
            out[out_pos:end_pos] += stretched[:end_pos - out_pos]
            
        p = np.max(np.abs(out))
        if p > 0: out /= p
        return out

    @staticmethod
    def sat_tube(x, drive=1.0):
        if drive <= 0: return x
        return (np.tanh(x * drive) / max(0.001, np.tanh(drive))).astype(np.float32)

    @staticmethod
    def sat_hard(x, t=1.0): return np.clip(x, -t, t).astype(np.float32)

    @staticmethod
    def sat_oversampled(x, drive=1.5):
        if drive <= 0: return x
        n = len(x)
        if n < 2: return DSP.sat_tube(x, drive)
        x_up = np.zeros(n * 2, dtype=np.float32); x_up[0::2] = x; x_up[1:-1:2] = (x[:-1] + x[1:]) / 2.0; x_up[-1] = x[-1]; y_up = DSP.sat_tube(x_up, drive); y_down = (y_up[0:-1:2] + y_up[1::2]) / 2.0
        return y_down[:n].astype(np.float32)

    @staticmethod
    def binaural_pan(mono, pan, sr=44100):
        if abs(pan) < 0.01: return mono.copy(), mono.copy()
        if pan > 0:
            left = DSP.filt_hp(mono, 4000, sr, n_taps=127) * (1.0 - pan * 0.6); right = mono * min(1.0, 1.0 + pan * 0.2)
        else:
            right = DSP.filt_hp(mono, 4000, sr, n_taps=127) * (1.0 + pan * 0.6); left = mono * min(1.0, 1.0 - pan * 0.2)
        
        # Haas Effect for extra stereo width
        haas_delay = int(abs(pan) * 0.012 * sr)  # up to 12ms delay
        l_out = np.zeros(len(mono) + haas_delay, dtype=np.float32)
        r_out = np.zeros(len(mono) + haas_delay, dtype=np.float32)
        
        if pan > 0:
            l_out[haas_delay:haas_delay + len(left)] += left
            r_out[:len(mono)] += right
        else:
            r_out[haas_delay:haas_delay + len(right)] += right
            l_out[:len(mono)] += left
        return l_out, r_out

    @staticmethod
    def compressor_rms(x, sr, threshold_db=-12.0, ratio=3.0, attack_ms=10, release_ms=100):
        if threshold_db >= 0 or ratio <= 1: return x.astype(np.float32)
        target_lin = 10 ** (threshold_db / 20); env = np.abs(x).astype(np.float32); ac = np.exp(-1.0 / (sr * (attack_ms / 1000.0))); rc = np.exp(-1.0 / (sr * (release_ms / 1000.0)))
        if _HAS_SCIPY:
            env_a = _scipy_lfilter([1 - ac], [1, -ac], env); env_r = _scipy_lfilter([1 - rc], [1, -rc], env); smooth_env = np.maximum(env_a, env_r).astype(np.float32)
        else: smooth_env = env
        gain = np.ones_like(x, dtype=np.float32); mask = smooth_env > target_lin
        gain[mask] = 1.0 - (1.0 - 1.0 / ratio) * (1.0 - target_lin / smooth_env[mask])
        return (x * gain).astype(np.float32)

    @staticmethod
    def lookahead_limiter(x, sr=44100, threshold_db=-1.0, release_ms=50):
        target_lin = 10 ** (threshold_db / 20); peak = np.max(np.abs(x))
        if peak <= target_lin: return x.astype(np.float32)
        env = np.abs(x).astype(np.float32); gain = np.ones_like(x, dtype=np.float32); mask = env > target_lin; gain[mask] = target_lin / env[mask]
        lookahead = int(sr * 0.002)
        if lookahead > 0 and len(gain) > lookahead: gain = np.roll(gain, -lookahead); gain[-lookahead:] = gain[-lookahead - 1]
        return (x * gain).astype(np.float32)

    @staticmethod
    def dc_block(x, sr=44100): return DSP.filt_hp(x, 20.0, sr, n_taps=63)

    @staticmethod
    def osc_sine(f, n, sr, phase=0.0): return np.sin(2 * np.pi * f * (np.arange(n, dtype=np.float64) / sr) + phase).astype(np.float32)

    @staticmethod
    def osc_triangle(f, n, sr):
        t = np.arange(n, dtype=np.float64) / sr
        return (2 * np.abs(2 * (f * t - np.floor(f * t + 0.5))) - 1.0).astype(np.float32)

    @staticmethod
    def osc_sweep(f1, f2, n, sr):
        if n <= 0: return np.zeros(0, dtype=np.float32)
        if f1 <= 0: f1 = 1.0
        if f2 <= 0: f2 = 1.0
        t = np.arange(n, dtype=np.float64) / sr; freq = f1 * (f2 / f1) ** (t / (n / sr)); return np.sin(2 * np.pi * np.cumsum(freq) / sr).astype(np.float32)

    @staticmethod
    def osc_808(f, n, sr, pitch_drop=0.5):
        t = np.arange(n, dtype=np.float64) / sr
        freq = np.zeros(n, dtype=np.float64)
        drop_len = int(sr * 0.05)
        if drop_len > 0:
            freq[:drop_len] = np.linspace(f, f * (1.0 - pitch_drop), drop_len)
            freq[drop_len:] = f * (1.0 - pitch_drop)
        else:
            freq[:] = f
        phase = 2 * np.pi * np.cumsum(freq) / sr
        return np.sin(phase).astype(np.float32)

    @staticmethod
    def osc_fm(c, mr, mi, n, sr, phase=0.0):
        t = np.arange(n, dtype=np.float64) / sr; return np.sin(2 * np.pi * c * t + mi * np.sin(2 * np.pi * c * mr * t) + phase).astype(np.float32)

    @staticmethod
    def osc_fm_multi(c, mrs, mis, n, sr):
        t = np.arange(n, dtype=np.float64) / sr; ph = 2 * np.pi * c * t
        for mr, mi in zip(mrs, mis): ph += mi * np.sin(2 * np.pi * c * mr * t)
        return np.sin(ph).astype(np.float32)

    @staticmethod
    def osc_fm_fb(c, mr, mi, fb, n, sr):
        if _HAS_NUMBA:
            return _nb_fm_fb(c, mr, mi, fb, n, sr)
        # Fallback if no Numba
        return DSP.osc_fm(c, mr, mi, n, sr)

    @staticmethod
    def osc_fm_dx7(c, m1, m2, i1, i2, fb, n, sr):
        if _HAS_NUMBA:
            return _nb_fm_dx7(c, m1, m2, i1, i2, fb, n, sr)
        # Fallback if no Numba
        return DSP.osc_fm(c, m1, i1, n, sr)

    @staticmethod
    def osc_saw(f, n, sr):
        t = np.arange(n, dtype=np.float64) / sr; sig = np.zeros(n, dtype=np.float64)
        for h in range(1, 17):
            fh = f * h
            if fh >= sr * 0.45: break
            sig += np.sin(2 * np.pi * fh * t) / h
        return (sig * 0.5).astype(np.float32)

    @staticmethod
    def osc_supersaw(f, n, sr, detune=0.1):
        t = np.arange(n, dtype=np.float64) / sr
        sig = np.zeros(n, dtype=np.float64)
        rng = np.random.RandomState(int(f))
        for d in [-2, -1, 0, 1, 2]:
            fd = f * (1.0 + d * detune * 0.01)
            phase = rng.uniform(0, 2*np.pi)
            saw = 2.0 * ((fd * t + phase/(2*np.pi)) - np.floor(0.5 + fd * t + phase/(2*np.pi)))
            sig += saw
        p = np.max(np.abs(sig))
        if p > 0: sig /= p
        return sig.astype(np.float32)

    @staticmethod
    def osc_reese(f, n, sr, detune=0.5):
        t = np.arange(n, dtype=np.float64) / sr
        sig = np.zeros(n, dtype=np.float64)
        for d in [-2, -1, 0, 1, 2]:
            fd = f * (1.0 + d * detune * 0.01)
            saw = 2.0 * ((fd * t) - np.floor(0.5 + fd * t))
            sig += saw
        sig = np.tanh(sig * 2.0)
        p = np.max(np.abs(sig))
        if p > 0: sig /= p
        return sig.astype(np.float32)

    @staticmethod
    def osc_pulse(f, n, sr, pw=0.5):
        t = np.arange(n, dtype=np.float64) / sr
        phase = (f * t) % 1.0
        return (phase < pw).astype(np.float32) * 2.0 - 1.0

    @staticmethod
    def osc_phase_distort(f, n, sr, bend=0.5):
        t = np.arange(n, dtype=np.float64) / sr
        phase = (f * t) % 1.0
        distorted_phase = phase + bend * np.sin(2 * np.pi * phase)
        return np.sin(2 * np.pi * distorted_phase).astype(np.float32)

    @staticmethod
    def osc_ring_mod(f1, f2, n, sr):
        t = np.arange(n, dtype=np.float64) / sr
        return (np.sin(2 * np.pi * f1 * t) * np.sin(2 * np.pi * f2 * t)).astype(np.float32)

    @staticmethod
    def osc_wavetable(f, n, sr, morph_rate=1.0):
        t = np.arange(n, dtype=np.float64) / sr
        phase = (f * t) % 1.0
        morph = (np.sin(2 * np.pi * morph_rate * t) + 1.0) / 2.0
        sine = np.sin(2 * np.pi * phase)
        saw = 2.0 * (phase - np.floor(phase + 0.5))
        square = np.where(phase < 0.5, 1.0, -1.0)
        w1 = np.clip(morph * 2.0, 0, 1)
        w2 = np.clip(morph * 2.0 - 1.0, 0, 1)
        sig = sine * (1 - w1) + saw * (w1 - w2) + square * w2
        return sig.astype(np.float32)

    @staticmethod
    def osc_additive(harmonics, n, sr):
        t = np.arange(n, dtype=np.float64) / sr
        sig = np.zeros(n, dtype=np.float64)
        for i, (f, a) in enumerate(harmonics):
            sig += np.sin(2 * np.pi * f * t) * a
        p = np.max(np.abs(sig))
        if p > 0: sig /= p
        return sig.astype(np.float32)

    @staticmethod
    def osc_vosim(f0, f1, n, sr, decay=5.0, width=0.3):
        t = np.arange(n, dtype=np.float64) / sr
        T0 = 1.0 / f0
        phase0 = (t * f0) % 1.0
        win = np.exp(-decay * phase0)
        win = np.where(phase0 < width, win, 0.0)
        sig = np.sin(2 * np.pi * f1 * t) * win
        p = np.max(np.abs(sig))
        if p > 0: sig /= p
        return sig.astype(np.float32)

    @staticmethod
    def osc_pulsar(f, n, sr, pulse_width=0.1, rng=None):
        if rng is None: rng = np.random.RandomState(int(f))
        t = np.arange(n, dtype=np.float64) / sr
        phase = (f * t) % 1.0
        mask = (phase < pulse_width).astype(np.float32)
        f_jitter = f * (2 ** rng.uniform(-0.02, 0.02, n))
        phase_jit = 2 * np.pi * np.cumsum(f_jitter) / sr
        sig = np.sin(phase_jit) * mask
        p = np.max(np.abs(sig))
        if p > 0: sig /= p
        return sig.astype(np.float32)

    @staticmethod
    def karplus_strong(f, n, sr, decay=0.99, brightness=0.5, exciter=None):
        if exciter is None:
            exciter = np.random.randn(max(1, int(sr / f))).astype(np.float32)
        if _HAS_NUMBA:
            return _nb_karplus_strong(f, n, sr, decay, brightness, exciter)
        if f <= 0: return np.zeros(n, dtype=np.float32)
        delay_len = max(1, int(sr / f))
        buf = np.zeros(delay_len, dtype=np.float32)
        for i in range(delay_len):
            if i < len(exciter): buf[i] = exciter[i]
        out = np.zeros(n, dtype=np.float32)
        filt_coef = 1.0 - brightness
        for i in range(n):
            idx = i % delay_len
            next_idx = (i + 1) % delay_len
            out[i] = buf[idx]
            buf[idx] = decay * (buf[idx] * filt_coef + buf[next_idx] * (1 - filt_coef))
        p = np.max(np.abs(out))
        if p > 0: out /= p
        return out

    @staticmethod
    def noise_white(n, rng): return rng.randn(n).astype(np.float32)

    @staticmethod
    def noise_pink(n, rng, sr=44100):
        bs = min(n, sr * 5); white = rng.randn(bs).astype(np.float32); fft = np.fft.rfft(white); freqs = np.fft.rfftfreq(bs, 1.0 / sr)
        if len(freqs) > 1: freqs[0] = max(freqs[1], 1.0); fft[1:] /= np.sqrt(np.abs(freqs[1:])); fft[0] = 0
        pink = np.fft.irfft(fft, n=bs).astype(np.float32); p = np.max(np.abs(pink))
        if p > 1e-6: pink /= p
        reps, rem = n // bs, n % bs; out = np.tile(pink, reps) if reps > 0 else np.array([], dtype=np.float32)
        if rem > 0: out = np.concatenate([out, pink[:rem]])
        return out

    @staticmethod
    def noise_brown(n, rng):
        white = rng.randn(n).astype(np.float32); cum = np.cumsum(white); brown = cum - np.mean(cum); p = np.max(np.abs(brown))
        if p > 0: brown = brown / p
        return (brown * 0.5).astype(np.float32)

    @staticmethod
    def noise_blue(n, rng, sr=44100):
        bs = min(n, sr * 2)
        white = rng.randn(bs).astype(np.float32)
        fft = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(bs, 1.0 / sr)
        if len(freqs) > 1:
            fft[1:] *= np.sqrt(freqs[1:]) 
            fft[0] = 0
        blue = np.fft.irfft(fft, n=bs).astype(np.float32)
        p = np.max(np.abs(blue))
        if p > 1e-6: blue /= p
        reps, rem = n // bs, n % bs
        out = np.tile(blue, reps) if reps > 0 else np.array([], dtype=np.float32)
        if rem > 0: out = np.concatenate([out, blue[:rem]])
        return out

    @staticmethod
    def modal(modes, n, sr, inharmonicity=0.0, exciter=None):
        if exciter is None:
            ex_len = max(1, int(sr * 0.002))
            exciter = np.random.randn(ex_len).astype(np.float32)
        out = np.zeros(n, dtype=np.float32)
        f0 = modes[0][0] if modes else 440.0
        for freq, decay, amp in modes:
            af = freq
            if inharmonicity > 0: nh = freq / f0; af = freq * np.sqrt(1 + inharmonicity * nh * nh)
            t = np.arange(n, dtype=np.float64) / sr
            ir = (np.sin(2 * np.pi * af * t) * np.exp(-decay * t)).astype(np.float32)
            mode_sig = DSP.convolve(exciter, ir)[:n]
            out += mode_sig * amp
        p = np.max(np.abs(out))
        if p > 1e-6: out /= p
        return out

    @staticmethod
    def normalize(x, target_db=-1.0):
        p = np.max(np.abs(x)) if len(x) > 0 else 0.0
        if p < 1e-10: return x
        return x * (10 ** (target_db / 20) / p)

    @staticmethod
    def to_int16(x, target_db=-1.0):
        x = DSP.normalize(x, target_db); x = DSP.sat_hard(x, 1.0); np.nan_to_num(x, copy=False, nan=0.0, posinf=1.0, neginf=-1.0); return np.clip(x * 32767, -32768, 32767).astype(np.int16)

    @staticmethod
    def normalize_loudness(x, sr, target_lufs=-14.0, true_peak_db=-1.0): return _normalize_to_lufs(x, sr, target_lufs, true_peak_db)

    @staticmethod
    def master_mono(x, sr=44100, target_db=-0.1):
        if len(x) == 0: return x
        x = DSP.dc_block(x, sr); x = DSP.filt_hp(x, 25.0, sr, n_taps=127)
        sub = DSP.filt_bp(x, 40, 120, sr, n_taps=255); sub = DSP.sat_tube(sub * 1.4, 1.3); x = x + sub * 0.25
        x = x + DSP.filt_lp(x, 150, sr, n_taps=127) * 0.12
        x = x - DSP.filt_bp(x, 250, 500, sr, n_taps=255) * 0.12
        x = x + DSP.filt_hp(x, 10000, sr, n_taps=63) * 0.20
        x = DSP.compressor_rms(x, sr, threshold_db=-22.0, ratio=1.8, attack_ms=30, release_ms=220)
        x = DSP.compressor_rms(x, sr, threshold_db=-12.0, ratio=2.5, attack_ms=8, release_ms=100)
        x = DSP.sat_oversampled(x, drive=1.3)
        return np.clip(DSP.lookahead_limiter(x, sr, threshold_db=target_db), -1.0, 1.0).astype(np.float32)

    @staticmethod
    def master_stereo_asmr(mix_l, mix_r, sr=44100, target_db=-1.0):
        if len(mix_l) == 0: return mix_l, mix_r
        mix_l = DSP.dc_block(mix_l, sr); mix_r = DSP.dc_block(mix_r, sr)
        mix_l = DSP.filt_hp(mix_l, 25.0, sr, n_taps=127); mix_r = DSP.filt_hp(mix_r, 25.0, sr, n_taps=127)

        mid = (mix_l + mix_r) * 0.5; side = (mix_l - mix_r) * 0.5
        mid = DSP.filt_lp(mid, 10000, sr, n_taps=127) 
        side = DSP.filt_hp(side, 2000, sr, n_taps=127) 
        mix_l = mid + side; mix_r = mid - side

        sub_l = DSP.filt_bp(mix_l, 40, 120, sr, n_taps=255); sub_r = DSP.filt_bp(mix_r, 40, 120, sr, n_taps=255)
        sub_l = DSP.sat_tube(sub_l * 1.4, 1.3); sub_r = DSP.sat_tube(sub_r * 1.4, 1.3)
        mix_l = mix_l + sub_l * 0.25; mix_r = mix_r + sub_r * 0.25

        air_l = DSP.filt_hp(mix_l, 15000, sr, n_taps=63); air_r = DSP.filt_hp(mix_r, 15000, sr, n_taps=63)
        mix_l = mix_l + air_l * 0.15; mix_r = mix_r + air_r * 0.15

        ess_l = DSP.filt_bp(mix_l, 5000, 8500, sr, n_taps=255); ess_r = DSP.filt_bp(mix_r, 5000, 8500, sr, n_taps=255)
        ess_env_l = np.abs(ess_l).astype(np.float32); ess_env_r = np.abs(ess_r).astype(np.float32); ess_thresh = 0.12
        ess_gain_l = np.where(ess_env_l > ess_thresh, ess_thresh / np.maximum(ess_env_l, 1e-10), 1.0).astype(np.float32)
        ess_gain_r = np.where(ess_env_r > ess_thresh, ess_thresh / np.maximum(ess_env_r, 1e-10), 1.0).astype(np.float32)
        if _HAS_SCIPY: ess_gain_l = _scipy_lfilter([1.0], [1.0, -0.99], ess_gain_l).astype(np.float32); ess_gain_r = _scipy_lfilter([1.0], [1.0, -0.99], ess_gain_r).astype(np.float32)
        mix_l = mix_l - ess_l * (1.0 - ess_gain_l) * 0.4; mix_r = mix_r - ess_r * (1.0 - ess_gain_r) * 0.4

        mix_l = DSP.sat_oversampled(mix_l, drive=1.25); mix_r = DSP.sat_oversampled(mix_r, drive=1.25)

        mix_l = DSP.compressor_rms(mix_l, sr, threshold_db=-24.0, ratio=1.6, attack_ms=40, release_ms=250); mix_r = DSP.compressor_rms(mix_r, sr, threshold_db=-24.0, ratio=1.6, attack_ms=40, release_ms=250)
        mix_l = DSP.compressor_rms(mix_l, sr, threshold_db=-12.0, ratio=2.5, attack_ms=8, release_ms=100); mix_r = DSP.compressor_rms(mix_r, sr, threshold_db=-12.0, ratio=2.5, attack_ms=8, release_ms=100)

        mix_l = DSP.lookahead_limiter(mix_l, sr, threshold_db=target_db, release_ms=80); mix_r = DSP.lookahead_limiter(mix_r, sr, threshold_db=target_db, release_ms=80)
        return (np.clip(mix_l, -1.0, 1.0).astype(np.float32), np.clip(mix_r, -1.0, 1.0).astype(np.float32))

@dataclass(slots=True)
class Voice:
    osc: str = "sine"; freq: float = 440.0; freq_end: float = 0.0; fm_ratio: float = 1.0; fm_index: float = 0.0; fm_ratios: Optional[List[float]] = None; fm_indices: Optional[List[float]] = None
    attack: float = 0.001; decay_rate: float = 30.0; attack_curve: float = 2.0; filt: str = "none"; f_cutoff: float = 8000.0; f_high: float = 12000.0; f_q: float = 1.0; drive: float = 0.0; gain: float = 0.5; delay: float = 0.0
    modes: Optional[List[Tuple[float, float, float]]] = None; inharmonicity: float = 0.0
    delay_ms: float = 0.0; feedback: float = 0.0; delay_mix: float = 0.0
    vowel: str = 'a'
    bend: float = 0.5
    f2: float = 0.0
    depth: float = 0.0
    rate: float = 0.0
    bits: float = 8.0
    downsample: float = 2.0
    harmonics: Optional[List[Tuple[float, float]]] = None
    width: float = 0.3
    pitch_var: float = 0.1
    pos_var: float = 0.2
    grain_size_ms: float = 50.0
    overlap: float = 0.5
    m1: float = 0.0
    m2: float = 0.0
    i1: float = 0.0
    i2: float = 0.0
    fb: float = 0.0
    threshold: float = 0.8

class SoundRenderer:
    __slots__ = ("sr",)
    def __init__(self, sr: int = 44100): self.sr = sr
    def render(self, voices: List[Voice], duration: float, rng: np.random.RandomState) -> np.ndarray:
        n = int(self.sr * duration)
        if n <= 0: return np.zeros(0, dtype=np.float32)
        mix = np.zeros(n, dtype=np.float32)
        for v in voices:
            sig = self._render_voice(v, n, rng); d_samp = int(v.delay * self.sr)
            if d_samp > 0:
                if d_samp >= n: continue
                sig = np.concatenate([np.zeros(d_samp, dtype=np.float32), sig[:n - d_samp]])
            if len(sig) < n: sig = np.concatenate([sig, np.zeros(n - len(sig), dtype=np.float32)])
            else: sig = sig[:n]
            mix += sig
        return DSP.dc_block(mix, self.sr)

    def _render_voice(self, v: Voice, n: int, rng: np.random.RandomState) -> np.ndarray:
        match v.osc:
            case "sine": sig = DSP.osc_sine(v.freq, n, self.sr)
            case "triangle": sig = DSP.osc_triangle(v.freq, n, self.sr)
            case "sweep": sig = DSP.osc_sweep(v.freq, v.freq_end, n, self.sr)
            case "808": sig = DSP.osc_808(v.freq, n, self.sr, v.pitch_drop if hasattr(v, 'pitch_drop') else 0.5)
            case "fm": sig = DSP.osc_fm_multi(v.freq, v.fm_ratios, v.fm_indices, n, self.sr) if v.fm_ratios else DSP.osc_fm(v.freq, v.fm_ratio, v.fm_index, n, self.sr)
            case "fm_fb": sig = DSP.osc_fm_fb(v.freq, v.fm_ratio, v.fm_index, v.feedback, n, self.sr)
            case "fm_dx7": sig = DSP.osc_fm_dx7(v.freq, v.m1, v.m2, v.i1, v.i2, v.fb, n, self.sr)
            case "noise_white": sig = DSP.noise_white(n, rng)
            case "noise_pink": sig = DSP.noise_pink(n, rng, self.sr)
            case "noise_brown": sig = DSP.noise_brown(n, rng)
            case "noise_blue": sig = DSP.noise_blue(n, rng, self.sr)
            case "modal":
                ex_len = max(1, int(self.sr * 0.002))
                exciter = rng.randn(ex_len).astype(np.float32)
                sig = DSP.modal(v.modes, n, self.sr, v.inharmonicity, exciter) if v.modes else np.zeros(n, dtype=np.float32)
            case "saw": sig = DSP.osc_saw(v.freq, n, self.sr)
            case "supersaw": sig = DSP.osc_supersaw(v.freq, n, self.sr, v.attack_curve)
            case "reese": sig = DSP.osc_reese(v.freq, n, self.sr, v.attack_curve)
            case "pulse": sig = DSP.osc_pulse(v.freq, n, self.sr, v.f_q)
            case "phase_distort": sig = DSP.osc_phase_distort(v.freq, n, self.sr, v.bend)
            case "ring_mod": sig = DSP.osc_ring_mod(v.freq, v.f2, n, self.sr)
            case "wavetable": sig = DSP.osc_wavetable(v.freq, n, self.sr, v.rate)
            case "additive":
                if v.harmonics:
                    t = np.arange(n, dtype=np.float64) / self.sr
                    sig = np.zeros(n, dtype=np.float64)
                    for i, (f, a) in enumerate(v.harmonics):
                        freq = f * (v.freq_end / f) ** (t / (n / self.sr)) if v.freq_end > 0 else f
                        sig += np.sin(2 * np.pi * np.cumsum(freq) / self.sr) * a
                    p = np.max(np.abs(sig))
                    if p > 0: sig /= p
                    sig = sig.astype(np.float32)
                else: sig = np.zeros(n, dtype=np.float32)
            case "vosim": sig = DSP.osc_vosim(v.freq, v.f2, n, self.sr, v.decay_rate, v.width)
            case "pulsar": sig = DSP.osc_pulsar(v.freq, n, self.sr, v.f_q, rng)
            case "karplus":
                ex_len = max(1, int(self.sr / max(1.0, v.freq)))
                exciter = rng.randn(ex_len).astype(np.float32)
                sig = DSP.karplus_strong(v.freq, n, self.sr, decay=v.decay_rate if v.decay_rate > 0 else 0.99, brightness=v.attack_curve, exciter=exciter)
            case _: sig = np.zeros(n, dtype=np.float32)
        sig *= DSP.env_exp(n, self.sr, attack=v.attack, decay_rate=v.decay_rate, attack_curve=v.attack_curve)
        match v.filt:
            case "lp": sig = DSP.filt_lp(sig, v.f_cutoff, self.sr)
            case "hp": sig = DSP.filt_hp(sig, v.f_cutoff, self.sr)
            case "bp": sig = DSP.filt_bp(sig, v.f_cutoff, v.f_high, self.sr)
            case "peak": sig = sig + DSP.filt_peak(sig, v.f_cutoff, v.f_high, self.sr) * 0.5
            case "resonator": sig = DSP.filt_resonator(sig, v.f_cutoff, v.f_q, self.sr)
            case "formant": sig = DSP.filt_formant(sig, self.sr, v.vowel)
            case "delay": sig = DSP.feedback_delay(sig, self.sr, delay_ms=v.delay_ms, feedback=v.feedback, cutoff=v.f_q, mix=v.delay_mix)
            case "flanger": sig = DSP.filt_flanger(sig, self.sr, depth=v.depth, rate=v.rate, feedback=v.feedback, mix=v.delay_mix)
            case "chorus": sig = DSP.fx_chorus(sig, self.sr, rate=v.rate, depth=v.depth, mix=v.delay_mix)
            case "phaser": sig = DSP.fx_phaser(sig, self.sr, rate=v.rate, depth=v.depth, mix=v.delay_mix)
            case "spring_reverb": sig = DSP.filt_spring_reverb(sig, self.sr, decay=v.feedback, damp=v.f_cutoff)
            case "bitcrush": sig = DSP.fx_bitcrush(sig, bits=v.bits, downsample=v.downsample)
            case "wow_flutter": sig = DSP.fx_wow_flutter(sig, self.sr, rate=v.rate, depth=v.depth)
            case "tape_sat": sig = DSP.fx_tape_sat(sig, drive=v.drive, sr=self.sr)
            case "foldback": sig = DSP.fx_foldback(sig, threshold=v.threshold)
            case "granular": sig = DSP.fx_granular(sig, self.sr, grain_size_ms=v.grain_size_ms, overlap=v.overlap, pitch_var=v.pitch_var, pos_var=v.pos_var, rng=rng)
        if v.drive > 0 and v.filt != "tape_sat": sig = DSP.sat_tube(sig, v.drive)
        return sig * v.gain

class VoiceFactory:
    @staticmethod
    def transient_tick(freq, gain=0.5, delay=0.0, rng=None):
        if rng: freq = freq * (2 ** rng.uniform(-0.05, 0.05))
        # Mixed blue noise with a fast sine sweep for a micro-zap
        return Voice(osc="noise_blue", attack=0.00001, decay_rate=1000.0, attack_curve=3.0, filt="hp", f_cutoff=4000, drive=1.5, gain=gain, delay=delay)
    @staticmethod
    def click_noise(freq, gain=0.5, delay=0.0, rng=None):
        if rng: freq = freq * (2 ** rng.uniform(-0.05, 0.05))
        return Voice(osc="noise_white", attack=0.0001, decay_rate=300.0, attack_curve=3.0, filt="bp", f_cutoff=freq*0.7, f_high=freq*1.8, drive=1.8, gain=gain*0.8, delay=delay)
    @staticmethod
    def dirt(gain=0.05, delay=0.0, rng=None):
        return Voice(osc="noise_blue", attack=0.00002, decay_rate=2000.0, attack_curve=3.0, filt="bp", f_cutoff=6000, f_high=9000, drive=1.0, gain=gain, delay=delay)
    @staticmethod
    def tactile_bump(freq, gain=0.3, delay=0.0, rng=None):
        if rng: freq = freq * (2 ** rng.uniform(-0.02, 0.02))
        return Voice(osc="noise_pink", attack=0.0005, decay_rate=150.0, attack_curve=1.5, filt="bp", f_cutoff=freq*0.5, f_high=freq*1.5, drive=1.2, gain=gain, delay=delay)
    @staticmethod
    def spring_rattle(freq, gain=0.1, delay=0.0, rng=None):
        if rng: freq = freq * (2 ** rng.uniform(-0.05, 0.05))
        return Voice(osc="noise_white", attack=0.0002, decay_rate=200.0, attack_curve=1.5, filt="resonator", f_cutoff=freq*2.0, f_q=5.0, drive=1.0, gain=gain, delay=delay)
    @staticmethod
    def fm(freq, fm_idx=0.4, gain=0.35, cutoff=3500, delay=0.0, rng=None):
        if rng: freq = freq * (2 ** rng.uniform(-0.02, 0.02))
        return Voice(osc="fm", freq=freq, fm_ratio=2.0, fm_index=fm_idx, attack=0.0003, decay_rate=80.0, attack_curve=2.0, filt="lp", f_cutoff=cutoff, drive=1.2, gain=gain, delay=delay)
    @staticmethod
    def fm_fb(freq, fm_idx=2.0, fb=0.5, gain=0.35, cutoff=3500, delay=0.0, rng=None):
        if rng: freq = freq * (2 ** rng.uniform(-0.02, 0.02))
        return Voice(osc="fm_fb", freq=freq, fm_ratio=1.0, fm_index=fm_idx, feedback=fb, attack=0.001, decay_rate=10.0, attack_curve=2.0, filt="lp", f_cutoff=cutoff, drive=1.0, gain=gain, delay=delay)
    @staticmethod
    def housing(freq, gain=0.30, delay=0.0, rng=None):
        if rng: freq = freq * (2 ** rng.uniform(-0.01, 0.01))
        return Voice(osc="modal", modes=[(freq, 30.0, 1.0), (freq * 1.5, 45.0, 0.3), (freq * 2.1, 60.0, 0.1)], attack=0.0008, decay_rate=0.0, attack_curve=1.5, filt="lp", f_cutoff=3000, gain=gain, delay=delay)
    @staticmethod
    def thock(freq, gain=0.45, delay=0.002, rng=None):
        if rng: freq = freq * (2 ** rng.uniform(-0.01, 0.01))
        # Blended sweep with a low sine wave for massive physical weight
        return Voice(osc="sweep", freq=freq*1.1, freq_end=freq*0.9, attack=0.001, decay_rate=50.0, attack_curve=2.5, filt="lp", f_cutoff=max(400.0, freq * 3.5), gain=gain, delay=delay)
    @staticmethod
    def sub(freq, gain=0.30, delay=0.003, rng=None):
        if rng: freq = freq * (2 ** rng.uniform(-0.005, 0.005))
        return Voice(osc="sine", freq=freq, attack=0.002, decay_rate=30.0, attack_curve=2.0, filt="lp", f_cutoff=250, gain=gain, delay=delay)
    @staticmethod
    def sub_808(freq, gain=0.60, delay=0.0, rng=None):
        if rng: freq = freq * (2 ** rng.uniform(-0.005, 0.005))
        return Voice(osc="808", freq=freq, attack=0.005, decay_rate=20.0, attack_curve=0.5, filt="lp", f_cutoff=200, gain=gain, delay=delay)
    @staticmethod
    def topre_dome(freq, gain=0.45, delay=0.0, rng=None):
        if rng: freq = freq * (2 ** rng.uniform(-0.01, 0.01))
        return Voice(osc="triangle", freq=freq, attack=0.001, decay_rate=60.0, attack_curve=2.0, filt="lp", f_cutoff=max(400.0, freq * 3.0), drive=1.5, gain=gain, delay=delay)
    @staticmethod
    def metallic_pipe(freq, gain=0.45, delay=0.001, rng=None):
        if rng: freq = freq * (2 ** rng.uniform(-0.05, 0.05))
        return Voice(osc="karplus", freq=freq, decay_rate=0.999, attack_curve=0.1, gain=gain, delay=delay)
    @staticmethod
    def glass_chime(freq, gain=0.45, delay=0.002, rng=None):
        if rng: freq = freq * (2 ** rng.uniform(-0.01, 0.01))
        return Voice(osc="modal", modes=[(freq, 8.0, 1.0), (freq*2.76, 5.0, 0.5), (freq*5.40, 3.0, 0.25), (freq*8.93, 2.0, 0.1)], attack=0.001, decay_rate=0.0, attack_curve=2.0, filt="lp", f_cutoff=8000, gain=gain, delay=delay)
    @staticmethod
    def vocal_whisper(gain=0.40, delay=0.0, rng=None):
        return Voice(osc="noise_pink", attack=0.005, decay_rate=60.0, attack_curve=1.5, filt="formant", f_cutoff=500, f_high=120, f_q=0, drive=0.0, gain=gain, delay=delay, vowel='a')
    @staticmethod
    def vosim_voice(f0, f1, gain=0.40, delay=0.0, rng=None):
        return Voice(osc="vosim", freq=f0, f2=f1, attack=0.005, decay_rate=10.0, attack_curve=2.0, width=0.3, filt="lp", f_cutoff=4000, gain=gain, delay=delay)
    @staticmethod
    def reese_bass(freq, gain=0.55, delay=0.0, rng=None):
        if rng: freq = freq * (2 ** rng.uniform(-0.005, 0.005))
        return Voice(osc="reese", freq=freq, attack=0.01, decay_rate=20.0, attack_curve=0.5, filt="lp", f_cutoff=400, drive=1.5, gain=gain, delay=delay)
    @staticmethod
    def rattle(lo, hi, gain=0.08, delay=0.002): return Voice(osc="noise_white", attack=0.0005, decay_rate=100.0, attack_curve=1.0, filt="bp", f_cutoff=lo, f_high=hi, drive=1.5, gain=gain, delay=delay)
    @staticmethod
    def water_drop(freq, gain=0.45, delay=0.0, rng=None):
        harmonics = [(freq, 1.0), (freq*1.5, 0.4), (freq*2.0, 0.2), (freq*3.0, 0.1)]
        return Voice(osc="additive", harmonics=harmonics, freq=harmonics[0][0], freq_end=harmonics[0][0]*0.5, attack=0.001, decay_rate=80.0, attack_curve=2.0, filt="spring_reverb", feedback=0.6, f_cutoff=5000, gain=gain, delay=delay)
    @staticmethod
    def piano_hammer(freq, gain=0.35, delay=0.0, rng=None):
        return Voice(osc="noise_white", attack=0.0001, decay_rate=300.0, attack_curve=1.0, filt="bp", f_cutoff=2000, f_high=4000, drive=1.0, gain=gain, delay=delay)
    @staticmethod
    def dx7_epiano(freq, gain=0.55, delay=0.0, rng=None):
        if rng: freq = freq * (2 ** rng.uniform(-0.01, 0.01))
        return Voice(osc="fm_dx7", freq=freq, m1=freq*1.0, m2=freq*14.0, i1=2.0, i2=1.5, fb=0.5, attack=0.001, decay_rate=15.0, attack_curve=1.5, filt="lp", f_cutoff=6000, gain=gain, delay=delay)
_VF = VoiceFactory

# The 10 elite presets
SOUND_PRESETS: Dict[str, Dict[str, str]] = {
    "Mechanical": {"desc": "Cherry MX Blue: sharp click jacket, tactile bump, deep housing resonance, and microscopic dirt."},
    "ASMR Deep Thock": {"desc": "Ultra-deep sub-bass thock with soft pink noise, optimized for brain tingles."},
    "Heavenly ASMR": {"desc": "Angellic, wide-stereo glass chimes paired with soft sub-bass. Processed with tape saturation."},
    "Lo-Fi Chill": {"desc": "Warm, heavily tape-saturated, bitcrushed, and wow/flutter keys for relaxing coding."},
    "Neuro Bass": {"desc": "Aggressive Reese bass and dense sub-harmonic impacts for dark, intense coding."},
    "Retro 80s E.Piano": {"desc": "True DX7 6-operator FM synthesis matrix for a classic 1980s electric piano."},
    "Water Drop ASMR": {"desc": "Additive synthesis of a water droplet with pitch drop and lush spring reverb tail."},
    "Cyberpunk UI": {"desc": "Analog foldback distortion and metallic ring modulation for harsh digital sci-fi transients."},
    "Vocal Whisper": {"desc": "Formant-filtered pink noise sounding like a soft human whisper. Highly ASMR-inducing."},
    "Goblin Cave": {"desc": "Dripping wet cave ambience with VOSIM alien chatter and Karplus-Strong drops."}
}

CLICK_DURATIONS: Dict[str, float] = {k: 0.12 for k in SOUND_PRESETS.keys()}
CLICK_DURATIONS["ASMR Deep Thock"] = 0.18
CLICK_DURATIONS["Heavenly ASMR"] = 0.22
CLICK_DURATIONS["Neuro Bass"] = 0.22
CLICK_DURATIONS["Vocal Whisper"] = 0.15
CLICK_DURATIONS["Water Drop ASMR"] = 0.30
CLICK_DURATIONS["Retro 80s E.Piano"] = 0.35
CLICK_DURATIONS["Goblin Cave"] = 0.25

def _mk_click(sr, dur, seed, voices):
    rng = np.random.RandomState(seed); R = SoundRenderer(sr)
    return DSP.to_int16(R.render(voices, dur, rng))

_PRESET_FACTORIES: Dict[str, Dict[str, Callable]] = {
    "Mechanical": {
        "click": lambda sr,d,s: _mk_click(sr,d,s,[_VF.transient_tick(4000, 0.45, rng=np.random.RandomState(s)), _VF.click_noise(4000, 0.45, rng=np.random.RandomState(s)), _VF.dirt(0.05, 0.0005, rng=np.random.RandomState(s)), _VF.tactile_bump(2000, 0.20, 0.0005, rng=np.random.RandomState(s)), _VF.spring_rattle(4000, 0.05, 0.001, rng=np.random.RandomState(s)), _VF.housing(400, 0.30, 0.001, rng=np.random.RandomState(s)), _VF.thock(180, 0.55, 0.002, rng=np.random.RandomState(s)), _VF.sub(80, 0.35, 0.003, rng=np.random.RandomState(s))]), 
        "space": lambda sr,d,s: _mk_click(sr,0.12,s,[_VF.transient_tick(2500, 0.40, rng=np.random.RandomState(s)), _VF.click_noise(2500, 0.40, rng=np.random.RandomState(s)), _VF.dirt(0.04, 0.0005, rng=np.random.RandomState(s)), _VF.thock(120, 0.60, 0.002, rng=np.random.RandomState(s)), _VF.sub(60, 0.40, 0.003, rng=np.random.RandomState(s))]), 
        "enter": lambda sr,d,s: _mk_click(sr,0.14,s,[_VF.transient_tick(2000, 0.45, rng=np.random.RandomState(s)), _VF.click_noise(2000, 0.45, rng=np.random.RandomState(s)), _VF.dirt(0.04, 0.0005, rng=np.random.RandomState(s)), _VF.thock(100, 0.65, 0.002, rng=np.random.RandomState(s)), _VF.sub(50, 0.45, 0.003, rng=np.random.RandomState(s))]),
        "release": lambda sr,d,s: _mk_click(sr,0.06,s,[_VF.transient_tick(3500, 0.25, rng=np.random.RandomState(s)), _VF.click_noise(3500, 0.20, rng=np.random.RandomState(s)), _VF.dirt(0.03, 0.0005, rng=np.random.RandomState(s)), _VF.spring_rattle(3500, 0.05, 0.001, rng=np.random.RandomState(s)), _VF.housing(300, 0.15, 0.001, rng=np.random.RandomState(s))])
    },
    "ASMR Deep Thock": {
        "click": lambda sr,d,s: _mk_click(sr,d,s,[_VF.thock(120, 0.75, 0.001, rng=np.random.RandomState(s)), _VF.sub(55, 0.55, 0.002, rng=np.random.RandomState(s)), Voice(osc="noise_pink",attack=0.005,decay_rate=40.0,filt="lp",f_cutoff=800,gain=0.15), _VF.housing(200, 0.20, 0.003, rng=np.random.RandomState(s))]), 
        "space": lambda sr,d,s: _mk_click(sr,0.14,s,[_VF.thock(90, 0.80, 0.001, rng=np.random.RandomState(s)), _VF.sub(40, 0.60, 0.002, rng=np.random.RandomState(s))]), 
        "enter": lambda sr,d,s: _mk_click(sr,0.18,s,[_VF.thock(70, 0.85, 0.001, rng=np.random.RandomState(s)), _VF.sub(30, 0.65, 0.002, rng=np.random.RandomState(s))]),
        "release": lambda sr,d,s: _mk_click(sr,0.06,s,[_VF.housing(150, 0.15, 0.002, rng=np.random.RandomState(s)), _VF.sub(40, 0.10, 0.002, rng=np.random.RandomState(s))])
    },
    "Heavenly ASMR": {
        "click": lambda sr,d,s: _mk_click(sr,d,s,[_VF.fm_fb(800, 2.0, 0.5, 0.35, 0.002, rng=np.random.RandomState(s)), _VF.sub(100, 0.45, 0.001, rng=np.random.RandomState(s))]), 
        "space": lambda sr,d,s: _mk_click(sr,0.16,s,[_VF.fm_fb(500, 2.0, 0.5, 0.30, 0.002, rng=np.random.RandomState(s)), _VF.sub(70, 0.55, 0.001, rng=np.random.RandomState(s))]), 
        "enter": lambda sr,d,s: _mk_click(sr,0.22,s,[_VF.fm_fb(600, 2.0, 0.5, 0.25, 0.002, rng=np.random.RandomState(s)), _VF.sub(60, 0.60, 0.001, rng=np.random.RandomState(s))]),
        "release": lambda sr,d,s: _mk_click(sr,0.08,s,[_VF.fm_fb(700, 2.0, 0.5, 0.15, 0.002, rng=np.random.RandomState(s))])
    },
    "Lo-Fi Chill": {
        "click": lambda sr,d,s: _mk_click(sr,d,s,[_VF.thock(150, 0.55, 0.004, rng=np.random.RandomState(s)), Voice(osc="sine", freq=100, attack=0.005, decay_rate=15.0, filt="tape_sat", drive=1.5, gain=0.25), Voice(osc="noise_pink",attack=0.005,decay_rate=15.0,filt="chorus", rate=0.2, depth=0.005, delay_mix=0.5, f_cutoff=500,gain=0.25)]), 
        "space": lambda sr,d,s: _mk_click(sr,0.14,s,[_VF.thock(100, 0.60, 0.004, rng=np.random.RandomState(s)), Voice(osc="noise_pink",attack=0.005,decay_rate=10.0,filt="bitcrush", bits=8.0, downsample=2.0, gain=0.15)]), 
        "enter": lambda sr,d,s: _mk_click(sr,0.18,s,[_VF.thock(80, 0.65, 0.004, rng=np.random.RandomState(s)), Voice(osc="noise_pink",attack=0.005,decay_rate=10.0,filt="bitcrush", bits=6.0, downsample=3.0, gain=0.10)]),
        "release": lambda sr,d,s: _mk_click(sr,0.06,s,[_VF.thock(140, 0.20, 0.004, rng=np.random.RandomState(s)), Voice(osc="noise_pink",attack=0.005,decay_rate=10.0,filt="lp",f_cutoff=400,gain=0.10)])
    },
    "Neuro Bass": {
        "click": lambda sr,d,s: _mk_click(sr,d,s,[_VF.reese_bass(100, 0.55, 0.0, rng=np.random.RandomState(s)), _VF.sub_808(80, 0.60, 0.0, rng=np.random.RandomState(s)), _VF.transient_tick(3000, 0.30, rng=np.random.RandomState(s))]), 
        "space": lambda sr,d,s: _mk_click(sr,0.16,s,[_VF.reese_bass(80, 0.60, 0.0, rng=np.random.RandomState(s)), _VF.sub_808(60, 0.65, 0.0, rng=np.random.RandomState(s))]), 
        "enter": lambda sr,d,s: _mk_click(sr,0.22,s,[_VF.reese_bass(60, 0.65, 0.0, rng=np.random.RandomState(s)), _VF.sub_808(50, 0.70, 0.0, rng=np.random.RandomState(s))]),
        "release": lambda sr,d,s: _mk_click(sr,0.08,s,[_VF.sub(50, 0.15, 0.0, rng=np.random.RandomState(s))])
    },
    "Retro 80s E.Piano": {
        "click": lambda sr,d,s: _mk_click(sr,d,s,[_VF.piano_hammer(2000, 0.25, rng=np.random.RandomState(s)), _VF.dx7_epiano(220, 0.55, 0.001, rng=np.random.RandomState(s))]), 
        "space": lambda sr,d,s: _mk_click(sr,0.35,s,[_VF.piano_hammer(1500, 0.25, rng=np.random.RandomState(s)), _VF.dx7_epiano(150, 0.60, 0.001, rng=np.random.RandomState(s))]), 
        "enter": lambda sr,d,s: _mk_click(sr,0.45,s,[_VF.piano_hammer(1200, 0.25, rng=np.random.RandomState(s)), _VF.dx7_epiano(120, 0.65, 0.001, rng=np.random.RandomState(s))]),
        "release": lambda sr,d,s: _mk_click(sr,0.2,s,[_VF.piano_hammer(2500, 0.10, rng=np.random.RandomState(s)), _VF.dx7_epiano(400, 0.20, 0.001, rng=np.random.RandomState(s))])
    },
    "Water Drop ASMR": {
        "click": lambda sr,d,s: _mk_click(sr,d,s,[_VF.water_drop(800, 0.45, 0.0, rng=np.random.RandomState(s))]), 
        "space": lambda sr,d,s: _mk_click(sr,0.3,s,[_VF.water_drop(500, 0.50, 0.0, rng=np.random.RandomState(s))]), 
        "enter": lambda sr,d,s: _mk_click(sr,0.4,s,[_VF.water_drop(600, 0.55, 0.0, rng=np.random.RandomState(s))]),
        "release": lambda sr,d,s: _mk_click(sr,0.15,s,[_VF.water_drop(700, 0.25, 0.0, rng=np.random.RandomState(s))])
    },
    "Cyberpunk UI": {
        "click": lambda sr,d,s: _mk_click(sr,d,s,[_VF.transient_tick(3000, 0.30, rng=np.random.RandomState(s)), Voice(osc="sine", freq=800, attack=0.001, decay_rate=80.0, filt="foldback", threshold=0.5, f_cutoff=5000, gain=0.40), _VF.sub(80, 0.40, 0.003, rng=np.random.RandomState(s))]), 
        "space": lambda sr,d,s: _mk_click(sr,0.14,s,[Voice(osc="sine", freq=600, attack=0.001, decay_rate=100.0, filt="foldback", threshold=0.6, f_cutoff=4000, gain=0.45), _VF.sub(60, 0.45, 0.003, rng=np.random.RandomState(s))]), 
        "enter": lambda sr,d,s: _mk_click(sr,0.18,s,[Voice(osc="sine", freq=400, attack=0.001, decay_rate=120.0, filt="foldback", threshold=0.7, f_cutoff=3000, gain=0.50), _VF.sub(50, 0.50, 0.003, rng=np.random.RandomState(s))]),
        "release": lambda sr,d,s: _mk_click(sr,0.06,s,[Voice(osc="sine", freq=2500, attack=0.001, decay_rate=50.0, filt="foldback", threshold=0.4, f_cutoff=6000, gain=0.25)])
    },
    "Vocal Whisper": {
        "click": lambda sr,d,s: _mk_click(sr,d,s,[_VF.vocal_whisper(0.40, 0.0, rng=np.random.RandomState(s)), _VF.sub(80, 0.20, 0.001, rng=np.random.RandomState(s))]), 
        "space": lambda sr,d,s: _mk_click(sr,0.12,s,[_VF.vocal_whisper(0.45, 0.0, rng=np.random.RandomState(s))]), 
        "enter": lambda sr,d,s: _mk_click(sr,0.14,s,[_VF.vocal_whisper(0.50, 0.0, rng=np.random.RandomState(s))]),
        "release": lambda sr,d,s: _mk_click(sr,0.06,s,[_VF.vocal_whisper(0.20, 0.0, rng=np.random.RandomState(s))])
    },
    "Goblin Cave": {
        "click": lambda sr,d,s: _mk_click(sr,d,s,[_VF.vosim_voice(80, 300, 0.40, 0.0, rng=np.random.RandomState(s)), _VF.metallic_pipe(400, 0.30, 0.001, rng=np.random.RandomState(s)), Voice(osc="karplus", freq=200, decay_rate=0.99, attack_curve=0.1, attack=0.001, filt="spring_reverb", feedback=0.8, f_cutoff=3000, gain=0.35)]), 
        "space": lambda sr,d,s: _mk_click(sr,0.25,s,[_VF.vosim_voice(60, 250, 0.45, 0.0, rng=np.random.RandomState(s)), Voice(osc="karplus", freq=150, decay_rate=0.99, attack_curve=0.1, attack=0.001, filt="spring_reverb", feedback=0.8, f_cutoff=2500, gain=0.40)]), 
        "enter": lambda sr,d,s: _mk_click(sr,0.35,s,[_VF.vosim_voice(40, 200, 0.50, 0.0, rng=np.random.RandomState(s)), Voice(osc="karplus", freq=100, decay_rate=0.99, attack_curve=0.1, attack=0.001, filt="spring_reverb", feedback=0.8, f_cutoff=2000, gain=0.45)]),
        "release": lambda sr,d,s: _mk_click(sr,0.15,s,[_VF.vosim_voice(300, 1000, 0.20, 0.0, rng=np.random.RandomState(s))])
    }
}

class SimpleSoundGen:
    _SHIFT_MAP = {'~': '`', '!': '1', '@': '2', '#': '3', '$': '4', '%': '5', '^': '6', '&': '7', '*': '8', '(': '9', ')': '0', '_': '-', '+': '=', '{': '[', '}': ']', '|': '\\', ':': ';', '"': "'", '<': ',', '>': '.', '?': '/'}
    def __init__(self, sr: int = 44100, preset: str = "Mechanical", stereo: bool = False, reverb_amt: float = 0.1, target_lufs: float = -14.0):
        self.sr, self.preset, self.stereo, self.reverb_amt, self.target_lufs = sr, preset, stereo, reverb_amt, target_lufs
        factories = _PRESET_FACTORIES.get(preset, _PRESET_FACTORIES["Mechanical"]); click_dur = CLICK_DURATIONS.get(preset, 0.06)
        self.key_sounds = {}; seed_base = 0
        
        for ch in "`1234567890-=[]\\": snd = factories["click"](sr, click_dur, seed_base); self.key_sounds[ch.lower()] = snd; seed_base += 1
        for ch in "qwertyuiop": snd = factories["click"](sr, click_dur, seed_base); self.key_sounds[ch.lower()] = snd; seed_base += 1
        for ch in "asdfghjkl;'": snd = factories["click"](sr, click_dur, seed_base); self.key_sounds[ch.lower()] = snd; seed_base += 1
        for ch in "zxcvbnm,./": snd = factories["click"](sr, click_dur, seed_base); self.key_sounds[ch.lower()] = snd; seed_base += 1
        
        self.mod_sounds = {}; mod_seed = 500
        for mod_name in ["tab", "caps", "shift", "ctrl", "alt", "win", "fn", "menu", "esc"]: snd = factories["space"](sr, 0.1, mod_seed); self.mod_sounds[mod_name] = snd; mod_seed += 1
        self.fallback_clicks = [factories["click"](sr, click_dur, 900 + i) for i in range(8)]
        self.spaces = [factories["space"](sr, 0.1, 600 + i) for i in range(3)]
        self.enters = [factories["enter"](sr, 0.1, 700 + i) for i in range(2)]
        
        self.releases = []
        if "release" in factories:
            for i in range(8):
                self.releases.append(factories["release"](sr, 0.06, 400 + i))
        else:
            for i in range(8):
                self.releases.append(self._make_release_tick(sr, 300 + i))
                
        self._ir_l = self._ir_r = None
        if self.reverb_amt > 0: self._get_reverb_ir()

    @staticmethod
    def _pitch_shift(sample: np.ndarray, mult: float, sr: int = 44100) -> np.ndarray:
        if mult == 1.0 or len(sample) == 0: return sample
        s = sample.astype(np.float32) / 32767.0
        if mult > 1.0: s = DSP.filt_lp(s, min(8500, (sr / 2.0 / mult) * 0.9), sr, n_taps=63)
        n = len(s); new_n = max(1, int(n / mult)); indices = np.linspace(0, n - 1, new_n); left = np.floor(indices).astype(int); right = np.minimum(left + 1, n - 1); frac = indices - left; shifted = s[left] * (1 - frac) + s[right] * frac
        if mult < 1.0: shifted = DSP.filt_hp(shifted, 30.0, sr, n_taps=63); sub = DSP.filt_bp(shifted, 40, 120, sr, n_taps=127); shifted = shifted + sub * 0.25
        return np.clip(shifted * 32767, -32768, 32767).astype(np.int16)

    def _make_release_tick(self, sr=44100, seed=0) -> np.ndarray:
        rng = np.random.RandomState(seed); R = SoundRenderer(sr); v = [Voice(osc="noise_white", attack=0.0001, decay_rate=600.0, attack_curve=2.0, filt="hp", f_cutoff=5000, drive=1.5, gain=0.18)]; return DSP.to_int16(R.render(v, 0.035, rng))

    def _get_reverb_ir(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._ir_l is None:
            sr = self.sr; n = int(sr * 0.8); rng = np.random.RandomState(42); t = np.arange(n, dtype=np.float32) / sr; ir_l, ir_r = np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
            for i, (dt, g) in enumerate(zip(np.linspace(0.005, 0.080, 40), np.linspace(0.8, 0.1, 40))):
                ds = int(dt * sr)
                if ds < n:
                    if i % 2 == 0: ir_l[ds] += g * (1.0 if rng.rand() > 0.5 else -1.0); ir_r[min(ds + int(0.002 * sr), n - 1)] += g * 0.5 * (1.0 if rng.rand() > 0.5 else -1.0)
                    else: ir_r[ds] += g * (1.0 if rng.rand() > 0.5 else -1.0); ir_l[min(ds + int(0.0015 * sr), n - 1)] += g * 0.5 * (1.0 if rng.rand() > 0.5 else -1.0)
            if np.max(np.abs(ir_l)) > 0: ir_l /= np.max(np.abs(ir_l))
            if np.max(np.abs(ir_r)) > 0: ir_r /= np.max(np.abs(ir_r))
            self._ir_l, self._ir_r = DSP.filt_hp(DSP.filt_lp(ir_l, 7000, sr, n_taps=127), 150, sr, n_taps=63), DSP.filt_hp(DSP.filt_lp(ir_r, 7000, sr, n_taps=127), 150, sr, n_taps=63)
        return self._ir_l, self._ir_r

    def _pick(self, char: str, rng: random.Random) -> Tuple[np.ndarray, float]:
        if char == "__release__": return rng.choice(self.releases), rng.uniform(0.3, 0.5) # Quieter release
        if char == "\n": return rng.choice(self.enters), rng.uniform(1.10, 1.25)
        if char == " ": return self.spaces[rng.randint(0, 2)], rng.uniform(1.05, 1.15)
        if char == "\t": return self.mod_sounds["tab"], rng.uniform(0.95, 1.05)
        cl = char.lower()
        if cl in self._SHIFT_MAP: cl = self._SHIFT_MAP[cl]
        if cl in self.key_sounds: return self.key_sounds[cl], rng.uniform(0.92, 1.08)
        return rng.choice(self.fallback_clicks), rng.uniform(0.92, 1.08)

    def generate_pcm(self, timestamps: List[Tuple[float, str]], volume: float = 0.5) -> np.ndarray:
        if not timestamps: return np.zeros(0, dtype=np.int16)
        sr = self.sr; rng = random.Random(98765); all_ts: List[Tuple[float, str]] = []
        for ts, ch in timestamps:
            all_ts.append((ts, ch))
            if ch.isalnum() or ch in ".,;[]'": all_ts.append((ts + 0.045 + rng.uniform(0, 0.015), "__release__"))
        n = int(sr * (max(ts for ts, _ in all_ts) + 0.3)); raw_sounds: List[Tuple[np.ndarray, float, int]] = []
        for ts, ch in all_ts:
            snd, vm = self._pick(ch, rng); s = int(ts * sr)
            if s >= n: continue
            snd_float = snd.astype(np.float32) / 32767.0
            if len(snd_float) == 0: continue
            snd_mono = snd_float * ((rng.gauss(1.0, 0.05) * vm) ** 0.8) * volume; pan = 0.0
            if self.stereo:
                cl = ch.lower()
                if cl in self._SHIFT_MAP: cl = self._SHIFT_MAP[cl]
                if cl in "qazwsx12345edcrfv tgbyhn": pan = rng.uniform(-0.7, -0.3)
                elif cl in "yp[]\\=-7890uio phjklnm,./;'": pan = rng.uniform(0.3, 0.7)
                elif ch in (" ", "\n", "\t"): pan = rng.uniform(-0.1, 0.1)
                else: pan = rng.uniform(-0.3, 0.3)
            raw_sounds.append((snd_mono, pan, s))
        if not raw_sounds: return np.zeros(n, dtype=np.int16)
        if self.stereo:
            max_len = n + int(0.012 * sr) + 2; mix_l, mix_r = np.zeros(max_len, dtype=np.float32), np.zeros(max_len, dtype=np.float32) # extra len for haas
            for snd_mono, pan, s in raw_sounds:
                l_sig, r_sig = DSP.binaural_pan(snd_mono, pan, sr); e = min(s + len(l_sig), max_len); mix_l[s:e] += l_sig[:e - s]; mix_r[s:e] += r_sig[:e - s]
            if self.reverb_amt > 0:
                ir_l, ir_r = self._get_reverb_ir(); mix_l = mix_l * (1.0 - self.reverb_amt) + DSP.convolve(mix_l, ir_l)[:max_len] * self.reverb_amt; mix_r = mix_r * (1.0 - self.reverb_amt) + DSP.convolve(mix_r, ir_r)[:max_len] * self.reverb_amt
            mix_l, mix_r = mix_l[:n], mix_r[:n]
            mix_l, mix_r = DSP.master_stereo_asmr(mix_l, mix_r, sr)
            combined = (mix_l + mix_r) * 0.5
            cl = _compute_lufs(combined, sr)
            if np.isfinite(cl) and cl > -70:
                gdb = self.target_lufs - cl; gl = 10.0 ** (gdb / 20.0); mix_l *= gl; mix_r *= gl
            mix_l = DSP.lookahead_limiter(mix_l, sr, threshold_db=-1.0); mix_r = DSP.lookahead_limiter(mix_r, sr, threshold_db=-1.0)
            out = np.empty(n * 2, dtype=np.int16)
            out[0::2] = np.clip(mix_l * 32767, -32768, 32767).astype(np.int16); out[1::2] = np.clip(mix_r * 32767, -32768, 32767).astype(np.int16); return out
        mono_sounds = [s[0] for s in raw_sounds]; starts = np.array([s[2] for s in raw_sounds], dtype=np.int64); lengths = np.array([len(s) for s in mono_sounds], dtype=np.int64); flat = np.concatenate(mono_sounds) if mono_sounds else np.array([], dtype=np.float32); offsets = np.zeros(len(mono_sounds), dtype=np.int64)
        for i in range(1, len(mono_sounds)): offsets[i] = offsets[i - 1] + lengths[i - 1]
        mix = np.zeros(n, dtype=np.float32)
        if len(flat) > 0:
            if _HAS_NUMBA: _nb_mix(mix, flat, offsets, lengths, starts)
            else:
                for i in range(len(starts)): s_idx, ln, o = int(starts[i]), int(lengths[i]), int(offsets[i]); mix[s_idx:s_idx + ln] += flat[o:o + ln]
        if self.reverb_amt > 0:
            ir_l, ir_r = self._get_reverb_ir(); mix = mix * (1.0 - self.reverb_amt) + DSP.convolve(mix, (ir_l + ir_r) * 0.5)[:n] * self.reverb_amt
        mix = DSP.master_mono(mix, sr)
        mix = DSP.normalize_loudness(mix, sr, self.target_lufs)
        return np.clip(mix * 32767.0, -32768, 32767).astype(np.int16)

    def generate_track(self, timestamps: List[Tuple[float, str]], filepath: str, volume: float = 0.5) -> None:
        if not timestamps: return
        pcm = self.generate_pcm(timestamps, volume)
        if len(pcm) == 0: return
        with wave.open(filepath, "w") as wf: wf.setnchannels(2 if self.stereo else 1); wf.setsampwidth(2); wf.setframerate(self.sr); wf.writeframes(pcm.astype('<i2').tobytes())

def _pcm_to_wav_bytes(pcm: np.ndarray, sr: int = 44100, channels: int = 1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "w") as wf: wf.setnchannels(channels); wf.setsampwidth(2); wf.setframerate(sr); wf.writeframes(pcm.tobytes())
    return buf.getvalue()