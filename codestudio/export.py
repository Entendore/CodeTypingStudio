from __future__ import annotations
from typing import Optional, List, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, Future
import os, math, threading, subprocess, shutil, numpy as np, wave
from PySide6.QtCore import (Qt, QTimer, QElapsedTimer, QUrl, QBuffer, QIODevice,
                            Signal, Slot, QObject, QThread, QPointF)
from PySide6.QtGui import (QImage, QPixmap, QColor, QKeyEvent, QPainter, QFont)
from PySide6.QtWidgets import (QWidget, QLabel, QHBoxLayout, QVBoxLayout, QPushButton,
                               QSlider, QSizePolicy, QGraphicsDropShadowEffect,
                               QFrame, QComboBox)
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer

from .core import (EXPORT_FORMATS, ENCODERS, RenderingError, FFmpegError, log,
                   _subprocess_startup_info, temp_directory, THEMES)
from .rendering import (Tokenizer, TypingProfile, TypingAnimator, KeyboardOverlay,
                        CodeRenderer, MultiPaneRenderer)
from .audio import SimpleSoundGen, _pcm_to_wav_bytes


# ============================================================
# Export (original export.py)
# ============================================================

@dataclass(slots=True)
class ExportConfig:
    junior_code: str; senior_code: str; name: str; fmt: str; res: str; theme: str; j_font: str; s_font: str; font_size_auto: bool; font_size: int; chrome: bool; ln: bool; cursor_glow: bool; watermark: bool; bg_image_path: str; show_kb: bool; kb_layout: str; kb_pos: str; kb_op: float; j_wpm: int; j_pause: float; s_wpm: int; s_pause: float; start_pause: float; end_pause: float; lang: str; pane_layout: str; pad: int; gap: int; target_duration: float = 0.0

class ExportConfigBuilder:
    def __init__(self): self._fields: dict = {}
    def junior_code(self, code: str) -> "ExportConfigBuilder": self._fields["junior_code"] = code; return self
    def senior_code(self, code: str) -> "ExportConfigBuilder": self._fields["senior_code"] = code; return self
    def name(self, n: str) -> "ExportConfigBuilder": self._fields["name"] = n; return self
    def fmt(self, f: str) -> "ExportConfigBuilder": self._fields["fmt"] = f; return self
    def res(self, r: str) -> "ExportConfigBuilder": self._fields["res"] = r; return self
    def theme(self, t: str) -> "ExportConfigBuilder": self._fields["theme"] = t; return self
    def j_font(self, f: str) -> "ExportConfigBuilder": self._fields["j_font"] = f; return self
    def s_font(self, f: str) -> "ExportConfigBuilder": self._fields["s_font"] = f; return self
    def font_size_auto(self, v: bool) -> "ExportConfigBuilder": self._fields["font_size_auto"] = v; return self
    def font_size(self, s: int) -> "ExportConfigBuilder": self._fields["font_size"] = s; return self
    def chrome(self, c: bool) -> "ExportConfigBuilder": self._fields["chrome"] = c; return self
    def ln(self, l: bool) -> "ExportConfigBuilder": self._fields["ln"] = l; return self
    def cursor_glow(self, g: bool) -> "ExportConfigBuilder": self._fields["cursor_glow"] = g; return self
    def watermark(self, w: bool) -> "ExportConfigBuilder": self._fields["watermark"] = w; return self
    def bg_image_path(self, p: str) -> "ExportConfigBuilder": self._fields["bg_image_path"] = p; return self
    def show_kb(self, s: bool) -> "ExportConfigBuilder": self._fields["show_kb"] = s; return self
    def kb_layout(self, l: str) -> "ExportConfigBuilder": self._fields["kb_layout"] = l; return self
    def kb_pos(self, p: str) -> "ExportConfigBuilder": self._fields["kb_pos"] = p; return self
    def kb_op(self, o: float) -> "ExportConfigBuilder": self._fields["kb_op"] = o; return self
    def j_wpm(self, w: int) -> "ExportConfigBuilder": self._fields["j_wpm"] = w; return self
    def j_pause(self, p: float) -> "ExportConfigBuilder": self._fields["j_pause"] = p; return self
    def s_wpm(self, w: int) -> "ExportConfigBuilder": self._fields["s_wpm"] = w; return self
    def s_pause(self, p: float) -> "ExportConfigBuilder": self._fields["s_pause"] = p; return self
    def start_pause(self, s: float) -> "ExportConfigBuilder": self._fields["start_pause"] = s; return self
    def end_pause(self, e: float) -> "ExportConfigBuilder": self._fields["end_pause"] = e; return self
    def lang(self, l: str) -> "ExportConfigBuilder": self._fields["lang"] = l; return self
    def pane_layout(self, p: str) -> "ExportConfigBuilder": self._fields["pane_layout"] = p; return self
    def pad(self, p: int) -> "ExportConfigBuilder": self._fields["pad"] = p; return self
    def gap(self, g: int) -> "ExportConfigBuilder": self._fields["gap"] = g; return self
    def target_duration(self, t: float) -> "ExportConfigBuilder": self._fields["target_duration"] = t; return self
    def build(self) -> ExportConfig: return ExportConfig(**self._fields)

def _build_panes_from_config(cfg: ExportConfig) -> MultiPaneRenderer:
    fmt_data = EXPORT_FORMATS.get(cfg.fmt, EXPORT_FORMATS["YouTube (16:9)"]); w, h = fmt_data["resolutions"].get(cfg.res, (1920, 1080)); theme = THEMES[cfg.theme]
    jp = TypingProfile(name="Junior Dev", wpm=cfg.j_wpm, pause_freq=cfg.j_pause, start_pause=cfg.start_pause, end_pause=cfg.end_pause, error_chance=0.04)
    sp = TypingProfile(name="Senior Dev", wpm=cfg.s_wpm, pause_freq=cfg.s_pause, start_pause=cfg.start_pause, end_pause=cfg.end_pause, error_chance=0.01)
    if cfg.target_duration > 0:
        for profile, code in [(jp, cfg.junior_code), (sp, cfg.senior_code)]:
            req_wpm = TypingAnimator.find_wpm_for_target_duration(code, cfg.target_duration, profile.start_pause, profile.end_pause, profile.error_chance); profile.wpm = max(10, min(500, req_wpm))
    panes: List[Tuple[CodeRenderer, TypingAnimator]] = []
    for i, (code, profile, title, accent_key) in enumerate([(cfg.junior_code, jp, f"👶 Junior Dev — {cfg.name}", "junior_accent"), (cfg.senior_code, sp, f"🧓 Senior Dev — {cfg.name}", "senior_accent")]):
        match cfg.pane_layout:
            case "Side by Side (50/50)": pw, ph = (w - cfg.gap) // 2, h
            case "Stacked (50/50)": pw, ph = w, (h - cfg.gap) // 2
            case "Junior Big (70/30)": pw = int((w - cfg.gap) * 0.7) if i == 0 else (w - cfg.gap) - int((w - cfg.gap) * 0.7); ph = h
            case "Senior Big (70/30)": pw = int((w - cfg.gap) * 0.3) if i == 0 else (w - cfg.gap) - int((w - cfg.gap) * 0.3); ph = h
            case _: pw, ph = (w - cfg.gap) // 2, h
        language = cfg.lang
        if language == "Auto": ext = os.path.splitext(cfg.name)[1].lower(); ext_map = {".py": "Python", ".js": "JavaScript", ".cpp": "CFamily", ".cs": "CSharp"}; language = ext_map.get(ext, "Text")
        code_lines = code.count("\n") + 1; kb_overlay = None; kb_h = 0
        if cfg.show_kb: kb_overlay = KeyboardOverlay(pw, ph, cfg.kb_layout, theme, cfg.kb_op, (ph - 2 * cfg.pad) // 3, cfg.kb_pos); kb_h = kb_overlay.height_needed()
        current_font = cfg.j_font if i == 0 else cfg.s_font
        font_size = (CodeRenderer.auto_font_size(code_lines, pw, ph, cfg.pad, cfg.chrome, cfg.ln, 4, code, current_font, kb_h) if cfg.font_size_auto else max(8, int(cfg.font_size)))
        renderer = CodeRenderer(width=pw, height=ph, theme_name=cfg.theme, font_family=current_font, font_size=font_size, title_text=title, language=language, keyboard_overlay=kb_overlay, padding=cfg.pad, show_window_chrome=cfg.chrome, show_line_numbers=cfg.ln, bg_image_path=cfg.bg_image_path if i == 0 else None, total_code_lines=code_lines, cursor_glow=cfg.cursor_glow, show_watermark=cfg.watermark and i == 0, accent_color=getattr(theme, accent_key, "#7aa2f7"), code=code)
        animator = TypingAnimator(code, profile, seed=i * 1000 + 42); panes.append((renderer, animator))
    return MultiPaneRenderer(panes=panes, frame_w=w, frame_h=h, num_panes=2, pane_layout=cfg.pane_layout, gap=cfg.gap)

class VideoExporter(QThread):
    progress = Signal(int); status = Signal(str); finished_ok = Signal(str); error = Signal(str); shorts_ready = Signal(bool, float)
    CHUNK_DURATION: float = 300.0
    def __init__(self, export_config: ExportConfig, output: str, fps: int = 30, sound_params: Optional[Tuple[str, bool, float]] = None, sound_gens: Optional[List[SimpleSoundGen]] = None, volume: float = 0.5, encoder_name: str = "YouTube Pro (H.264/AAC)", bgm_path: Optional[str] = None, bgm_volume: float = 0.3, bgm_duck: float = 0.6, generate_thumbnail: bool = True, generate_chapters: bool = True, parent=None):
        super().__init__(parent); self.setObjectName("VideoExporter"); self.config = export_config; self.output = output; self.fps = fps; self.sound_params = sound_params; self.sound_gens = sound_gens; self.volume = volume; self.encoder_name = encoder_name; self.bgm_path = bgm_path; self.bgm_volume = bgm_volume; self.bgm_duck = bgm_duck; self.generate_thumbnail = generate_thumbnail; self.generate_chapters = generate_chapters; self.multi_pane = _build_panes_from_config(self.config); self._cancel = threading.Event(); self._last_duration = 0.0; self.finished.connect(self.deleteLater)
    def cancel(self) -> None: self._cancel.set()
    def is_cancelled(self) -> bool: return self._cancel.is_set()
    def run(self) -> None:
        audio_executor: Optional[ThreadPoolExecutor] = None
        try:
            anim_dur = self.multi_pane.duration(); self._last_duration = anim_dur; total_dur = self.config.target_duration if self.config.target_duration > 0 else anim_dur; num_chunks = max(1, math.ceil(total_dur / self.CHUNK_DURATION))
            with temp_directory() as tmp_dir:
                audio_executor = ThreadPoolExecutor(max_workers=1) if self.sound_params or self.sound_gens else None; audio_future: Optional[Future] = None; final_audio = os.path.join(tmp_dir, "final_audio.wav")
                if audio_executor:
                    self.status.emit("Starting audio generation...")
                    if self.sound_gens:
                        audio_future = audio_executor.submit(self._gen_full_audio_per_pane, self.multi_pane, final_audio)
                    else:
                        all_ts = self.multi_pane.char_timestamps_all()
                        if all_ts: audio_future = audio_executor.submit(self._gen_full_audio, all_ts, final_audio)
                video_chunks: List[str] = []
                for c_idx in range(num_chunks):
                    if self._cancel.is_set(): break
                    start_t = c_idx * self.CHUNK_DURATION; end_t = min(total_dur, start_t + self.CHUNK_DURATION); start_frame = int(start_t * self.fps); n_chunk = math.ceil(end_t * self.fps) - start_frame
                    self.status.emit(f"Rendering Video Chunk {c_idx + 1}/{num_chunks}..."); chunk_vid = os.path.join(tmp_dir, f"vid_{c_idx}.mp4")
                    if not self._render_chunk_video(start_frame, n_chunk, chunk_vid): raise RenderingError(f"Failed to render video for chunk {c_idx + 1}")
                    video_chunks.append(chunk_vid); self.progress.emit(int(((c_idx + 1) / num_chunks) * 80))
                if self._cancel.is_set(): self.error.emit("Cancelled"); return
                self.status.emit("Stitching video chunks..."); final_video = os.path.join(tmp_dir, "final_video.mp4"); self._concat_files(video_chunks, final_video)
                if audio_future:
                    self.status.emit("Waiting for audio generation to finish..."); audio_ok = audio_future.result()
                    if audio_ok: self.status.emit("Muxing final video and audio..."); self._mux_final(final_video, final_audio, self.output)
                    else: shutil.move(final_video, self.output)
                else: shutil.move(final_video, self.output)
                if self.generate_thumbnail:
                    try: self._generate_thumbnail_candidates(); self.status.emit("Thumbnail candidates saved.")
                    except Exception as e: log.warning("Thumbnail generation failed: %s", e)
                if self.generate_chapters:
                    try: self._generate_chapters_file(); self._generate_description(); self.status.emit("YouTube chapters + description saved.")
                    except Exception as e: log.warning("Chapters/description generation failed: %s", e)
                w, h = self.multi_pane.frame_w, self.multi_pane.frame_h
                self.shorts_ready.emit(h > w and 15 <= anim_dur <= 60, anim_dur)
                self.progress.emit(100); self.status.emit(f"Done → {self.output}"); self.finished_ok.emit(self.output)
        except Exception as e: self.error.emit(str(e))
        finally:
            if audio_executor: audio_executor.shutdown(wait=True)
    def _gen_full_audio(self, ts: List[Tuple[float, str]], filepath: str) -> bool:
        try:
            preset, stereo, reverb_amt = self.sound_params; sg = SimpleSoundGen(sr=44100, preset=preset, stereo=stereo, reverb_amt=reverb_amt); sg.generate_track(ts, filepath, self.volume)
            return os.path.exists(filepath) and os.path.getsize(filepath) > 44
        except Exception: return False
    def _gen_full_audio_per_pane(self, mp: MultiPaneRenderer, filepath: str) -> bool:
        try:
            pcm_tracks = []
            is_stereo = any(g.stereo for g in self.sound_gens if g)
            for pane_idx, (renderer, animator) in enumerate(mp.panes):
                gen = self.sound_gens[pane_idx] if pane_idx < len(self.sound_gens) else self.sound_gens[0]
                if gen:
                    pane_ts = animator.char_timestamps()
                    if pane_ts:
                        pcm = gen.generate_pcm(pane_ts, self.volume)
                        pcm_tracks.append(pcm)
            if not pcm_tracks: return False
            max_len = max(len(t) for t in pcm_tracks)
            if is_stereo:
                mixed_l = np.zeros(max_len // 2, dtype=np.float32); mixed_r = np.zeros(max_len // 2, dtype=np.float32)
                for pane_idx, f in enumerate(pcm_tracks):
                    gen = self.sound_gens[pane_idx] if pane_idx < len(self.sound_gens) else self.sound_gens[0]
                    if gen and gen.stereo and len(f) > 1:
                        half = len(f) // 2
                        mixed_l[:half] += f[0::2]; mixed_r[:half] += f[1::2]
                    else:
                        mixed_l[:len(f)] += f; mixed_r[:len(f)] += f
                pk = max(float(np.max(np.abs(mixed_l))), float(np.max(np.abs(mixed_r))))
                if pk > 0: mixed_l *= 0.95 / pk; mixed_r *= 0.95 / pk
                out = np.empty(max_len, dtype=np.int16)
                out[0::2] = np.clip(mixed_l * 32767, -32768, 32767).astype(np.int16); out[1::2] = np.clip(mixed_r * 32767, -32768, 32767).astype(np.int16)
            else:
                mixed = np.zeros(max_len, dtype=np.float32)
                for pane_idx, f in enumerate(pcm_tracks):
                    gen = self.sound_gens[pane_idx] if pane_idx < len(self.sound_gens) else self.sound_gens[0]
                    if gen and gen.stereo and len(f) > 1:
                        half = len(f) // 2
                        mixed[:half] += (f[0::2] + f[1::2]) * 0.5
                    else:
                        mixed[:len(f)] += f
                pk = float(np.max(np.abs(mixed)))
                if pk > 0: mixed *= 0.95 / pk
                out = np.clip(mixed * 32767, -32768, 32767).astype(np.int16)
            with wave.open(filepath, "w") as wf: wf.setnchannels(2 if is_stereo else 1); wf.setsampwidth(2); wf.setframerate(self.sound_gens[0].sr); wf.writeframes(out.tobytes())
            return os.path.exists(filepath) and os.path.getsize(filepath) > 44
        except Exception as e: log.error("Per-pane audio failed: %s", e); return False
    def _concat_files(self, paths: List[str], out_path: str) -> None:
        concat_path = os.path.join(os.path.dirname(out_path), "concat_list.txt")
        with open(concat_path, "w", encoding="utf-8") as f:
            for p in paths: f.write(f"file '{p}'\n")
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_path, "-c", "copy", out_path]
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, startupinfo=_subprocess_startup_info())
        _, stderr = proc.communicate(timeout=max(120, len(paths) * 60))
        if proc.returncode != 0: raise FFmpegError(f"Concat failed: {stderr.decode(errors='ignore')[-400:]}")
    def _mux_final(self, vid_path: str, aud_path: str, out_path: str) -> None:
        if self.bgm_path and os.path.exists(self.bgm_path):
            sc_ratio = max(1.0, 1.0 + self.bgm_duck * 9.0); sc_threshold = max(0.005, 0.10 - self.bgm_duck * 0.08); sc_makeup = 1.0 + self.bgm_duck * 1.0
            cmd = ["ffmpeg", "-y", "-i", vid_path, "-i", aud_path, "-i", self.bgm_path, "-c:v", "copy", "-filter_complex", f"[2:a]highpass=f=60,lowpass=f=5500,volume={self.bgm_volume},afade=t=in:st=0:d=2[bgm_eq];[bgm_eq][1:a]sidechaincompress=threshold={sc_threshold:.3f}:ratio={sc_ratio:.2f}:attack=0.015:release=0.3:makeup={sc_makeup:.2f}[bgm_duck];[1:a][bgm_duck]amix=inputs=2:duration=longest:normalize=0,alimiter=limit=0.95[aout]", "-map", "0:v", "-map", "[aout]", "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-movflags", "+faststart", out_path]
        else:
            cmd = ["ffmpeg", "-y", "-i", vid_path, "-i", aud_path, "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-movflags", "+faststart", out_path]
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, startupinfo=_subprocess_startup_info())
        _, stderr = proc.communicate(timeout=max(120, os.path.getsize(vid_path) // (1024 * 1024) * 2))
        if proc.returncode != 0: raise FFmpegError(f"Final mux failed: {stderr.decode(errors='ignore')[-400:]}")
    def _render_chunk_video(self, start_frame: int, n_frames: int, out_path: str, allow_fallback: bool = True) -> bool:
        w, h, fps = self.multi_pane.frame_w, self.multi_pane.frame_h, self.fps
        cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgra", "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:0"]
        enc_cfg = ENCODERS.get(self.encoder_name, ENCODERS["YouTube Pro (H.264/AAC)"]); cmd += enc_cfg["base"]
        if enc_cfg.get("use_dynamic_gop", False): gop = int(fps * 2); cmd += ["-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0"]
        if enc_cfg.get("use_dynamic_vbv", False): pixels = w * h; maxrate = int((pixels / 2073600.0) * 12000000); cmd += ["-maxrate", f"{maxrate}", "-bufsize", f"{maxrate * 2}"]
        if enc_cfg.get("color_tags", False): cmd += ["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"]
        cmd += ["-an", out_path]
        try: proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, startupinfo=_subprocess_startup_info())
        except FileNotFoundError: self.error.emit("FFmpeg not found on PATH."); return False
        scratch = QImage(w, h, QImage.Format_RGB32)
        for fi in range(n_frames):
            if self._cancel.is_set(): break
            t = (start_frame + fi) / self.fps; state = self.multi_pane.compute_state(t); qimg = self.multi_pane.render_frame(t, scratch, precomputed_state=state); ptr = qimg.constBits()
            try: img_bytes = bytes(ptr)
            except TypeError: img_bytes = ptr.tobytes()
            except AttributeError: ptr.setsize(qimg.sizeInBytes()); img_bytes = bytes(ptr)
            try: proc.stdin.write(img_bytes); proc.stdin.flush()
            except BrokenPipeError:
                proc.kill(); proc.wait(timeout=5); return False
        proc.stdin.close()
        try:
            proc.wait(timeout=600)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait(timeout=5); return False
        if proc.returncode != 0:
            if allow_fallback and self.encoder_name != "x264 (CPU, Fast)":
                self.encoder_name = "x264 (CPU, Fast)"; self.status.emit("Hardware encoder failed. Retrying with CPU fallback..."); return self._render_chunk_video(start_frame, n_frames, out_path, allow_fallback=False)
            return False
        return True
    def _generate_thumbnail_candidates(self):
        paths = []; base = os.path.splitext(self.output)[0]; vw, vh = self.multi_pane.frame_w, self.multi_pane.frame_h; is_vertical = vh > vw
        tw, th = (1080, 1920) if is_vertical else (1280, 720)
        for i, pct in enumerate([0.25, 0.50, 0.75, 0.95]):
            tp = f"{base}_thumb_{i+1}.jpg"; t = self._last_duration * pct
            state = self.multi_pane.compute_state(t); qimg = self.multi_pane.render_frame(t, None, precomputed_state=state)
            sc = qimg.scaled(tw, th, Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation)
            x = (sc.width() - tw) // 2; y = (sc.height() - th) // 2; sc.copy(x, y, tw, th).save(tp, "JPG", 95); paths.append(tp)
        return paths
    def _generate_chapters_file(self):
        cp = os.path.splitext(self.output)[0] + "_chapters.txt"
        markers = Tokenizer.find_structural_markers(self.config.junior_code, self.multi_pane.panes[0][0].language)
        chapters = []
        for co, _, label in markers:
            for ts, idx, ch, _ in self.multi_pane.panes[0][1].timeline:
                if idx == co: chapters.append((ts, label)); break
        chapters = [(0.0, "Intro")] + [c for c in chapters if c[0] > 0]
        if not chapters or chapters[-1][0] < self._last_duration - 5: chapters.append((self._last_duration - 2.0, "Outro"))
        filtered = []; last_ts = -100
        for ts, label in chapters:
            if ts - last_ts >= 10 or ts == 0: filtered.append((ts, label)); last_ts = ts
        def _fmt(s): return f"{int(s//60)}:{int(s%60):02d}" if s < 3600 else f"{int(s//3600)}:{int((s%3600)//60):02d}:{int(s%60):02d}"
        ct = "\n".join(f"{_fmt(ts)} {label}" for ts, label in filtered)
        with open(cp, "w", encoding="utf-8") as f: f.write(ct)
        return cp
    def _generate_description(self):
        dp = os.path.splitext(self.output)[0] + "_description.txt"; title = os.path.splitext(os.path.basename(self.output))[0]; ct = ""
        cp = os.path.splitext(self.output)[0] + "_chapters.txt"
        if os.path.exists(cp):
            with open(cp, "r", encoding="utf-8") as f: ct = f.read()
        desc = f"""{title.replace('_', ' ').title()}

A real-time typing animation of this code, rendered with procedural ASMR audio.

Timestamps:
{ct}

#coding #programming #typingtutorial #code #asmr

---
Generated by Code Typing Studio
Encoder: {self.encoder_name}
Resolution: {self.multi_pane.frame_w}x{self.multi_pane.frame_h}
Duration: {self._last_duration:.1f}s
"""
        with open(dp, "w", encoding="utf-8") as f: f.write(desc)
        return dp


# ============================================================
# Preview (merged from preview.py)
# ============================================================

def _fmt_dur(s: float) -> str:
    """Format seconds as M:SS or H:MM:SS; '--:--' for invalid."""
    if s is None or s < 0 or s != s:  # NaN or negative
        return "--:--"
    total = int(round(s))
    h, rem = divmod(total, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"

class PreviewImageLabel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None
        self._zoom: float = 1.0
        self._pan: QPointF = QPointF(0, 0)
        self._dragging: bool = False
        self._last_pos = QPointF()
        self.setMinimumSize(480, 270)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background: #11111b; border-radius: 8px;")

    def set_preview_image(self, qimg: QImage) -> None:
        self._pixmap = QPixmap.fromImage(qimg)
        self.update()

    def reset_view(self):
        self._zoom = 1.0
        self._pan = QPointF(0, 0)
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        
        if self._pixmap is None:
            p.setPen(QColor("#585b70"))
            p.setFont(QFont("Arial", 12))
            p.drawText(self.rect(), Qt.AlignCenter, "Select a file to preview\nor drop files into the batch panel")
            return

        # Calculate base rect to keep aspect ratio
        base_w = self.width()
        base_h = self.height()
        pix_w = self._pixmap.width()
        pix_h = self._pixmap.height()
        
        scale_w = base_w / pix_w
        scale_h = base_h / pix_h
        scale = min(scale_w, scale_h)
        
        render_w = int(pix_w * scale)
        render_h = int(pix_h * scale)
        render_x = (base_w - render_w) // 2
        render_y = (base_h - render_h) // 2
        
        # Apply transforms
        p.translate(self._pan)
        p.scale(self._zoom, self._zoom)
        
        # Draw image centered
        p.drawPixmap(render_x, render_y, render_w, render_h, self._pixmap)
        
    def wheelEvent(self, event) -> None:
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y() / 120.0
            self._zoom = max(1.0, min(5.0, self._zoom + delta * 0.2))
            if self._zoom == 1.0:
                self._pan = QPointF(0, 0)
            self.update()
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._zoom > 1.0:
            self._dragging = True
            self._last_pos = event.position()
            self.setCursor(Qt.ClosedHandCursor)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._dragging:
            self._pan += event.position() - self._last_pos
            self._last_pos = event.position()
            self.update()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._dragging:
            self._dragging = False
            self.setCursor(Qt.ArrowCursor)
        else:
            super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        self.reset_view()

class PreviewController(QObject):
    state_changed = Signal(float, float)
    audio_failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.multi_pane: Optional[MultiPaneRenderer] = None
        self.scratch_image: Optional[QImage] = None
        self.sound_gen: Optional[SimpleSoundGen] = None
        self.audio_volume: float = 0.5
        self.progress: float = 0.0
        self.anim_t: float = 0.0
        self.animating: bool = False
        self.loop: bool = False
        self.target_duration: float = 0.0
        self.speed: float = 1.0
        
        self._cached_pcm: Optional[np.ndarray] = None
        self._cached_pcm_hash: Optional[tuple] = None
        
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.PreciseTimer)
        self._timer.setInterval(16) # ~60fps
        self._timer.timeout.connect(self._advance)
        self._clock = QElapsedTimer()
        
        self._audio_player: Optional[QMediaPlayer] = None
        self._audio_out: Optional[QAudioOutput] = None
        self._audio_buf: Optional[QBuffer] = None

    def attach_audio(self, player: QMediaPlayer, output: QAudioOutput) -> None:
        self._audio_player = player
        self._audio_out = output

    def set_renderer(self, mp: MultiPaneRenderer, scratch: Optional[QImage] = None) -> None:
        self.multi_pane = mp
        self.scratch_image = scratch or QImage(mp.frame_w, mp.frame_h, QImage.Format_RGB32)

    def set_sound(self, sg: Optional[SimpleSoundGen], volume: float = 0.5) -> None:
        self.sound_gen = sg
        self.audio_volume = max(0.0, min(1.0, volume))
        self._cached_pcm = None
        self._cached_pcm_hash = None

    def total_duration(self) -> float:
        if self.multi_pane is None: return 0.0
        if self.target_duration > 0: return self.target_duration
        return self.multi_pane.duration()

    def play(self) -> None:
        if self.multi_pane is None: return
        total = self.total_duration()
        if self.progress >= 0.999:
            self.progress, self.anim_t = 0.0, 0.0
        else:
            self.anim_t = total * self.progress
        self.animating = True
        self._clock.start()
        self._maybe_start_audio()
        self._timer.start()

    def pause(self) -> None:
        self.animating = False
        self._timer.stop()
        self._stop_audio()

    def toggle(self) -> bool:
        if self.animating:
            self.pause()
            return False
        self.play()
        return True

    def seek(self, progress: float) -> None:
        self.pause()
        self.progress = max(0.0, min(1.0, progress))
        self.anim_t = self.total_duration() * self.progress
        self.state_changed.emit(self.anim_t, self.total_duration())

    def seek_delta(self, delta_seconds: float) -> None:
        total = self.total_duration()
        new_t = max(0.0, min(total, total * self.progress + delta_seconds))
        self.anim_t = new_t
        self.progress = new_t / total if total > 0 else 0.0
        self.state_changed.emit(self.anim_t, total)

    def reset(self) -> None:
        self.pause()
        self.progress = 0.0
        self.anim_t = 0.0
        self.state_changed.emit(0.0, self.total_duration())

    def set_loop(self, enabled: bool) -> None:
        self.loop = enabled

    def set_speed(self, speed: float) -> None:
        self.speed = speed
        if self._audio_player is not None:
            self._audio_player.setPlaybackRate(speed)

    def render_frame_at(self, progress: float) -> Optional[QImage]:
        if self.multi_pane is None: return None
        total = self.total_duration()
        t = min(progress * total, self.multi_pane.duration())
        state = self.multi_pane.compute_state(t)
        return self.multi_pane.render_frame(t, self.scratch_image, precomputed_state=state)

    def _advance(self) -> None:
        dt = min(self._clock.restart() / 1000.0, 0.1)
        self.anim_t += dt * self.speed
        total = self.total_duration()
        if self.anim_t >= total:
            if self.loop:
                self.anim_t = 0.0
                self._maybe_start_audio() # restart audio on loop
            else:
                self.anim_t = total
                self.pause()
        self.progress = min(1.0, self.anim_t / total) if total > 0 else 0.0
        self.state_changed.emit(self.anim_t, total)

    def _maybe_start_audio(self) -> None:
        if self.sound_gen is None or self._audio_player is None: return
        try:
            ts = self.multi_pane.char_timestamps_all() if self.multi_pane else []
            if not ts: return
            cache_key = (id(self.sound_gen), len(ts), self.audio_volume)
            if self._cached_pcm_hash != cache_key or self._cached_pcm is None:
                self._cached_pcm = self.sound_gen.generate_pcm(ts, self.audio_volume)
                self._cached_pcm_hash = cache_key
            pcm = self._cached_pcm
            if len(pcm) == 0: return
            channels = 2 if self.sound_gen.stereo else 1
            wav_bytes = _pcm_to_wav_bytes(pcm, self.sound_gen.sr, channels)
            self._stop_audio()
            self._audio_buf = QBuffer()
            self._audio_buf.setData(wav_bytes)
            self._audio_buf.open(QIODevice.ReadOnly)
            self._audio_out.setVolume(self.audio_volume)
            self._audio_player.setSourceDevice(self._audio_buf, QUrl())
            self._audio_player.setPlaybackRate(self.speed)
            self._audio_player.play()
        except Exception as e:
            log.warning("Preview audio failed: %s", e)
            self.audio_failed.emit(str(e))

    def _stop_audio(self) -> None:
        if self._audio_player is not None:
            self._audio_player.stop()
            self._audio_player.setSource(QUrl())
        if self._audio_buf is not None:
            try:
                self._audio_buf.close()
            except Exception:
                pass
            self._audio_buf = None

class PreviewWidget(QWidget):
    progress_changed = Signal(float)
    play_toggled = Signal(bool)
    prev_file_requested = Signal()
    next_file_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.controller = PreviewController(self)
        self._build_ui()
        self.controller.state_changed.connect(self._on_state_changed)
        self.controller.audio_failed.connect(lambda e: log.warning("Preview audio: %s", e))
        
        # Enable keyboard focus for shortcuts
        self.setFocusPolicy(Qt.StrongFocus)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        
        # --- Info Bar ---
        info_bar = QHBoxLayout()
        info_bar.setSpacing(6)
        self.prev_preview_btn = QPushButton("◀")
        self.prev_preview_btn.setFixedWidth(24)
        self.prev_preview_btn.setToolTip("Preview previous file (↑)")
        self.prev_preview_btn.setEnabled(False)
        self.prev_preview_btn.clicked.connect(self.prev_file_requested.emit)
        info_bar.addWidget(self.prev_preview_btn)
        
        self.info_label = QLabel("No file selected")
        self.info_label.setStyleSheet("color: #f9e2af; font-size: 11px; font-weight: bold; font-family: monospace;")
        info_bar.addWidget(self.info_label)
        
        self.preview_idx_label = QLabel("")
        self.preview_idx_label.setStyleSheet("color: #cdd6f4; font-size: 11px; font-family: monospace;")
        info_bar.addWidget(self.preview_idx_label)
        
        self.next_preview_btn = QPushButton("▶")
        self.next_preview_btn.setFixedWidth(24)
        self.next_preview_btn.setToolTip("Preview next file (↓)")
        self.next_preview_btn.setEnabled(False)
        self.next_preview_btn.clicked.connect(self.next_file_requested.emit)
        info_bar.addWidget(self.next_preview_btn)
        
        info_bar.addStretch()
        
        self.batch_label = QLabel("📦 Batch: 0 files")
        self.batch_label.setStyleSheet("color: #a6e3a1; font-size: 11px; font-weight: bold; font-family: monospace;")
        info_bar.addWidget(self.batch_label)
        layout.addLayout(info_bar)
        # ---------------------------------
        
        # --- Video Frame ---
        video_frame = QFrame()
        video_frame.setStyleSheet("QFrame { background: #11111b; border-radius: 8px; }")
        v_layout = QVBoxLayout(video_frame)
        v_layout.setContentsMargins(0, 0, 0, 0)
        
        self.image_label = PreviewImageLabel()
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setOffset(0, 4)
        shadow.setColor(QColor(0, 0, 0, 120))
        self.image_label.setGraphicsEffect(shadow)
        v_layout.addWidget(self.image_label, 1)
        layout.addWidget(video_frame, 1)
        # ------------------
        
        # --- Stats Bar ---
        stats = QHBoxLayout()
        self.time_label = QLabel("0:00 / 0:00")
        self.time_label.setStyleSheet("color: #a6e3a1; font-size: 13px; font-weight: bold; font-family: monospace; padding-left: 4px;")
        stats.addWidget(self.time_label)
        
        self.stats_label = QLabel("")
        self.stats_label.setStyleSheet("color: #a6adc8; font-size: 11px; font-family: monospace;")
        stats.addWidget(self.stats_label)
        stats.addStretch()
        
        self.pct_label = QLabel("0%")
        self.pct_label.setStyleSheet("color: #89b4fa; font-size: 11px; font-weight: bold; font-family: monospace; padding-right: 4px;")
        stats.addWidget(self.pct_label)
        layout.addLayout(stats)
        # ----------------
        
        # --- Controls Bar ---
        ctrl = QHBoxLayout()
        ctrl.setSpacing(4)
        
        self.frame_back_btn = QPushButton("⏮")
        self.frame_back_btn.setFixedWidth(36)
        self.frame_back_btn.setToolTip("Previous Frame (Shift+Left)")
        self.frame_back_btn.clicked.connect(lambda: self.controller.seek_delta(-1 / 30))
        ctrl.addWidget(self.frame_back_btn)
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setToolTip("Seek")
        self.slider.valueChanged.connect(self._on_slider)
        ctrl.addWidget(self.slider, 1)
        
        self.frame_fwd_btn = QPushButton("⏭")
        self.frame_fwd_btn.setFixedWidth(36)
        self.frame_fwd_btn.setToolTip("Next Frame (Shift+Right)")
        self.frame_fwd_btn.clicked.connect(lambda: self.controller.seek_delta(1 / 30))
        ctrl.addWidget(self.frame_fwd_btn)
        
        self.speed_cb = QComboBox()
        self.speed_cb.addItems(["1.0x", "0.5x", "0.75x", "1.5x", "2.0x"])
        self.speed_cb.setFixedWidth(65)
        self.speed_cb.setToolTip("Playback Speed")
        self.speed_cb.currentTextChanged.connect(lambda txt: self.controller.set_speed(float(txt.replace('x', ''))))
        ctrl.addWidget(self.speed_cb)
        
        self.loop_btn = QPushButton("🔁")
        self.loop_btn.setFixedWidth(36)
        self.loop_btn.setCheckable(True)
        self.loop_btn.setToolTip("Loop (L)")
        self.loop_btn.toggled.connect(self.controller.set_loop)
        ctrl.addWidget(self.loop_btn)
        
        self.play_btn = QPushButton("▶ Animate")
        self.play_btn.setFixedWidth(100)
        self.play_btn.setToolTip("Play/Pause (Space)")
        self.play_btn.setStyleSheet(
            "QPushButton { background: #a6e3a1; color: #1e1e2e; font-weight: bold; border: none; border-radius: 6px; padding: 6px; }"
            "QPushButton:hover { background: #94e2d5; }"
            "QPushButton:pressed { background: #89b4fa; }"
        )
        self.play_btn.clicked.connect(self._toggle_play)
        ctrl.addWidget(self.play_btn)
        
        layout.addLayout(ctrl)
        # -------------------
        
        # --- Help Label ---
        help_label = QLabel("Space: Play/Pause  ·  ←/→: Seek 1s (Shift=1 frame)  ·  L: Loop  ·  R: Reset  ·  ↑/↓: Change File  ·  Ctrl+Wheel: Zoom")
        help_label.setStyleSheet("color: #585b70; font-size: 10px;")
        help_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(help_label)
        # ------------------
        
    def set_info(self, file_name: str, file_est: float, batch_idx: int, batch_count: int, batch_est: float, min_est: float = 0.0, max_est: float = 0.0) -> None:
        self.info_label.setText(f"📄 {file_name}  |  Est: {_fmt_dur(file_est)}")
        if batch_count > 0:
            self.preview_idx_label.setText(f"({batch_idx}/{batch_count})")
            self.batch_label.setText(f"📦 {batch_count} files | Total: {_fmt_dur(batch_est)} | Min: {_fmt_dur(min_est)} | Max: {_fmt_dur(max_est)}")
        else:
            self.preview_idx_label.setText("")
            self.batch_label.setText("📦 Batch: 0 files")
        has_multiple = batch_count > 1
        self.prev_preview_btn.setEnabled(has_multiple)
        self.next_preview_btn.setEnabled(has_multiple)
        
    def set_multi_pane(self, mp: MultiPaneRenderer, sound_gen: Optional[SimpleSoundGen] = None, volume: float = 0.5) -> None:
        self.controller.set_renderer(mp)
        self.controller.set_sound(sound_gen, volume)
        self.controller.reset()
        self.image_label.reset_view()
        self._render_current()

    def attach_audio(self, player: QMediaPlayer, output: QAudioOutput) -> None:
        self.controller.attach_audio(player, output)

    def set_target_duration(self, seconds: float) -> None:
        self.controller.target_duration = max(0.0, seconds)
        self.controller.reset()
        self._render_current()

    def stop(self) -> None:
        self.controller.pause()

    def _toggle_play(self) -> None:
        playing = self.controller.toggle()
        self.play_btn.setText("⏸ Pause" if playing else "▶ Animate")
        self.play_toggled.emit(playing)

    def _on_slider(self, value: int) -> None:
        self.controller.seek(value / 1000.0)
        self._render_current()

    def _on_state_changed(self, t: float, total: float) -> None:
        self._render_current()
        cur_m, cur_s = divmod(int(t), 60)
        tot_m, tot_s = divmod(int(total), 60)
        self.time_label.setText(f"{cur_m}:{cur_s:02d} / {tot_m}:{tot_s:02d}")
        self.pct_label.setText(f"{int(self.controller.progress * 100)}%")
        self.slider.blockSignals(True)
        self.slider.setValue(int(self.controller.progress * 1000))
        self.slider.blockSignals(False)
        self.progress_changed.emit(self.controller.progress)

    def _render_current(self) -> None:
        qimg = self.controller.render_frame_at(self.controller.progress)
        if qimg is not None:
            self.image_label.set_preview_image(qimg)

    # Keyboard Shortcuts Implementation
    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        mod = event.modifiers()
        
        if key in (Qt.Key_Space, Qt.Key_Play, Qt.Key_MediaPlay):
            if not event.isAutoRepeat():
                self._toggle_play()
        elif key == Qt.Key_Left:
            delta = 1.0/30.0 if (mod & Qt.ShiftModifier) else 1.0
            self.controller.seek_delta(-delta)
            self._render_current()
        elif key == Qt.Key_Right:
            delta = 1.0/30.0 if (mod & Qt.ShiftModifier) else 1.0
            self.controller.seek_delta(delta)
            self._render_current()
        elif key == Qt.Key_L:
            if not event.isAutoRepeat():
                self.loop_btn.toggle()
        elif key == Qt.Key_R:
            if not event.isAutoRepeat():
                self.controller.reset()
                self.image_label.reset_view()
                self._render_current()
        elif key == Qt.Key_Up:
            if not event.isAutoRepeat():
                self.prev_file_requested.emit()
        elif key == Qt.Key_Down:
            if not event.isAutoRepeat():
                self.next_file_requested.emit()
        elif (mod & Qt.ControlModifier) and key in (Qt.Key_Plus, Qt.Key_Equal):
            self.image_label._zoom = min(5.0, self.image_label._zoom + 0.2)
            self.image_label.update()
        elif (mod & Qt.ControlModifier) and key == Qt.Key_Minus:
            self.image_label._zoom = max(1.0, self.image_label._zoom - 0.2)
            if self.image_label._zoom == 1.0: self.image_label._pan = QPointF(0, 0)
            self.image_label.update()
        elif key == Qt.Key_0:
            self.image_label.reset_view()
        else:
            super().keyPressEvent(event)