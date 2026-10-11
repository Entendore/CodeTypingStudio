"""
Project 1: Junior vs Senior Dev — Code Typing Studio (Expert Edition + Batch)
"""
from __future__ import annotations
import sys, os, json
from dataclasses import dataclass
from typing import List, Optional
from codestudio import *
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QScrollArea

@dataclass
class FileItem:
    path: str
    checked: bool = False
    status: str = "Pending"
    output_path: Optional[str] = None
    est_duration: float = 0.0
    name: str = ""
    language: str = "Python"
    junior_code: str = ""
    senior_code: str = ""
    size_bytes: int = 0

class BatchTableWidget(QTableWidget):
    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if event.button() != Qt.LeftButton: return
        pos = event.position().toPoint() if hasattr(event, 'position') else event.pos()
        item = self.itemAt(pos)
        if not item: return
        if item.column() != 0:
            ci = self.item(item.row(), 0)
            if ci is not None:
                ns = Qt.CheckState.Checked if ci.checkState() == Qt.CheckState.Unchecked else Qt.CheckState.Unchecked
                ci.setCheckState(ns)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle("Project 1: Junior vs Senior Dev Studio"); self.setMinimumSize(1200, 800); self.resize(1400, 900)
        self._items: List[FileItem] = []; self._item_paths = set(); self._exporter: Optional[VideoExporter] = None; self._loading_settings = False; self._bg_image_path = ""; self._bgm_path = ""
        self._batch_queue: List[FileItem] = []; self._batch_total = 0; self._batch_done = 0; self._batch_cancel = False; self._current_item: Optional[FileItem] = None
        self._preview_idx = 0
        self._audio_player = QMediaPlayer(self); self._audio_out = QAudioOutput(self); self._audio_player.setAudioOutput(self._audio_out)
        ensure_examples_for_project("jr_vs_sr")
        self._build_ui(); self._connect_signals(); self._load_settings(); self._scan_input_dir()
        QTimer.singleShot(300, self._rebuild_preview)

    def _build_ui(self) -> None:
        central = QWidget(); self.setCentralWidget(central); root = QHBoxLayout(central); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)
        left = QWidget(); left.setFixedWidth(360); ll = QVBoxLayout(left); ll.setContentsMargins(0, 0, 0, 0); ll.setSpacing(0)
        br1 = QHBoxLayout()
        self.scan_btn = QPushButton("🔄 Scan"); self.scan_btn.clicked.connect(self._scan_input_dir); br1.addWidget(self.scan_btn)
        self.add_btn = QPushButton("➕ Add..."); self.add_btn.clicked.connect(self._add_files); br1.addWidget(self.add_btn)
        br1.addStretch()
        self.all_btn = QPushButton("✅ All"); self.all_btn.setFixedWidth(52); self.all_btn.clicked.connect(self._select_all); br1.addWidget(self.all_btn)
        self.none_btn = QPushButton("❌ None"); self.none_btn.setFixedWidth(58); self.none_btn.clicked.connect(self._select_none); br1.addWidget(self.none_btn)
        ll.addLayout(br1)
        self.file_table = BatchTableWidget(0, 5)
        self.file_table.setHorizontalHeaderLabels(["", "File", "Lang", "Est", "Status"])
        self.file_table.setSelectionBehavior(QTableWidget.SelectRows); self.file_table.setEditTriggers(QTableWidget.NoEditTriggers); self.file_table.verticalHeader().setVisible(False)
        hdr = self.file_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Fixed); hdr.resizeSection(0, 32)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.Fixed); hdr.resizeSection(2, 55)
        hdr.setSectionResizeMode(3, QHeaderView.Fixed); hdr.resizeSection(3, 55)
        hdr.setSectionResizeMode(4, QHeaderView.Fixed); hdr.resizeSection(4, 90)
        self.file_table.itemChanged.connect(self._on_item_changed); ll.addWidget(self.file_table)
        self.stats_label = QLabel("0 files · 0 selected · 0:00 total")
        self.stats_label.setStyleSheet("color: #a6adc8; font-size: 11px; padding: 6px 10px; background: #181825; border-top: 1px solid #11111b;")
        ll.addWidget(self.stats_label)
        root.addWidget(left)
        right = QWidget(); rl = QVBoxLayout(right); rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(0)
        self.preview = PreviewWidget(); self.preview.attach_audio(self._audio_player, self._audio_out)
        self.preview.prev_file_requested.connect(self._preview_prev_file)
        self.preview.next_file_requested.connect(self._preview_next_file)
        rl.addWidget(self.preview, 1)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); container = QWidget(); cl = QVBoxLayout(container)
        grp = QGroupBox("Appearance & Export"); fl = QFormLayout(grp)
        self.theme_cb = QComboBox(); self.theme_cb.addItems(list(THEMES.keys())); self.theme_cb.setCurrentText("Dracula"); fl.addRow("Theme:", self.theme_cb)
        self.export_format_cb = QComboBox(); self.export_format_cb.addItems(list(EXPORT_FORMATS.keys())); fl.addRow("Format:", self.export_format_cb)
        self.pane_layout_cb = QComboBox(); self.pane_layout_cb.addItems(["Side by Side (50/50)", "Stacked (50/50)", "Junior Big (70/30)", "Senior Big (70/30)"]); fl.addRow("Pane Layout:", self.pane_layout_cb)
        self.sound_preset_cb = QComboBox(); self.sound_preset_cb.addItems(list(SOUND_PRESETS.keys())); fl.addRow("Audio Preset:", self.sound_preset_cb)
        self.encoder_cb = QComboBox(); self.encoder_cb.addItems(list(ENCODERS.keys())); fl.addRow("Encoder:", self.encoder_cb)
        self.kb_chk = QCheckBox("Show Keyboard"); fl.addRow(self.kb_chk)
        self.watermark_chk = QCheckBox("Watermark"); fl.addRow(self.watermark_chk)
        self.watermark_edit = QLineEdit(); self.watermark_edit.setPlaceholderText("Watermark Text"); fl.addRow("WM Text:", self.watermark_edit)
        self.j_wpm = QSpinBox(); self.j_wpm.setRange(30, 300); self.j_wpm.setValue(90); fl.addRow("Junior WPM:", self.j_wpm)
        self.s_wpm = QSpinBox(); self.s_wpm.setRange(30, 300); self.s_wpm.setValue(130); fl.addRow("Senior WPM:", self.s_wpm)
        self.bgm_edit = QLineEdit(); self.bgm_edit.setPlaceholderText("BGM MP3 (ducked)"); self.bgm_edit.textChanged.connect(self._update_bgm); fl.addRow("BGM:", self.bgm_edit)
        cl.addWidget(grp)
        btn_row = QHBoxLayout()
        self.export_btn = QPushButton("🎬 Export Checked"); self.export_btn.setStyleSheet("QPushButton { background: #a6e3a1; color: #1e1e2e; font-weight: bold; border: none; border-radius: 6px; padding: 10px; } QPushButton:hover { background: #94e2d5; }"); self.export_btn.clicked.connect(self._start_export); btn_row.addWidget(self.export_btn, 1)
        self.cancel_btn = QPushButton("⏹ Cancel Batch"); self.cancel_btn.setStyleSheet("QPushButton { background: #f38ba8; color: #1e1e2e; font-weight: bold; border: none; border-radius: 6px; padding: 10px; } QPushButton:hover { background: #eba0ac; }"); self.cancel_btn.clicked.connect(self._cancel_export); self.cancel_btn.hide(); btn_row.addWidget(self.cancel_btn, 1)
        cl.addLayout(btn_row)
        scroll.setWidget(container); rl.addWidget(scroll); root.addWidget(right, 1)

    def _connect_signals(self) -> None:
        for w in [self.theme_cb, self.export_format_cb, self.pane_layout_cb, self.sound_preset_cb, self.encoder_cb, self.j_wpm, self.s_wpm, self.watermark_edit]:
            if hasattr(w, 'currentTextChanged'): w.currentTextChanged.connect(self._on_param_changed)
            if hasattr(w, 'valueChanged'): w.valueChanged.connect(self._on_param_changed)
        self.kb_chk.toggled.connect(self._on_param_changed)
        self.watermark_chk.toggled.connect(self._on_param_changed)

    def _on_param_changed(self, *_) -> None:
        self._update_est_durations(); self._rebuild_preview()

    def _update_bgm(self, text): self._bgm_path = text.strip()

    def _select_all(self) -> None:
        for it in self._items: it.checked = True
        self._preview_idx = 0; self._refresh_table(); self._update_stats(); self._rebuild_preview()

    def _select_none(self) -> None:
        for it in self._items: it.checked = False
        self._preview_idx = 0; self._refresh_table(); self._update_stats(); self._rebuild_preview()

    def _load_item(self, path: str) -> Optional[FileItem]:
        try:
            name, language, jr, sr = load_jr_vs_sr_json(path)
        except Exception as e:
            log.warning("Failed to parse %s: %s", path, e); return None
        item = FileItem(path=path, checked=True, name=name, language=language, junior_code=jr, senior_code=sr, size_bytes=os.path.getsize(path))
        self._recompute_est(item)
        return item

    def _recompute_est(self, item: FileItem) -> None:
        d_jr = TypingAnimator.estimate_duration(item.junior_code, self.j_wpm.value(), 0.5, 1.5, 0.15, 0.04)
        d_sr = TypingAnimator.estimate_duration(item.senior_code, self.s_wpm.value(), 0.5, 1.5, 0.05, 0.01)
        item.est_duration = max(d_jr, d_sr)

    def _update_est_durations(self) -> None:
        for it in self._items: self._recompute_est(it)
        self._refresh_table(); self._update_stats()

    def _scan_input_dir(self) -> None:
        self._items.clear(); self._item_paths.clear(); self._preview_idx = 0
        if not os.path.isdir(str(INPUT_DIR)): return
        for fname in sorted(os.listdir(str(INPUT_DIR))):
            if not fname.lower().endswith(".json"): continue
            p = os.path.join(str(INPUT_DIR), fname)
            item = self._load_item(p)
            if item: self._items.append(item); self._item_paths.add(p)
        self._refresh_table(); self._update_stats(); self._rebuild_preview()

    def _add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Select JSON Files", "", "JSON Files (*.json)")
        added = False
        for p in paths:
            if p in self._item_paths: continue
            item = self._load_item(p)
            if not item: QMessageBox.warning(self, "Invalid JSON", f"Could not parse {p}"); continue
            self._items.append(item); self._item_paths.add(p); added = True
        if added: self._preview_idx = 0; self._refresh_table(); self._update_stats(); self._rebuild_preview()

    def _refresh_table(self) -> None:
        self.file_table.blockSignals(True); self.file_table.setRowCount(len(self._items))
        for i, item in enumerate(self._items):
            chk = QTableWidgetItem(); chk.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled); chk.setCheckState(Qt.CheckState.Checked if item.checked else Qt.CheckState.Unchecked); self.file_table.setItem(i, 0, chk)
            self.file_table.setItem(i, 1, QTableWidgetItem(os.path.basename(item.path)))
            self.file_table.setItem(i, 2, QTableWidgetItem(item.language[:8]))
            est_item = QTableWidgetItem(format_duration(item.est_duration))
            jr_lines = item.junior_code.count("\n") + 1; sr_lines = item.senior_code.count("\n") + 1
            est_item.setToolTip(f"Estimated: {item.est_duration:.1f}s\nSize: {item.size_bytes} bytes\nJunior lines: {jr_lines}\nSenior lines: {sr_lines}")
            self.file_table.setItem(i, 3, est_item)
            self.file_table.setItem(i, 4, QTableWidgetItem(item.status))
        self.file_table.blockSignals(False)

    def _update_stats(self) -> None:
        total = len(self._items); sel = sum(1 for it in self._items if it.checked)
        sel_dur = sum(it.est_duration for it in self._items if it.checked)
        self.stats_label.setText(f"{total} files · {sel} selected · {format_duration(sel_dur)} total")

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() == 0 and 0 <= item.row() < len(self._items):
            self._items[item.row()].checked = (item.checkState() == Qt.CheckState.Checked)
            self._preview_idx = 0; self._update_stats(); self._rebuild_preview()

    def _preview_prev_file(self):
        checked = [it for it in self._items if it.checked]
        if len(checked) < 2: return
        self._preview_idx = (self._preview_idx - 1) % len(checked)
        self._rebuild_preview()

    def _preview_next_file(self):
        checked = [it for it in self._items if it.checked]
        if len(checked) < 2: return
        self._preview_idx = (self._preview_idx + 1) % len(checked)
        self._rebuild_preview()

    def _build_panes(self) -> MultiPaneRenderer:
        fmt = self.export_format_cb.currentText(); fmt_data = EXPORT_FORMATS.get(fmt, EXPORT_FORMATS["YouTube (16:9)"]); res_name = "1920x1080"; w, h = fmt_data["resolutions"].get(res_name, (1920, 1080)); theme_name = self.theme_cb.currentText(); theme = THEMES[theme_name]
        checked = [it for it in self._items if it.checked]
        if not checked:
            jr = sr = "# No JSON files selected.\n# Add .json files defining junior_code and senior_code."; filename = "empty.json"; lang = "Text"
        else:
            cur_item = checked[self._preview_idx]
            jr = cur_item.junior_code; sr = cur_item.senior_code; filename = os.path.basename(cur_item.path); lang = cur_item.language
        jp = TypingProfile(name="Junior Dev", wpm=self.j_wpm.value(), pause_freq=0.15, error_chance=0.04); sp = TypingProfile(name="Senior Dev", wpm=self.s_wpm.value(), pause_freq=0.05, error_chance=0.01)
        panes = []
        wm_text = self.watermark_edit.text() if self.watermark_edit.text() else "Code Typing Studio"
        for i, (code, profile, title, accent_key) in enumerate([(jr, jp, "👶 Junior", "junior_accent"), (sr, sp, "🧓 Senior", "senior_accent")]):
            pw, ph = (w - 20) // 2, h
            kb_overlay = KeyboardOverlay(pw, ph, "QWERTY", theme, 0.82, (ph - 48) // 3, "bottom_center") if self.kb_chk.isChecked() else None
            fs = CodeRenderer.auto_font_size(code.count("\n")+1, pw, ph, 24, True, True, 4, code, "Consolas", kb_overlay.height_needed() if kb_overlay else 0)
            renderer = CodeRenderer(pw, ph, theme_name, "Consolas", fs, True, True, 24, 4, f"{title} — {filename}", lang, kb_overlay, None, code.count("\n")+1, True, self.watermark_chk.isChecked(), wm_text, getattr(theme, accent_key), code=code)
            animator = TypingAnimator(code, profile, seed=i * 1000 + 42); panes.append((renderer, animator))
        return MultiPaneRenderer(panes=panes, frame_w=w, frame_h=h, num_panes=2, pane_layout=self.pane_layout_cb.currentText())

    def _rebuild_preview(self, *_) -> None:
        if self._loading_settings: return
        checked = [it for it in self._items if it.checked]
        if not checked:
            self.preview.set_info("No files selected", 0.0, 0, 0, 0.0, 0.0, 0.0)
            mp = self._build_panes()
            sg = SimpleSoundGen(sr=44100, preset=self.sound_preset_cb.currentText(), stereo=True, reverb_amt=0.1)
            self.preview.set_multi_pane(mp, sg, 0.5)
            return

        if self._preview_idx >= len(checked): self._preview_idx = 0
        cur_item = checked[self._preview_idx]
        batch_est = sum(it.est_duration for it in checked)
        min_est = min(it.est_duration for it in checked)
        max_est = max(it.est_duration for it in checked)
        self.preview.set_info(os.path.basename(cur_item.path), cur_item.est_duration, self._preview_idx + 1, len(checked), batch_est, min_est, max_est)
        
        mp = self._build_panes()
        sg = SimpleSoundGen(sr=44100, preset=self.sound_preset_cb.currentText(), stereo=True, reverb_amt=0.1)
        self.preview.set_multi_pane(mp, sg, 0.5)

    def _start_export(self) -> None:
        checked = [it for it in self._items if it.checked]
        if not checked: QMessageBox.information(self, "Nothing Selected", "Check at least one JSON file."); return
        for item in self._items:
            item.status = "Queued" if item in checked else "Pending"; item.output_path = None
        self._refresh_table()
        self._batch_queue = list(checked); self._batch_total = len(checked); self._batch_done = 0; self._batch_cancel = False
        self.export_btn.hide(); self.cancel_btn.show()
        self._process_next_in_batch()

    def _process_next_in_batch(self) -> None:
        if self._batch_cancel: self._on_batch_complete(True); return
        if not self._batch_queue: self._on_batch_complete(False); return
        item = self._batch_queue.pop(0); self._batch_done += 1; self._current_item = item
        item.status = f"Rendering 0% ({self._batch_done}/{self._batch_total})"; self._refresh_table()
        wm_text = self.watermark_edit.text() if self.watermark_edit.text() else "Code Typing Studio"
        cfg = (ExportConfigBuilder().junior_code(item.junior_code).senior_code(item.senior_code).name(os.path.basename(item.path)).fmt(self.export_format_cb.currentText()).res("1920x1080").theme(self.theme_cb.currentText()).j_font("Consolas").s_font("Consolas").font_size_auto(True).font_size(22).chrome(True).ln(True).cursor_glow(True).watermark(self.watermark_chk.isChecked()).watermark_text(wm_text).bg_image_path("").show_kb(self.kb_chk.isChecked()).kb_layout("QWERTY").kb_pos("bottom_center").kb_op(0.82).j_wpm(self.j_wpm.value()).j_pause(0.15).s_wpm(self.s_wpm.value()).s_pause(0.05).start_pause(0.5).end_pause(1.5).lang(item.language).pane_layout(self.pane_layout_cb.currentText()).pad(24).gap(20).build())
        out_name = f"{os.path.splitext(os.path.basename(item.path))[0]}_jr_vs_sr.mp4"; output_path = os.path.join(str(OUTPUT_DIR), out_name)
        sound_params = (self.sound_preset_cb.currentText(), True, 0.1)
        self._exporter = VideoExporter(cfg, output_path, 30, sound_params, None, 0.5, self.encoder_cb.currentText(), bgm_path=self._bgm_path if self._bgm_path else None, generate_thumbnail=True, generate_chapters=True)
        self._exporter.progress.connect(lambda p: self._update_progress(item, p)); self._exporter.status.connect(lambda s: log.info(s)); self._exporter.finished_ok.connect(lambda p: self._on_export_done(item, p)); self._exporter.error.connect(lambda e: self._on_export_failed(item, e)); self._exporter.start()

    def _update_progress(self, item: FileItem, val: int) -> None:
        item.status = f"Rendering {val}% ({self._batch_done}/{self._batch_total})"; self._refresh_table()

    def _on_export_done(self, item: FileItem, path: str) -> None:
        item.status = "Done"; item.output_path = path; self._refresh_table(); self._process_next_in_batch()

    def _on_export_failed(self, item: FileItem, err: str) -> None:
        item.status = "Failed"; self._refresh_table(); QMessageBox.warning(self, "Export Failed", f"{os.path.basename(item.path)}:\n{err}"); self._process_next_in_batch()

    def _cancel_export(self) -> None:
        self._batch_cancel = True
        if self._exporter: self._exporter.cancel()
        self._batch_queue.clear()
        if self._current_item: self._current_item.status = "Cancelled"; self._refresh_table()

    def _on_batch_complete(self, was_cancelled: bool) -> None:
        self.export_btn.show(); self.cancel_btn.hide(); self._current_item = None
        done = sum(1 for it in self._items if it.status == "Done"); failed = sum(1 for it in self._items if it.status == "Failed")
        msg = f"Done: {done}, Failed: {failed}" + (f", Cancelled: {self._batch_total - done - failed}" if was_cancelled else "")
        QMessageBox.information(self, "Batch Cancelled" if was_cancelled else "Batch Complete", msg)

    def _load_settings(self) -> None:
        self._loading_settings = True
        try:
            if os.path.exists(str(SETTINGS_FILE)):
                with open(str(SETTINGS_FILE), "r", encoding="utf-8") as f:
                    data = json.load(f)
                s = data.get("project1", {})
                if "theme" in s: self.theme_cb.setCurrentText(s["theme"])
                if "fmt" in s: self.export_format_cb.setCurrentText(s["fmt"])
                if "layout" in s: self.pane_layout_cb.setCurrentText(s["layout"])
                if "audio" in s: self.sound_preset_cb.setCurrentText(s["audio"])
                if "encoder" in s: self.encoder_cb.setCurrentText(s["encoder"])
                if "kb" in s: self.kb_chk.setChecked(s["kb"])
                if "wm" in s: self.watermark_chk.setChecked(s["wm"])
                if "wm_text" in s: self.watermark_edit.setText(s["wm_text"])
                if "j_wpm" in s: self.j_wpm.setValue(s["j_wpm"])
                if "s_wpm" in s: self.s_wpm.setValue(s["s_wpm"])
                if "bgm" in s: self.bgm_edit.setText(s["bgm"])
        except Exception as e:
            log.warning("Failed to load settings: %s", e)
        self._loading_settings = False

    def _save_settings(self) -> None:
        data = {}
        if os.path.exists(str(SETTINGS_FILE)):
            try:
                with open(str(SETTINGS_FILE), "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception: pass
        data["project1"] = {
            "theme": self.theme_cb.currentText(),
            "fmt": self.export_format_cb.currentText(),
            "layout": self.pane_layout_cb.currentText(),
            "audio": self.sound_preset_cb.currentText(),
            "encoder": self.encoder_cb.currentText(),
            "kb": self.kb_chk.isChecked(),
            "wm": self.watermark_chk.isChecked(),
            "wm_text": self.watermark_edit.text(),
            "j_wpm": self.j_wpm.value(),
            "s_wpm": self.s_wpm.value(),
            "bgm": self.bgm_edit.text(),
        }
        try:
            with open(str(SETTINGS_FILE), "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            log.warning("Failed to save settings: %s", e)

    def closeEvent(self, event) -> None:
        if self._exporter: self._exporter.cancel()
        self._save_settings(); self.preview.stop(); super().closeEvent(event)

STYLE = "QMainWindow, QDialog { background: #1e1e2e; } QGroupBox { color: #cdd6f4; font-weight: bold; font-size: 13px; border: 1px solid #45475a; border-radius: 8px; margin-top: 12px; padding-top: 16px; } QGroupBox::title { subcontrol-origin: margin; left: 12px; } QLabel { color: #cdd6f4; } QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTimeEdit { background: #313244; color: #cdd6f4; border: 1px solid #45475a; border-radius: 4px; padding: 4px 8px; min-height: 24px; } QComboBox::drop-down { border: none; } QCheckBox { color: #cdd6f4; spacing: 6px; } QPushButton { background: #313244; color: #cdd6f4; border: 1px solid #45475a; border-radius: 6px; padding: 6px 16px; font-size: 13px; } QPushButton:hover { background: #45475a; } QProgressBar { background: #313244; border: none; border-radius: 4px; text-align: center; color: #cdd6f4; min-height: 20px; } QProgressBar::chunk { background: #89b4fa; border-radius: 4px; } QTableWidget { background: #181825; color: #cdd6f4; gridline-color: #313244; border: none; outline: none; } QTableWidget::item { padding: 4px 6px; border-bottom: 1px solid #11111b; } QTableWidget::item:selected { background: #89b4fa; color: #1e1e2e; } QHeaderView::section { background: #1e1e1e; color: #a6adc8; border: none; padding: 4px; font-size: 10px; } QToolTip { background: #313244; color: #cdd6f4; border: 1px solid #45475a; border-radius: 4px; padding: 4px 8px; }"
def main() -> int:
    setup_project_logging("project1_jr_vs_sr")
    app = QApplication(sys.argv); app.setStyle("Fusion"); app.setStyleSheet(STYLE)
    pal = QPalette(); pal.setColor(QPalette.Window, QColor("#1e1e2e")); pal.setColor(QPalette.WindowText, QColor("#cdd6f4")); pal.setColor(QPalette.Base, QColor("#313244")); pal.setColor(QPalette.Text, QColor("#cdd6f4")); pal.setColor(QPalette.Button, QColor("#313244")); pal.setColor(QPalette.ButtonText, QColor("#cdd6f4")); pal.setColor(QPalette.Highlight, QColor("#89b4fa")); pal.setColor(QPalette.HighlightedText, QColor("#1e1e2e")); app.setPalette(pal)
    win = MainWindow(); win.show(); return app.exec()
if __name__ == "__main__": sys.exit(main())