"""
Project 2: Code Typing Video MultiWindow Generator Studio
"""
from __future__ import annotations
import sys, os, json
from dataclasses import dataclass
from typing import List, Optional, Tuple
from codestudio import *
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QScrollArea

EDITOR_LAYOUTS = {"1 Window (Single)": 1, "2 Windows (Split)": 2, "4 Windows (Grid)": 4}

@dataclass
class FileItem:
    path: str; checked: bool = False; status: str = "Pending"; output_path: Optional[str] = None
    est_duration: float = 0.0; size_bytes: int = 0; code: str = ""

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
        super().__init__(); self.setWindowTitle("Project 2: MultiWindow Studio"); self.setMinimumSize(1200, 800); self.resize(1400, 900)
        self._items: List[FileItem] = []; self._item_paths = set(); self._exporter: Optional[VideoExporter] = None; self._loading_settings = False; self._bg_image_path = ""
        self._batch_queue: List[FileItem] = []; self._batch_total = 0; self._batch_done = 0; self._batch_cancel = False; self._current_item: Optional[FileItem] = None
        self._preview_idx = 0
        self._audio_player = QMediaPlayer(self); self._audio_out = QAudioOutput(self); self._audio_player.setAudioOutput(self._audio_out)
        ensure_examples_for_project("multi_window")
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
        self.layout_cb = QComboBox(); self.layout_cb.addItems(list(EDITOR_LAYOUTS.keys())); self.layout_cb.setCurrentText("2 Windows (Split)"); fl.addRow("Editor Layout:", self.layout_cb)
        self.encoder_cb = QComboBox(); self.encoder_cb.addItems(list(ENCODERS.keys())); fl.addRow("Encoder:", self.encoder_cb)
        self.kb_chk = QCheckBox("Show Keyboard"); self.kb_chk.setChecked(True); fl.addRow(self.kb_chk)
        self.watermark_chk = QCheckBox("Watermark"); fl.addRow(self.watermark_chk)
        self.watermark_edit = QLineEdit(); self.watermark_edit.setPlaceholderText("Watermark Text"); fl.addRow("WM Text:", self.watermark_edit)
        self.wpm_sp = QSpinBox(); self.wpm_sp.setRange(30, 300); self.wpm_sp.setValue(100); fl.addRow("WPM:", self.wpm_sp)
        cl.addWidget(grp)
        ag = QGroupBox("Per-Pane Audio Profiles"); al = QFormLayout(ag)
        self.per_pane_audio_chk = QCheckBox("Enable Per-Pane Audio"); self.per_pane_audio_chk.setChecked(True); al.addRow(self.per_pane_audio_chk)
        self.pane_audio_cbs = []
        for i in range(4):
            cb = QComboBox(); cb.addItems(list(SOUND_PRESETS.keys())); cb.setCurrentText("Mechanical"); al.addRow(f"Pane {i+1}:", cb); self.pane_audio_cbs.append(cb)
        cl.addWidget(ag)
        btn_row = QHBoxLayout()
        self.export_btn = QPushButton("🎬 Export Checked"); self.export_btn.setStyleSheet("QPushButton { background: #a6e3a1; color: #1e1e2e; font-weight: bold; border: none; border-radius: 6px; padding: 10px; } QPushButton:hover { background: #94e2d5; }"); self.export_btn.clicked.connect(self._start_export); btn_row.addWidget(self.export_btn, 1)
        self.cancel_btn = QPushButton("⏹ Cancel Batch"); self.cancel_btn.setStyleSheet("QPushButton { background: #f38ba8; color: #1e1e2e; font-weight: bold; border: none; border-radius: 6px; padding: 10px; } QPushButton:hover { background: #eba0ac; }"); self.cancel_btn.clicked.connect(self._cancel_export); self.cancel_btn.hide(); btn_row.addWidget(self.cancel_btn, 1)
        cl.addLayout(btn_row)
        scroll.setWidget(container); rl.addWidget(scroll); root.addWidget(right, 1)

    def _connect_signals(self) -> None:
        for w in [self.theme_cb, self.export_format_cb, self.layout_cb, self.encoder_cb, self.wpm_sp, self.watermark_edit] + self.pane_audio_cbs:
            if hasattr(w, 'currentTextChanged'): w.currentTextChanged.connect(self._on_param_changed)
            if hasattr(w, 'valueChanged'): w.valueChanged.connect(self._on_param_changed)
        self.kb_chk.toggled.connect(self._on_param_changed)
        self.watermark_chk.toggled.connect(self._on_param_changed)
        self.per_pane_audio_chk.toggled.connect(self._on_param_changed)

    def _on_param_changed(self, *_) -> None:
        self._update_est_durations(); self._rebuild_preview()

    def _select_all(self) -> None:
        for it in self._items: it.checked = True
        self._preview_idx = 0; self._refresh_table(); self._update_stats(); self._rebuild_preview()

    def _select_none(self) -> None:
        for it in self._items: it.checked = False
        self._preview_idx = 0; self._refresh_table(); self._update_stats(); self._rebuild_preview()

    def _load_item(self, path: str) -> Optional[FileItem]:
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f: code = f.read()
        except Exception: return None
        item = FileItem(path=path, checked=True, code=code, size_bytes=os.path.getsize(path))
        self._recompute_est(item)
        return item

    def _recompute_est(self, item: FileItem) -> None:
        item.est_duration = TypingAnimator.estimate_duration(item.code, self.wpm_sp.value(), 0.5, 1.5, 0.15, 0.01)

    def _update_est_durations(self) -> None:
        for it in self._items: self._recompute_est(it)
        self._refresh_table(); self._update_stats()

    def _scan_input_dir(self) -> None:
        self._items.clear(); self._item_paths.clear(); self._preview_idx = 0
        if not os.path.isdir(str(INPUT_DIR)): return
        for fname in sorted(os.listdir(str(INPUT_DIR))):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in CODE_EXTENSIONS: continue
            p = os.path.join(str(INPUT_DIR), fname)
            item = self._load_item(p)
            if item: self._items.append(item); self._item_paths.add(p)
        self._refresh_table(); self._update_stats(); self._rebuild_preview()

    def _add_files(self) -> None:
        es = " ".join(f"*{e}" for e in sorted(CODE_EXTENSIONS))
        paths, _ = QFileDialog.getOpenFileNames(self, "Select Code Files", "", f"Code Files ({es})")
        added = False
        for p in paths:
            if p in self._item_paths: continue
            item = self._load_item(p)
            if item: self._items.append(item); self._item_paths.add(p); added = True
        if added: self._preview_idx = 0; self._refresh_table(); self._update_stats(); self._rebuild_preview()

    def _refresh_table(self) -> None:
        self.file_table.blockSignals(True); self.file_table.setRowCount(len(self._items))
        for i, item in enumerate(self._items):
            chk = QTableWidgetItem(); chk.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled); chk.setCheckState(Qt.CheckState.Checked if item.checked else Qt.CheckState.Unchecked); self.file_table.setItem(i, 0, chk)
            self.file_table.setItem(i, 1, QTableWidgetItem(os.path.basename(item.path)))
            ext = os.path.splitext(item.path)[1].lower(); lang = EXT_TO_LANGUAGE.get(ext, "Text")[:8]
            self.file_table.setItem(i, 2, QTableWidgetItem(lang))
            est_item = QTableWidgetItem(format_duration(item.est_duration))
            est_item.setToolTip(f"Estimated: {item.est_duration:.1f}s\nSize: {item.size_bytes} bytes\nLines: {item.code.count(chr(10))+1}")
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

    def _build_panes(self) -> Tuple[MultiPaneRenderer, List[SimpleSoundGen]]:
        fmt = self.export_format_cb.currentText(); fmt_data = EXPORT_FORMATS.get(fmt, EXPORT_FORMATS["YouTube (16:9)"]); res_name = "1920x1080"; w, h = fmt_data["resolutions"].get(res_name, (1920, 1080)); theme_name = self.theme_cb.currentText(); theme = THEMES[theme_name]
        num_panes = EDITOR_LAYOUTS.get(self.layout_cb.currentText(), 2)
        checked = [it for it in self._items if it.checked]
        if not checked: code_files = [("# No files selected.", "empty.py")] * num_panes
        else:
            code_files = []
            for i in range(num_panes):
                cur_item = checked[(self._preview_idx + i) % len(checked)]
                code_files.append((cur_item.code, os.path.basename(cur_item.path)))
                
        panes = []; sound_gens = []
        wm_text = self.watermark_edit.text() if self.watermark_edit.text() else "Code Typing Studio"
        for i in range(num_panes):
            code, filename = code_files[i]; ext = os.path.splitext(filename)[1].lower(); lang = EXT_TO_LANGUAGE.get(ext, "Text")
            if num_panes == 1: pw, ph = w, h
            elif num_panes == 2: pw, ph = (w - 20) // 2, h
            else: pw, ph = (w - 20) // 2, (h - 20) // 2
            kb_overlay = KeyboardOverlay(pw, ph, "QWERTY", theme, 0.82, (ph - 48) // 3, "bottom_center") if self.kb_chk.isChecked() else None
            fs = CodeRenderer.auto_font_size(code.count("\n")+1, pw, ph, 24, True, True, 4, code, "Consolas", kb_overlay.height_needed() if kb_overlay else 0)
            renderer = CodeRenderer(pw, ph, theme_name, "Consolas", fs, True, True, 24, 4, filename, lang, kb_overlay, None, code.count("\n")+1, True, self.watermark_chk.isChecked() and i == 0, wm_text, getattr(theme, "junior_accent", "#7aa2f7"), code=code)
            animator = TypingAnimator(code, TypingProfile(wpm=self.wpm_sp.value()), seed=i * 1000 + 42); panes.append((renderer, animator))
            if self.per_pane_audio_chk.isChecked():
                preset = self.pane_audio_cbs[i].currentText()
                sound_gens.append(SimpleSoundGen(sr=44100, preset=preset, stereo=True, reverb_amt=0.1))
            else:
                sound_gens.append(SimpleSoundGen(sr=44100, preset=self.pane_audio_cbs[0].currentText(), stereo=True, reverb_amt=0.1))
        mp = MultiPaneRenderer(panes=panes, frame_w=w, frame_h=h, num_panes=num_panes, pane_layout="Side by Side (50/50)")
        return mp, sound_gens

    def _rebuild_preview(self, *_) -> None:
        if self._loading_settings: return
        checked = [it for it in self._items if it.checked]
        if not checked:
            self.preview.set_info("No files selected", 0.0, 0, 0, 0.0, 0.0, 0.0)
            mp, sound_gens = self._build_panes(); sg = sound_gens[0] if sound_gens else None
            self.preview.set_multi_pane(mp, sg, 0.5)
            return

        if self._preview_idx >= len(checked): self._preview_idx = 0
        cur_item = checked[self._preview_idx]
        batch_est = sum(it.est_duration for it in checked)
        min_est = min(it.est_duration for it in checked)
        max_est = max(it.est_duration for it in checked)
        self.preview.set_info(os.path.basename(cur_item.path), cur_item.est_duration, self._preview_idx + 1, len(checked), batch_est, min_est, max_est)
        
        mp, sound_gens = self._build_panes(); sg = sound_gens[0] if sound_gens else None
        self.preview.set_multi_pane(mp, sg, 0.5)

    def _start_export(self) -> None:
        checked = [it for it in self._items if it.checked]
        if not checked: QMessageBox.information(self, "Nothing Selected", "Check at least one file."); return
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
        num_panes = EDITOR_LAYOUTS.get(self.layout_cb.currentText(), 2)
        ext = os.path.splitext(item.path)[1].lower(); lang = EXT_TO_LANGUAGE.get(ext, "Text")
        wm_text = self.watermark_edit.text() if self.watermark_edit.text() else "Code Typing Studio"
        pane_layout_str = "Side by Side (50/50)"  # MultiPaneRenderer computes rectangles dynamically based on num_panes
        cfg = (ExportConfigBuilder().junior_code(item.code).senior_code(item.code).name(os.path.basename(item.path)).fmt(self.export_format_cb.currentText()).res("1920x1080").theme(self.theme_cb.currentText()).j_font("Consolas").s_font("Consolas").font_size_auto(True).font_size(22).chrome(True).ln(True).cursor_glow(True).watermark(self.watermark_chk.isChecked()).watermark_text(wm_text).bg_image_path("").show_kb(self.kb_chk.isChecked()).kb_layout("QWERTY").kb_pos("bottom_center").kb_op(0.82).j_wpm(self.wpm_sp.value()).j_pause(0.15).s_wpm(self.wpm_sp.value()).s_pause(0.05).start_pause(0.5).end_pause(1.5).lang(lang).pane_layout(pane_layout_str).pad(24).gap(20).build())
        out_name = f"{os.path.splitext(os.path.basename(item.path))[0]}_multiwindow.mp4"; output_path = os.path.join(str(OUTPUT_DIR), out_name)
        sound_gens = []
        if self.per_pane_audio_chk.isChecked():
            for i in range(num_panes): sound_gens.append(SimpleSoundGen(sr=44100, preset=self.pane_audio_cbs[i].currentText(), stereo=True, reverb_amt=0.1))
        else: sound_gens = [SimpleSoundGen(sr=44100, preset=self.pane_audio_cbs[0].currentText(), stereo=True, reverb_amt=0.1)] * num_panes
        self._exporter = VideoExporter(cfg, output_path, 30, None, sound_gens, 0.5, self.encoder_cb.currentText())
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
                s = data.get("project2", {})
                if "theme" in s: self.theme_cb.setCurrentText(s["theme"])
                if "fmt" in s: self.export_format_cb.setCurrentText(s["fmt"])
                if "layout" in s: self.layout_cb.setCurrentText(s["layout"])
                if "encoder" in s: self.encoder_cb.setCurrentText(s["encoder"])
                if "kb" in s: self.kb_chk.setChecked(s["kb"])
                if "wm" in s: self.watermark_chk.setChecked(s["wm"])
                if "wm_text" in s: self.watermark_edit.setText(s["wm_text"])
                if "wpm" in s: self.wpm_sp.setValue(s["wpm"])
                if "per_pane_audio" in s: self.per_pane_audio_chk.setChecked(s["per_pane_audio"])
                if isinstance(s.get("pane_audio"), list):
                    for i, val in enumerate(s["pane_audio"]):
                        if i < len(self.pane_audio_cbs):
                            self.pane_audio_cbs[i].setCurrentText(val)
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
        data["project2"] = {
            "theme": self.theme_cb.currentText(),
            "fmt": self.export_format_cb.currentText(),
            "layout": self.layout_cb.currentText(),
            "encoder": self.encoder_cb.currentText(),
            "kb": self.kb_chk.isChecked(),
            "wm": self.watermark_chk.isChecked(),
            "wm_text": self.watermark_edit.text(),
            "wpm": self.wpm_sp.value(),
            "per_pane_audio": self.per_pane_audio_chk.isChecked(),
            "pane_audio": [cb.currentText() for cb in self.pane_audio_cbs],
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
    setup_project_logging("project2_multi_window")
    app = QApplication(sys.argv); app.setStyle("Fusion"); app.setStyleSheet(STYLE)
    pal = QPalette(); pal.setColor(QPalette.Window, QColor("#1e1e2e")); pal.setColor(QPalette.WindowText, QColor("#cdd6f4")); pal.setColor(QPalette.Base, QColor("#313244")); pal.setColor(QPalette.Text, QColor("#cdd6f4")); pal.setColor(QPalette.Button, QColor("#313244")); pal.setColor(QPalette.ButtonText, QColor("#cdd6f4")); pal.setColor(QPalette.Highlight, QColor("#89b4fa")); pal.setColor(QPalette.HighlightedText, QColor("#1e1e2e")); app.setPalette(pal)
    win = MainWindow(); win.show(); return app.exec()
if __name__ == "__main__": sys.exit(main())