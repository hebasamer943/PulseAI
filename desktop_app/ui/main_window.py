from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QSlider, QProgressBar,
    QStackedWidget, QMessageBox, QScrollArea, QSplitter,
    QFileDialog, QInputDialog
)

import pyqtgraph as pg

from desktop_app.io_sources import PTBDBSource, CSVSource, BeatSample
from desktop_app.image_pipeline import extract_ecg_from_image
from desktop_app.live_serial import SerialECGSource
from desktop_app.signal_quality import evaluate_quality


def card(title: str):
    frame = QFrame()
    frame.setObjectName("Card")
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(16, 16, 16, 16)
    lay.setSpacing(10)

    h = QLabel(title)
    h.setObjectName("H2")
    lay.addWidget(h)
    return frame, lay


@tf.keras.utils.register_keras_serializable()
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, length, d_model, **kwargs):
        super().__init__(**kwargs)
        self.length = length
        self.d_model = d_model

        pos = np.arange(length)[:, None]
        i = np.arange(d_model)[None, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        angles = pos * angle_rates

        pe = np.zeros((length, d_model), dtype=np.float32)
        pe[:, 0::2] = np.sin(angles[:, 0::2])
        pe[:, 1::2] = np.cos(angles[:, 1::2])
        self.pe = tf.constant(pe[None, :, :])

    def call(self, x):
        return x + self.pe[:, :tf.shape(x)[1], :]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"length": self.length, "d_model": self.d_model})
        return cfg


class ModelWrapper:
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir

        self.resnet_path = self.models_dir / "ptbdb_resnet1d_fix.keras"
        self.tcn_path = self.models_dir / "ptbdb_tcn.keras"
        self.trans_path = self.models_dir / "ptbdb_transformer_fix.keras"

        self.m_resnet = None
        self.m_tcn = None
        self.m_trans = None

        self.w_resnet = 0.50
        self.w_tcn = 0.30
        self.w_trans = 0.20

        self.unc_low = 0.30
        self.unc_high = 0.70

    def load(self):
        if not self.resnet_path.exists():
            raise FileNotFoundError("Missing Models/ptbdb_resnet1d_fix.keras")
        self.m_resnet = tf.keras.models.load_model(self.resnet_path, compile=False)
        print("Loaded ResNet:", self.resnet_path)

        if not self.tcn_path.exists():
            print("TCN missing (optional):", self.tcn_path)
        if not self.trans_path.exists():
            print("Transformer missing (optional):", self.trans_path)

    def _ensure_tcn(self):
        if self.m_tcn is None:
            if not self.tcn_path.exists():
                raise FileNotFoundError("Missing Models/ptbdb_tcn.keras")
            self.m_tcn = tf.keras.models.load_model(self.tcn_path, compile=False)
            print("Loaded TCN:", self.tcn_path)

    def _ensure_trans(self):
        if self.m_trans is None:
            if not self.trans_path.exists():
                raise FileNotFoundError("Missing Models/ptbdb_transformer_fix.keras")
            self.m_trans = tf.keras.models.load_model(
                self.trans_path,
                compile=False,
                custom_objects={"PositionalEncoding": PositionalEncoding},
            )
            print("Loaded Transformer:", self.trans_path)

    def _prep(self, beat: np.ndarray) -> np.ndarray:
        x = np.asarray(beat, dtype=np.float32).reshape(-1)
        if x.size > 187:
            x = x[:187]
        if x.size < 187:
            x = np.pad(x, (0, 187 - x.size))
        return x.reshape(1, 187, 1)

    def predict_proba_abn(self, beat: np.ndarray) -> float:
        if self.m_resnet is None:
            self.load()

        X = self._prep(beat)
        p_res = float(self.m_resnet.predict(X, verbose=0).reshape(-1)[0])

        if (p_res < self.unc_low) or (p_res > self.unc_high):
            return p_res

        self._ensure_tcn()
        self._ensure_trans()

        p_tcn = float(self.m_tcn.predict(X, verbose=0).reshape(-1)[0])
        p_tr = float(self.m_trans.predict(X, verbose=0).reshape(-1)[0])

        p_final = (self.w_resnet * p_res) + (self.w_tcn * p_tcn) + (self.w_trans * p_tr)
        return float(p_final)


class MainWindow(QMainWindow):
    def __init__(self, project_root: Path):
        super().__init__()
        self.project_root = project_root
        self.dataset_dir = project_root / "Dataset"
        self.models_dir = project_root / "Models"
        self.reports_dir = project_root / "Reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Default source
        self.source = PTBDBSource(self.dataset_dir)

        # Model
        self.model = ModelWrapper(self.models_dir)
        self.model.load()

        # Hardware
        self.serial_source: SerialECGSource | None = None
        self.hw_running = False
        self.last_bpm_display = 0.0

        # Pending image beat
        self._pending_image_beat = None

        # Thresholds
        self.threshold = 0.50      # CSV/Dataset
        self.hw_threshold = 0.65   # Hardware calibration

        # State
        self.auto_on = False
        self.session_rows = []
        self._error_showing = False

        # UI
        self.setWindowTitle("ECG Monitor")
        self.setMinimumSize(1100, 680)

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self._build_home()
        self._build_monitor()
        self.stack.setCurrentIndex(0)

        # Timers
        self.auto_timer = QTimer(self)
        self.auto_timer.setInterval(900)
        self.auto_timer.timeout.connect(self.next_beat)

        self.hw_timer = QTimer(self)
        self.hw_timer.setInterval(200)
        self.hw_timer.timeout.connect(self._update_hardware_ui)

        self.blink_timer = QTimer(self)
        self.blink_timer.setInterval(350)
        self._blink_state = True
        self.blink_timer.timeout.connect(self._blink)
        self.blink_timer.start()

        self._apply_mode_ui()

    def _stop_auto(self):
        self.auto_on = False
        self.auto_timer.stop()
        if hasattr(self, "btn_auto"):
            self.btn_auto.setText("Auto: OFF")

    def closeEvent(self, event):
        try:
            if self.serial_source is not None:
                self.serial_source.stop()
        except Exception:
            pass
        event.accept()

    # ---------- HOME ----------
    def _build_home(self):
        page = QWidget()
        root = QHBoxLayout(page)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(18)

        left, left_lay = card("Welcome")
        left.setMaximumWidth(420)

        t = QLabel("Pulse ECG Monitor")
        t.setObjectName("H1")
        left_lay.addWidget(t)

        self.btn_csv = QPushButton("Upload CSV")
        self.btn_csv.clicked.connect(self.load_csv_file)
        left_lay.addWidget(self.btn_csv)

        self.btn_img = QPushButton("Upload ECG Image (PNG/JPG)")
        self.btn_img.clicked.connect(self.load_image_file)
        left_lay.addWidget(self.btn_img)

        self.btn_hw = QPushButton("Connect Hardware")
        self.btn_hw.clicked.connect(self.connect_hardware)
        left_lay.addWidget(self.btn_hw)

        self.btn_start = QPushButton("Start Monitor")
        self.btn_start.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        left_lay.addWidget(self.btn_start)

        left_lay.addStretch(1)

        right, right_lay = card("Preview")
        self.preview_plot = pg.PlotWidget()
        self._setup_plot(self.preview_plot)
        right_lay.addWidget(self.preview_plot, 1)

        x = np.arange(187)
        y = 0.45 + 0.05 * np.sin(x / 25)
        self.preview_plot.plot(x, y, pen=pg.mkPen((255, 60, 60, 190), width=2))

        root.addWidget(left, 0)
        root.addWidget(right, 1)
        self.stack.addWidget(page)

    # ---------- MONITOR ----------
    def _build_monitor(self):
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(18, 18, 18, 18)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        side_container = QWidget()
        side = QVBoxLayout(side_container)
        side.setContentsMargins(0, 0, 0, 0)
        side.setSpacing(14)

        btn_back = QPushButton("← Back to Home")
        btn_back.clicked.connect(self._go_home)
        side.addWidget(btn_back)

        # Hardware card
        self.hw_card, hw_lay = card("Hardware")
        self.lbl_hw_status = QLabel("Status: —")
        self.lbl_hw_status.setObjectName("Muted")
        hw_lay.addWidget(self.lbl_hw_status)

        self.lbl_hw_bpm = QLabel("BPM: —")
        self.lbl_hw_bpm.setStyleSheet("font-size: 30pt; font-weight: 900;")
        hw_lay.addWidget(self.lbl_hw_bpm)

        row = QHBoxLayout()
        self.btn_hw_start = QPushButton("Start")
        self.btn_hw_start.clicked.connect(self.hw_start)
        row.addWidget(self.btn_hw_start)

        self.btn_hw_stop = QPushButton("Stop")
        self.btn_hw_stop.clicked.connect(self.hw_stop)
        self.btn_hw_stop.setEnabled(False)
        row.addWidget(self.btn_hw_stop)

        hw_lay.addLayout(row)
        side.addWidget(self.hw_card)

        # Prediction
        pred_card, pred_lay = card("Prediction")
        self.badge = QLabel("—")
        self.badge.setObjectName("Badge")
        pred_lay.addWidget(self.badge)

        self.lbl_true = QLabel("True: —")
        self.lbl_pred = QLabel("Pred: —")
        self.lbl_prob = QLabel("P(abn): —")
        self.lbl_idx = QLabel("Index: —")
        for w in (self.lbl_true, self.lbl_pred, self.lbl_prob, self.lbl_idx):
            w.setObjectName("Muted")
            pred_lay.addWidget(w)
        side.addWidget(pred_card)

        # Signal Quality
        self.q_card, q_lay = card("Signal Quality")
        self.lbl_quality = QLabel("Quality: —")
        self.lbl_quality.setObjectName("Muted")
        q_lay.addWidget(self.lbl_quality)

        self.pb_quality = QProgressBar()
        self.pb_quality.setRange(0, 100)
        self.pb_quality.setValue(0)
        q_lay.addWidget(self.pb_quality)

        self.lbl_qhint = QLabel("—")
        self.lbl_qhint.setObjectName("Muted")
        q_lay.addWidget(self.lbl_qhint)
        side.addWidget(self.q_card)

        # Controls (CSV/Dataset only)
        c_card, c_lay = card("Controls")

        self.btn_next = QPushButton("Next beat")
        self.btn_next.clicked.connect(self.next_beat)
        c_lay.addWidget(self.btn_next)

        self.btn_auto = QPushButton("Auto: OFF")
        self.btn_auto.clicked.connect(self.toggle_auto)
        c_lay.addWidget(self.btn_auto)

        self.btn_save = QPushButton("Save report")
        self.btn_save.clicked.connect(self.save_report)
        c_lay.addWidget(self.btn_save)

        self.btn_reset = QPushButton("Reset session")
        self.btn_reset.clicked.connect(self.reset_session)
        c_lay.addWidget(self.btn_reset)

        self.lbl_thr = QLabel(f"Threshold: {self.threshold:.2f}")
        self.lbl_thr.setObjectName("Muted")
        c_lay.addWidget(self.lbl_thr)

        self.slider_thr = QSlider(Qt.Horizontal)
        self.slider_thr.setMinimum(1)
        self.slider_thr.setMaximum(99)
        self.slider_thr.setValue(int(self.threshold * 100))
        self.slider_thr.valueChanged.connect(self._thr_changed)
        c_lay.addWidget(self.slider_thr)

        side.addWidget(c_card)

        # Session
        s_card, s_lay = card("Session")
        self.lbl_seen = QLabel("Seen: 0")
        self.lbl_true_abn = QLabel("TrueAbn: 0")
        self.lbl_pred_abn = QLabel("PredAbn: 0")
        for w in (self.lbl_seen, self.lbl_true_abn, self.lbl_pred_abn):
            w.setObjectName("Muted")
            s_lay.addWidget(w)
        side.addWidget(s_card)

        side.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(side_container)
        scroll.setMinimumWidth(380)
        splitter.addWidget(scroll)

        plot_card, plot_lay = card("ECG Waveform")
        self.plot = pg.PlotWidget()
        self._setup_plot(self.plot)
        plot_lay.addWidget(self.plot, 1)
        self.curve = self.plot.plot([], [], pen=pg.mkPen((255, 60, 60, 210), width=2))

        splitter.addWidget(plot_card)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.stack.addWidget(page)

    def _setup_plot(self, w: pg.PlotWidget):
        w.setBackground("#0b0b0f")
        w.showGrid(x=True, y=True, alpha=0.18)
        axis_pen = pg.mkPen((255, 255, 255, 120))
        w.getAxis("left").setPen(axis_pen)
        w.getAxis("bottom").setPen(axis_pen)
        w.getAxis("left").setTextPen(axis_pen)
        w.getAxis("bottom").setTextPen(axis_pen)
        w.getAxis("left").setLabel("Amplitude", color="#ff3c3c")
        w.getAxis("bottom").setLabel("Sample", color="#ff3c3c")

    def _go_home(self):
        if isinstance(self.source, SerialECGSource) and self.hw_running:
            self.hw_stop()
        else:
            self._stop_auto()
        self.stack.setCurrentIndex(0)

    def _apply_mode_ui(self):
        is_hw = isinstance(self.source, SerialECGSource)

        self.hw_card.setVisible(is_hw)
        self.q_card.setVisible(is_hw)

        # Hide CSV controls in hardware mode
        self.btn_next.setVisible(not is_hw)
        self.btn_auto.setVisible(not is_hw)
        self.lbl_thr.setVisible(not is_hw)
        self.slider_thr.setVisible(not is_hw)

        self.btn_hw_start.setEnabled(is_hw and not self.hw_running)
        self.btn_hw_stop.setEnabled(is_hw and self.hw_running)

    def _thr_changed(self, v: int):
        self.threshold = v / 100.0
        self.lbl_thr.setText(f"Threshold: {self.threshold:.2f}")

    def _set_badge_normal(self):
        self.badge.setText("NORMAL")
        self.badge.setStyleSheet(
            "color: #50ff8c; font-weight: 900; padding: 10px 14px;"
            "border-radius: 12px; background: rgba(80,255,140,0.12);"
            "border: 1px solid rgba(80,255,140,0.35);"
        )

    def _set_badge_abnormal(self):
        self.badge.setText("ABNORMAL")
        self.badge.setStyleSheet(
            "color: #ff3c3c; font-weight: 900; padding: 10px 14px;"
            "border-radius: 12px; background: rgba(255,60,60,0.15);"
            "border: 1px solid rgba(255,60,60,0.40);"
        )

    # ---------- Hardware ----------
    def hw_start(self):
        if not isinstance(self.source, SerialECGSource):
            return
        try:
            self.source.start()
        except Exception as e:
            QMessageBox.critical(self, "Hardware Error", str(e))
            return

        self.hw_running = True
        self.btn_hw_start.setEnabled(False)
        self.btn_hw_stop.setEnabled(True)

        # enable autorange while running
        try:
            self.plot.enableAutoRange("xy", True)
        except Exception:
            pass

        self.hw_timer.start()

        # auto classification
        self.auto_on = True
        self.auto_timer.setInterval(200)
        self.auto_timer.start()

        self.lbl_hw_status.setText(f"Status: {self.source.port} • RUNNING")
        self._apply_mode_ui()

    def hw_stop(self):
        self._stop_auto()
        self.hw_timer.stop()

        try:
            if isinstance(self.source, SerialECGSource):
                self.source.stop()
        except Exception:
            pass

        self.hw_running = False
        self.btn_hw_start.setEnabled(True)
        self.btn_hw_stop.setEnabled(False)

        # freeze plot scale
        try:
            self.plot.disableAutoRange()
        except Exception:
            pass

        if self.last_bpm_display > 0:
            self.lbl_hw_bpm.setText(f"Last BPM: {self.last_bpm_display:.0f}")
        else:
            self.lbl_hw_bpm.setText("Last BPM: —")

        self.lbl_hw_status.setText("Status: STOPPED")
        self._apply_mode_ui()

    def _update_hardware_ui(self):
        if not isinstance(self.source, SerialECGSource):
            return

        port = getattr(self.source, "port", "—")
        run = "RUNNING" if self.hw_running else "STOPPED"
        self.lbl_hw_status.setText(f"Status: {port} • {run}")

        bpm = float(getattr(self.source, "last_bpm", 0.0) or 0.0)
        if bpm > 0:
            self.last_bpm_display = bpm
            self.lbl_hw_bpm.setText(f"BPM: {bpm:.0f}")
        else:
            if self.hw_running and self.last_bpm_display > 0:
                self.lbl_hw_bpm.setText(f"BPM: {self.last_bpm_display:.0f}")

        buf = getattr(self.source, "sig_buf", None)
        if buf and len(buf) > 50:
            sr = int(getattr(self.source, "sample_rate", 150) or 150)
            n = min(len(buf), max(200, sr * 5))
            seg = np.asarray(buf[-n:], dtype=np.float32)

            seg = seg - np.median(seg)
            seg = seg / (np.max(np.abs(seg)) + 1e-9)

            x = np.arange(seg.size)
            y = 0.45 + 0.45 * seg
            self.curve.setData(x, y)

    #  CSV/Dataset 
    def toggle_auto(self):
        self.auto_on = not self.auto_on
        if self.auto_on:
            self.auto_timer.setInterval(900)
            self.auto_timer.start()
            self.btn_auto.setText("Auto: ON")
        else:
            self.auto_timer.stop()
            self.btn_auto.setText("Auto: OFF")

    def reset_session(self):
        self.session_rows.clear()
        self.lbl_seen.setText("Seen: 0")
        self.lbl_true_abn.setText("TrueAbn: 0")
        self.lbl_pred_abn.setText("PredAbn: 0")
        self.badge.setText("—")
        self.badge.setStyleSheet("")
        self.lbl_true.setText("True: —")
        self.lbl_pred.setText("Pred: —")
        self.lbl_prob.setText("P(abn): —")
        self.lbl_idx.setText("Index: —")
        self.lbl_quality.setText("Quality: —")
        self.pb_quality.setValue(0)
        self.lbl_qhint.setText("—")

    def save_report(self):
        if not self.session_rows:
            QMessageBox.information(self, "Report", "No session data yet.")
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.reports_dir / f"session_{ts}.csv"
        import csv
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["time", "index", "true", "pred", "p_abn", "thr", "q_score", "q_label"])
            w.writerows(self.session_rows)
        QMessageBox.information(self, "Report Saved", f"Saved:\n{path}")

    def _blink(self):
        self._blink_state = not self._blink_state
        alpha = 230 if self._blink_state else 140
        self.curve.setPen(pg.mkPen((255, 60, 60, alpha), width=2))

    # ---------- Classification ----------
    def next_beat(self):
        # don't classify when hardware is stopped
        if isinstance(self.source, SerialECGSource) and (not self.hw_running):
            return

        if self._pending_image_beat is not None:
            sample = BeatSample(beat=self._pending_image_beat, true_label=None, index=0, source="IMAGE")
            self._pending_image_beat = None
        else:
            try:
                if self.auto_on and isinstance(self.source, SerialECGSource) and hasattr(self.source, "try_next"):
                    last = None
                    while True:
                        s = self.source.try_next()
                        if s is None:
                            break
                        last = s
                    if last is None:
                        return
                    sample = last
                else:
                    sample = self.source.next()
            except Exception as e:
                self._stop_auto()
                QMessageBox.critical(self, "Source error", str(e))
                return

        beat = sample.beat
        idx = sample.index
        true_label = sample.true_label
        is_hw = isinstance(self.source, SerialECGSource)

        # Quality 
        q_score = None
        q_label = "—"
        if is_hw:
            q = evaluate_quality(beat)
            q_score = float(q.score)
            q_label = q.label
            self.lbl_quality.setText(f"Quality: {q.label} ({q.score:.2f})")
            self.pb_quality.setValue(int(q.score * 100))
            self.lbl_qhint.setText(q.hint)

        # Normalize + light smoothing
        b = np.asarray(beat, dtype=np.float32).reshape(-1)
        b = b - np.median(b)
        b = b / (np.max(np.abs(b)) + 1e-9)
        k = np.ones(5, dtype=np.float32) / 5.0
        b = np.convolve(b, k, mode="same")

        # Predict (polarity selection without bias)
        try:
            if is_hw:
                p1 = float(self.model.predict_proba_abn(b))
                p2 = float(self.model.predict_proba_abn(-b))
                # choose the orientation with higher confidence (farther from 0.5)
                if abs(p2 - 0.5) > abs(p1 - 0.5):
                    p_abn = p2
                    pol = "flipped"
                else:
                    p_abn = p1
                    pol = "normal"
            else:
                p_abn = float(self.model.predict_proba_abn(b))
                pol = "—"
        except Exception as e:
            QMessageBox.critical(self, "Model error", str(e))
            return

        thr = self.hw_threshold if is_hw else self.threshold
        pred_label = 1 if p_abn >= thr else 0

        if pred_label == 1:
            self._set_badge_abnormal()
        else:
            self._set_badge_normal()

        self.lbl_true.setText(
            f"True: {'Abnormal (1)' if true_label == 1 else 'Normal (0)' if true_label == 0 else '—'}"
        )
        self.lbl_pred.setText(f"Pred: {'Abnormal (1)' if pred_label == 1 else 'Normal (0)'}")

        extra = f" | pol={pol}" if is_hw else ""
        if is_hw and q_score is not None and q_score < 0.35:
            extra += " | low-quality"
        self.lbl_prob.setText(f"P(abn): {p_abn:.3f} | thr={thr:.2f}{extra}")
        self.lbl_idx.setText(f"Index: {idx}")

        # Plot beat only for non-hardware
        if not is_hw:
            x = np.arange(b.size)
            y = 0.45 + 0.45 * b
            self.curve.setData(x, y)

        self.session_rows.append([
            datetime.now().isoformat(timespec="seconds"),
            idx,
            true_label,
            pred_label,
            float(p_abn),
            float(thr),
            q_score,
            q_label
        ])

        seen = len(self.session_rows)
        true_abn = sum(1 for r in self.session_rows if r[2] == 1)
        pred_abn = sum(1 for r in self.session_rows if r[3] == 1)
        self.lbl_seen.setText(f"Seen: {seen}")
        self.lbl_true_abn.setText(f"TrueAbn: {true_abn}")
        self.lbl_pred_abn.setText(f"PredAbn: {pred_abn}")

    # ---------- Source switching ----------
    def load_csv_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV", str(self.project_root), "CSV (*.csv)")
        if not path:
            return
        if self.hw_running:
            self.hw_stop()
        self.source = CSVSource(Path(path))
        self._apply_mode_ui()
        QMessageBox.information(self, "CSV Loaded", "CSV loaded. Go to Monitor and press Next beat.")
        self.stack.setCurrentIndex(1)

    def load_image_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", str(self.project_root),
                                              "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        if self.hw_running:
            self.hw_stop()

        res = extract_ecg_from_image(path)
        if not res.ok:
            QMessageBox.critical(self, "Image Error", res.msg)
            return
        self._pending_image_beat = res.beat_187
        self._apply_mode_ui()
        QMessageBox.information(self, "Image Loaded", "Image loaded. Go to Monitor and press Next beat.")
        self.stack.setCurrentIndex(1)

    def connect_hardware(self):
        port, ok = QInputDialog.getText(self, "Hardware Port", "Enter COM port (e.g., COM6):")
        if not ok or not port.strip():
            return

        try:
            try:
                if self.serial_source is not None:
                    self.serial_source.stop()
            except Exception:
                pass

            self.serial_source = SerialECGSource(port.strip())
            self.source = self.serial_source
            self.hw_running = False
            self.last_bpm_display = 0.0

            self.lbl_hw_status.setText(f"Status: {port.strip()} • CONNECTED (press Start)")
            self.lbl_hw_bpm.setText("BPM: —")
            self._apply_mode_ui()

            QMessageBox.information(self, "Hardware", f"Connected to {port.strip()}.\nGo to Monitor and press Start.")
            self.stack.setCurrentIndex(1)

        except Exception as e:
            QMessageBox.critical(self, "Hardware Error", str(e))