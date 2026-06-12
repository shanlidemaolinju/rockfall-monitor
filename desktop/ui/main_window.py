"""
主窗口 — 表现层 (四级分级预警版)
==================================
布局: 左侧视频 | 右侧控制面板
控制面板: 检测统计 + 按钮 + 日志面板
四级预警: 红色(Ⅰ级)/橙色(Ⅱ级)/黄色(Ⅲ级)/蓝色(Ⅳ级) + 声光报警
"""

from datetime import datetime

from PyQt6.QtCore import Qt
from PyQt6.QtMultimedia import QSoundEffect
from PyQt6.QtCore import QUrl
from PyQt6.QtWidgets import (
    QMainWindow, QLabel, QVBoxLayout, QWidget,
    QPushButton, QApplication, QHBoxLayout, QSplitter,
    QTextEdit, QGroupBox, QGridLayout, QFrame, QSlider, QCheckBox,
)
from PyQt6.QtGui import QFont

from desktop.ui.video_widget import VideoCaptureWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._init_ui()

        # ---- 视频控件 ----
        self.video_widget = VideoCaptureWidget()
        self.video_widget.stats_changed.connect(self._on_stats)
        self._alert_level = "green"
        self.video_widget.log_message.connect(self._on_log)

        # ---- 声音报警 ----
        self._sound_enabled = True
        self._sound_alarm_active = False

        # ---- 控制面板 ----
        right_panel = self._build_panel()

        # ---- 布局: 视频(左) | 面板(右) ----
        splitter = QSplitter(Qt.Orientation.Horizontal)
        left_container = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.video_widget)

        # 底部按钮栏
        btn_bar = QHBoxLayout()
        self.btn_open = QPushButton("选择视频")
        self.btn_open.clicked.connect(self.video_widget.open_file)
        self.btn_open.setMinimumHeight(32)
        btn_bar.addWidget(self.btn_open)

        self.btn_camera = QPushButton("RTSP摄像头")
        self.btn_camera.clicked.connect(self.video_widget.open_camera)
        self.btn_camera.setMinimumHeight(32)
        btn_bar.addWidget(self.btn_camera)

        self.btn_webcam = QPushButton("USB摄像头")
        self.btn_webcam.clicked.connect(lambda: self.video_widget.load_camera(0))
        self.btn_webcam.setMinimumHeight(32)
        btn_bar.addWidget(self.btn_webcam)

        self.btn_stop = QPushButton("停止")
        self.btn_stop.clicked.connect(self.video_widget.stop)
        self.btn_stop.setMinimumHeight(32)
        btn_bar.addWidget(self.btn_stop)

        left_layout.addLayout(btn_bar)
        left_container.setLayout(left_layout)

        splitter.addWidget(left_container)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 3)  # 视频占 3/4
        splitter.setStretchFactor(1, 1)  # 面板占 1/4

        self.setCentralWidget(splitter)

    # ============================================================
    # 控制面板
    # ============================================================

    def _build_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(8)

        # ---- 统计区 ----
        stats_group = QGroupBox("检测统计")
        stats_grid = QGridLayout()

        self.lbl_count = QLabel("0")
        self.lbl_count.setStyleSheet("font-size: 24pt; font-weight: bold; color: #00cc66;")
        self.lbl_count.setAlignment(Qt.AlignmentFlag.AlignCenter)
        stats_grid.addWidget(QLabel("当前检测"), 0, 0)
        stats_grid.addWidget(self.lbl_count, 1, 0)

        self.lbl_fps = QLabel("0")
        self.lbl_fps.setAlignment(Qt.AlignmentFlag.AlignCenter)
        stats_grid.addWidget(QLabel("FPS"), 0, 1)
        stats_grid.addWidget(self.lbl_fps, 1, 1)

        self.lbl_conf = QLabel("-")
        self.lbl_conf.setAlignment(Qt.AlignmentFlag.AlignCenter)
        stats_grid.addWidget(QLabel("最大置信度"), 2, 0)
        stats_grid.addWidget(self.lbl_conf, 3, 0)

        self.lbl_tracks = QLabel("-")
        self.lbl_tracks.setAlignment(Qt.AlignmentFlag.AlignCenter)
        stats_grid.addWidget(QLabel("跟踪ID"), 2, 1)
        stats_grid.addWidget(self.lbl_tracks, 3, 1)

        stats_group.setLayout(stats_grid)
        layout.addWidget(stats_group)

        # ---- 按钮区 ----
        btn_group = QGroupBox("操作")
        btn_layout = QVBoxLayout()

        self.btn_roi = QPushButton("ROI 框选区域")
        self.btn_roi.setCheckable(True)
        self.btn_roi.toggled.connect(self.video_widget.toggle_roi_mode)
        self.btn_roi.clicked.connect(lambda: self.btn_roi.setText(
            "ROI 框选中..." if self.btn_roi.isChecked() else "ROI 框选区域"))
        self.btn_roi.setMinimumHeight(32)
        btn_layout.addWidget(self.btn_roi)

        self.btn_redo = QPushButton("自动检测边坡")
        self.btn_redo.clicked.connect(self.video_widget.redo_detection)
        self.btn_redo.setMinimumHeight(32)
        btn_layout.addWidget(self.btn_redo)

        self.btn_reset_roi = QPushButton("重置ROI并重检")
        self.btn_reset_roi.clicked.connect(self.video_widget.reset_and_redetect)
        self.btn_reset_roi.setMinimumHeight(32)
        btn_layout.addWidget(self.btn_reset_roi)

        self.btn_simulate = QPushButton("模拟预警 (演示)")
        self.btn_simulate.clicked.connect(self.video_widget.simulate_alert)
        self.btn_simulate.setMinimumHeight(32)
        btn_layout.addWidget(self.btn_simulate)

        self.btn_clear_log = QPushButton("清空日志")
        self.btn_clear_log.clicked.connect(self._clear_log)
        self.btn_clear_log.setMinimumHeight(32)
        btn_layout.addWidget(self.btn_clear_log)

        btn_group.setLayout(btn_layout)
        layout.addWidget(btn_group)

        # ---- 参数调节 ----
        param_group = QGroupBox("参数调节")
        param_grid = QGridLayout()

        # Row 0: 置信度阈值 (0.05–0.95, default 0.30) — YOLO基础阈值
        lbl_conf = QLabel("YOLO置信度阈值")
        sld_conf = QSlider(Qt.Orientation.Horizontal)
        sld_conf.setRange(0, 100)
        sld_conf.setValue(28)
        val_conf = QLabel("0.30")
        val_conf.setMinimumWidth(40)
        sld_conf.valueChanged.connect(
            lambda v, lbl=val_conf, s=sld_conf: self._on_slider_conf(v, lbl))
        param_grid.addWidget(lbl_conf, 0, 0)
        param_grid.addWidget(sld_conf, 0, 1)
        param_grid.addWidget(val_conf, 0, 2)

        # Row 1: 跟踪IoU阈值 (0.05–0.95, default 0.30)
        lbl_iou = QLabel("跟踪IoU阈值")
        sld_iou = QSlider(Qt.Orientation.Horizontal)
        sld_iou.setRange(0, 100)
        sld_iou.setValue(28)
        val_iou = QLabel("0.30")
        val_iou.setMinimumWidth(40)
        sld_iou.valueChanged.connect(
            lambda v, lbl=val_iou: self._on_slider_iou(v, lbl))
        param_grid.addWidget(lbl_iou, 1, 0)
        param_grid.addWidget(sld_iou, 1, 1)
        param_grid.addWidget(val_iou, 1, 2)

        # Row 2: 红色/橙色分界阈值 (0.50–1.00, default 0.90)
        lbl_red = QLabel("Ⅰ级红色阈值 (>此值)")
        sld_red = QSlider(Qt.Orientation.Horizontal)
        sld_red.setRange(0, 100)
        sld_red.setValue(90)
        val_red = QLabel("0.90")
        val_red.setMinimumWidth(40)
        sld_red.valueChanged.connect(
            lambda v, lbl=val_red: self._on_slider_orange_high(v, lbl))
        param_grid.addWidget(lbl_red, 2, 0)
        param_grid.addWidget(sld_red, 2, 1)
        param_grid.addWidget(val_red, 2, 2)

        # Row 3: 橙色/黄色分界阈值 (0.30–0.95, default 0.70)
        lbl_yellow = QLabel("Ⅱ级橙色阈值 (>此值)")
        sld_yellow = QSlider(Qt.Orientation.Horizontal)
        sld_yellow.setRange(0, 100)
        sld_yellow.setValue(70)
        val_yellow = QLabel("0.70")
        val_yellow.setMinimumWidth(40)
        sld_yellow.valueChanged.connect(
            lambda v, lbl=val_yellow: self._on_slider_yellow_high(v, lbl))
        param_grid.addWidget(lbl_yellow, 3, 0)
        param_grid.addWidget(sld_yellow, 3, 1)
        param_grid.addWidget(val_yellow, 3, 2)

        # Row 4: 黄色/蓝色分界阈值 (0.10–0.80, default 0.50)
        lbl_blue = QLabel("Ⅲ级黄色阈值 (>此值)")
        sld_blue = QSlider(Qt.Orientation.Horizontal)
        sld_blue.setRange(0, 100)
        sld_blue.setValue(50)
        val_blue = QLabel("0.50")
        val_blue.setMinimumWidth(40)
        sld_blue.valueChanged.connect(
            lambda v, lbl=val_blue: self._on_slider_blue_high(v, lbl))
        param_grid.addWidget(lbl_blue, 4, 0)
        param_grid.addWidget(sld_blue, 4, 1)
        param_grid.addWidget(val_blue, 4, 2)

        # 声音报警开关
        self.chk_sound = QCheckBox("🔊 启用声音报警 (Ⅰ级红色)")
        self.chk_sound.setChecked(True)
        self.chk_sound.toggled.connect(self._on_sound_toggle)
        param_grid.addWidget(self.chk_sound, 5, 0, 1, 3)

        param_group.setLayout(param_grid)
        layout.addWidget(param_group)

        # ---- 日志区 ----
        log_group = QGroupBox("运行日志")
        log_layout = QVBoxLayout()
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.document().setMaximumBlockCount(500)
        self.log_view.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.log_view)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        panel.setLayout(layout)
        panel.setMinimumWidth(260)
        return panel

    # ============================================================
    # 信号回调
    # ============================================================

    def _on_stats(self, count: int, max_conf: float, fps: float, track_ids: list, alert_level: str = "green"):
        prev_level = self._alert_level
        self._alert_level = alert_level
        # 四级预警颜色
        colors = {
            "red": "#ff4444", "orange": "#ff8c42", "yellow": "#ffaa00",
            "blue": "#58a6ff", "green": "#00cc66",
        }
        color = colors.get(alert_level, "#00cc66")
        self.lbl_count.setText(str(count))
        self.lbl_count.setStyleSheet(f"font-size: 24pt; font-weight: bold; color: {color};")
        self.lbl_fps.setText(f"{fps:.1f}")
        self.lbl_conf.setText(f"{max_conf:.2f}" if count > 0 else "-")
        self.lbl_tracks.setText(", ".join(f"#{i}" for i in track_ids) if track_ids else "-")

        # Ⅰ级红色预警: 触发声音报警
        if alert_level == "red" and prev_level != "red" and self._sound_enabled:
            self._trigger_sound_alarm()

    def _on_log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_view.append(f"[{ts}] {msg}")

    def _clear_log(self):
        self.log_view.clear()

    # ---- 参数滑块回调 ----

    def _on_slider_conf(self, value: int, label: QLabel):
        f = round(value * 0.009 + 0.05, 2)
        label.setText(f"{f:.2f}")
        if self.video_widget and self.video_widget.detector:
            self.video_widget.detector.confidence = f

    def _on_slider_iou(self, value: int, label: QLabel):
        f = round(value * 0.009 + 0.05, 2)
        label.setText(f"{f:.2f}")
        if self.video_widget and self.video_widget.tracker:
            self.video_widget.tracker.iou_threshold = f

    def _on_slider_orange_high(self, value: int, label: QLabel):
        """Ⅰ级红色/Ⅱ级橙色分界 (0.50-1.00)"""
        f = round(value * 0.005 + 0.50, 2)
        label.setText(f"{f:.2f}")
        if self.video_widget and self.video_widget.detector:
            self.video_widget.detector.alert_orange_conf_high = f

    def _on_slider_yellow_high(self, value: int, label: QLabel):
        """Ⅱ级橙色/Ⅲ级黄色分界 (0.30-0.95)"""
        f = round(value * 0.0065 + 0.30, 2)
        label.setText(f"{f:.2f}")
        if self.video_widget and self.video_widget.detector:
            self.video_widget.detector.alert_yellow_conf_high = f

    def _on_slider_blue_high(self, value: int, label: QLabel):
        """Ⅲ级黄色/Ⅳ级蓝色分界 (0.10-0.80)"""
        f = round(value * 0.007 + 0.10, 2)
        label.setText(f"{f:.2f}")
        if self.video_widget and self.video_widget.detector:
            self.video_widget.detector.alert_blue_conf_high = f

    def _on_sound_toggle(self, checked: bool):
        self._sound_enabled = checked

    # ============================================================
    # 声光报警
    # ============================================================

    def _trigger_sound_alarm(self):
        """Ⅰ级红色预警: 触发声光报警 (系统蜂鸣 + 窗口闪烁)"""
        if self._sound_alarm_active:
            return
        self._sound_alarm_active = True

        # 异步执行蜂鸣序列, 不阻塞 UI
        from PyQt6.QtCore import QTimer
        self._beep_count = 0
        self._beep_timer = QTimer(self)
        self._beep_timer.timeout.connect(self._do_beep)
        self._beep_timer.start(500)  # 每500ms蜂鸣一次

        # 窗口标题闪烁
        self._orig_title = self.windowTitle()
        self._flash_timer = QTimer(self)
        self._flash_timer.timeout.connect(self._do_flash_title)
        self._flash_timer.start(800)

        # 10秒后自动停止
        QTimer.singleShot(10000, self._stop_sound_alarm)

        self._on_log("🚨 Ⅰ级红色预警 — 声光报警已触发")

    def _do_beep(self):
        """单次蜂鸣"""
        try:
            import winsound
            winsound.MessageBeep(winsound.MB_ICONHAND)
        except ImportError:
            QApplication.beep()
        self._beep_count += 1

    def _do_flash_title(self):
        """窗口标题闪烁"""
        if "🚨" in self.windowTitle():
            self.setWindowTitle(self._orig_title)
        else:
            self.setWindowTitle("🚨 " + self._orig_title)

    def _stop_sound_alarm(self):
        """停止声光报警"""
        self._sound_alarm_active = False
        if hasattr(self, '_beep_timer') and self._beep_timer.isActive():
            self._beep_timer.stop()
        if hasattr(self, '_flash_timer') and self._flash_timer.isActive():
            self._flash_timer.stop()
        self.setWindowTitle(self._orig_title)
        self._on_log("🔕 声光报警已停止")

    # ============================================================
    # 窗口设置
    # ============================================================

    def _init_ui(self):
        self.setWindowTitle("落石检测系统 - 钦州监测")
        screen = QApplication.primaryScreen()
        sz = screen.size()
        self.resize(int(sz.width() * 0.85), int(sz.height() * 0.8))
