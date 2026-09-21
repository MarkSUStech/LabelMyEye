"""LabelMyEye main window."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import List, Optional

from PySide6.QtCore import QSettings, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QSlider,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from labelmyeye import __version__
from labelmyeye import auto_annotate as aa
from labelmyeye.canvas import (
    AI_MACULAR,
    AI_OPTIC,
    CREATE,
    EDIT,
    MOVE_BEFORE,
    Canvas,
)
from labelmyeye.dr_geometry import DRParams
from labelmyeye.file_panel import FileBrowserPanel
from labelmyeye.history import History
from labelmyeye.label_file import (
    image_from_meta,
    load_labelme,
    load_project,
    resolve_image_path,
    save_labelme,
    save_project,
)
from labelmyeye.shape import Shape


class _SamEncoderWorker(QThread):
    """Runs the SAM image encoder off the GUI thread."""

    done = Signal()
    failed = Signal(str)

    def __init__(self, segmenter, pil_image, parent=None):
        super().__init__(parent)
        self._segmenter = segmenter
        self._pil = pil_image

    def run(self) -> None:
        try:
            self._segmenter.set_image(self._pil)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))
            return
        self.done.emit()


class DRParamsDialog(QDialog):
    """Tunable PD coefficients for the derived DR structures."""

    def __init__(self, params: DRParams, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DR 参数（PD = 视盘半径）")
        self.setMinimumWidth(360)
        form = QFormLayout(self)

        def spin(val, lo, hi, step, tip):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setSingleStep(step)
            s.setDecimals(2)
            s.setValue(float(val))
            s.setToolTip(tip)
            return s

        self.sp_pp_t = spin(params.k_pp_temporal, 0.5, 6.0, 0.5,
                            "后极部椭圆：颞侧（黄斑侧）从绿椭圆边缘\n"
                            "向外扩展的视盘直径数（标准 2PD）")
        self.sp_pp_n = spin(params.k_pp_nasal, 0.5, 6.0, 0.5,
                            "后极部椭圆：鼻侧（视盘侧）从绿椭圆边缘\n"
                            "向外扩展的视盘直径数（标准 1PD）")
        self.sp_pp_v = spin(params.k_pp_vertical, 0.5, 6.0, 0.5,
                            "后极部椭圆：上下方向从绿椭圆边缘\n"
                            "向外扩展的视盘直径数（标准 2PD）")
        self.sp_k_mac = spin(params.k_mac, 0.4, 2.0, 0.05,
                             "黄斑圆参考半径 = k_mac × 视盘半径")
        self.sp_d_sup = spin(params.ci_depth_sup, 0.0, 0.9, 0.05,
                             "central_inner 上方凹陷深度（占局部边界半径比例）\n"
                             "越小越宽松")
        self.sp_d_inf = spin(params.ci_depth_inf, 0.0, 0.9, 0.05,
                             "central_inner 下方凹陷深度（占局部边界半径比例）")
        self.sp_w = spin(params.ci_width, 8.0, 90.0, 2.0,
                         "central_inner 凹口过渡平滑度（高斯 σ，角度）\n"
                         "越大，凹口与后极部的衔接越平滑")
        form.addRow("后极部颞侧扩展 (PD):", self.sp_pp_t)
        form.addRow("后极部鼻侧扩展 (PD):", self.sp_pp_n)
        form.addRow("后极部上下扩展 (PD):", self.sp_pp_v)
        form.addRow("黄斑半径系数 k_mac:", self.sp_k_mac)
        form.addRow("上方凹陷深度:", self.sp_d_sup)
        form.addRow("下方凹陷深度:", self.sp_d_inf)
        form.addRow("凹口平滑度 (σ):", self.sp_w)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self) -> DRParams:
        return DRParams(
            k_pp_temporal=self.sp_pp_t.value(),
            k_pp_nasal=self.sp_pp_n.value(),
            k_pp_vertical=self.sp_pp_v.value(),
            k_mac=self.sp_k_mac.value(),
            ci_depth_sup=self.sp_d_sup.value(),
            ci_depth_inf=self.sp_d_inf.value(),
            ci_width=self.sp_w.value(),
        )


class DualOpenDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Open Dual Project (Before / After)")
        self.setMinimumWidth(520)
        layout = QFormLayout(self)

        self.after_img = QLineEdit()
        self.after_json = QLineEdit()
        self.before_img = QLineEdit()
        self.before_json = QLineEdit()

        layout.addRow("After image:", self._browse_row(self.after_img, "image"))
        layout.addRow("After JSON:", self._browse_row(self.after_json, "json"))
        layout.addRow("Before image:", self._browse_row(self.before_img, "image"))
        layout.addRow("Before JSON:", self._browse_row(self.before_json, "json"))

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def _browse_row(self, line: QLineEdit, kind: str) -> QWidget:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(line)
        btn = QLabel('<a href="#">Browse…</a>')
        btn.setTextInteractionFlags(Qt.TextBrowserInteraction)
        btn.linkActivated.connect(lambda: self._browse(line, kind))
        # use a real button for reliability
        from PySide6.QtWidgets import QPushButton

        b = QPushButton("…")
        b.setFixedWidth(32)
        b.clicked.connect(lambda: self._browse(line, kind))
        h.addWidget(b)
        return row

    def _browse(self, line: QLineEdit, kind: str) -> None:
        if kind == "image":
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select image",
                "",
                "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All (*.*)",
            )
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select LabelMe JSON", "", "JSON (*.json);;All (*.*)"
            )
        if path:
            line.setText(path)
            # auto-fill sibling
            if kind == "image":
                stem = os.path.splitext(path)[0] + ".json"
                if os.path.isfile(stem):
                    if line is self.after_img and not self.after_json.text():
                        self.after_json.setText(stem)
                    if line is self.before_img and not self.before_json.text():
                        self.before_json.setText(stem)
            else:
                for ext in (".png", ".jpg", ".jpeg"):
                    cand = os.path.splitext(path)[0] + ext
                    if os.path.isfile(cand):
                        if line is self.after_json and not self.after_img.text():
                            self.after_img.setText(cand)
                        if line is self.before_json and not self.before_img.text():
                            self.before_img.setText(cand)
                        break

    def values(self):
        return {
            "after_image": self.after_img.text().strip(),
            "after_json": self.after_json.text().strip(),
            "before_image": self.before_img.text().strip(),
            "before_json": self.before_json.text().strip(),
        }


def _desc(low_conf: bool) -> str:
    return aa._AUTO_LOWCONF_DESC if low_conf else aa._AUTO_DESC


def _is_auto(shape: Shape) -> bool:
    return (shape.description or "").startswith("auto")


class _ModelDownloadWorker(QThread):
    """Downloads the SAM ONNX models in the background."""

    progress = Signal(int, int, object)   # done_files, total_files, file_frac|None
    done = Signal()
    failed = Signal(str)

    def run(self) -> None:
        from labelmyeye import model_setup
        try:
            model_setup.download_models(
                progress=lambda a, b, c: self.progress.emit(a, b, c))
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))
            return
        self.done.emit()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"LabelMyEye {__version__}")
        self.resize(1280, 840)

        self.canvas = Canvas()
        self.history = History()
        self.history.bind(self.canvas.snapshot, self._restore_state)
        self.canvas.on_before_edit = self.history.push

        self.after_image_path: Optional[str] = None
        self.before_image_path: Optional[str] = None
        self.after_json_path: Optional[str] = None
        self.before_json_path: Optional[str] = None
        self.project_path: Optional[str] = None
        self.label_history: List[str] = list(aa.DR_LABELS)

        # AI segmentation state
        self.dr_params = DRParams()
        self._sam = None
        self._sam_key = None
        self._sam_worker = None
        self._pending_click = None

        self._build_ui()
        self._build_menus()
        self._connect()

        # Project Files dock (folder browser for batch annotation)
        self.file_panel = FileBrowserPanel()
        self.file_panel.open_requested.connect(self._open_from_browser)
        self.file_panel.folder_changed.connect(self._remember_folder)
        self.file_dock = QDockWidget("Project Files", self)
        self.file_dock.setWidget(self.file_panel)
        self.file_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.file_dock)
        act_files = self.file_dock.toggleViewAction()
        act_files.setText("Project Files 面板")
        self.view_menu.addAction(act_files)

        self.settings = QSettings("LabelMyEye", "LabelMyEye")
        last_folder = self.settings.value("project_folder", "", str) or ""
        if last_folder and os.path.isdir(last_folder):
            self.file_panel.set_folder(last_folder)

        self.statusBar().showMessage("Ready — Open a single image or a Before/After dual project")

    def _build_ui(self) -> None:
        splitter = QSplitter()
        splitter.addWidget(self.canvas)

        side = QWidget()
        side.setMinimumWidth(240)
        side.setMaximumWidth(360)
        v = QVBoxLayout(side)

        v.addWidget(QLabel("Mode"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Create Polygon", CREATE)
        self.mode_combo.addItem("Edit", EDIT)
        self.mode_combo.addItem("Register Before (move/scale/rotate)", MOVE_BEFORE)
        self.mode_combo.addItem("AI: 点击标注视盘", AI_OPTIC)
        self.mode_combo.addItem("AI: 点击标注黄斑", AI_MACULAR)
        v.addWidget(self.mode_combo)

        v.addWidget(QLabel("Active layer (new polygons)"))
        self.layer_combo = QComboBox()
        self.layer_combo.addItem("After / Single", "after")
        self.layer_combo.addItem("Before", "before")
        v.addWidget(self.layer_combo)

        self.chk_show_after = QCheckBox("Show after image")
        self.chk_show_after.setChecked(True)
        self.chk_show_before = QCheckBox("Show before image")
        self.chk_show_before.setChecked(True)
        self.chk_labels_after = QCheckBox("Show after labels")
        self.chk_labels_after.setChecked(True)
        self.chk_labels_before = QCheckBox("Show before labels")
        self.chk_labels_before.setChecked(True)
        for w in (
            self.chk_show_after,
            self.chk_show_before,
            self.chk_labels_after,
            self.chk_labels_before,
        ):
            v.addWidget(w)

        v.addWidget(QLabel("Before opacity"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(45)
        v.addWidget(self.opacity_slider)

        rot_row = QHBoxLayout()
        rot_row.addWidget(QLabel("Before rotation"))
        self.rotation_label = QLabel("0.0°")
        self.rotation_label.setMinimumWidth(48)
        rot_row.addWidget(self.rotation_label)
        v.addLayout(rot_row)
        self.rotation_slider = QSlider(Qt.Horizontal)
        self.rotation_slider.setRange(-1800, 1800)  # tenths of a degree
        self.rotation_slider.setValue(0)
        self.rotation_slider.setToolTip("Drag to rotate Before image (keeps view center fixed)")
        v.addWidget(self.rotation_slider)
        rot_btns = QHBoxLayout()
        from PySide6.QtWidgets import QPushButton

        self.btn_rot_ccw = QPushButton("↺ -1°")
        self.btn_rot_cw = QPushButton("+1° ↻")
        self.btn_rot_reset = QPushButton("0°")
        for b in (self.btn_rot_ccw, self.btn_rot_reset, self.btn_rot_cw):
            rot_btns.addWidget(b)
        v.addLayout(rot_btns)

        v.addWidget(QLabel("Shapes"))
        self.shape_list = QListWidget()
        v.addWidget(self.shape_list, stretch=1)

        help_txt = QLabel(
            "<b>Shortcuts</b><br>"
            "N: create · E: edit<br>"
            "Edit: drag vertex / polygon<br>"
            "Click edge: insert vertex<br>"
            "Del / Backspace: delete vertex<br>"
            "F2 / double-click: edit label<br>"
            "Right-click: edit menu<br>"
            "Wheel: zoom · Shift+Drag: pan<br>"
            "Ctrl+Wheel: scale before<br>"
            "Ctrl+Shift+Wheel / [ ]: rotate<br>"
            "Enter: close polygon · F: fit<br>"
            "Ctrl+Z / Y: undo / redo"
        )
        help_txt.setWordWrap(True)
        help_txt.setStyleSheet("color:#aaa; font-size:11px;")
        v.addWidget(help_txt)

        splitter.addWidget(side)
        splitter.setStretchFactor(0, 1)
        self.setCentralWidget(splitter)

        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)
        self.tb = tb

    def _build_menus(self) -> None:
        m = self.menuBar()
        file_m = m.addMenu("&File")
        act_open = QAction("Open Image…", self)
        act_open.setShortcut(QKeySequence.Open)
        act_open.triggered.connect(self.open_single)
        file_m.addAction(act_open)

        act_open_json = QAction("Open LabelMe JSON…", self)
        act_open_json.triggered.connect(self.open_json)
        file_m.addAction(act_open_json)

        act_dual = QAction("Open Dual Project (Before/After)…", self)
        act_dual.setShortcut("Ctrl+D")
        act_dual.triggered.connect(self.open_dual)
        file_m.addAction(act_dual)

        act_folder = QAction("Quick Dual from Folder…", self)
        act_folder.triggered.connect(self.open_dual_folder)
        file_m.addAction(act_folder)

        file_m.addSeparator()
        act_save = QAction("Save", self)
        act_save.setShortcut(QKeySequence.Save)
        act_save.triggered.connect(self.save)
        file_m.addAction(act_save)

        act_export = QAction("Export Before-relative JSON…", self)
        act_export.setShortcut("Ctrl+E")
        act_export.triggered.connect(self.export_before)
        file_m.addAction(act_export)

        act_save_proj = QAction("Save Dual Project…", self)
        act_save_proj.triggered.connect(self.save_dual_project)
        file_m.addAction(act_save_proj)

        act_load_proj = QAction("Load Dual Project File…", self)
        act_load_proj.triggered.connect(self.load_dual_project)
        file_m.addAction(act_load_proj)

        file_m.addSeparator()
        act_quit = QAction("Quit", self)
        act_quit.setShortcut(QKeySequence.Quit)
        act_quit.triggered.connect(self.close)
        file_m.addAction(act_quit)

        edit_m = m.addMenu("&Edit")
        act_undo = QAction("Undo", self)
        act_undo.setShortcut(QKeySequence.Undo)
        act_undo.triggered.connect(self.undo)
        edit_m.addAction(act_undo)

        act_redo = QAction("Redo", self)
        act_redo.setShortcut(QKeySequence.Redo)
        act_redo.triggered.connect(self.redo)
        edit_m.addAction(act_redo)

        edit_m.addSeparator()
        act_edit_label = QAction("Edit Label…", self)
        act_edit_label.setShortcut("F2")
        act_edit_label.triggered.connect(self._edit_selected_label)
        edit_m.addAction(act_edit_label)

        act_del_vertex = QAction("Delete Vertex", self)
        act_del_vertex.setShortcut("Backspace")
        act_del_vertex.triggered.connect(self.canvas.delete_selected_vertex)
        edit_m.addAction(act_del_vertex)

        act_del = QAction("Delete Polygon", self)
        act_del.setShortcut(QKeySequence.Delete)
        act_del.triggered.connect(self._delete_polygon_or_vertex)
        edit_m.addAction(act_del)

        view_m = m.addMenu("&View")
        self.view_menu = view_m
        act_fit = QAction("Fit to Window", self)
        act_fit.setShortcut("F")
        act_fit.triggered.connect(self.canvas.fit_view)
        view_m.addAction(act_fit)

        act_reset_reg = QAction("Reset Before Registration", self)
        act_reset_reg.triggered.connect(self.reset_registration)
        view_m.addAction(act_reset_reg)

        ai_m = m.addMenu("&AI (DR)")
        act_install = QAction("一键安装 SAM 模型…", self)
        act_install.triggered.connect(self._install_sam_models)
        ai_m.addAction(act_install)
        ai_m.addSeparator()

        act_ai_optic = QAction("框选标注视盘（拖框）", self)
        act_ai_optic.setShortcut("Ctrl+1")
        act_ai_optic.triggered.connect(lambda: self._set_ai_mode(AI_OPTIC))
        ai_m.addAction(act_ai_optic)

        act_ai_mac = QAction("框选标注黄斑（拖框）", self)
        act_ai_mac.setShortcut("Ctrl+2")
        act_ai_mac.triggered.connect(lambda: self._set_ai_mode(AI_MACULAR))
        ai_m.addAction(act_ai_mac)

        ai_m.addSeparator()
        act_regen = QAction("从视盘/黄斑重算派生结构", self)
        act_regen.setShortcut("Ctrl+3")
        act_regen.triggered.connect(self._regenerate_derived)
        ai_m.addAction(act_regen)

        act_params = QAction("DR 参数…", self)
        act_params.triggered.connect(self._show_dr_params)
        ai_m.addAction(act_params)

        # toolbar mirrors
        for act in (act_open, act_dual, act_save, act_export, act_undo,
                    act_redo, act_fit, act_ai_optic, act_ai_mac):
            self.tb.addAction(act)

    def _connect(self) -> None:
        self.mode_combo.currentIndexChanged.connect(self._on_mode)
        self.layer_combo.currentIndexChanged.connect(self._on_layer)
        self.opacity_slider.valueChanged.connect(self._on_opacity)
        self.rotation_slider.valueChanged.connect(self._on_rotation_slider)
        self.btn_rot_ccw.clicked.connect(lambda: self._nudge_rotation(-1.0))
        self.btn_rot_cw.clicked.connect(lambda: self._nudge_rotation(1.0))
        self.btn_rot_reset.clicked.connect(lambda: self._nudge_rotation(-self.canvas.before_rotation))
        self.canvas.before_transform_changed.connect(self._sync_rotation_ui)
        self.chk_show_after.toggled.connect(self._toggle_show_after)
        self.chk_show_before.toggled.connect(self._toggle_show_before)
        self.chk_labels_after.toggled.connect(self._toggle_labels_after)
        self.chk_labels_before.toggled.connect(self._toggle_labels_before)
        self.canvas.shapes_changed.connect(self._refresh_shape_list)
        self.canvas.request_label.connect(self._prompt_label)
        self.canvas.status_message.connect(self.statusBar().showMessage)
        self.canvas.mode_changed.connect(self._sync_mode_combo)
        self.canvas.ai_box.connect(self._on_ai_box)
        self.canvas.on_edit_label = self._edit_label_for_shape
        self.shape_list.currentRowChanged.connect(self._on_list_select)
        self.shape_list.itemDoubleClicked.connect(self._on_list_double_click)

    def _on_mode(self) -> None:
        self.canvas.set_mode(self.mode_combo.currentData())

    def _sync_mode_combo(self, mode: str) -> None:
        idx = self.mode_combo.findData(mode)
        if idx >= 0 and self.mode_combo.currentIndex() != idx:
            self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentIndex(idx)
            self.mode_combo.blockSignals(False)

    def _edit_selected_label(self) -> None:
        if self.canvas.selected:
            self._edit_label_for_shape(self.canvas.selected)

    def _edit_label_for_shape(self, shape: Shape) -> None:
        labels = list(self.label_history) if self.label_history else []
        current = shape.label or ""
        if current and current not in labels:
            labels = [current] + labels
        if not labels:
            labels = ["lesion"]
        start = labels.index(current) if current in labels else 0
        label, ok = QInputDialog.getItem(
            self, "Edit label", "Label name:", labels, start, True
        )
        if ok and label.strip() and label.strip() != shape.label:
            self.history.push()
            shape.label = label.strip()
            if shape.label not in self.label_history:
                self.label_history.insert(0, shape.label)
            self.canvas.shapes_changed.emit()
            self.canvas.update()

    def _delete_polygon_or_vertex(self) -> None:
        if self.canvas._selected_vertex_sticky is not None and self.canvas.selected:
            if self.canvas.delete_selected_vertex():
                return
        self.canvas.delete_selected()

    def _on_list_double_click(self, _item) -> None:
        self._edit_selected_label()

    def _on_layer(self) -> None:
        layer = self.layer_combo.currentData()
        if not self.canvas.dual_mode and layer == "before":
            self.statusBar().showMessage("Before layer only available in dual mode")
            self.layer_combo.setCurrentIndex(0)
            return
        self.canvas.set_active_layer(layer if self.canvas.dual_mode else "single")

    def _on_opacity(self, v: int) -> None:
        self.canvas.before_opacity = v / 100.0
        self.canvas.update()

    def _on_rotation_slider(self, v: int) -> None:
        degrees = v / 10.0
        if abs(degrees - self.canvas.before_rotation) < 1e-6:
            return
        aw, ah = self.canvas.after_size
        pivot = (aw / 2.0, ah / 2.0) if aw else None
        self.canvas.set_before_rotation(degrees, pivot)

    def _nudge_rotation(self, delta: float) -> None:
        if not self.canvas.dual_mode:
            return
        aw, ah = self.canvas.after_size
        pivot = (aw / 2.0, ah / 2.0) if aw else None
        self.canvas.rotate_before(delta, pivot)

    def _sync_rotation_ui(self) -> None:
        deg = self.canvas.before_rotation
        self.rotation_label.setText(f"{deg:.1f}°")
        slider_v = int(round(deg * 10))
        slider_v = max(-1800, min(1800, slider_v))
        if self.rotation_slider.value() != slider_v:
            self.rotation_slider.blockSignals(True)
            self.rotation_slider.setValue(slider_v)
            self.rotation_slider.blockSignals(False)

    def _toggle_show_after(self, v: bool) -> None:
        self.canvas.show_after = v
        self.canvas.update()

    def _toggle_show_before(self, v: bool) -> None:
        self.canvas.show_before = v
        self.canvas.update()

    def _toggle_labels_after(self, v: bool) -> None:
        self.canvas.show_after_labels = v
        self.canvas.update()

    def _toggle_labels_before(self, v: bool) -> None:
        self.canvas.show_before_labels = v
        self.canvas.update()

    def _restore_state(self, state) -> None:
        self.canvas.restore(state)
        self._refresh_shape_list()

    def undo(self) -> None:
        if self.history.undo():
            self.statusBar().showMessage("Undo")
        self._refresh_shape_list()

    def redo(self) -> None:
        if self.history.redo():
            self.statusBar().showMessage("Redo")
        self._refresh_shape_list()

    def _refresh_shape_list(self) -> None:
        self.shape_list.blockSignals(True)
        self.shape_list.clear()
        for i, s in enumerate(self.canvas.shapes):
            tag = f"[{s.layer[0].upper()}] {s.label or '(no label)'}  ({len(s.points)} pts)"
            item = QListWidgetItem(tag)
            self.shape_list.addItem(item)
            if s.selected:
                self.shape_list.setCurrentRow(i)
        self.shape_list.blockSignals(False)

    def _on_list_select(self, row: int) -> None:
        if row < 0 or row >= len(self.canvas.shapes):
            return
        self.canvas.select_shape(self.canvas.shapes[row])

    def _prompt_label(self) -> None:
        labels = self.label_history or ["lesion", "optic", "macular"]
        label, ok = QInputDialog.getItem(
            self, "Polygon label", "Label name:", labels, 0, True
        )
        if ok and label.strip():
            label = label.strip()
            if label not in self.label_history:
                self.label_history.insert(0, label)
            self.canvas.commit_pending(label)
        else:
            self.canvas.cancel_pending()
            self.statusBar().showMessage("Polygon discarded")

    # ---------- AI (DR) ----------
    def _set_ai_mode(self, mode: str) -> None:
        if aa.model_paths() is None:
            QMessageBox.warning(
                self, "AI 模型缺失",
                "未找到分割模型文件 assets/models/mobile_sam_encoder.onnx。\n"
                "请先运行 scripts/prepare_sam_onnx.py 下载并导出模型。")
            return
        idx = self.mode_combo.findData(mode)
        if idx >= 0:
            self.mode_combo.setCurrentIndex(idx)
            what = "视盘" if mode == AI_OPTIC else "黄斑"
            self.statusBar().showMessage(
                f"AI 模式：按住拖出一个框包住{what}，松开后自动分割并生成标注")

    def _current_after_pil(self):
        pil = getattr(self, "_after_pil", None)
        if pil is not None:
            return pil
        if self.after_image_path and os.path.isfile(self.after_image_path):
            from PIL import Image
            self._after_pil = Image.open(self.after_image_path).convert("RGB")
            return self._after_pil
        return None

    def _sam_image_key(self) -> Optional[str]:
        if self.after_image_path and os.path.isfile(self.after_image_path):
            try:
                mtime = os.path.getmtime(self.after_image_path)
            except OSError:
                mtime = 0
            return f"{self.after_image_path}|{mtime}"
        return None

    def _on_ai_box(self, x0: float, y0: float, x1: float, y1: float,
                   label: str) -> None:
        pil = self._current_after_pil()
        if pil is None:
            self.statusBar().showMessage("AI 标注：请先打开图像")
            return
        if self._sam is None:
            try:
                self._sam = aa.get_segmenter()
            except Exception as e:  # noqa: BLE001
                QMessageBox.warning(self, "AI 初始化失败", str(e))
                return
        key = self._sam_image_key()
        if self._sam.ready and key is not None and key == self._sam_key:
            self._apply_ai_box((x0, y0, x1, y1), label)
            return
        # need the encoder for this image — run in background
        self._pending_click = (x0, y0, x1, y1, label)
        self._sam.clear_image()
        self._sam_key = None
        self.statusBar().showMessage("AI 模型编码图像中（约 1-3 秒）…")
        self._sam_worker = _SamEncoderWorker(self._sam, pil)
        self._sam_worker.done.connect(self._on_encoder_done)
        self._sam_worker.failed.connect(self._on_encoder_failed)
        self._sam_worker.start()

    def _on_encoder_done(self) -> None:
        self._sam_key = self._sam_image_key()
        self.statusBar().showMessage("AI 编码完成，正在分割…")
        pending = self._pending_click
        self._pending_click = None
        if pending:
            self._apply_ai_box(tuple(pending[:4]), pending[4])

    def _on_encoder_failed(self, msg: str) -> None:
        self._pending_click = None
        QMessageBox.warning(self, "AI 编码失败", msg)

    def _apply_ai_box(self, box, label: str) -> None:
        try:
            mask, info = self._sam.segment_box(box)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "AI 分割失败", str(e))
            return
        from labelmyeye.sam_seg import fit_circle_from_mask
        fit = fit_circle_from_mask(mask)
        if fit is None:
            self.statusBar().showMessage("AI：未分割到有效区域，请调整框位置")
            return
        scale = info["scale"]
        cx, cy = fit["cx"] / scale, fit["cy"] / scale
        r_mask = fit["r"] / scale

        layer = "single" if not self.canvas.dual_mode else "after"
        shapes = list(self.canvas.shapes)
        circles = aa.find_circles(shapes)
        warnings = list(fit["warnings"])
        if label == aa.OPTIC:
            r = r_mask
        else:
            # the macula circle is a convention-sized landmark: centre from
            # the segmentation, radius from the k_mac * disc-radius rule
            if aa.OPTIC in circles:
                r = self.dr_params.k_mac * circles[aa.OPTIC][2]
            else:                       # optic not drawn yet — interim value
                r = self.dr_params.k_mac * r_mask
            if r_mask > 2.5 * r or (r_mask > 0 and r_mask < 0.3 * r):
                warnings.append("macula mask size unusual")
        low_conf = bool(warnings) or info["iou"] < 0.5

        new_circle = (cx, cy, r)
        other = circles.get(aa.MACULAR if label == aa.OPTIC else aa.OPTIC)

        gid = aa.next_group_id(shapes)
        for s in shapes:
            if s.label == (aa.MACULAR if label == aa.OPTIC else aa.OPTIC):
                if s.group_id is not None:
                    gid = s.group_id
                break

        added = aa.circle_shape(label, cx, cy, r, layer, gid,
                                _desc(low_conf))
        keep = [s for s in shapes
                if not (s.label == label and _is_auto(s))]
        out = list(keep)
        out.append(added)

        if label == aa.OPTIC and other is not None:
            # optic radius now known — normalise the stored macula circle
            other = (other[0], other[1], self.dr_params.k_mac * r)
            out = [s if not (s.label == aa.MACULAR and _is_auto(s)) else
                   aa.circle_shape(aa.MACULAR, *other, layer, gid,
                                   _desc(low_conf))
                   for s in out]

        status = f"AI {label}: r={r:.0f}px iou={info['iou']:.2f}"
        if other is not None:
            out += self._derived_for(other, new_circle, layer, gid, low_conf)
            status += "  — 已生成派生结构（绿/橙/紫）"
        else:
            status += ("  — 请切换到黄斑模式" if label == aa.OPTIC
                       else "  — 请切换到视盘模式") + "再框一次以生成派生结构"
        self.history.push()
        self.canvas.set_shapes(out)
        if warnings:
            status += "  ⚠ " + ";".join(warnings)
        if low_conf:
            status += "  ⚠ 低置信，请人工核对"
        self.statusBar().showMessage(status)

    def _derived_for(self, disc, mac, layer, gid, low_conf):
        """Derived structures + replacement of stale auto-derived shapes."""
        return aa.derived_shapes(disc, mac, layer, gid, self.dr_params,
                                 low_conf)

    def _regenerate_derived(self) -> None:
        circles = aa.find_circles(self.canvas.shapes)
        if aa.OPTIC not in circles or aa.MACULAR not in circles:
            QMessageBox.information(
                self, "重算派生结构",
                "需要同时存在 optic 和 macular 标注（点击标注或手绘均可）。")
            return
        disc, mac = circles[aa.OPTIC], circles[aa.MACULAR]
        layer = "single" if not self.canvas.dual_mode else "after"
        gid = aa.next_group_id(self.canvas.shapes)
        for s in self.canvas.shapes:
            if s.label == aa.OPTIC and s.group_id is not None:
                gid = s.group_id
                break
        new_derived = aa.derived_shapes(disc, mac, layer, gid, self.dr_params)
        keep = [s for s in self.canvas.shapes
                if s.label not in (aa.MACULAR_AREA, aa.POSTERIOR_POLE,
                                   aa.CENTRAL_INNER)]
        self.history.push()
        self.canvas.set_shapes(keep + new_derived)
        self.statusBar().showMessage("已根据当前视盘/黄斑重算 绿/橙/紫 派生结构")

    def _show_dr_params(self) -> None:
        dlg = DRParamsDialog(self.dr_params, self)
        if dlg.exec() != QDialog.Accepted:
            return
        self.dr_params = dlg.values().validated()
        self.statusBar().showMessage(
            f"DR 参数已更新：颞侧 {self.dr_params.k_pp_temporal:.1f}PD / "
            f"鼻侧 {self.dr_params.k_pp_nasal:.1f}PD / 上下 "
            f"{self.dr_params.k_pp_vertical:.1f}PD，"
            f"凹陷 {self.dr_params.ci_depth_sup:.2f}/{self.dr_params.ci_depth_inf:.2f} "
            f"宽度 {self.dr_params.ci_width:.0f}°"
            "（可用 Ctrl+3 重算派生结构）")

    # ---------- 一键安装 SAM 模型 ----------
    def _install_sam_models(self) -> None:
        from labelmyeye import model_setup
        if model_setup.models_present():
            QMessageBox.information(
                self, "SAM 模型",
                "模型已安装（assets/models/），无需重复安装。")
            return
        box = QMessageBox(self)
        box.setWindowTitle("一键安装 SAM 模型")
        box.setText("尚未找到 SAM 分割模型（约 45 MB）。\n\n"
                    "点击 Yes 将自动从网上下载并安装；\n"
                    "点击 No 改用本机 PyTorch 导出（如已安装）。")
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        if box.exec() != QMessageBox.Yes:
            # 尝试本地导出兜底
            cmd = model_setup.local_export_cmd()
            if cmd:
                self.statusBar().showMessage("正在本地导出 SAM 模型（可能需要几分钟）…")
                self._export_proc = subprocess.Popen(
                    cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(cmd[1]))))

                def _poll():
                    if self._export_proc.poll() is None:
                        QTimer.singleShot(500, _poll)
                        return
                    if self._export_proc.returncode == 0:
                        self.statusBar().showMessage("SAM 模型本地导出完成！")
                    else:
                        QMessageBox.warning(self, "导出失败",
                                            f"导出命令返回 {self._export_proc.returncode}，"
                                            "请查看控制台输出。")
                QTimer.singleShot(500, _poll)
            else:
                QMessageBox.warning(
                    self, "SAM 模型",
                    "本机没有 PyTorch，无法本地导出。\n请先安装：pip install torch timm "
                    "mobile_sam 后重试，或从项目 Release 手动下载 onnx 文件。")
            return

        self._model_worker = _ModelDownloadWorker(self)
        dlg = QProgressDialog("正在下载 SAM 模型（约 45 MB）…", "", 0, 0, self)
        dlg.setWindowTitle("一键安装 SAM 模型")
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setCancelButton(None)
        dlg.setMinimumDuration(0)
        dlg.show()
        self._model_dlg = dlg

        def _on_progress(done: int, total: int, frac) -> None:
            pct = int(((done + (frac or 0)) / max(total, 1)) * 100)
            dlg.setValue(min(pct, 99))

        def _on_done() -> None:
            dlg.setValue(100)
            dlg.close()
            self.statusBar().showMessage("SAM 模型安装完成！现在可以用 AI (DR) 菜单框选标注了。")
            QMessageBox.information(self, "SAM 模型",
                                    "安装完成！\n用 AI (DR) 菜单的框选标注即可开始。")

        def _on_fail(msg: str) -> None:
            dlg.close()
            cmd = model_setup.local_export_cmd()
            extra = ("\n\n本机检测到 PyTorch，可用菜单重试并选择本地导出。" if cmd else
                     "\n\n也可手动运行 scripts/prepare_sam_onnx.py 导出模型。")
            QMessageBox.warning(self, "SAM 模型下载失败", msg + extra)

        self._model_worker.progress.connect(_on_progress)
        self._model_worker.done.connect(_on_done)
        self._model_worker.failed.connect(_on_fail)
        self._model_worker.start()

    # ---------- open / save ----------
    def _remember_folder(self, folder: str) -> None:
        self.settings.setValue("project_folder", folder)

    def _open_from_browser(self, path: str) -> None:
        if self.after_image_path and os.path.normpath(
                path) == os.path.normpath(self.after_image_path):
            return
        self.load_single_image(path)

    def open_single(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All (*.*)",
        )
        if path:
            self.load_single_image(path)

    def load_single_image(self, path: str) -> None:
        from PIL import Image

        img = Image.open(path).convert("RGB")
        self._after_pil = img
        self.canvas.clear_images()
        self.canvas.set_after_image(img)
        self.canvas.dual_mode = False
        self.canvas.active_layer = "single"
        self.after_image_path = path
        self.before_image_path = None
        self.after_json_path = os.path.splitext(path)[0] + ".json"
        self.before_json_path = None
        shapes = []
        if os.path.isfile(self.after_json_path):
            shapes, _ = load_labelme(self.after_json_path, layer="single")
            for s in shapes:
                s.layer = "single"
        self.canvas.set_shapes(shapes)
        self.history.clear()
        self.setWindowTitle(f"LabelMyEye — {os.path.basename(path)}")
        self.statusBar().showMessage(f"Loaded {path}")

    def open_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open LabelMe JSON", "", "JSON (*.json);;All (*.*)"
        )
        if not path:
            return
        shapes, meta = load_labelme(path, layer="single")
        for s in shapes:
            s.layer = "single"
        img_path, pil = image_from_meta(meta, path)
        if pil is None:
            QMessageBox.warning(self, "Error", "Could not load image for this JSON.")
            return
        self._after_pil = pil
        self.canvas.clear_images()
        self.canvas.set_after_image(pil)
        self.canvas.dual_mode = False
        self.canvas.active_layer = "single"
        self.after_image_path = img_path or resolve_image_path(path, meta.get("imagePath"))
        self.after_json_path = path
        self.before_image_path = None
        self.before_json_path = None
        self.canvas.set_shapes(shapes)
        self.history.clear()
        self.setWindowTitle(f"LabelMyEye — {os.path.basename(path)}")
        self.statusBar().showMessage(f"Loaded {path}")

    def open_dual(self) -> None:
        dlg = DualOpenDialog(self)
        if dlg.exec() != QDialog.Accepted:
            return
        v = dlg.values()
        self._load_dual(v["after_image"], v["after_json"], v["before_image"], v["before_json"])

    def open_dual_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select folder with before/after files")
        if not folder:
            return
        after_img = before_img = after_json = before_json = None
        for name in os.listdir(folder):
            low = name.lower()
            path = os.path.join(folder, name)
            if not os.path.isfile(path):
                continue
            if "after" in low and low.endswith((".png", ".jpg", ".jpeg")):
                after_img = path
            elif "before" in low and low.endswith((".png", ".jpg", ".jpeg")):
                before_img = path
            elif "after" in low and low.endswith(".json") and "introduction" not in low and "meta" not in low:
                after_json = path
            elif "before" in low and low.endswith(".json"):
                before_json = path
        if not after_img or not before_img:
            QMessageBox.warning(
                self,
                "Not found",
                "Need files with 'before' and 'after' in the name (png/jpg + optional json).",
            )
            return
        if not after_json:
            after_json = os.path.splitext(after_img)[0] + ".json"
        if not before_json:
            before_json = os.path.splitext(before_img)[0] + ".json"
        self._load_dual(after_img, after_json, before_img, before_json)

    def _load_dual(
        self,
        after_img: str,
        after_json: str,
        before_img: str,
        before_json: str,
    ) -> None:
        from PIL import Image

        if not after_img or not os.path.isfile(after_img):
            QMessageBox.warning(self, "Error", "After image is required.")
            return
        if not before_img or not os.path.isfile(before_img):
            QMessageBox.warning(self, "Error", "Before image is required.")
            return

        after_pil = Image.open(after_img).convert("RGB")
        before_pil = Image.open(before_img).convert("RGB")
        self._after_pil = after_pil

        after_shapes: List[Shape] = []
        before_shapes: List[Shape] = []
        if after_json and os.path.isfile(after_json):
            after_shapes, _ = load_labelme(after_json, layer="after")
        if before_json and os.path.isfile(before_json):
            before_shapes, _ = load_labelme(before_json, layer="before")

        self.canvas.clear_images()
        self.canvas.set_after_image(after_pil)
        self.canvas.set_before_image(before_pil)
        for s in after_shapes:
            s.layer = "after"
        for s in before_shapes:
            s.layer = "before"
        self.canvas.set_shapes(after_shapes + before_shapes)
        self.canvas.active_layer = "after"
        self.layer_combo.setCurrentIndex(0)

        self.after_image_path = after_img
        self.before_image_path = before_img
        self.after_json_path = after_json if after_json else os.path.splitext(after_img)[0] + ".json"
        self.before_json_path = before_json if before_json else os.path.splitext(before_img)[0] + ".json"
        self.history.clear()
        self.setWindowTitle(
            f"LabelMyEye — {os.path.basename(before_img)} ⊕ {os.path.basename(after_img)}"
        )
        self.statusBar().showMessage(
            f"Dual loaded: {len(before_shapes)} before + {len(after_shapes)} after shapes. "
            "Use Register mode to align, then Export Before-relative JSON."
        )
        # switch to registration mode hint
        idx = self.mode_combo.findData(MOVE_BEFORE)
        if idx >= 0:
            self.mode_combo.setCurrentIndex(idx)

    def save(self) -> None:
        if self.canvas.dual_mode:
            # save each layer back to its own json (original coords)
            self._save_layers_separately()
        else:
            if not self.after_image_path:
                QMessageBox.information(self, "Save", "No image loaded.")
                return
            path = self.after_json_path or os.path.splitext(self.after_image_path)[0] + ".json"
            if not path or not os.path.isdir(os.path.dirname(path) or "."):
                path, _ = QFileDialog.getSaveFileName(
                    self, "Save LabelMe JSON", path or "annotation.json", "JSON (*.json)"
                )
                if not path:
                    return
            w, h = self.canvas.after_size
            shapes = [s for s in self.canvas.shapes]
            save_labelme(path, shapes, self.after_image_path, h, w, embed_image=False)
            self.after_json_path = path
            if self.file_panel.folder:
                self.file_panel.mark_annotated(self.after_image_path)
            self.statusBar().showMessage(f"Saved {path}")

    def _save_layers_separately(self) -> None:
        if self.after_json_path and self.after_image_path:
            w, h = self.canvas.after_size
            after_shapes = [s for s in self.canvas.shapes if s.layer == "after"]
            save_labelme(
                self.after_json_path,
                after_shapes,
                self.after_image_path,
                h,
                w,
                embed_image=False,
            )
        if self.before_json_path and self.before_image_path:
            w, h = self.canvas.before_size
            before_shapes = [s for s in self.canvas.shapes if s.layer == "before"]
            save_labelme(
                self.before_json_path,
                before_shapes,
                self.before_image_path,
                h,
                w,
                embed_image=False,
            )
        self.statusBar().showMessage("Saved before/after JSON (original coordinates)")

    def export_before(self) -> None:
        """Export all polygons transformed into before-image coordinates."""
        if self.canvas.dual_mode:
            if not self.canvas.before_size[0]:
                QMessageBox.information(self, "Export", "No before image loaded.")
                return
            default = self.before_json_path or "before_registered.json"
            base, ext = os.path.splitext(default)
            default = f"{base}_registered{ext or '.json'}"
            w, h = self.canvas.before_size
            img_path = self.before_image_path or "before.png"
            coord_name = "before"
        else:
            if not self.canvas.after_size[0]:
                QMessageBox.information(self, "Export", "No image loaded.")
                return
            default = self.after_json_path or "annotation.json"
            w, h = self.canvas.after_size
            img_path = self.after_image_path or "image.png"
            coord_name = "image"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Before-relative JSON", default, "JSON (*.json)"
        )
        if not path:
            return
        shapes = self.canvas.transformed_shapes_for_export()
        save_labelme(path, shapes, img_path, h, w, embed_image=False)
        self.statusBar().showMessage(
            f"Exported {len(shapes)} shapes ({coord_name} coordinates) → {path}"
        )
        QMessageBox.information(
            self,
            "Exported",
            f"Saved {len(shapes)} polygons in {coord_name}-image coordinates to:\n{path}",
        )

    def save_dual_project(self) -> None:
        if not self.canvas.dual_mode:
            QMessageBox.information(self, "Project", "Only available in dual mode.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Dual Project", "project.lmye.json", "LabelMyEye Project (*.lmye.json *.json)"
        )
        if not path:
            return
        proj = {
            "type": "labelmyeye_dual",
            "version": __version__,
            "after_image": self.after_image_path,
            "after_json": self.after_json_path,
            "before_image": self.before_image_path,
            "before_json": self.before_json_path,
            "before_transform": self.canvas.get_before_transform(),
            "shapes": [
                {
                    **s.to_labelme(),
                    "layer": s.layer,
                }
                for s in self.canvas.shapes
            ],
        }
        save_project(path, proj)
        self.project_path = path
        self.statusBar().showMessage(f"Project saved: {path}")

    def load_dual_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Dual Project", "", "LabelMyEye Project (*.lmye.json *.json)"
        )
        if not path:
            return
        proj = load_project(path)
        self._load_dual(
            proj.get("after_image", ""),
            proj.get("after_json", ""),
            proj.get("before_image", ""),
            proj.get("before_json", ""),
        )
        if "before_transform" in proj:
            self.canvas.set_before_transform(proj["before_transform"])
            self.opacity_slider.setValue(int(self.canvas.before_opacity * 100))
        if "shapes" in proj:
            shapes = []
            for d in proj["shapes"]:
                layer = d.get("layer", "after")
                shapes.append(Shape.from_labelme(d, layer=layer))
            self.canvas.set_shapes(shapes)
        self.project_path = path

    def reset_registration(self) -> None:
        if not self.canvas.dual_mode or not self.canvas.before_size[0]:
            return
        self.history.push()
        aw, ah = self.canvas.after_size
        bw, bh = self.canvas.before_size
        self.canvas.before_scale = min(aw / bw, ah / bh)
        self.canvas.before_tx = (aw - bw * self.canvas.before_scale) / 2
        self.canvas.before_ty = (ah - bh * self.canvas.before_scale) / 2
        self.canvas.before_rotation = 0.0
        self.canvas.update()
        self.statusBar().showMessage("Before registration reset")


def main() -> None:
    # large images: avoid Qt pixmap size issues somewhat by staying on PIL→QImage path
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    app = QApplication(sys.argv)
    app.setApplicationName("LabelMyEye")
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    # optional CLI: python -m labelmyeye [folder]
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        os.chdir(sys.argv[1])
        win.open_dual_folder()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
