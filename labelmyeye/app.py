"""LabelMyEye main window."""

from __future__ import annotations

import os
import sys
from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
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
from labelmyeye.canvas import CREATE, EDIT, MOVE_BEFORE, Canvas
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
        self.label_history: List[str] = []

        self._build_ui()
        self._build_menus()
        self._connect()

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
        self.mode_combo.addItem("Register Before (move/scale)", MOVE_BEFORE)
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
        act_fit = QAction("Fit to Window", self)
        act_fit.setShortcut("F")
        act_fit.triggered.connect(self.canvas.fit_view)
        view_m.addAction(act_fit)

        act_reset_reg = QAction("Reset Before Registration", self)
        act_reset_reg.triggered.connect(self.reset_registration)
        view_m.addAction(act_reset_reg)

        # toolbar mirrors
        for act in (act_open, act_dual, act_save, act_export, act_undo, act_redo, act_fit):
            self.tb.addAction(act)

    def _connect(self) -> None:
        self.mode_combo.currentIndexChanged.connect(self._on_mode)
        self.layer_combo.currentIndexChanged.connect(self._on_layer)
        self.opacity_slider.valueChanged.connect(self._on_opacity)
        self.chk_show_after.toggled.connect(self._toggle_show_after)
        self.chk_show_before.toggled.connect(self._toggle_show_before)
        self.chk_labels_after.toggled.connect(self._toggle_labels_after)
        self.chk_labels_before.toggled.connect(self._toggle_labels_before)
        self.canvas.shapes_changed.connect(self._refresh_shape_list)
        self.canvas.request_label.connect(self._prompt_label)
        self.canvas.status_message.connect(self.statusBar().showMessage)
        self.canvas.mode_changed.connect(self._sync_mode_combo)
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

    # ---------- open / save ----------
    def open_single(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All (*.*)",
        )
        if not path:
            return
        from PIL import Image

        img = Image.open(path).convert("RGB")
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
        # prefill from sample if present
        cwd = os.getcwd()
        sample_after = os.path.join(cwd, "4245209_after.png")
        sample_before = os.path.join(cwd, "4245209_before.png")
        if os.path.isfile(sample_after):
            dlg.after_img.setText(sample_after)
            dlg.after_json.setText(os.path.join(cwd, "4245209_after.json"))
        if os.path.isfile(sample_before):
            dlg.before_img.setText(sample_before)
            dlg.before_json.setText(os.path.join(cwd, "4245209_before.json"))
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
