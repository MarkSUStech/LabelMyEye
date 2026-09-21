"""Project Files panel: folder browser with image thumbnails.

Lists the images of a folder (local or network share) with lazy
background thumbnail loading; single-click opens an image for annotation.
Files whose sibling .json already exists are marked with a check prefix.
"""

from __future__ import annotations

import os

from PIL import Image
from PySide6.QtCore import QSize, QThread, Signal
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
THUMB = (100, 75)
DONE_MARK = "✓ "


class _ThumbLoader(QThread):
    """Loads thumbnails one by one; results emitted as QImage."""

    thumb_ready = Signal(int, QImage)
    loading_done = Signal()

    def __init__(self, files, parent=None):
        super().__init__(parent)
        self._files = files
        self._stopped = False

    def stop(self) -> None:
        self._stopped = True

    def run(self) -> None:
        for i, path in enumerate(self._files):
            if self._stopped:
                return
            try:
                img = Image.open(path).convert("RGB")
                img.thumbnail(THUMB)
                data = img.tobytes()
                qimg = QImage(data, img.width, img.height,
                              3 * img.width, QImage.Format_RGB888).copy()
                if not qimg.isNull():
                    self.thumb_ready.emit(i, qimg)
            except Exception:  # noqa: BLE001 — unreadable image: skip icon
                continue
        self.loading_done.emit()


class FileBrowserPanel(QWidget):
    """'Project Files' panel: pick a folder, click an image to annotate."""

    open_requested = Signal(str)
    folder_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._folder = ""
        self._files: list[str] = []
        self._loader: _ThumbLoader | None = None
        self._generation = 0

        v = QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(4)

        header = QHBoxLayout()
        title = QLabel("<b>Project Files</b>")
        header.addWidget(title)
        header.addStretch(1)
        self.btn_open = QPushButton("Open Folder")
        self.btn_open.setStyleSheet(
            "QPushButton {background:#2563eb; color:white;"
            "border:none; border-radius:4px; padding:4px 10px;}"
            "QPushButton:hover {background:#1d4ed8;}")
        self.btn_open.clicked.connect(self._pick_folder)
        header.addWidget(self.btn_open)
        btn_refresh = QPushButton("⟳")
        btn_refresh.setFixedWidth(28)
        btn_refresh.setToolTip("刷新列表（重新检查 ✓ 标注标记）")
        btn_refresh.clicked.connect(self.refresh)
        header.addWidget(btn_refresh)
        v.addLayout(header)

        self.folder_label = QLabel("未选择文件夹")
        self.folder_label.setToolTip("")
        self.folder_label.setStyleSheet("color:#666; font-size:11px;")
        self.folder_label.setWordWrap(False)
        v.addWidget(self.folder_label)

        self.count_label = QLabel("")
        self.count_label.setStyleSheet("color:#666; font-size:11px;")
        v.addWidget(self.count_label)

        self.list = QListWidget()
        from PySide6.QtCore import QSize
        self.list.setIconSize(QSize(*THUMB))
        self.list.setUniformItemSizes(True)
        self.list.setWordWrap(False)
        self.list.itemClicked.connect(self._emit_open)
        self.list.itemActivated.connect(self._emit_open)
        v.addWidget(self.list, stretch=1)

        hint = QLabel("单击图片打开标注 · ✓ = 已有 JSON")
        hint.setStyleSheet("color:#999; font-size:11px;")
        v.addWidget(hint)

    # ------------------------------------------------------------- public
    def set_folder(self, folder: str) -> None:
        folder = folder.strip()
        if not folder or not os.path.isdir(folder):
            return
        self._folder = folder
        self.folder_label.setText(folder)
        self.folder_label.setToolTip(folder)
        try:
            names = sorted(os.listdir(folder))
        except OSError as e:
            self.count_label.setText(f"无法读取文件夹: {e}")
            return
        self._files = [os.path.join(folder, n) for n in names
                       if n.lower().endswith(IMAGE_EXTS)]
        self.list.clear()
        for path in self._files:
            name = os.path.basename(path)
            item = QListWidgetItem(self._mark(name, path))
            item.setToolTip(path)
            self.list.addItem(item)
        self.count_label.setText(f"{len(self._files)} images")
        self._start_loader()
        self.folder_changed.emit(folder)

    def refresh(self) -> None:
        if self._folder:
            self.set_folder(self._folder)

    def mark_annotated(self, image_path: str) -> None:
        """Flag an image as annotated (after saving its JSON)."""
        image_path = os.path.normpath(image_path)
        for i, path in enumerate(self._files):
            if os.path.normpath(path) == image_path:
                item = self.list.item(i)
                if item is not None and not item.text().startswith(DONE_MARK):
                    item.setText(DONE_MARK + os.path.basename(path))
                break

    @property
    def folder(self) -> str:
        return self._folder

    # ------------------------------------------------------------ internal
    @staticmethod
    def _mark(name: str, path: str) -> str:
        json_path = os.path.splitext(path)[0] + ".json"
        return (DONE_MARK + name) if os.path.isfile(json_path) else name

    def _start_loader(self) -> None:
        if self._loader is not None:
            self._loader.stop()
        self._generation += 1
        gen = self._generation
        self._loader = _ThumbLoader(self._files)
        self._loader.thumb_ready.connect(
            lambda i, qimg, g=gen: self._set_thumb(i, qimg, g))
        self._loader.start()

    def _set_thumb(self, index: int, qimg: QImage, generation: int) -> None:
        if generation != self._generation or index >= self.list.count():
            return
        item = self.list.item(index)
        if item is not None:
            item.setIcon(QIcon(QPixmap.fromImage(qimg)))

    def _emit_open(self, item: QListWidgetItem) -> None:
        name = item.text()
        if name.startswith(DONE_MARK):
            name = name[len(DONE_MARK):]
        path = os.path.join(self._folder, name)
        if os.path.isfile(path):
            self.open_requested.emit(path)

    def _pick_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Select image folder", self._folder or "")
        if folder:
            self.set_folder(folder)
