#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal, QThread
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from detect import collect_detections, write_csv, process_video


@dataclass
class RunParams:
    source_path: Path
    model_name: str
    confidence: float
    device: str
    save_images: bool
    output_dir: Path
    csv_path: Optional[Path]


class Worker(QObject):
    log = pyqtSignal(str)
    done = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, params: RunParams) -> None:
        super().__init__()
        self.params = params

    def run(self) -> None:
        try:
            p = self.params
            self.log.emit(f"Running detection on: {p.source_path}")
            video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
            if p.source_path.is_file() and p.source_path.suffix.lower() in video_exts:
                out_dir = process_video(
                    source=p.source_path,
                    output_dir=p.output_dir,
                    model_name=p.model_name,
                    conf=p.confidence,
                    device=p.device,
                )
                msg = f"Annotated video saved under: {out_dir}"
                self.log.emit(msg)
                self.done.emit(msg)
                return

            detections = collect_detections(
                source=p.source_path,
                model_name=p.model_name,
                conf=p.confidence,
                device=p.device,
                save_images=p.save_images,
                output_dir=p.output_dir,
            )
            self.log.emit(f"Detections: {len(detections)}")
            if not detections:
                self.done.emit("No detections found.")
                return

            csv_path = p.csv_path or (Path.home() / "detections.csv")
            write_csv(detections, csv_path)
            msg = f"CSV saved to: {csv_path}"
            self.log.emit(msg)
            self.done.emit(msg)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("YOLO Detection (PyQt)")
        self.resize(820, 520)

        self.source_path: Optional[Path] = None
        self.csv_path: Optional[Path] = None
        self.output_dir = Path("runs/detect")

        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        # Source row
        src_layout = QHBoxLayout()
        self.lbl_source = QLabel("No source selected")
        btn_file = QPushButton("Select File")
        btn_file.clicked.connect(self.choose_file)
        btn_dir = QPushButton("Select Folder")
        btn_dir.clicked.connect(self.choose_folder)
        src_layout.addWidget(self.lbl_source, 1)
        src_layout.addWidget(btn_file)
        src_layout.addWidget(btn_dir)
        root_layout.addLayout(src_layout)

        # Params grid
        grid = QGridLayout()
        grid.addWidget(QLabel("Model"), 0, 0)
        self.txt_model = QLineEdit("yolov8n.pt")
        grid.addWidget(self.txt_model, 0, 1)

        grid.addWidget(QLabel("Confidence"), 0, 2)
        self.txt_conf = QLineEdit("0.25")
        grid.addWidget(self.txt_conf, 0, 3)

        grid.addWidget(QLabel("Device"), 0, 4)
        self.txt_device = QLineEdit("")
        grid.addWidget(self.txt_device, 0, 5)
        root_layout.addLayout(grid)

        # Outputs row
        out_layout = QHBoxLayout()
        self.chk_save_images = QCheckBox("Save annotated images")
        btn_csv = QPushButton("Choose CSV Path")
        btn_csv.clicked.connect(self.choose_csv)
        self.lbl_csv = QLabel("No CSV path selected")
        out_layout.addWidget(self.chk_save_images)
        out_layout.addWidget(btn_csv)
        out_layout.addWidget(self.lbl_csv, 1)
        root_layout.addLayout(out_layout)

        # Run row
        run_layout = QHBoxLayout()
        self.btn_run = QPushButton("Run Detection")
        self.btn_run.clicked.connect(self.run_detection)
        run_layout.addWidget(self.btn_run)
        root_layout.addLayout(run_layout)

        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        root_layout.addWidget(self.txt_log, 1)

        self._thread: Optional[QThread] = None
        self._worker: Optional[Worker] = None

    def choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select image or video file",
            str(Path.home()),
            "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp);;Videos (*.mp4 *.avi *.mov *.mkv *.webm);;All files (*)",
        )
        if path:
            self.source_path = Path(path)
            self.lbl_source.setText(path)

    def choose_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select images folder", str(Path.home()))
        if path:
            self.source_path = Path(path)
            self.lbl_source.setText(path)

    def choose_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save detections CSV", str(Path.home()), "CSV files (*.csv)")
        if path:
            self.csv_path = Path(path)
            self.lbl_csv.setText(path)

    def append_log(self, msg: str) -> None:
        self.txt_log.append(msg)

    def on_done(self, msg: str) -> None:
        self.append_log(msg)
        QMessageBox.information(self, "Completed", msg)
        self.btn_run.setEnabled(True)

    def on_error(self, msg: str) -> None:
        QMessageBox.critical(self, "Error", msg)
        self.btn_run.setEnabled(True)

    def run_detection(self) -> None:
        if not self.source_path:
            QMessageBox.warning(self, "Missing source", "Please select a file or folder of images")
            return
        try:
            conf = float(self.txt_conf.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid confidence", "Confidence must be a number between 0 and 1")
            return

        params = RunParams(
            source_path=self.source_path,
            model_name=(self.txt_model.text().strip() or "yolov8n.pt"),
            confidence=conf,
            device=self.txt_device.text().strip(),
            save_images=self.chk_save_images.isChecked(),
            output_dir=self.output_dir,
            csv_path=self.csv_path,
        )

        self.btn_run.setEnabled(False)
        self.txt_log.clear()

        self._thread = QThread(self)
        self._worker = Worker(params)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self.append_log)
        self._worker.done.connect(self.on_done)
        self._worker.error.connect(self.on_error)
        self._worker.done.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()


def main() -> int:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())


