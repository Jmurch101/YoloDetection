#!/usr/bin/env python3
"""
Simple Tkinter GUI for YOLO detections → CSV.

- Select a file or a folder of images
- Choose model, confidence threshold, device
- Choose CSV output path
- Run detection and save CSV (optionally save annotated images)
"""

import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from typing import Optional

from detect import collect_detections, write_csv


class YoloGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("YOLO Image Detection → CSV")
        self.geometry("640x360")

        # State
        self.source_path: Optional[Path] = None
        self.csv_path: Optional[Path] = None
        self.output_dir: Path = Path("runs/detect")

        # Widgets
        self._build_widgets()

    def _build_widgets(self) -> None:
        pad = {"padx": 8, "pady": 6}

        # Source selection
        frm_source = tk.LabelFrame(self, text="Source")
        frm_source.pack(fill="x", **pad)

        self.lbl_source = tk.Label(frm_source, text="No source selected")
        self.lbl_source.pack(side="left", expand=True, fill="x", **pad)

        tk.Button(frm_source, text="Select File", command=self._choose_file).pack(side="left", **pad)
        tk.Button(frm_source, text="Select Folder", command=self._choose_folder).pack(side="left", **pad)

        # Model/params
        frm_params = tk.LabelFrame(self, text="Parameters")
        frm_params.pack(fill="x", **pad)

        tk.Label(frm_params, text="Model").grid(row=0, column=0, sticky="w", **pad)
        self.var_model = tk.StringVar(value="yolov8n.pt")
        tk.Entry(frm_params, textvariable=self.var_model, width=24).grid(row=0, column=1, **pad)

        tk.Label(frm_params, text="Confidence").grid(row=0, column=2, sticky="w", **pad)
        self.var_conf = tk.StringVar(value="0.25")
        tk.Entry(frm_params, textvariable=self.var_conf, width=8).grid(row=0, column=3, **pad)

        tk.Label(frm_params, text="Device").grid(row=0, column=4, sticky="w", **pad)
        self.var_device = tk.StringVar(value="")
        tk.Entry(frm_params, textvariable=self.var_device, width=8).grid(row=0, column=5, **pad)

        # Output options
        frm_output = tk.LabelFrame(self, text="Outputs")
        frm_output.pack(fill="x", **pad)

        self.var_save_images = tk.BooleanVar(value=False)
        tk.Checkbutton(frm_output, text="Save annotated images", variable=self.var_save_images).pack(side="left", **pad)

        tk.Button(frm_output, text="Choose CSV Path", command=self._choose_csv).pack(side="left", **pad)
        self.lbl_csv = tk.Label(frm_output, text="No CSV path selected")
        self.lbl_csv.pack(side="left", expand=True, fill="x", **pad)

        # Run
        frm_run = tk.Frame(self)
        frm_run.pack(fill="x", **pad)
        self.btn_run = tk.Button(frm_run, text="Run Detection", command=self._run_async)
        self.btn_run.pack(side="left", **pad)

        self.txt_log = tk.Text(self, height=10)
        self.txt_log.pack(expand=True, fill="both", **pad)

    def _log(self, msg: str) -> None:
        self.txt_log.insert("end", msg + "\n")
        self.txt_log.see("end")

    def _choose_file(self) -> None:
        path = filedialog.askopenfilename(title="Select image file")
        if path:
            self.source_path = Path(path)
            self.lbl_source.config(text=str(self.source_path))

    def _choose_folder(self) -> None:
        path = filedialog.askdirectory(title="Select images folder")
        if path:
            self.source_path = Path(path)
            self.lbl_source.config(text=str(self.source_path))

    def _choose_csv(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save detections CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.csv_path = Path(path)
            self.lbl_csv.config(text=str(self.csv_path))

    def _run_async(self) -> None:
        thread = threading.Thread(target=self._run, daemon=True)
        self.btn_run.config(state="disabled")
        thread.start()

    def _run(self) -> None:
        try:
            if not self.source_path:
                messagebox.showwarning("Missing source", "Please select a file or folder of images")
                return

            model = self.var_model.get().strip()
            device = self.var_device.get().strip()
            try:
                conf = float(self.var_conf.get())
            except ValueError:
                messagebox.showwarning("Invalid confidence", "Confidence must be a number between 0 and 1")
                return

            self._log(f"Running detection on: {self.source_path}")
            detections = collect_detections(
                source=self.source_path,
                model_name=model or "yolov8n.pt",
                conf=conf,
                device=device,
                save_images=self.var_save_images.get(),
                output_dir=self.output_dir,
            )
            self._log(f"Detections: {len(detections)}")

            if not detections:
                messagebox.showinfo("Completed", "No detections found.")
                return

            if not self.csv_path:
                # Default CSV path next to source
                default_csv = Path.home() / "detections.csv"
                self.csv_path = default_csv
                self.lbl_csv.config(text=str(self.csv_path))

            write_csv(detections, self.csv_path)
            self._log(f"CSV saved to: {self.csv_path}")
            messagebox.showinfo("Completed", f"CSV saved to: {self.csv_path}")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
        finally:
            self.btn_run.config(state="normal")


def main() -> None:
    app = YoloGui()
    app.mainloop()


if __name__ == "__main__":
    main()


