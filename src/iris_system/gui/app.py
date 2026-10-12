from __future__ import annotations

import json
import os
import queue
import csv
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from iris_system.config import default_config, project_root
from iris_system.data.manifest import compile_dataset, save_manifest
from iris_system.data.validation import summarize_records, validate_records


def _cli_option(key: str, *, loss: bool = False) -> str:
    """Translate internal snake_case parameter keys to argparse kebab-case options."""
    prefix = "--loss-" if loss else "--"
    return prefix + key.replace("_", "-")


class IrisApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("虹膜分割模型")
        self.root.geometry("1140x880")
        self.root.minsize(1040, 800)
        self.process: subprocess.Popen | None = None
        self.log_queue: queue.Queue[str | None] = queue.Queue()
        self.datasets: list[str] = []
        self._build()
        self.root.after(100, self._poll_log)
        self.root.after(2000, self._auto_refresh_monitor)

    def _build(self) -> None:
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)
        self.data_page = ttk.Frame(notebook)
        self.train_page = ttk.Frame(notebook)
        self.eval_page = ttk.Frame(notebook)
        self.annotate_page = ttk.Frame(notebook)
        self.models_page = ttk.Frame(notebook)
        self.monitor_page = ttk.Frame(notebook)
        notebook.add(self.data_page, text="数据")
        notebook.add(self.train_page, text="训练")
        notebook.add(self.eval_page, text="评估")
        notebook.add(self.annotate_page, text="自动标注")
        notebook.add(self.models_page, text="模型与版本")
        notebook.add(self.monitor_page, text="训练曲线")
        self._build_data_page()
        self._build_train_page()
        self._build_eval_page()
        self._build_annotate_page()
        self._build_models_page()
        self._build_monitor_page()

        log_frame = ttk.LabelFrame(self.root, text="运行日志")
        log_frame.pack(fill="both", expand=False, padx=10, pady=(0, 10))
        self.log = tk.Text(log_frame, height=11, state="disabled", wrap="word")
        self.log.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        scroll.pack(side="right", fill="y")
        self.log.configure(yscrollcommand=scroll.set)
        controls = ttk.Frame(self.root)
        controls.pack(fill="x", padx=10, pady=(0, 10))
        self.stop_button = ttk.Button(controls, text="安全停止", command=self._stop, state="disabled")
        self.stop_button.pack(side="right")

    def _build_data_page(self) -> None:
        buttons = ttk.Frame(self.data_page)
        buttons.pack(fill="x", padx=10, pady=10)
        ttk.Button(buttons, text="添加数据集", command=self._add_dataset).pack(side="left")
        ttk.Button(buttons, text="移除", command=self._remove_dataset).pack(side="left", padx=6)
        ttk.Button(buttons, text="重新检查", command=self._inspect_all).pack(side="left")
        ttk.Button(buttons, text="导出锁定清单", command=self._export_manifest).pack(side="left", padx=6)
        columns = ("path", "samples", "boundary", "eye_region", "fill", "pupil", "ignored", "status")
        self.data_tree = ttk.Treeview(self.data_page, columns=columns, show="headings", height=17)
        headings = {
            "path": "数据集",
            "samples": "有效样本",
            "boundary": "Boundary/Complete",
            "eye_region": "眼球区域",
            "fill": "Fill",
            "pupil": "瞳孔",
            "ignored": "忽略原图",
            "status": "状态",
        }
        widths = {"path": 285, "samples": 75, "boundary": 120, "eye_region": 80, "fill": 55, "pupil": 60, "ignored": 70, "status": 160}
        for column in columns:
            self.data_tree.heading(column, text=headings[column])
            self.data_tree.column(column, width=widths[column], anchor="w")
        self.data_tree.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def _build_train_page(self) -> None:
        frame = ttk.Frame(self.train_page)
        frame.pack(fill="x", padx=12, pady=12)
        self.train_mode = tk.StringVar(value="rebuild")
        ttk.Label(frame, text="训练模式").grid(row=0, column=0, sticky="w", pady=4)
        modes = ttk.Frame(frame)
        modes.grid(row=0, column=1, sticky="w")
        for text, value in (("增量迭代（默认）", "incremental"), ("恢复中断", "resume"), ("完整重构", "rebuild")):
            ttk.Radiobutton(modes, text=text, value=value, variable=self.train_mode).pack(side="left", padx=(0, 12))
        self.base_model = tk.StringVar(value=str(project_root() / "models" / "current.pth"))
        self.resume_model = tk.StringVar()
        self.val_source = tk.StringVar()
        self.run_name = tk.StringVar()
        self.promote = tk.BooleanVar(value=False)
        config = default_config()
        training = config["training"]
        loss = config["loss"]
        self.training_vars = {
            "max_epochs": tk.StringVar(value=str(training["max_epochs"])),
            "batch_size": tk.StringVar(value=str(training["batch_size"])),
            "gradient_accumulation": tk.StringVar(value=str(training["gradient_accumulation"])),
            "learning_rate": tk.StringVar(value=str(training["learning_rate"])),
            "encoder_lr_factor": tk.StringVar(value=str(training["encoder_lr_factor"])),
            "incremental_lr_factor": tk.StringVar(value=str(training["incremental_lr_factor"])),
            "weight_decay": tk.StringVar(value=str(training["weight_decay"])),
            "warmup_epochs": tk.StringVar(value=str(training["warmup_epochs"])),
            "freeze_encoder_epochs": tk.StringVar(value=str(training["freeze_encoder_epochs"])),
            "early_stopping_patience": tk.StringVar(value=str(training["early_stopping_patience"])),
            "num_workers": tk.StringVar(value=str(training["num_workers"])),
            "seed": tk.StringVar(value=str(training["seed"])),
        }
        self.amp_enabled = tk.BooleanVar(value=bool(training["amp"]))
        self.loss_vars = {
            "iris": tk.StringVar(value=str(loss["iris"])),
            "eye_region": tk.StringVar(value=str(loss["eye_region"])),
            "pupil": tk.StringVar(value=str(loss["pupil"])),
            "boundary": tk.StringVar(value=str(loss["occlusion_boundary"])),
            "hierarchy": tk.StringVar(value=str(loss["hierarchy"])),
        }
        self._path_row(frame, 1, "基础模型", self.base_model, file=True)
        self._path_row(frame, 2, "恢复点", self.resume_model, file=True)
        self._path_row(frame, 3, "开发验证集/manifest（可选）", self.val_source, file_or_dir=True)
        ttk.Label(frame, text="版本名称").grid(row=4, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.run_name, width=70).grid(row=4, column=1, sticky="ew", pady=4)
        ttk.Checkbutton(frame, text="训练完成后设为当前正式模型", variable=self.promote).grid(row=5, column=1, sticky="w", pady=4)
        ttk.Label(
            frame,
            text=(
                "增量迭代会使用“数据”页中的全部累计数据集；只添加新数据而移除历史数据可能导致遗忘。\n"
                "开发验证集留空时不计算 Dice、禁用早停，并采用最后一个 epoch；选择后按 val_dice 选择最佳 epoch。"
            ),
            foreground="#8a4b08",
        ).grid(row=6, column=0, columnspan=3, sticky="w", pady=(10, 4))

        parameters = ttk.LabelFrame(frame, text="本次训练参数")
        parameters.grid(row=7, column=0, columnspan=4, sticky="ew", pady=(6, 2))
        preset_row = ttk.Frame(parameters)
        preset_row.grid(row=0, column=0, columnspan=4, sticky="w", padx=8, pady=(7, 3))
        ttk.Button(
            preset_row, text="应用完整重构建议", command=lambda: self._apply_training_preset("rebuild")
        ).pack(side="left")
        ttk.Button(
            preset_row, text="应用增量训练建议", command=lambda: self._apply_training_preset("incremental")
        ).pack(side="left", padx=6)
        ttk.Label(
            preset_row,
            text="预设只是起点；仅在选择开发验证集时启用 early stopping。",
            foreground="#555555",
        ).pack(side="left", padx=8)

        fields = (
            ("最大 Epoch", "max_epochs", "Batch size", "batch_size"),
            ("梯度累计", "gradient_accumulation", "基础学习率", "learning_rate"),
            ("编码器 LR 倍率", "encoder_lr_factor", "增量 LR 倍率", "incremental_lr_factor"),
            ("Weight decay", "weight_decay", "Warmup epoch", "warmup_epochs"),
            ("冻结编码器 epoch", "freeze_encoder_epochs", "早停耐心值（有验证集）", "early_stopping_patience"),
            ("数据加载进程", "num_workers", "随机种子", "seed"),
        )
        for row, (left_label, left_key, right_label, right_key) in enumerate(fields, start=1):
            ttk.Label(parameters, text=left_label).grid(row=row, column=0, sticky="w", padx=(8, 4), pady=3)
            ttk.Entry(parameters, textvariable=self.training_vars[left_key], width=16).grid(
                row=row, column=1, sticky="w", padx=(0, 18), pady=3
            )
            ttk.Label(parameters, text=right_label).grid(row=row, column=2, sticky="w", padx=(8, 4), pady=3)
            ttk.Entry(parameters, textvariable=self.training_vars[right_key], width=16).grid(
                row=row, column=3, sticky="w", padx=(0, 8), pady=3
            )

        ttk.Checkbutton(parameters, text="启用混合精度 AMP", variable=self.amp_enabled).grid(
            row=7, column=0, columnspan=2, sticky="w", padx=8, pady=3
        )
        loss_frame = ttk.Frame(parameters)
        loss_frame.grid(row=7, column=2, columnspan=2, sticky="w", padx=8, pady=3)
        for label, key in (("虹膜", "iris"), ("眼球区域", "eye_region"), ("瞳孔", "pupil"),
                           ("遮挡线", "boundary"), ("层级", "hierarchy")):
            ttk.Label(loss_frame, text=f"{label} loss").pack(side="left", padx=(0, 3))
            ttk.Entry(loss_frame, textvariable=self.loss_vars[key], width=6).pack(side="left", padx=(0, 8))
        ttk.Label(
            parameters,
            text="增量实际学习率 = 基础学习率 × 增量倍率；编码器再乘编码器倍率。恢复中断会恢复优化器状态。",
            foreground="#555555",
        ).grid(row=8, column=0, columnspan=4, sticky="w", padx=8, pady=(3, 7))
        parameters.columnconfigure(3, weight=1)

        ttk.Label(
            frame,
            text="默认启用：全图/局部双视图训练、浅层细节分支、EMA 验证和对称 TTA。",
            foreground="#0b6b3a",
        ).grid(row=8, column=0, columnspan=4, sticky="w", pady=(8, 2))
        self._apply_training_preset("rebuild")
        ttk.Button(frame, text="开始训练", command=self._start_training).grid(row=9, column=1, sticky="w", pady=10)
        frame.columnconfigure(1, weight=1)

    def _apply_training_preset(self, mode: str) -> None:
        values = default_config()["training"]
        if mode == "rebuild":
            values.update({
                "max_epochs": 120,
                "learning_rate": 1e-4,
                "encoder_lr_factor": 0.1,
                "incremental_lr_factor": 0.33,
                "warmup_epochs": 5,
                "freeze_encoder_epochs": 5,
                "early_stopping_patience": 20,
            })
        else:
            values.update({
                "max_epochs": 50,
                "learning_rate": 1e-4,
                "encoder_lr_factor": 0.1,
                "incremental_lr_factor": 0.33,
                "warmup_epochs": 3,
                "freeze_encoder_epochs": 0,
                "early_stopping_patience": 12,
            })
        self.train_mode.set(mode)
        for key, variable in self.training_vars.items():
            variable.set(str(values[key]))
        self.amp_enabled.set(bool(values["amp"]))

    def _training_parameter_arguments(self) -> list[str]:
        integer_keys = {
            "max_epochs", "batch_size", "gradient_accumulation", "warmup_epochs",
            "freeze_encoder_epochs", "early_stopping_patience", "num_workers", "seed",
        }
        cli_names = {key: _cli_option(key) for key in self.training_vars}
        arguments: list[str] = []
        for key, variable in self.training_vars.items():
            raw = variable.get().strip()
            try:
                int(raw) if key in integer_keys else float(raw)
            except ValueError as exc:
                raise ValueError(f"训练参数 {key} 不是有效数字：{raw}") from exc
            arguments += [cli_names[key], raw]
        for key, variable in self.loss_vars.items():
            raw = variable.get().strip()
            try:
                float(raw)
            except ValueError as exc:
                raise ValueError(f"损失权重 {key} 不是有效数字：{raw}") from exc
            arguments += [_cli_option(key, loss=True), raw]
        arguments.append("--amp" if self.amp_enabled.get() else "--no-amp")
        return arguments

    def _build_eval_page(self) -> None:
        frame = ttk.Frame(self.eval_page)
        frame.pack(fill="x", padx=12, pady=12)
        self.eval_model = tk.StringVar(value=str(project_root() / "models" / "current.pth"))
        self.eval_kind = tk.StringVar(value="root")
        self.eval_source = tk.StringVar()
        self.eval_output = tk.StringVar()
        self.eval_preset = tk.StringVar(value="high")
        self.eval_target = tk.StringVar(value="iris")
        self._path_row(frame, 0, "模型", self.eval_model, file=True)
        ttk.Label(frame, text="评估来源").grid(row=1, column=0, sticky="w", pady=4)
        kinds = ttk.Frame(frame)
        kinds.grid(row=1, column=1, sticky="w")
        ttk.Radiobutton(kinds, text="普通数据集", variable=self.eval_kind, value="root").pack(side="left")
        ttk.Radiobutton(kinds, text="Benchmark manifest", variable=self.eval_kind, value="benchmark").pack(side="left", padx=12)
        self._path_row(frame, 2, "数据/Benchmark", self.eval_source, file_or_dir=True)
        self._path_row(frame, 3, "报告输出目录", self.eval_output, directory=True, save=True)
        ttk.Label(frame, text="推理预设").grid(row=4, column=0, sticky="w", pady=4)
        ttk.Combobox(frame, textvariable=self.eval_preset, values=("fast", "standard", "high"), state="readonly", width=18).grid(row=4, column=1, sticky="w")
        ttk.Label(frame, text="评估目标").grid(row=5, column=0, sticky="w", pady=4)
        ttk.Combobox(
            frame, textvariable=self.eval_target,
            values=("iris", "eye_region", "sclera"), state="readonly", width=18,
        ).grid(row=5, column=1, sticky="w")
        ttk.Button(frame, text="开始评估", command=self._start_evaluation).grid(row=6, column=1, sticky="w", pady=12)
        frame.columnconfigure(1, weight=1)

    def _build_annotate_page(self) -> None:
        frame = ttk.Frame(self.annotate_page)
        frame.pack(fill="x", padx=12, pady=12)
        self.ann_model = tk.StringVar(value=str(project_root() / "models" / "current.pth"))
        self.ann_images = tk.StringVar()
        self.ann_preset = tk.StringVar(value="high")
        self.ann_target = tk.StringVar(value="iris")
        self.ann_doubt = tk.StringVar(value="0.1")
        self._path_row(frame, 0, "模型", self.ann_model, file=True)
        self._path_row(frame, 1, "原图文件夹", self.ann_images, directory=True)
        ttk.Label(frame, text="推理预设").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Combobox(frame, textvariable=self.ann_preset, values=("fast", "standard", "high"), state="readonly", width=18).grid(row=2, column=1, sticky="w")
        ttk.Label(frame, text="标注目标").grid(row=3, column=0, sticky="w", pady=4)
        targets = ttk.Frame(frame)
        targets.grid(row=3, column=1, sticky="w")
        for text, value in (("虹膜", "iris"), ("巩膜", "sclera"), ("巩膜 + 虹膜", "both")):
            ttk.Radiobutton(targets, text=text, value=value, variable=self.ann_target).pack(side="left", padx=(0, 12))
        ttk.Label(frame, text="Doubt 数量/比例").grid(row=4, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.ann_doubt, width=20).grid(row=4, column=1, sticky="w")
        ttk.Label(
            frame,
            text="输出固定保存到项目目录 auto_annotations/<当前时间>/",
            foreground="#555555",
        ).grid(row=5, column=0, columnspan=3, sticky="w", pady=(10, 2))
        ttk.Button(frame, text="开始自动标注", command=self._start_annotation).grid(row=6, column=1, sticky="w", pady=12)
        frame.columnconfigure(1, weight=1)

    def _build_models_page(self) -> None:
        ttk.Button(self.models_page, text="刷新", command=self._refresh_models).pack(anchor="w", padx=10, pady=10)
        self.model_tree = ttk.Treeview(self.models_page, columns=("name", "mode", "parent", "promoted", "path"), show="headings")
        for column, width in (("name", 180), ("mode", 90), ("parent", 220), ("promoted", 70), ("path", 360)):
            self.model_tree.heading(column, text=column)
            self.model_tree.column(column, width=width, anchor="w")
        self.model_tree.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self._refresh_models()

    def _build_monitor_page(self) -> None:
        controls = ttk.Frame(self.monitor_page)
        controls.pack(fill="x", padx=12, pady=12)
        self.monitor_log = tk.StringVar()
        ttk.Label(controls, text="train_log.csv").pack(side="left")
        ttk.Entry(controls, textvariable=self.monitor_log, width=82).pack(
            side="left", fill="x", expand=True, padx=8
        )
        ttk.Button(controls, text="选择", command=self._pick_train_log).pack(side="left")
        ttk.Button(controls, text="最近一次", command=self._use_latest_train_log).pack(side="left", padx=6)
        ttk.Button(controls, text="刷新", command=self._refresh_curve).pack(side="left")
        self.curve_status = tk.StringVar(value="训练开始后选择 runs/<运行名称>/train_log.csv")
        ttk.Label(self.monitor_page, textvariable=self.curve_status, foreground="#555555").pack(
            fill="x", padx=12, pady=(0, 6)
        )
        self.curve_canvas = tk.Canvas(self.monitor_page, background="white", height=520)
        self.curve_canvas.pack(fill="both", expand=True, padx=12, pady=(0, 12))

    def _pick_train_log(self) -> None:
        selected = filedialog.askopenfilename(
            title="选择训练日志",
            initialdir=str(project_root() / "runs"),
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if selected:
            self.monitor_log.set(selected)
            self._refresh_curve()

    def _use_latest_train_log(self) -> None:
        logs = list((project_root() / "runs").glob("*/train_log.csv"))
        if not logs:
            messagebox.showinfo("训练曲线", "还没有找到训练日志。")
            return
        latest = max(logs, key=lambda path: path.stat().st_mtime)
        self.monitor_log.set(str(latest))
        self._refresh_curve()

    def _auto_refresh_monitor(self) -> None:
        if getattr(self, "monitor_log", None) and self.monitor_log.get():
            self._refresh_curve(silent=True)
        self.root.after(2000, self._auto_refresh_monitor)

    def _refresh_curve(self, *, silent: bool = False) -> None:
        path = Path(self.monitor_log.get()) if self.monitor_log.get() else None
        if path is None or not path.is_file():
            if not silent:
                self.curve_status.set("请选择有效的 train_log.csv")
            return
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        except Exception as exc:
            if not silent:
                messagebox.showerror("读取失败", str(exc))
            return
        if not rows:
            return
        epochs = [float(row["epoch"]) for row in rows]
        train_loss = [float(row["train_loss"]) for row in rows]
        val_dice = [float(row["val_dice"]) for row in rows if row.get("val_dice") not in (None, "")]
        val_epochs = [float(row["epoch"]) for row in rows if row.get("val_dice") not in (None, "")]
        canvas = self.curve_canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 900)
        height = max(canvas.winfo_height(), 500)
        self._draw_series(canvas, (55, 35, width - 25, height * 0.48), epochs, train_loss,
                          title="训练损失 train_loss", color="#1769aa", lower_is_better=True)
        if val_dice:
            self._draw_series(canvas, (55, height * 0.55, width - 25, height - 35), val_epochs, val_dice,
                              title="开发集 Dice val_dice", color="#16803c", lower_is_better=False)
            self.curve_status.set(
                f"epoch {int(epochs[-1])}｜train_loss {train_loss[-1]:.5f}｜"
                f"val_dice {val_dice[-1]:.5f}｜最佳 {max(val_dice):.5f}"
            )
        else:
            self.curve_status.set(f"epoch {int(epochs[-1])}｜train_loss {train_loss[-1]:.5f}｜无开发集指标")

    @staticmethod
    def _draw_series(canvas, box, xs, ys, *, title: str, color: str, lower_is_better: bool) -> None:
        left, top, right, bottom = (float(value) for value in box)
        canvas.create_text(left, top - 14, text=title, anchor="w", fill="#222222")
        canvas.create_rectangle(left, top, right, bottom, outline="#b0b0b0")
        if len(xs) < 1:
            return
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        if y_max == y_min:
            y_max = y_min + 1.0
        x_span = max(x_max - x_min, 1.0)
        points = []
        for x, y in zip(xs, ys):
            px = left + (x - x_min) / x_span * (right - left)
            py = bottom - (y - y_min) / (y_max - y_min) * (bottom - top)
            points.extend((px, py))
        if len(points) >= 4:
            canvas.create_line(*points, fill=color, width=2, smooth=True)
        else:
            canvas.create_oval(points[0] - 2, points[1] - 2, points[0] + 2, points[1] + 2, fill=color)
        best = min(ys) if lower_is_better else max(ys)
        canvas.create_text(left + 5, top + 5, text=f"最大 {y_max:.4f}", anchor="nw", fill="#666666")
        canvas.create_text(left + 5, bottom - 5, text=f"最小 {y_min:.4f}", anchor="sw", fill="#666666")
        canvas.create_text(right - 5, top + 5, text=f"最佳 {best:.4f}", anchor="ne", fill=color)

    def _path_row(self, frame, row, label, variable, *, file=False, directory=False, file_or_dir=False, save=False):
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=variable, width=70).grid(row=row, column=1, sticky="ew", pady=4)

        if file_or_dir:
            def choose_directory():
                value = filedialog.askdirectory()
                if value:
                    variable.set(value)

            def choose_manifest():
                value = filedialog.askopenfilename(
                    filetypes=(("JSON Lines manifest", "*.jsonl"), ("All files", "*.*")),
                )
                if value:
                    variable.set(value)

            ttk.Button(frame, text="选择文件夹", command=choose_directory).grid(row=row, column=2, padx=5)
            ttk.Button(frame, text="选择清单", command=choose_manifest).grid(row=row, column=3, padx=(0, 5))
            return

        def choose():
            if file:
                value = filedialog.askopenfilename()
            else:
                value = filedialog.askdirectory()
            if value:
                variable.set(value)
        ttk.Button(frame, text="选择", command=choose).grid(row=row, column=2, padx=5)

    def _add_dataset(self) -> None:
        path = filedialog.askdirectory()
        if path and path not in self.datasets:
            self.datasets.append(path)
            self._inspect_dataset(path)

    def _remove_dataset(self) -> None:
        selected = self.data_tree.selection()
        for item in selected:
            path = self.data_tree.item(item, "values")[0]
            if path in self.datasets:
                self.datasets.remove(path)
            self.data_tree.delete(item)

    def _inspect_all(self) -> None:
        self.data_tree.delete(*self.data_tree.get_children())
        for path in self.datasets:
            self._inspect_dataset(path)

    def _export_manifest(self) -> None:
        if not self.datasets:
            messagebox.showerror("错误", "请先添加数据集。")
            return
        destination = filedialog.asksaveasfilename(
            defaultextension=".jsonl",
            filetypes=(("JSON Lines", "*.jsonl"), ("All files", "*.*")),
        )
        if not destination:
            return
        try:
            records = []
            for path in self.datasets:
                records.extend(compile_dataset(path, compute_hashes=True))
            issues = validate_records(records)
            errors = [item for item in issues if item.level == "error"]
            if errors:
                raise ValueError("；".join(f"{item.sample_id}: {item.message}" for item in errors[:5]))
            save_manifest(records, destination)
            messagebox.showinfo("完成", f"已锁定 {len(records)} 个样本：\n{destination}")
        except Exception as exc:
            messagebox.showerror("导出失败", str(exc))

    def _inspect_dataset(self, path: str) -> None:
        try:
            records = compile_dataset(path, compute_hashes=False)
            summary = summarize_records(records)
            issues = validate_records(records, deep=False)
            image_dir = next((Path(path) / name for name in ("JPEGImages", "Images", "images") if (Path(path) / name).is_dir()), None)
            image_count = len(list(image_dir.iterdir())) if image_dir else len(records)
            ignored = max(0, image_count - len(records))
            boundary = summary["boundary"]
            status = "通过" if not issues else f"{len(issues)} 个提示"
            self.data_tree.insert("", "end", values=(
                path,
                summary["samples"],
                f"{boundary.get('boundary', 0)}/{boundary.get('complete', 0)}",
                summary["with_eye_region"],
                summary["with_fill"],
                summary["with_pupil"],
                ignored,
                status,
            ))
        except Exception as exc:
            self.data_tree.insert("", "end", values=(path, 0, "0/0", 0, 0, 0, "?", f"错误: {exc}"))

    def _start_training(self) -> None:
        if not self.datasets:
            messagebox.showerror("错误", "请先在数据页添加累计训练数据集。")
            return
        val_path = Path(self.val_source.get().strip())
        if self.val_source.get().strip() and val_path.is_file() and val_path.suffix.casefold() != ".jsonl":
            messagebox.showerror("验证集类型错误", "验证集文件只能是 .jsonl manifest；图片数据请选择其数据集文件夹。")
            return
        if self.train_mode.get() == "incremental" and not Path(self.base_model.get().strip()).is_file():
            messagebox.showerror(
                "缺少基础模型",
                "增量训练需要现有模型。首次训练请选择“完整重构”；以后可以使用增量迭代。",
            )
            return
        if self.train_mode.get() == "resume" and not Path(self.resume_model.get().strip()).is_file():
            messagebox.showerror("缺少恢复点", "恢复中断模式需要选择 last.pth。")
            return
        command = [sys.executable, "-m", "iris_system.cli.train", "--mode", self.train_mode.get()]
        try:
            command += self._training_parameter_arguments()
        except ValueError as exc:
            messagebox.showerror("训练参数错误", str(exc))
            return
        for path in self.datasets:
            command += ["--root", path]
        if self.base_model.get().strip():
            command += ["--base-model", self.base_model.get().strip()]
        if self.resume_model.get().strip():
            command += ["--resume-model", self.resume_model.get().strip()]
        val = self.val_source.get().strip()
        if val:
            command += ["--val-manifest" if Path(val).is_file() else "--val-root", val]
        if self.run_name.get().strip():
            command += ["--run-name", self.run_name.get().strip()]
        if self.promote.get():
            command.append("--promote")
        self._run(command)

    def _start_evaluation(self) -> None:
        source = self.eval_source.get().strip()
        model = self.eval_model.get().strip()
        if not source or not model:
            messagebox.showerror("错误", "请选择模型和评估数据。")
            return
        source_path = Path(source)
        if self.eval_kind.get() == "benchmark" and (
            not source_path.is_file() or source_path.suffix.casefold() != ".jsonl"
        ):
            messagebox.showerror("Benchmark 类型错误", "Benchmark 必须选择 .jsonl 锁定清单。")
            return
        if self.eval_kind.get() == "root" and not source_path.is_dir():
            messagebox.showerror("数据集类型错误", "普通评估数据必须选择数据集文件夹。")
            return
        command = [sys.executable, "-m", "iris_system.cli.evaluate", "--model", model,
                   "--benchmark" if self.eval_kind.get() == "benchmark" else "--root", source,
                   "--preset", self.eval_preset.get(), "--target", self.eval_target.get()]
        if self.eval_output.get().strip():
            command += ["--output", self.eval_output.get().strip()]
        self._run(command)

    def _start_annotation(self) -> None:
        if not self.ann_model.get().strip() or not self.ann_images.get().strip():
            messagebox.showerror("错误", "请选择模型和原图文件夹。")
            return
        command = [sys.executable, "-m", "iris_system.cli.annotate",
                   "--model", self.ann_model.get().strip(), "--images", self.ann_images.get().strip(),
                   "--preset", self.ann_preset.get(), "--target", self.ann_target.get(),
                   "--doubt", self.ann_doubt.get().strip()]
        self._run(command)

    def _run(self, command: list[str]) -> None:
        if self.process is not None:
            messagebox.showwarning("运行中", "已有任务正在运行。")
            return
        self._append_log("$ " + subprocess.list2cmdline(command) + "\n")
        environment = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        source_root = str(project_root() / "src")
        environment["PYTHONPATH"] = source_root + os.pathsep + environment.get("PYTHONPATH", "")
        self.process = subprocess.Popen(
            command,
            cwd=project_root(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=environment,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        self.stop_button.configure(state="normal")
        threading.Thread(target=self._read_process, daemon=True).start()

    def _read_process(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for line in self.process.stdout:
            self.log_queue.put(line)
        self.process.wait()
        self.log_queue.put(None)

    def _poll_log(self) -> None:
        try:
            while True:
                value = self.log_queue.get_nowait()
                if value is None:
                    code = self.process.returncode if self.process else -1
                    self._append_log(f"任务结束，退出码 {code}\n")
                    self.process = None
                    self.stop_button.configure(state="disabled")
                    self._refresh_models()
                else:
                    self._append_log(value)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_log)

    def _append_log(self, value: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", value)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _stop(self) -> None:
        if self.process is not None:
            self.process.terminate()
            self._append_log("已请求停止；最近 epoch 的 last.pth 可用于恢复。\n")

    def _refresh_models(self) -> None:
        if not hasattr(self, "model_tree"):
            return
        self.model_tree.delete(*self.model_tree.get_children())
        registry = project_root() / "models" / "registry.json"
        if not registry.exists():
            return
        try:
            entries = json.loads(registry.read_text(encoding="utf-8"))
        except Exception:
            entries = []
        for entry in reversed(entries):
            self.model_tree.insert("", "end", values=(
                entry.get("name"), entry.get("mode"), entry.get("parent"),
                entry.get("promoted"), entry.get("checkpoint"),
            ))


def main() -> None:
    root = tk.Tk()
    IrisApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
