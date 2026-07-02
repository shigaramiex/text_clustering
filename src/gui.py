import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from src.pipeline import process_genre_folder


class ClusteringApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("ニュース記事サブクラスタリング")
        self.root.geometry("640x480")

        self.selected_dir = tk.StringVar()
        self.k_min = tk.IntVar(value=2)
        self.k_max = tk.IntVar(value=10)
        self._message_queue: queue.Queue[str] = queue.Queue()
        self._worker_thread: threading.Thread | None = None

        self._build_widgets()
        self.root.after(100, self._drain_message_queue)

    def _build_widgets(self) -> None:
        folder_frame = ttk.Frame(self.root, padding=10)
        folder_frame.pack(fill=tk.X)

        ttk.Label(folder_frame, text="対象フォルダ（テキストファイルが直接入っているフォルダ）:").pack(
            side=tk.LEFT
        )
        ttk.Entry(folder_frame, textvariable=self.selected_dir, width=50).pack(
            side=tk.LEFT, padx=5, fill=tk.X, expand=True
        )
        ttk.Button(folder_frame, text="参照...", command=self._choose_folder).pack(
            side=tk.LEFT
        )

        param_frame = ttk.Frame(self.root, padding=10)
        param_frame.pack(fill=tk.X)

        ttk.Label(param_frame, text="クラスタ数探索範囲 k =").pack(side=tk.LEFT)
        ttk.Spinbox(
            param_frame, from_=2, to=20, textvariable=self.k_min, width=5
        ).pack(side=tk.LEFT, padx=5)
        ttk.Label(param_frame, text="〜").pack(side=tk.LEFT)
        ttk.Spinbox(
            param_frame, from_=2, to=30, textvariable=self.k_max, width=5
        ).pack(side=tk.LEFT, padx=5)

        self.run_button = ttk.Button(
            self.root, text="実行", command=self._start_run
        )
        self.run_button.pack(pady=5)

        log_frame = ttk.Frame(self.root, padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(log_frame, state=tk.DISABLED, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _choose_folder(self) -> None:
        chosen = filedialog.askdirectory(
            title="テキストファイル(.txt)が直接入っているフォルダを選択"
        )
        if chosen:
            self.selected_dir.set(chosen)

    def _log(self, message: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _start_run(self) -> None:
        target_dir = self.selected_dir.get().strip()
        if not target_dir:
            messagebox.showwarning("未選択", "対象フォルダを選択してください")
            return
        if not Path(target_dir).is_dir():
            messagebox.showerror("エラー", "指定されたフォルダが存在しません")
            return
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return

        self.run_button.configure(state=tk.DISABLED)
        self._worker_thread = threading.Thread(
            target=self._run_pipeline, args=(target_dir,), daemon=True
        )
        self._worker_thread.start()

    def _run_pipeline(self, target_dir: str) -> None:
        try:
            summary = process_genre_folder(
                Path(target_dir),
                k_min=self.k_min.get(),
                k_max=self.k_max.get(),
                progress_callback=self._message_queue.put,
            )
            if summary["output_dir"] is not None:
                self._message_queue.put(
                    f"完了: {summary['total_files']}件のファイルを"
                    f"{summary['num_clusters']}個のクラスタに分類し、"
                    f"{summary['output_dir']} にコピーしました"
                    "（元のフォルダは変更していません）"
                )
            else:
                self._message_queue.put(
                    f"完了: ファイルが{summary['total_files']}件のみのため"
                    "クラスタリングをスキップしました"
                )
        except Exception as exc:  # noqa: BLE001 - surface to the log panel
            self._message_queue.put(f"予期しないエラー: {exc}")
        finally:
            self._message_queue.put("__DONE__")

    def _drain_message_queue(self) -> None:
        try:
            while True:
                message = self._message_queue.get_nowait()
                if message == "__DONE__":
                    self.run_button.configure(state=tk.NORMAL)
                else:
                    self._log(message)
        except queue.Empty:
            pass
        self.root.after(100, self._drain_message_queue)


def main() -> None:
    root = tk.Tk()
    ClusteringApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
