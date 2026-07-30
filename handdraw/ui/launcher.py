"""Dark-mode launcher built with CustomTkinter.

Replaces the FreeSimpleGUI window.  Notable behaviours:

* A single Tk root owns every dialog (``parent=self``), so no stray hidden root
  windows are ever created or leaked.
* The PDF is opened on a worker thread behind an indeterminate progress bar;
  the annotation loop then runs on the main thread with the launcher hidden,
  which keeps all OpenCV window calls on the thread that created them.
* Failures surface as readable modal dialogs instead of tracebacks.
"""

from __future__ import annotations

import threading
import traceback
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk

from .. import models, paths
from ..document import PdfDocument
from ..errors import AppError
from ..session import AnnotationSession
from ..settings import AppSettings

ACCENT = "#F09A3E"
ACCENT_HOVER = "#D8842B"
SURFACE = "#1B1A19"
CARD = "#232120"
CARD_BORDER = "#332F2C"
TEXT_MUTED = "#A9A29D"

_FONT_FAMILY = "Segoe UI"


class Dialog(ctk.CTkToplevel):
    """Small modal message box that matches the app's styling."""

    def __init__(self, master: ctk.CTk, title: str, message: str, detail: str = "",
                 tone: str = "error") -> None:
        super().__init__(master)
        self.title(title)
        self.configure(fg_color=SURFACE)
        self.resizable(False, False)
        self.transient(master)

        accent = {"error": "#E5484D", "success": "#5BBD72", "info": ACCENT}.get(tone, ACCENT)

        wrapper = ctk.CTkFrame(self, fg_color=CARD, corner_radius=14, border_width=1,
                               border_color=CARD_BORDER)
        wrapper.pack(padx=18, pady=18, fill="both", expand=True)

        ctk.CTkLabel(
            wrapper, text=message, font=ctk.CTkFont(_FONT_FAMILY, 15, "bold"),
            text_color=accent, wraplength=420, justify="left",
        ).pack(anchor="w", padx=20, pady=(18, 6))

        if detail:
            ctk.CTkLabel(
                wrapper, text=detail, font=ctk.CTkFont(_FONT_FAMILY, 12),
                text_color=TEXT_MUTED, wraplength=420, justify="left",
            ).pack(anchor="w", padx=20, pady=(0, 12))

        ctk.CTkButton(
            wrapper, text="OK", width=110, height=34, corner_radius=10,
            fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color="#1B1A19",
            font=ctk.CTkFont(_FONT_FAMILY, 13, "bold"), command=self.destroy,
        ).pack(anchor="e", padx=20, pady=(0, 18))

        self.update_idletasks()
        self._center_on(master)
        self.grab_set()
        self.focus_force()

    def _center_on(self, master: ctk.CTk) -> None:
        try:
            x = master.winfo_rootx() + (master.winfo_width() - self.winfo_width()) // 2
            y = master.winfo_rooty() + (master.winfo_height() - self.winfo_height()) // 3
            self.geometry(f"+{max(0, x)}+{max(0, y)}")
        except Exception:  # pragma: no cover - geometry is cosmetic
            pass


class LauncherApp(ctk.CTk):
    """The launcher window."""

    def __init__(self) -> None:
        super().__init__()
        ctk.set_appearance_mode("dark")

        self.title("AI Hand Draw")
        self.geometry("780x660")
        self.minsize(700, 620)
        self.configure(fg_color=SURFACE)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.pdf_path: Path | None = None
        self.output_dir: Path = paths.default_output_dir()
        self._busy = False

        self._build()
        self._refresh_start_state()

    # ----------------------------------------------------------------- build
    def _card(self, parent: ctk.CTkBaseClass) -> ctk.CTkFrame:
        card = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=16, border_width=1,
                            border_color=CARD_BORDER)
        card.pack(fill="x", pady=(0, 14))
        return card

    def _build(self) -> None:
        root = ctk.CTkFrame(self, fg_color="transparent")
        root.pack(fill="both", expand=True, padx=28, pady=24)

        # --- header ------------------------------------------------------
        header = ctk.CTkFrame(root, fg_color="transparent")
        header.pack(fill="x", pady=(0, 20))
        ctk.CTkLabel(
            header, text="AI Hand Draw", font=ctk.CTkFont(_FONT_FAMILY, 30, "bold"),
            text_color="#F5F3F1",
        ).pack(anchor="w")
        ctk.CTkLabel(
            header,
            text="Annotate any PDF in mid-air - pinch to draw, open your palm to erase, "
                 "point and swipe to turn the page.",
            font=ctk.CTkFont(_FONT_FAMILY, 13), text_color=TEXT_MUTED,
            wraplength=680, justify="left",
        ).pack(anchor="w", pady=(6, 0))

        # --- document ----------------------------------------------------
        document = self._card(root)
        self._section(document, "Document")
        row = ctk.CTkFrame(document, fg_color="transparent")
        row.pack(fill="x", padx=20, pady=(0, 18))
        self.pdf_entry = ctk.CTkEntry(
            row, placeholder_text="Choose a PDF file...", height=40, corner_radius=10,
            font=ctk.CTkFont(_FONT_FAMILY, 13), fg_color="#1A1918", border_color=CARD_BORDER,
        )
        self.pdf_entry.pack(side="left", fill="x", expand=True)
        self.pdf_entry.configure(state="readonly")
        ctk.CTkButton(
            row, text="Browse", width=110, height=40, corner_radius=10,
            fg_color="#2E2B29", hover_color="#3A3633", font=ctk.CTkFont(_FONT_FAMILY, 13),
            command=self._choose_pdf,
        ).pack(side="left", padx=(10, 0))

        # --- output ------------------------------------------------------
        output = self._card(root)
        self._section(output, "Save annotations to")
        row = ctk.CTkFrame(output, fg_color="transparent")
        row.pack(fill="x", padx=20, pady=(0, 18))
        self.output_entry = ctk.CTkEntry(
            row, height=40, corner_radius=10, font=ctk.CTkFont(_FONT_FAMILY, 13),
            fg_color="#1A1918", border_color=CARD_BORDER,
        )
        self.output_entry.pack(side="left", fill="x", expand=True)
        self._set_entry(self.output_entry, str(self.output_dir), readonly=True)
        ctk.CTkButton(
            row, text="Change", width=110, height=40, corner_radius=10,
            fg_color="#2E2B29", hover_color="#3A3633", font=ctk.CTkFont(_FONT_FAMILY, 13),
            command=self._choose_output,
        ).pack(side="left", padx=(10, 0))

        # --- options -----------------------------------------------------
        options = self._card(root)
        self._section(options, "Capture")
        grid = ctk.CTkFrame(options, fg_color="transparent")
        grid.pack(fill="x", padx=20, pady=(0, 18))
        grid.grid_columnconfigure((0, 1, 2), weight=1, uniform="opt")

        self.camera_menu = self._labelled_menu(grid, 0, "Camera", ["0", "1", "2", "3"], "0")
        self.dpi_menu = self._labelled_menu(grid, 1, "Page quality",
                                            ["110 - fast", "150 - balanced", "200 - sharp"],
                                            "150 - balanced")

        toggles = ctk.CTkFrame(grid, fg_color="transparent")
        toggles.grid(row=0, column=2, sticky="w", padx=(12, 0))
        ctk.CTkLabel(toggles, text="Options", font=ctk.CTkFont(_FONT_FAMILY, 12),
                     text_color=TEXT_MUTED).pack(anchor="w", pady=(0, 6))
        self.mirror_switch = self._switch(toggles, "Mirror camera", True)
        self.preview_switch = self._switch(toggles, "Show camera preview", True)

        # --- action ------------------------------------------------------
        action = ctk.CTkFrame(root, fg_color="transparent")
        action.pack(fill="x", pady=(4, 0))

        self.start_button = ctk.CTkButton(
            action, text="Start annotating", height=52, corner_radius=12,
            fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color="#1B1A19",
            font=ctk.CTkFont(_FONT_FAMILY, 16, "bold"), command=self._start,
        )
        self.start_button.pack(fill="x")

        self.progress = ctk.CTkProgressBar(action, height=4, corner_radius=2,
                                           progress_color=ACCENT, fg_color="#2A2725")
        self.progress.set(0)  # packed only while something is loading

        self.status = ctk.CTkLabel(action, text="Ready.", font=ctk.CTkFont(_FONT_FAMILY, 12),
                                   text_color=TEXT_MUTED, anchor="w")
        self.status.pack(fill="x", pady=(8, 0))

    def _section(self, parent: ctk.CTkFrame, title: str) -> None:
        ctk.CTkLabel(parent, text=title, font=ctk.CTkFont(_FONT_FAMILY, 12, "bold"),
                     text_color=TEXT_MUTED).pack(anchor="w", padx=20, pady=(16, 8))

    def _labelled_menu(self, parent: ctk.CTkFrame, column: int, label: str,
                       values: list[str], default: str) -> ctk.CTkOptionMenu:
        holder = ctk.CTkFrame(parent, fg_color="transparent")
        holder.grid(row=0, column=column, sticky="ew", padx=(0, 12))
        ctk.CTkLabel(holder, text=label, font=ctk.CTkFont(_FONT_FAMILY, 12),
                     text_color=TEXT_MUTED).pack(anchor="w", pady=(0, 6))
        menu = ctk.CTkOptionMenu(
            holder, values=values, height=36, corner_radius=10,
            font=ctk.CTkFont(_FONT_FAMILY, 13), fg_color="#2E2B29", button_color="#3A3633",
            button_hover_color="#474240", dropdown_fg_color=CARD,
        )
        menu.set(default)
        menu.pack(fill="x")
        return menu

    def _switch(self, parent: ctk.CTkFrame, text: str, default: bool) -> ctk.CTkSwitch:
        switch = ctk.CTkSwitch(
            parent, text=text, font=ctk.CTkFont(_FONT_FAMILY, 13), progress_color=ACCENT,
            button_color="#F5F3F1", text_color="#D9D4D0",
        )
        switch.select() if default else switch.deselect()
        switch.pack(anchor="w", pady=3)
        return switch

    # ----------------------------------------------------------------- utils
    @staticmethod
    def _set_entry(entry: ctk.CTkEntry, value: str, readonly: bool = False) -> None:
        entry.configure(state="normal")
        entry.delete(0, "end")
        entry.insert(0, value)
        if readonly:
            entry.configure(state="readonly")

    def _set_status(self, text: str, tone: str = "muted") -> None:
        colors = {"muted": TEXT_MUTED, "error": "#E5484D", "success": "#5BBD72", "accent": ACCENT}
        self.status.configure(text=text, text_color=colors.get(tone, TEXT_MUTED))

    def _set_busy(self, busy: bool, message: str = "") -> None:
        self._busy = busy
        state = "disabled" if busy else "normal"
        for widget in (self.start_button, self.camera_menu, self.dpi_menu,
                       self.mirror_switch, self.preview_switch):
            widget.configure(state=state)
        if busy:
            self.progress.pack(fill="x", pady=(12, 0), before=self.status)
            self.progress.configure(mode="indeterminate")
            self.progress.start()
            self._set_status(message or "Working...", "accent")
        else:
            self.progress.stop()
            self.progress.configure(mode="determinate")
            self.progress.set(0)
            self.progress.pack_forget()
        self._refresh_start_state()

    def _refresh_start_state(self) -> None:
        ready = self.pdf_path is not None and not self._busy
        self.start_button.configure(state="normal" if ready else "disabled")

    def _error(self, message: str, detail: str = "") -> None:
        Dialog(self, "Something went wrong", message, detail, tone="error")
        self._set_status(message, "error")

    # --------------------------------------------------------------- actions
    def _choose_pdf(self) -> None:
        selected = filedialog.askopenfilename(
            parent=self,  # reuses this root: no orphaned Tk windows
            title="Select a PDF to annotate",
            filetypes=[("PDF documents", "*.pdf"), ("All files", "*.*")],
        )
        if not selected:
            return
        path = Path(selected)
        if path.suffix.lower() != ".pdf":
            self._error("That file is not a PDF.", path.name)
            return
        self.pdf_path = path
        self._set_entry(self.pdf_entry, str(path), readonly=True)
        self._set_status(f"Ready to annotate {path.name}.")
        self._refresh_start_state()

    def _choose_output(self) -> None:
        selected = filedialog.askdirectory(
            parent=self, title="Choose where annotations are saved",
            initialdir=str(self.output_dir),
        )
        if not selected:
            return
        directory = Path(selected)
        if not paths.is_writable(directory):
            self._error("That folder is not writable.", str(directory))
            return
        self.output_dir = directory
        self._set_entry(self.output_entry, str(directory), readonly=True)

    def _settings(self) -> AppSettings:
        dpi = int(self.dpi_menu.get().split(" ")[0])
        return AppSettings(
            pdf_path=self.pdf_path,
            render_dpi=dpi,
            camera_index=int(self.camera_menu.get()),
            mirror=bool(self.mirror_switch.get()),
            show_preview=bool(self.preview_switch.get()),
            output_dir=self.output_dir,
        )

    def _start(self) -> None:
        if self._busy or self.pdf_path is None:
            return
        settings = self._settings()
        self._set_busy(True, f"Opening {self.pdf_path.name}...")
        threading.Thread(target=self._load_worker, args=(settings,), daemon=True).start()

    def _load_worker(self, settings: AppSettings) -> None:
        """Runs off the UI thread: opening and rasterising can take a moment."""
        try:
            model = self._ensure_model()
            document = PdfDocument(
                settings.pdf_path,  # type: ignore[arg-type]
                dpi=settings.render_dpi,
                max_pixels=settings.max_page_pixels,
            )
            document.render(0)  # warm the first page so the window opens instantly
        except AppError as exc:
            # Bind before scheduling: Python unbinds `exc` when the except
            # block ends, so a lambda closing over it would raise NameError.
            message, detail = exc.message, exc.detail
            self.after(0, lambda: self._load_failed(message, detail))
        except Exception as exc:  # pragma: no cover - unexpected failures
            detail = str(exc)
            self.after(0, lambda: self._load_failed("The PDF could not be opened.", detail))
        else:
            self.after(0, lambda: self._run_session(settings, document, model))

    def _ensure_model(self) -> Path:
        """Fetch the hand-tracking model once, reporting real progress."""
        if models.find_model() is not None:
            return models.ensure_model()

        self.after(0, lambda: self._download_started())

        def report(fraction: float) -> None:
            self.after(0, lambda: self.progress.set(fraction))

        path = models.download_model(progress=report)
        self.after(0, lambda: self._download_finished())
        return path

    def _download_started(self) -> None:
        self.progress.stop()
        self.progress.configure(mode="determinate")
        self.progress.set(0)
        self._set_status("Downloading the hand-tracking model (7.5 MB, one time only)...", "accent")

    def _download_finished(self) -> None:
        self.progress.configure(mode="indeterminate")
        self.progress.start()
        self._set_status("Opening the document...", "accent")

    def _load_failed(self, message: str, detail: str) -> None:
        self._set_busy(False)
        self._error(message, detail)

    def _run_session(self, settings: AppSettings, document: PdfDocument,
                     model: Path | None = None) -> None:
        """Runs the OpenCV loop on the main thread with the launcher hidden."""
        self._set_status("Session running - close the annotation window to come back.", "accent")
        self.progress.stop()
        self.update_idletasks()
        self.withdraw()

        saved: list[Path] = []
        failure: tuple[str, str] | None = None
        try:
            saved = AnnotationSession(settings, document, model_path=model).run()
        except AppError as exc:
            failure = (exc.message, exc.detail)
        except Exception:  # pragma: no cover - unexpected failures
            failure = ("The annotation session stopped unexpectedly.",
                       traceback.format_exc(limit=3))
        finally:
            document.close()
            self.deiconify()
            self.lift()
            self.focus_force()
            self._set_busy(False)

        if failure is not None:
            self._error(*failure)
        elif saved:
            self._set_status(f"Saved {len(saved)} file(s) to {self.output_dir}", "success")
            Dialog(self, "Saved", f"Saved {len(saved)} file(s).", str(self.output_dir),
                   tone="success")
        else:
            self._set_status("Session ended. Nothing was saved.")

    # ----------------------------------------------------------------- close
    def _on_close(self) -> None:
        self.quit()
        self.destroy()


def main() -> int:
    app = LauncherApp()
    try:
        app.mainloop()
    finally:
        try:
            app.destroy()
        except Exception:
            pass
    return 0
