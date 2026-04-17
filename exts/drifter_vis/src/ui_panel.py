"""
ui_panel.py — omni.ui control panel for the River Drifter Visualisation.

Layout
------
  [📁 Load CSV…]  [Build Scene]
  ──────────────────────────────
  ▶ Play  ⏸ Pause  ⏹ Stop
  Speed:  [━━●━━━━━━] 1.0×
  Frame:  [━━━━●━━━] 1847 / 3553
  Camera: [Overview ▾]
  Physics Validation: [☐]
  ────── Live Readouts ──────
  Speed:        0.42 m/s
  Accel:        2.31 m/s²
  GPS:          36.0793°N  79.0082°W
  Altitude:     131.4 m
  Time:         00:08:47.6
  ──────────────────────────
  Status: Ready – 3553 rows, 527.6s

All callbacks are injected at construction so the panel has no direct
dependency on other modules (testable in isolation; easily rewired).
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .utils import format_hms

log = logging.getLogger(__name__)

# Guard import so the module can be imported outside Kit
try:
    import omni.ui as ui
    import omni.timeline
    _UI_AVAILABLE = True
except ImportError:
    _UI_AVAILABLE = False
    log.debug("omni.ui not available — DrifterUIPanel in stub mode")


class DrifterUIPanel:
    """
    omni.ui control panel window.

    Parameters
    ----------
    on_load_csv   : callback(path: str) — called when user selects a CSV
    on_build      : callback() — called when "Build Scene" is pressed
    on_play       : callback() — timeline play
    on_pause      : callback() — timeline pause
    on_stop       : callback() — timeline stop + rewind
    on_speed      : callback(scale: float) — playback speed changed
    on_seek       : callback(frame: int) — scrubber moved
    on_camera     : callback(mode_name: str) — camera dropdown changed
    on_physics_toggled : callback(enabled: bool) — physics checkbox toggled
    on_raycast    : callback() — called when "Raycast Terrain" is pressed
    """

    WINDOW_TITLE = "River Drifter Control"
    WINDOW_WIDTH = 340
    WINDOW_HEIGHT = 480

    CAMERA_OPTIONS = ["Overview", "Third Person"]

    def __init__(
        self,
        on_load_csv:        Optional[Callable[[str], None]]  = None,
        on_build:           Optional[Callable[[], None]]     = None,
        on_play:            Optional[Callable[[], None]]     = None,
        on_pause:           Optional[Callable[[], None]]     = None,
        on_stop:            Optional[Callable[[], None]]     = None,
        on_speed:           Optional[Callable[[float], None]]= None,
        on_seek:            Optional[Callable[[int], None]]  = None,
        on_camera:          Optional[Callable[[str], None]]  = None,
        on_physics_toggled: Optional[Callable[[bool], None]] = None,
        on_raycast:         Optional[Callable[[], None]]     = None,
    ) -> None:
        self._on_load_csv        = on_load_csv        or (lambda p: None)
        self._on_build           = on_build           or (lambda: None)
        self._on_play            = on_play            or (lambda: None)
        self._on_pause           = on_pause           or (lambda: None)
        self._on_stop            = on_stop            or (lambda: None)
        self._on_speed           = on_speed           or (lambda s: None)
        self._on_seek            = on_seek            or (lambda f: None)
        self._on_camera          = on_camera          or (lambda m: None)
        self._on_physics_toggled = on_physics_toggled or (lambda e: None)
        self._on_raycast         = on_raycast         or (lambda: None)

        # Internal state
        self._total_frames: int = 0
        self._df: Optional[pd.DataFrame] = None
        self._timeline_sub = None

        # UI widget references (set during _build_ui)
        self._window = None
        self._status_label = None
        self._speed_label = None
        self._frame_label = None
        self._frame_slider = None
        self._speed_slider = None
        self._readout_speed = None
        self._readout_accel = None
        self._readout_gps   = None
        self._readout_alt   = None
        self._readout_time  = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show(self) -> None:
        """Create and display the panel window."""
        if not _UI_AVAILABLE:
            log.warning("omni.ui not available — panel not shown")
            return
        if self._window is None:
            self._build_ui()
        self._window.visible = True

    def hide(self) -> None:
        if self._window:
            self._window.visible = False

    def destroy(self) -> None:
        self._unsubscribe_timeline()
        if self._window:
            self._window.destroy()
            self._window = None

    def set_status(self, message: str) -> None:
        log.info("Status: %s", message)
        if self._status_label:
            self._status_label.text = message

    def set_total_frames(self, n: int) -> None:
        self._total_frames = n
        if self._frame_slider and _UI_AVAILABLE:
            self._frame_slider.max = int(max(1, n - 1))
        if self._frame_label:
            self._frame_label.text = f"0 / {n}"

    def set_dataframe(self, df: pd.DataFrame) -> None:
        """Supply the loaded DataFrame for live readout updates."""
        self._df = df
        self._subscribe_timeline()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self._window = ui.Window(
            self.WINDOW_TITLE,
            width=self.WINDOW_WIDTH,
            height=self.WINDOW_HEIGHT,
        )
        with self._window.frame:
            with ui.VStack(spacing=4):
                self._build_file_row()
                ui.Separator()
                self._build_playback_row()
                self._build_speed_row()
                self._build_frame_row()
                ui.Separator()
                self._build_camera_row()
                self._build_physics_row()
                self._build_raycast_row()
                ui.Separator()
                self._build_readouts()
                ui.Separator()
                self._status_label = ui.Label(
                    "Ready",
                    word_wrap=True,
                    style={"color": 0xFF_A0_A0_A0},
                )

    def _build_file_row(self) -> None:
        with ui.HStack(height=30):
            ui.Button(
                "Load CSV…",
                width=120,
                clicked_fn=self._on_load_clicked,
                tooltip="Open a drifter CSV file",
            )
            ui.Spacer(width=8)
            ui.Button(
                "Build Scene",
                width=120,
                clicked_fn=lambda: self._on_build(),
                tooltip="(Re)build the USD scene with the loaded data",
            )

    def _build_playback_row(self) -> None:
        with ui.HStack(height=30):
            ui.Button("▶ Play",  width=80, clicked_fn=lambda: self._on_play())
            ui.Spacer(width=4)
            ui.Button("⏸ Pause", width=80, clicked_fn=lambda: self._on_pause())
            ui.Spacer(width=4)
            ui.Button("⏹ Stop",  width=80, clicked_fn=lambda: self._on_stop())

    def _build_speed_row(self) -> None:
        with ui.HStack(height=24):
            ui.Label("Speed:", width=55)
            self._speed_slider = ui.FloatSlider(
                min=0.1, max=10.0, step=0.1, width=200,
                tooltip="Playback speed multiplier",
            )
            self._speed_slider.model.set_value(1.0)
            self._speed_slider.model.add_value_changed_fn(self._on_speed_changed)
            ui.Spacer(width=4)
            self._speed_label = ui.Label("1.0×", width=40)

    def _build_frame_row(self) -> None:
        with ui.HStack(height=24):
            ui.Label("Frame:", width=55)
            self._frame_slider = ui.IntSlider(
                min=0, max=max(1, self._total_frames - 1), width=180,
                tooltip="Timeline scrubber",
            )
            self._frame_slider.model.add_value_changed_fn(self._on_frame_changed)
            ui.Spacer(width=4)
            self._frame_label = ui.Label(f"0 / {self._total_frames}", width=75)

    def _build_camera_row(self) -> None:
        with ui.HStack(height=24):
            ui.Label("Camera:", width=60)
            cam_combo = ui.ComboBox(
                0, *self.CAMERA_OPTIONS,
                width=200,
            )
            cam_combo.model.add_item_changed_fn(self._on_camera_changed)

    def _build_physics_row(self) -> None:
        with ui.HStack(height=24):
            ui.Label("Physics Validation:", width=140)
            cb = ui.CheckBox()
            cb.model.add_value_changed_fn(
                lambda m: self._on_physics_toggled(m.get_value_as_bool())
            )

    def _build_raycast_row(self) -> None:
        with ui.HStack(height=30):
            ui.Button(
                "Raycast Terrain",
                width=150,
                clicked_fn=lambda: self._on_raycast(),
                tooltip="Fire raycasts immediately to drape trajectory onto terrain",
            )

    def _build_readouts(self) -> None:
        with ui.VStack(spacing=2):
            ui.Label("── Live Readouts ──", style={"color": 0xFF_80_C0_FF})

            def _row(label: str):
                with ui.HStack(height=20):
                    ui.Label(label, width=90)
                    lbl = ui.Label("—")
                return lbl

            self._readout_speed = _row("Speed:")
            self._readout_accel = _row("Accel:")
            self._readout_gps   = _row("GPS:")
            self._readout_alt   = _row("Altitude:")
            self._readout_time  = _row("Time:")

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_load_clicked(self) -> None:
        if not _UI_AVAILABLE:
            return
        try:
            from omni.kit.window.filepicker import FilePickerDialog
            dialog = FilePickerDialog(
                "Select Drifter CSV",
                apply_button_label="Load",
                file_extension_options=[("*.csv", "CSV files")],
                click_apply_handler=self._on_file_selected,
            )
            dialog.show()
        except ImportError:
            log.warning("FilePickerDialog not available; using default CSV path")
            self._on_load_csv("")

    def _on_file_selected(self, filename: str, dirname: str) -> None:
        import os
        path = os.path.join(dirname, filename)
        self.set_status(f"Loading: {path}")
        self._on_load_csv(path)

    def _on_speed_changed(self, model) -> None:
        scale = model.get_value_as_float()
        if self._speed_label:
            self._speed_label.text = f"{scale:.1f}×"
        self._on_speed(scale)

    def _on_frame_changed(self, model) -> None:
        frame = model.get_value_as_int()
        if self._frame_label:
            self._frame_label.text = f"{frame} / {self._total_frames}"
        self._on_seek(frame)

    def _on_camera_changed(self, model, item) -> None:
        idx = model.get_item_value_model().get_value_as_int()
        mode_name = self.CAMERA_OPTIONS[idx] if idx < len(self.CAMERA_OPTIONS) else "Overview"
        self._on_camera(mode_name)

    # ------------------------------------------------------------------
    # Timeline subscription for live readouts
    # ------------------------------------------------------------------

    def _subscribe_timeline(self) -> None:
        if not _UI_AVAILABLE or self._df is None:
            return
        try:
            timeline = omni.timeline.get_timeline_interface()
            self._timeline_sub = omni.kit.app.get_app().get_update_event_stream() \
                .create_subscription_to_pop(self._on_update, name="drifter_ui_update")
        except Exception as exc:
            log.debug("Timeline subscription failed: %s", exc)

    def _unsubscribe_timeline(self) -> None:
        if self._timeline_sub:
            self._timeline_sub.unsubscribe()
            self._timeline_sub = None

    def _on_update(self, event) -> None:
        """Update live readout labels each frame."""
        if self._df is None or not _UI_AVAILABLE:
            return
        try:
            import omni.kit.app
            timeline = omni.timeline.get_timeline_interface()
            t = timeline.get_current_time()

            df = self._df
            idx = int(np.searchsorted(df["time_s"].values, t, side="right")) - 1
            idx = max(0, min(idx, len(df) - 1))

            row = df.iloc[idx]

            if self._readout_speed:
                self._readout_speed.text = f"{float(row['speed_ms']):.2f} m/s"
            if self._readout_accel:
                self._readout_accel.text = f"{float(row['accel_mag']):.2f} m/s²"
            if self._readout_gps:
                lat = float(row["Lat"])
                lon = float(row["Lon"])
                ns = "N" if lat >= 0 else "S"
                ew = "E" if lon >= 0 else "W"
                self._readout_gps.text = f"{abs(lat):.4f}°{ns}  {abs(lon):.4f}°{ew}"
            if self._readout_alt:
                self._readout_alt.text = f"{float(row['GPS_Alt']):.1f} m"
            if self._readout_time:
                self._readout_time.text = format_hms(float(row["time_s"]))

            # Update frame scrubber
            if self._frame_slider:
                self._frame_slider.model.set_value(idx)
            if self._frame_label:
                self._frame_label.text = f"{idx} / {self._total_frames}"

        except Exception as exc:
            log.debug("UI update error: %s", exc)
