"""Neurofeedback Fz: calibración basal y control online; se inicia desde el notebook.

La adquisición, los filtros y el guardado trabajan en un hilo. Tkinter solo dibuja
instantáneas; nunca espera muestras LSL. No se carga ningún clasificador.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk

import numpy as np
from scipy.signal import butter, iirnotch, lfilter, lfilter_zi, sosfilt, sosfilt_zi, welch

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    baseline_seconds: int = 120
    fs: int = 250
    window_seconds: float = 2.0
    update_seconds: float = 2.0
    warmup_seconds: float = 2.0
    notch_hz: float = 60.0
    bandpass_hz: tuple = (4.0, 30.0)
    artifact_uv: float = 150.0
    exploratory_mode: bool = False
    control_peak_factor: float = 1.5
    flat_std_uv: float = 0.1
    min_beta_uv2: float = 1e-6
    min_valid_fraction: float = 0.70
    beta_guard_factor: float = 3.0
    smoothing_tau_seconds: float = 1.0
    timestamp_gap_seconds: float = 0.25
    stream_name: str = ""
    fallback_unit: str = "uV"

    def __post_init__(self):
        if self.baseline_seconds not in (60, 120):
            raise ValueError("Selecciona 60 o 120 segundos de reposo.")
        if self.fs != 250 or not 0 < self.notch_hz < self.fs / 2:
            raise ValueError("Este protocolo requiere 250 Hz y un notch inferior a Nyquist.")
        if self.bandpass_hz != (4.0, 30.0):
            raise ValueError("El protocolo theta/beta utiliza band-pass de 4–30 Hz.")
        if self.window_seconds != 2 or self.update_seconds != 2:
            raise ValueError("Este protocolo utiliza ventanas de 2 s y avance online de 2 s.")
        if self.control_peak_factor < 1:
            raise ValueError("El límite relativo de amplitud debe ser al menos el del baseline.")


def band_power(frequencies, psd, low, high):
    mask = (frequencies >= low) & (frequencies <= high)
    return float(np.trapezoid(psd[mask], frequencies[mask]))


def estimate_tbr(window_uv, config: Config):
    """Potencias integradas en µV²; el modo exploratorio conserva una alerta de calidad."""
    x = np.asarray(window_uv, dtype=float)
    result = dict(valid=False, reason="", quality_warning="", peak_uv=None,
                  theta_uv2=None, beta_uv2=None, tbr=None)
    if x.shape != (round(config.fs * config.window_seconds),):
        return dict(result, reason="ventana_incompleta")
    if not np.isfinite(x).all():
        return dict(result, reason="datos_no_finitos")
    result["peak_uv"] = float(np.max(np.abs(x)))
    if result["peak_uv"] > config.artifact_uv:
        if not config.exploratory_mode:
            return dict(result, reason="amplitud_mayor_150_uV")
        result["quality_warning"] = "amplitud_mayor_150_uV"
    if np.std(x) < config.flat_std_uv:
        return dict(result, reason="senal_plana")
    f, psd = welch(x, fs=config.fs, window="hann", nperseg=config.fs,
                   noverlap=config.fs // 2, detrend="constant", scaling="density")
    theta = band_power(f, psd, 4, 8)
    beta = band_power(f, psd, 13, 30)
    if not np.isfinite([theta, beta]).all() or beta <= config.min_beta_uv2:
        return dict(result, reason="beta_insuficiente", theta_uv2=theta, beta_uv2=beta)
    return dict(result, valid=True, theta_uv2=theta, beta_uv2=beta, tbr=theta / beta)


def sphere_height(tbr, threshold):
    """Umbral -> 0.5; TBR menor -> más arriba. No usa el antiguo 13/8."""
    if not np.isfinite([tbr, threshold]).all() or threshold <= 0 or tbr < 0:
        raise ValueError("El TBR debe ser finito y no negativo; el umbral debe ser positivo.")
    return float(threshold / (threshold + tbr))


class StatefulFilter:
    def __init__(self, config):
        self.config = config
        self.b, self.a = iirnotch(config.notch_hz, 30, fs=config.fs)
        self.sos = butter(4, config.bandpass_hz, btype="bandpass", fs=config.fs, output="sos")
        self.reset()

    def reset(self):
        self.zi_notch = self.zi_band = None
        self.seen = 0

    def push(self, raw_uv):
        if self.zi_notch is None:
            self.zi_notch = lfilter_zi(self.b, self.a) * raw_uv
        notch, self.zi_notch = lfilter(self.b, self.a, [raw_uv], zi=self.zi_notch)
        if self.zi_band is None:
            self.zi_band = sosfilt_zi(self.sos) * notch[0]
        filtered, self.zi_band = sosfilt(self.sos, notch, zi=self.zi_band)
        self.seen += 1
        return float(filtered[0]), self.seen > round(self.config.warmup_seconds * self.config.fs)


class SessionFiles:
    """CSV incremental y resumen con umbral, configuración y causa de finalización."""
    def __init__(self, directory, config, metadata):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.raw_file = (self.directory / "fz_muestras.csv").open("w", newline="", encoding="utf-8")
        self.window_file = (self.directory / "tbr_ventanas.csv").open("w", newline="", encoding="utf-8")
        self.raw_writer = csv.writer(self.raw_file)
        self.window_writer = csv.writer(self.window_file)
        self.raw_writer.writerow(["lsl_timestamp_s", "phase", "Fz_raw_uV", "Fz_processed_uV",
                                  "filter_ready", "within_standard_amplitude_limit", "sample_valid"])
        self.window_writer.writerow(["start_lsl_s", "end_lsl_s", "phase", "valid", "reason", "quality_warning", "peak_uV",
                                     "theta_uV2", "beta_uV2", "tbr", "threshold", "tbr_smoothed", "height", "reward"])
        self.summary = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                            config=asdict(config), stream=metadata, status="recording", events=[])
        self.closed = False
        self.last_flush = time.monotonic()
        self.save_summary()

    def save_summary(self):
        target = self.directory / "resumen.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        temporary.replace(target)

    def sample(self, timestamp, phase, raw, processed, ready, within_standard_limit, valid):
        self.raw_writer.writerow([timestamp, phase, raw, processed, int(ready),
                                  int(within_standard_limit), int(valid)])
        if time.monotonic() - self.last_flush >= 1:
            self.flush()

    def window(self, event):
        self.window_writer.writerow([event.get(k) for k in
            ("start", "end", "phase", "valid", "reason", "quality_warning", "peak_uv", "theta_uv2", "beta_uv2", "tbr",
             "threshold", "smoothed", "height", "reward")])
        self.window_file.flush()

    def flush(self):
        self.raw_file.flush()
        self.window_file.flush()
        self.last_flush = time.monotonic()

    def close(self, status):
        if self.closed:
            return
        self.summary.update(status=status, ended_utc=datetime.now(timezone.utc).isoformat())
        try:
            self.save_summary()
        finally:
            self.raw_file.close()
            self.window_file.close()
            self.closed = True


class FeedbackSession:
    """Máquina de estados comprobable sin casco ni interfaz."""
    def __init__(self, config: Config, files: SessionFiles):
        self.config, self.files = config, files
        self.filter = StatefulFilter(config)
        self.phase = "baseline"
        self.window_n = round(config.fs * config.window_seconds)
        self.step_n = round(config.fs * config.update_seconds)
        self.buffer = deque(maxlen=self.window_n)
        self.times = deque(maxlen=self.window_n)
        self.baseline_samples = 0
        self.baseline_attempts = 0
        self.baseline_rejections = Counter()
        self.baseline_max_peak_uv = 0.0
        self.baseline_ratios = []
        self.baseline_betas = []
        self.baseline_peaks = []
        self.baseline_high_amplitude_windows = 0
        self.control_seen = 0
        self.next_control = self.window_n
        self.threshold = None
        self.beta_limit = None
        self.control_peak_limit_uv = None
        self.channel_diagnostics = None
        self.smoothed = None
        self.latest = None
        self.last_timestamp = None
        self.total_samples = 0
        self.discontinuities = 0
        self.message = "Preparando filtros. Relájate y mira el punto fijo."

    def invalidate(self, reason):
        self.filter.reset()
        self.buffer.clear()
        self.times.clear()
        self.control_seen = 0
        self.next_control = self.window_n
        self.smoothed = self.latest = None
        self.last_timestamp = None
        self.discontinuities += 1
        self.message = f"{reason}. Esperando señal continua y nuevas ventanas."
        self.files.summary["events"].append(dict(event=reason, sample=self.total_samples))
        # El CSV se escribe continuamente. Reescribir el JSON por cada salto puede
        # provocar bloqueos de archivo en Windows y detener la adquisición.

    def start_control(self):
        if self.phase not in ("ready", "paused") or self.threshold is None:
            raise RuntimeError("Primero completa un baseline válido.")
        self.phase = "control"
        self.buffer.clear()
        self.times.clear()
        self.control_seen = 0
        self.next_control = self.window_n
        self.smoothed = self.latest = None
        self.message = "Reuniendo 2 segundos nuevos para el control."
        self.files.summary["events"].append(dict(event="control_start", sample=self.total_samples))
        self.files.save_summary()

    def pause(self):
        if self.phase == "control":
            self.phase = "paused"
            self.smoothed = self.latest = None
            self.message = "Control en pausa. El umbral basal se conserva."
            self.files.summary["events"].append(dict(event="control_pause", sample=self.total_samples))
            self.files.save_summary()

    def finish_baseline(self):
        expected = self.config.baseline_seconds / self.config.window_seconds
        required = math.ceil(expected * self.config.min_valid_fraction)
        ratios = np.asarray(self.baseline_ratios, dtype=float)
        stats = dict(duration_received_s=self.baseline_samples / self.config.fs,
                     expected_windows=int(expected), evaluated_windows=self.baseline_attempts,
                     valid_windows=len(ratios), required_valid_windows=required,
                     rejection_reasons=dict(self.baseline_rejections),
                     max_window_peak_uv=self.baseline_max_peak_uv,
                     high_amplitude_windows=self.baseline_high_amplitude_windows,
                     quality_mode="exploratory" if self.config.exploratory_mode else "strict",
                     tbr_values=self.baseline_ratios,
                     mean_tbr=float(np.mean(ratios)) if len(ratios) else None,
                     median_tbr=float(np.median(ratios)) if len(ratios) else None,
                     percentile_30=float(np.percentile(ratios, 30)) if len(ratios) else None,
                     percentile_40=float(np.percentile(ratios, 40)) if len(ratios) else None,
                     beta_percentile_95_uv2=float(np.percentile(self.baseline_betas, 95)) if len(ratios) else None,
                     peak_percentile_95_uv=(float(np.percentile(self.baseline_peaks, 95))
                                            if self.baseline_peaks else self.config.artifact_uv),
                     threshold_method="median_of_valid_window_ratios")
        median = stats["median_tbr"]
        if len(ratios) < required or median is None or not np.isfinite(median) or median <= 0:
            self.phase = "failed"
            rejected_amplitude = self.baseline_rejections["amplitud_mayor_150_uV"]
            detail = (f" {rejected_amplitude} superaron ±{self.config.artifact_uv:g} µV; "
                      f"pico máximo {self.baseline_max_peak_uv:.0f} µV."
                      if rejected_amplitude else "")
            self.message = (f"Baseline insuficiente: {len(ratios)}/{self.baseline_attempts} ventanas válidas "
                            f"(mínimo {required}).{detail} Revisa el diagnóstico de canales y el modo elegido; "
                            "la causa de la amplitud alta no está confirmada.")
        else:
            self.threshold = median
            # Este límite solo suspende feedback ante beta extraordinariamente
            # alta. No diagnostica movimiento ni altera la mediana del TBR basal.
            self.beta_limit = stats["beta_percentile_95_uv2"] * self.config.beta_guard_factor
            self.control_peak_limit_uv = max(self.config.artifact_uv,
                                             stats["peak_percentile_95_uv"] * self.config.control_peak_factor)
            self.phase = "ready"
            if self.config.exploratory_mode and self.baseline_high_amplitude_windows:
                self.message = (f"Calibración EXPLORATORIA: {self.baseline_high_amplitude_windows} ventanas superaron "
                                f"±{self.config.artifact_uv:g} µV. Umbral TBR = {median:.3f}. "
                                "Pulsa Iniciar control.")
            else:
                self.message = f"Baseline listo. Umbral mediano = {median:.3f}. Pulsa Iniciar control."
        self.files.summary.update(baseline=stats, threshold=self.threshold,
                                  beta_guard_limit_uv2=self.beta_limit,
                                  control_peak_limit_uv=self.control_peak_limit_uv, status=self.phase)
        self.files.save_summary()
        self.files.flush()
        self.buffer.clear()
        self.times.clear()

    def push(self, timestamp, raw_uv):
        if not np.isfinite([timestamp, raw_uv]).all():
            self.files.sample(timestamp, self.phase, raw_uv, "", False, False, False)
            self.invalidate("Datos no finitos")
            return
        if self.last_timestamp is not None:
            dt = timestamp - self.last_timestamp
            # Los timestamps de UnicornLSL llegan con jitter y en ráfagas: un
            # intervalo no debe compararse con los 4 ms ideales de 250 Hz.
            # El filtro trabaja con el orden de las muestras y la tasa nominal.
            if dt <= 0 or dt > self.config.timestamp_gap_seconds:
                self.invalidate("Discontinuidad de timestamps")
        self.last_timestamp = timestamp
        self.total_samples += 1
        filtered, ready = self.filter.push(raw_uv)
        within_standard_limit = abs(filtered) <= self.config.artifact_uv
        self.files.sample(timestamp, self.phase, raw_uv, filtered, ready, within_standard_limit,
                          ready and (within_standard_limit or self.config.exploratory_mode))
        if not ready or self.phase not in ("baseline", "control"):
            return
        self.buffer.append(filtered)
        self.times.append(timestamp)
        if self.phase == "baseline":
            self.baseline_samples += 1
            self.message = "Relájate. Ojos abiertos, mirando el punto fijo."
            due = len(self.buffer) == self.window_n
        else:
            self.control_seen += 1
            due = self.control_seen >= self.next_control and len(self.buffer) == self.window_n
        if due:
            event = estimate_tbr(self.buffer, self.config)
            event.update(start=self.times[0], end=timestamp, phase=self.phase,
                         threshold=self.threshold, smoothed=None, height=None, reward=False)
            if self.phase == "baseline":
                self.baseline_attempts += 1
                if event["peak_uv"] is not None:
                    self.baseline_max_peak_uv = max(self.baseline_max_peak_uv, event["peak_uv"])
                if event["valid"]:
                    self.baseline_ratios.append(event["tbr"])
                    self.baseline_betas.append(event["beta_uv2"])
                    self.baseline_peaks.append(event["peak_uv"])
                    if event["quality_warning"]:
                        self.baseline_high_amplitude_windows += 1
                else:
                    self.baseline_rejections[event["reason"]] += 1
                self.buffer.clear()
                self.times.clear()
            else:
                self.next_control += self.step_n
                if event["valid"] and event["peak_uv"] > self.control_peak_limit_uv:
                    event.update(valid=False, reason="amplitud_atipica_respecto_baseline")
                if event["valid"] and event["beta_uv2"] > self.beta_limit:
                    event.update(valid=False, reason="beta_atipicamente_alta")
                if event["valid"]:
                    alpha = 1 - math.exp(-self.config.update_seconds / self.config.smoothing_tau_seconds)
                    self.smoothed = event["tbr"] if self.smoothed is None else (
                        alpha * event["tbr"] + (1 - alpha) * self.smoothed)
                    event.update(smoothed=self.smoothed,
                                 height=sphere_height(self.smoothed, self.threshold),
                                 reward=self.smoothed < self.threshold)
                    self.message = ("Control EXPLORATORIO: menor TBR eleva la esfera."
                                    if self.config.exploratory_mode
                                    else "Control activo: menor TBR eleva la esfera.")
                else:
                    self.smoothed = None
                    if event["reason"] == "beta_atipicamente_alta":
                        self.message = "Potencia beta atípicamente alta: feedback en pausa para esta ventana."
                    elif event["reason"] == "amplitud_atipica_respecto_baseline":
                        self.message = "Amplitud atípica respecto al reposo: feedback en pausa para esta ventana."
                    else:
                        self.message = f"Ventana rechazada: {event['reason']}. Sin recompensa."
                self.latest = event
            self.files.window(event)
        if self.phase == "baseline" and self.baseline_samples >= self.config.baseline_seconds * self.config.fs:
            self.finish_baseline()

    def snapshot(self):
        return dict(phase=self.phase, message=self.message, threshold=self.threshold,
                    beta_limit=self.beta_limit,
                    control_peak_limit_uv=self.control_peak_limit_uv,
                    exploratory_mode=self.config.exploratory_mode,
                    baseline_high_amplitude_windows=self.baseline_high_amplitude_windows,
                    channel_diagnostics=self.channel_diagnostics,
                    elapsed=self.baseline_samples / self.config.fs,
                    duration=self.config.baseline_seconds, valid_windows=len(self.baseline_ratios),
                    evaluated_windows=self.baseline_attempts,
                    required_valid_windows=math.ceil(self.config.baseline_seconds / self.config.window_seconds
                                                     * self.config.min_valid_fraction),
                    rejection_reasons=dict(self.baseline_rejections),
                    max_window_peak_uv=self.baseline_max_peak_uv, latest=self.latest,
                    samples=self.total_samples, discontinuities=self.discontinuities,
                    path=str(self.files.directory))


def stream_settings(info, config):
    """Usa etiqueta Fz; solo admite orden Unicorn conocido cuando faltan etiquetas."""
    if info.channel_count() != 8 or not np.isclose(info.nominal_srate(), config.fs, atol=0.1):
        raise RuntimeError("Selecciona el stream EEG separado: 8 canales y 250 Hz (modo Each).")
    labels, units = [], []
    node = info.desc().child("channels").child("channel")
    for _ in range(info.channel_count()):
        labels.append(node.child_value("label"))
        units.append(node.child_value("unit"))
        node = node.next_sibling()
    lower = [s.strip().lower() for s in labels]
    if "fz" in lower:
        index, mapping = lower.index("fz"), "metadata_label"
    elif info.name().upper().startswith("UN-") and info.name().upper().endswith("_EEG") and not any(
        label in ("c3", "cz", "c4", "pz", "oz", "po7", "po8") for label in lower
    ):
        index, mapping = 0, "unicorn_order_Fz_C3_Cz_C4_Pz_PO7_Oz_PO8"
    else:
        raise RuntimeError("No se pudo identificar Fz con seguridad en los metadatos del stream.")
    unit = units[index] or config.fallback_unit
    factors = dict(uv=1.0, microvolts=1.0, microvolt=1.0, v=1e6, volts=1e6, volt=1e6)
    channel_factors = []
    for channel, stated_unit in enumerate(units):
        chosen_unit = stated_unit or config.fallback_unit
        normalized = chosen_unit.strip().lower().replace("μ", "u").replace("µ", "u")
        if normalized not in factors:
            raise RuntimeError(f"Unidad desconocida para canal {channel + 1}: {chosen_unit!r}.")
        channel_factors.append(factors[normalized])
    metadata = dict(name=info.name(), source_id=info.source_id(), labels=labels,
                    fz_index=index, mapping=mapping, unit=unit,
                    unit_origin="metadata" if units[index] else "configured_fallback",
                    factor_to_uv=channel_factors[index], factors_to_uv=channel_factors,
                    fs=info.nominal_srate(),
                    timestamp_domain="LSL_local_clock_proc_clocksync")
    return index, channel_factors[index], metadata


def summarize_eeg_channels(samples_uv, fs, labels):
    """Resumen espectral descriptivo de los ocho canales; no certifica calidad EEG."""
    samples = np.asarray(samples_uv, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != 8 or len(samples) < fs * 2:
        raise ValueError("El diagnóstico necesita dos segundos de ocho canales EEG.")
    if not np.isfinite(samples).all():
        raise ValueError("El diagnóstico contiene muestras no finitas.")
    f, psd = welch(samples, fs=fs, axis=0, nperseg=fs, noverlap=fs // 2,
                   detrend="constant", scaling="density")
    band = (f >= 4) & (f <= 30)
    power = np.trapezoid(psd[band], f[band], axis=0)
    peak = f[band][np.argmax(psd[band], axis=0)]
    return [dict(channel=i + 1, label=labels[i], rms_4_30_uv=float(np.sqrt(max(power[i], 0))),
                 peak_hz=float(peak[i])) for i in range(8)]


class AcquisitionWorker(threading.Thread):
    def __init__(self, config, messages, output_root=ROOT / "sessions"):
        super().__init__(daemon=True)
        self.config, self.messages, self.output_root = config, messages, Path(output_root)
        self.stop_event = threading.Event()
        self.commands = queue.Queue()

    def publish(self, value):
        # Solo se descarta una instantánea de pantalla antigua; nunca muestras ni CSV.
        try:
            self.messages.put_nowait(value)
        except queue.Full:
            try:
                self.messages.get_nowait()
            except queue.Empty:
                pass
            self.messages.put_nowait(value)

    def run(self):
        from pylsl import StreamInlet, local_clock, proc_clocksync, resolve_streams
        inlet = files = session = None
        status = "interrupted"
        try:
            self.publish(dict(phase="connecting", message="Buscando el stream EEG del casco…"))
            streams = resolve_streams(wait_time=3)
            candidates = [s for s in streams if s.channel_count() == 8
                          and s.type().upper() in ("EEG", "DATA")
                          and (s.name() == self.config.stream_name if self.config.stream_name
                               else s.name().upper().startswith("UN-") and s.name().upper().endswith("_EEG"))]
            if not candidates:
                raise RuntimeError("No se detectó el stream EEG Unicorn por LSL. Inicia UnicornLSL "
                                   "en modo Each y vuelve a pulsar Iniciar reposo.")
            if len(candidates) > 1:
                names = ", ".join(s.name() for s in candidates)
                raise RuntimeError(f"Hay varios streams EEG Unicorn ({names}). Indica NOMBRE_STREAM en el notebook.")
            if self.stop_event.is_set():
                return
            inlet = StreamInlet(candidates[0], max_buflen=5, max_chunklen=32,
                                processing_flags=proc_clocksync, recover=True)
            inlet.open_stream(timeout=5)
            info = inlet.info(timeout=5)
            index, factor, metadata = stream_settings(info, self.config)
            inlet.time_correction(timeout=5)
            inlet.flush()
            directory = self.output_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            files = SessionFiles(directory, self.config, metadata)
            (directory / "stream.xml").write_text(info.as_xml(), encoding="utf-8")
            session = FeedbackSession(self.config, files)
            last_data = last_display = time.monotonic()
            last_channel_diagnostic = time.monotonic()
            channel_buffer = deque(maxlen=2 * self.config.fs)
            channel_labels = [label or fallback for label, fallback in zip(
                metadata["labels"], ("Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"))]
            stalled = False
            while not self.stop_event.is_set():
                while not self.commands.empty():
                    command = self.commands.get_nowait()
                    if command == "control":
                        session.start_control()
                        # Excluir paquetes pendientes de la etapa anterior al botón.
                        inlet.flush()
                    elif command == "pause":
                        session.pause()
                chunk, timestamps = inlet.pull_chunk(timeout=0.1, max_samples=32)
                now = time.monotonic()
                if len(timestamps):
                    if local_clock() - timestamps[-1] > 1.0:
                        if not stalled:
                            session.invalidate("Datos atrasados más de 1 segundo")
                        stalled = True
                        inlet.flush()
                    else:
                        last_data, stalled = now, False
                        for sample, stamp in zip(chunk, timestamps):
                            discontinuities_before = session.discontinuities
                            session.push(stamp, float(sample[index]) * factor)
                            if session.discontinuities != discontinuities_before:
                                channel_buffer.clear()
                                session.channel_diagnostics = None
                            channel_buffer.append(np.asarray(sample[:8], dtype=float)
                                                  * metadata["factors_to_uv"])
                            if session.phase == "failed":
                                break
                        if (len(channel_buffer) == channel_buffer.maxlen
                                and now - last_channel_diagnostic >= 1):
                            try:
                                session.channel_diagnostics = summarize_eeg_channels(
                                    channel_buffer, self.config.fs, channel_labels)
                            except ValueError:
                                session.channel_diagnostics = None
                            last_channel_diagnostic = now
                elif now - last_data > 1.0 and not stalled:
                    session.invalidate("Sin señal LSL")
                    stalled = True
                if now - last_data > 15:
                    raise RuntimeError("No llegan datos desde hace 15 segundos. Se guardó lo recibido; revisa LSL.")
                if now - last_display >= 0.1 or session.phase == "failed":
                    self.publish(session.snapshot())
                    last_display = now
                if session.phase == "failed":
                    status = "baseline_failed"
                    return
            status = "finished" if session.threshold is not None else "baseline_interrupted"
        except Exception as exc:
            status = "error"
            self.publish(dict(phase="error", message=str(exc), path=str(files.directory) if files else ""))
            if files:
                files.summary["error"] = str(exc)
        finally:
            if files:
                if session:
                    files.summary.update(samples=session.total_samples,
                                         discontinuities=session.discontinuities,
                                         baseline_received_s=session.baseline_samples / self.config.fs,
                                         last_channel_diagnostics=session.channel_diagnostics)
                files.close(status)
            if inlet:
                inlet.close_stream()


class NeurofeedbackApp:
    BG = "#0b1425"
    PANEL = "#152238"
    TEXT = "#ecf2ff"
    MUTED = "#a4b5d0"

    def __init__(self, root, config, output_root=ROOT / "sessions"):
        self.root, self.config = root, config
        self.output_root = Path(output_root)
        self.worker = None
        self.messages = queue.Queue(maxsize=1)
        self.state = dict(phase="idle", message="Conecta el casco a UnicornLSL y comienza el registro basal.")
        self.closing = False
        root.title("BCI Neurofeedback · Fz · Baseline y esfera")
        root.geometry("1120x760")
        root.minsize(950, 760)
        root.configure(bg=self.BG)
        root.protocol("WM_DELETE_WINDOW", self.close)
        header = tk.Frame(root, bg=self.BG)
        header.pack(fill="x", padx=28, pady=(22, 12))
        tk.Label(header, text="Neurofeedback · Fz", font=("Segoe UI", 25, "bold"),
                 bg=self.BG, fg=self.TEXT).pack(anchor="w")
        tk.Label(header, text="1. Reposo y calibración     →     2. Control con tu umbral basal",
                 font=("Segoe UI", 12), bg=self.BG, fg=self.MUTED).pack(anchor="w", pady=5)
        body = tk.Frame(root, bg=self.BG)
        body.pack(fill="both", expand=True, padx=28)
        side = tk.Frame(body, bg=self.PANEL, width=305)
        side.pack(side="right", fill="y", padx=(18, 0))
        side.pack_propagate(False)
        self.canvas = tk.Canvas(body, bg="#0e1a2f", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _: self.draw())
        duration_row = tk.Frame(side, bg=self.PANEL)
        duration_row.pack(fill="x", padx=20, pady=(18, 4))
        tk.Label(duration_row, text="Reposo (minutos)", bg=self.PANEL, fg=self.TEXT,
                 font=("Segoe UI", 11)).pack(side="left")
        self.minutes = tk.StringVar(value=str(config.baseline_seconds // 60))
        self.duration_box = ttk.Combobox(duration_row, textvariable=self.minutes, values=("1", "2"),
                                         state="readonly", width=4)
        self.duration_box.pack(side="right")
        self.exploratory = tk.BooleanVar(value=True)
        self.exploratory_box = tk.Checkbutton(
            side, text="Modo exploratorio (amplitud alta)", variable=self.exploratory,
            bg=self.PANEL, fg="#ffca76", activebackground=self.PANEL,
            activeforeground="#ffca76", selectcolor=self.PANEL,
            font=("Segoe UI", 10), anchor="w")
        self.exploratory_box.pack(fill="x", padx=20, pady=(4, 0))
        self.baseline_button = self.button(side, "1   Iniciar reposo", self.start_baseline, "#315fda")
        self.control_button = self.button(side, "2   Iniciar control", self.start_control, "#18866c")
        secondary = tk.Frame(side, bg=self.PANEL)
        secondary.pack(fill="x", padx=20, pady=(6, 0))
        self.pause_button = tk.Button(secondary, text="Pausar control", command=self.pause_control,
                                      bg="#3b4d68", fg="white", relief="flat", pady=8,
                                      disabledforeground="#9cabc2", font=("Segoe UI", 10))
        self.pause_button.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.stop_button = tk.Button(secondary, text="Finalizar sesión", command=self.stop_session,
                                     bg="#7f3948", fg="white", relief="flat", pady=8,
                                     disabledforeground="#9cabc2", font=("Segoe UI", 10))
        self.stop_button.pack(side="right", fill="x", expand=True, padx=(4, 0))
        self.progress = ttk.Progressbar(side, maximum=config.baseline_seconds)
        self.progress.pack(fill="x", padx=20, pady=(10, 6))
        self.timer = tk.Label(side, text="Sin registro basal", bg=self.PANEL, fg=self.MUTED,
                              font=("Segoe UI", 11))
        self.timer.pack(anchor="w", padx=20)
        self.metrics = tk.Label(side, text="Umbral pendiente", bg=self.PANEL, fg=self.TEXT,
                                justify="left", anchor="nw", font=("Segoe UI", 11), wraplength=265)
        self.metrics.pack(fill="both", expand=True, padx=20, pady=6)
        self.channels_label = tk.Label(side, text="Ocho canales: esperando datos LSL", bg=self.PANEL,
                                       fg=self.MUTED, justify="left", anchor="nw",
                                       font=("Consolas", 9), wraplength=265)
        self.channels_label.pack(fill="x", padx=20, pady=(0, 6))
        self.status = tk.Label(root, text="", bg=self.BG, fg=self.MUTED,
                               font=("Segoe UI", 11), wraplength=1040, anchor="w", justify="left")
        self.status.pack(fill="x", padx=28, pady=(12, 6))
        self.path_label = tk.Label(root, text="", bg=self.BG, fg=self.MUTED,
                                  font=("Segoe UI", 9), anchor="w", wraplength=1040, justify="left")
        self.path_label.pack(fill="x", padx=28, pady=(0, 15))
        self.tick()

    def button(self, frame, text, command, color):
        b = tk.Button(frame, text=text, command=command, bg=color, fg="white", relief="flat",
                      activebackground=color, activeforeground="white", cursor="hand2",
                      disabledforeground="#9cabc2", font=("Segoe UI", 12, "bold"), pady=10)
        b.pack(fill="x", padx=20, pady=(6, 0))
        return b

    def start_baseline(self):
        if self.worker and self.worker.is_alive():
            return
        while not self.messages.empty():
            self.messages.get_nowait()
        config = Config(**dict(asdict(self.config), baseline_seconds=int(self.minutes.get()) * 60,
                               exploratory_mode=self.exploratory.get()))
        self.worker = AcquisitionWorker(config, self.messages, self.output_root)
        self.state = dict(phase="connecting", message="Buscando el stream EEG…")
        self.worker.start()
        self.draw()

    def start_control(self):
        if self.worker and self.worker.is_alive():
            self.worker.commands.put("control")
            self.state["phase"] = "starting_control"

    def pause_control(self):
        if self.worker:
            self.worker.commands.put("pause")
            self.state["phase"] = "pausing"

    def stop_session(self):
        if self.worker:
            self.worker.stop_event.set()
            self.state = dict(self.state, phase="stopping", latest=None, message="Guardando y cerrando la sesión…")

    def close(self):
        self.closing = True
        self.stop_session()

    def tick(self):
        while not self.messages.empty():
            self.state = self.messages.get_nowait()
        alive = self.worker is not None and self.worker.is_alive()
        if self.closing and not alive:
            self.root.destroy()
            return
        if not alive and self.state["phase"] not in ("idle", "error", "failed", "finished"):
            self.state.update(phase="finished", latest=None, message="Sesión finalizada. Datos guardados.")
        phase = self.state["phase"]
        self.baseline_button.config(state="disabled" if alive or self.closing else "normal")
        self.duration_box.config(state="disabled" if alive else "readonly")
        self.exploratory_box.config(state="disabled" if alive else "normal")
        self.control_button.config(state="normal" if alive and phase in ("ready", "paused") else "disabled")
        self.pause_button.config(state="normal" if alive and phase == "control" else "disabled")
        self.stop_button.config(state="normal" if alive and not self.closing else "disabled")
        self.status.config(text=self.state.get("message", ""))
        self.path_label.config(text="Guardado: " + self.state["path"] if self.state.get("path") else "")
        elapsed, duration = self.state.get("elapsed", 0), self.state.get("duration", self.config.baseline_seconds)
        self.progress.config(maximum=duration, value=min(elapsed, duration))
        self.timer.config(text=f"Reposo registrado: {elapsed:.0f} / {duration:.0f} s")
        threshold = self.state.get("threshold")
        latest = self.state.get("latest")
        if phase in ("baseline", "connecting"):
            self.metrics.config(text="Relájate.\n\nOjos abiertos.\nMira el punto fijo.\nEvita movimientos.\nParpadea naturalmente cuando lo necesites.")
        elif phase == "failed":
            reasons = self.state.get("rejection_reasons", {})
            rejected_amplitude = reasons.get("amplitud_mayor_150_uV", 0)
            peak = self.state.get("max_window_peak_uv", 0)
            lines = [f"Válidas: {self.state.get('valid_windows', 0)}/"
                     f"{self.state.get('evaluated_windows', 0)} · mínimo "
                     f"{self.state.get('required_valid_windows', 0)}"]
            if rejected_amplitude:
                lines.extend([f"{rejected_amplitude} ventanas superaron ±{self.config.artifact_uv:g} µV.",
                              f"Pico observado: {peak:.0f} µV."])
            for reason, count in reasons.items():
                if reason != "amplitud_mayor_150_uV":
                    lines.append(f"{reason.replace('_', ' ')}: {count}")
            lines.append("La amplitud no indica concentración.")
            self.metrics.config(text="\n".join(lines))
        elif threshold is not None:
            lines = [f"Umbral {threshold:.3f} · {self.state.get('valid_windows', 0)} válidas"]
            if self.state.get("exploratory_mode"):
                lines.append(f"EXPLORATORIO · {self.state.get('baseline_high_amplitude_windows', 0)} alertas")
            if latest and latest["valid"]:
                lines += [f"θ {latest['theta_uv2']:.3f} · β {latest['beta_uv2']:.3f} µV²",
                          f"TBR {latest['tbr']:.3f} → {latest['smoothed']:.3f}",
                          "Debajo del umbral" if latest["reward"] else "Encima del umbral"]
                if self.state.get("exploratory_mode") and self.state.get("control_peak_limit_uv"):
                    lines.append(f"Pico {latest['peak_uv']:.0f}/{self.state['control_peak_limit_uv']:.0f} µV")
            self.metrics.config(text="\n".join(lines))
        else:
            self.metrics.config(text="La mediana de los TBR válidos de reposo será tu umbral de control.")
        channels = self.state.get("channel_diagnostics")
        if channels:
            rows = ["EEG RMS 4–30 Hz · Ch1=Fz"]
            for left, right in zip(channels[::2], channels[1::2]):
                rows.append(f"{left['label']:3} {left['rms_4_30_uv']:6.1f}   "
                            f"{right['label']:3} {right['rms_4_30_uv']:6.1f}")
            self.channels_label.config(text="\n".join(rows))
        else:
            self.channels_label.config(text="Ocho canales: esperando 2 s de EEG LSL")
        self.draw()
        self.root.after(100, self.tick)

    def draw(self):
        c = self.canvas
        c.delete("all")
        w, h = max(c.winfo_width(), 400), max(c.winfo_height(), 450)
        phase = self.state["phase"]
        if phase == "control":
            top, bottom, mid = 90, h - 80, (90 + h - 80) / 2
            c.create_line(55, mid, w - 55, mid, fill="#7c8da9", dash=(7, 6), width=2)
            c.create_text(60, mid - 17, anchor="w", text="Tu umbral basal", fill=self.MUTED, font=("Segoe UI", 11))
            title = ("TBR relativo · MODO EXPLORATORIO" if self.state.get("exploratory_mode")
                     else "Menor TBR → la esfera sube")
            c.create_text(w / 2, 35, text=title, fill=self.TEXT, font=("Segoe UI", 18, "bold"))
            latest = self.state.get("latest")
            valid = latest is not None and latest["valid"]
            if valid:
                y = bottom - latest["height"] * (bottom - top)
                color = "#51d7ae" if latest["reward"] else "#6693ff"
                c.create_oval(w / 2 - 38, y - 38, w / 2 + 38, y + 38, fill=color, outline="")
                c.create_oval(w / 2 - 20, y - 25, w / 2 - 2, y - 7, fill="#bbdeff", outline="")
            else:
                c.create_text(w / 2, mid + 58, text="Sin control válido\nEsperando una ventana limpia…",
                              fill=self.MUTED, font=("Segoe UI", 15), justify="center")
            c.create_text(w / 2, h - 30, text="Fz · ventana 2 s · actualización 2 s", fill=self.MUTED, font=("Segoe UI", 11))
        else:
            title = {"idle": "Prepara el registro basal", "connecting": "Conectando con LSL…",
                     "baseline": "Relájate y mira el punto fijo", "ready": "Calibración completada",
                     "paused": "Control en pausa", "error": "Revisa la conexión", "failed": "Repite el registro basal",
                     "finished": "Sesión guardada", "stopping": "Guardando…"}.get(phase, "Preparando control…")
            c.create_text(w / 2, 60, text=title, fill=self.TEXT, font=("Segoe UI", 18, "bold"), width=w - 50)
            c.create_text(w / 2, h / 2, text="+", fill=self.TEXT, font=("Segoe UI", 70))
            footer = "Ojos abiertos · postura cómoda · minimiza movimientos" if phase == "baseline" else "El control se habilita después de un baseline válido."
            c.create_text(w / 2, h - 50, text=footer, fill=self.MUTED, font=("Segoe UI", 11), width=w - 50)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-seconds", type=int, choices=(60, 120), default=120)
    parser.add_argument("--stream-name", default="")
    args = parser.parse_args()
    root = tk.Tk()
    NeurofeedbackApp(root, Config(baseline_seconds=args.baseline_seconds, stream_name=args.stream_name))
    root.mainloop()


if __name__ == "__main__":
    main()
