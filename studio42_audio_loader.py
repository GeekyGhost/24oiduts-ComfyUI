# studio42_audio_loader.py
import os
import json
import logging
from typing import Tuple

import torch
import folder_paths

logger = logging.getLogger(__name__)

# ---- Optional backends ------------------------------------------------------
try:
    import torchaudio
    import torchaudio.functional as TAF
    TORCHAUDIO_AVAILABLE = True
    logger.info("Studio42 Audio Loader: torchaudio available")
except Exception as e:
    TORCHAUDIO_AVAILABLE = False
    logger.warning(f"Studio42 Audio Loader: torchaudio not available: {e}")

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
    logger.info("Studio42 Audio Loader: soundfile available")
except Exception as e:
    SOUNDFILE_AVAILABLE = False
    logger.warning(f"Studio42 Audio Loader: soundfile not available: {e}")

SUPPORTED_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif", ".m4a", ".aac", ".wma")

def _list_audio_inputs():
    """Enumerate audio files in ComfyUI inputs/ directory (flat)."""
    input_dir = folder_paths.get_input_directory()
    try:
        files = [
            f for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(SUPPORTED_EXTS)
        ]
        return sorted(files, key=str.lower)
    except Exception as e:
        logger.error(f"Failed to list inputs/: {e}")
        return []

class Studio42AudioLoader:
    """
    Studio42 Audio Loader — supports:
      • File picker with native 'choose file to upload' button
      • OPTIONAL pass-through AUDIO input (bypasses file loading when connected)

    Outputs ComfyUI 'AUDIO' dict: {'waveform': [B,C,S] float32, 'sample_rate': int}
    """

    # ------------- ComfyUI UI spec -------------
    @classmethod
    def INPUT_TYPES(cls):
        files = _list_audio_inputs()
        if not files:
            files = ["<no audio in inputs/>"]

        return {
            "required": {
                # File selector — this flag renders the upload button
                "audio_file": (files, {
                    "audio_upload": True,
                    "tooltip": "Pick or upload an audio file (WAV, MP3, FLAC, OGG, AIFF, M4A, AAC, WMA). Ignored if 'audio_in' is connected."
                }),
                "output_sample_rate": ("INT", {
                    "default": 44100, "min": 8000, "max": 192000, "step": 1000,
                    "tooltip": "Target sample rate (44100=CD, 48000=pro)."
                }),
                "output_channels": (["keep_original", "mono", "stereo"], {
                    "default": "keep_original",
                    "tooltip": "Channel configuration."
                }),
            },
            "optional": {
                # NEW: pass-through input
                "audio_in": ("AUDIO", {
                    "tooltip": "Optional upstream AUDIO. If provided, file selection is ignored."
                }),
                # Processing options
                "enable_trim": ("BOOLEAN", {"default": False}),
                "trim_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1}),
                "trim_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1}),
                "max_duration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1}),
                "normalize_audio": ("BOOLEAN", {"default": False}),
                "fade_in": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "fade_out": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "volume_adjustment": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
                "generate_waveform_data": ("BOOLEAN", {"default": True}),
                "analyze_loudness": ("BOOLEAN", {"default": True}),
                "detect_silence": ("BOOLEAN", {"default": False}),
                "resampling_method": (["sinc_interp_kaiser", "sinc_interp_hann", "linear"], {"default": "sinc_interp_kaiser"}),
            },
        }

    RETURN_TYPES = ("AUDIO", "FLOAT", "INT", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("audio", "duration", "sample_rate", "audio_info", "waveform_data", "analysis")
    FUNCTION = "load_audio"
    CATEGORY = "Studio42/Audio Processing"

    @classmethod
    def IS_CHANGED(cls, audio_file, **kwargs):
        # When pass-through is connected Comfy handles change propagation.
        try:
            if folder_paths.exists_annotated_filepath(audio_file):
                path = folder_paths.get_annotated_filepath(audio_file)
                if os.path.exists(path):
                    return str(os.path.getmtime(path))
        except Exception:
            pass
        return ""

    @classmethod
    def VALIDATE_INPUTS(cls, audio_file, audio_in=None, **kwargs):
        # If pass-through is present, skip file validation.
        if audio_in is not None:
            return True
        try:
            if not folder_paths.exists_annotated_filepath(audio_file):
                return f"Audio file not found in inputs/: {audio_file}"
            path = folder_paths.get_annotated_filepath(audio_file)
            if not path.lower().endswith(SUPPORTED_EXTS):
                return f"Unsupported audio format: {os.path.splitext(path)[1]}"
            return True
        except Exception as e:
            return f"Could not validate audio file: {e}"

    # ------------------- Main op -------------------
    def load_audio(
        self,
        audio_file,
        output_sample_rate=44100,
        output_channels="keep_original",
        enable_trim=False,
        trim_start=0.0,
        trim_end=0.0,
        max_duration=0.0,
        normalize_audio=False,
        fade_in=0.0,
        fade_out=0.0,
        volume_adjustment=1.0,
        generate_waveform_data=True,
        analyze_loudness=True,
        detect_silence=False,
        resampling_method="sinc_interp_kaiser",
        audio_in=None,   # <-- NEW
    ):
        try:
            # ---------- Source selection ----------
            if audio_in is not None:
                # Expect dict {'waveform': [B,C,S]/[C,S], 'sample_rate': int}
                comfy_audio_in = audio_in
                if not isinstance(comfy_audio_in, dict) or "waveform" not in comfy_audio_in or "sample_rate" not in comfy_audio_in:
                    raise ValueError("audio_in must be an AUDIO dict with 'waveform' and 'sample_rate'.")
                audio_data = comfy_audio_in["waveform"]
                cur_sr = int(comfy_audio_in["sample_rate"])

                # Normalize shape to [C,S]
                if audio_data.ndim == 3:   # [B,C,S]
                    audio_data = audio_data[0]
                elif audio_data.ndim == 1: # [S]
                    audio_data = audio_data.unsqueeze(0)
                # Now audio_data is [C,S]
                source_name = "<pass-through>"
                orig_sr = cur_sr
            else:
                # Load from file
                file_path = folder_paths.get_annotated_filepath(audio_file)
                if TORCHAUDIO_AVAILABLE:
                    audio_data, orig_sr = self._load_with_torchaudio(file_path)
                elif SOUNDFILE_AVAILABLE:
                    audio_data, orig_sr = self._load_with_soundfile(file_path)
                else:
                    raise RuntimeError(
                        "No audio backend available. Install one of:\n"
                        "  pip install torchaudio  (recommended)\n"
                        "  pip install soundfile"
                    )
                source_name = os.path.basename(file_path)
                cur_sr = orig_sr

            orig_dur = audio_data.shape[-1] / cur_sr
            orig_ch = audio_data.shape[0] if audio_data.ndim > 1 else 1

            # ---------- Processing ----------
            if enable_trim:
                audio_data = self._apply_trimming(audio_data, cur_sr, trim_start, trim_end, max_duration)

            if volume_adjustment != 1.0:
                audio_data = audio_data * float(volume_adjustment)

            if (fade_in or fade_out):
                audio_data = self._apply_fades(audio_data, cur_sr, fade_in, fade_out)

            if abs(cur_sr - int(output_sample_rate)) > 100:
                audio_data = self._resample_audio(audio_data, cur_sr, int(output_sample_rate), resampling_method)
                cur_sr = int(output_sample_rate)

            audio_data = self._convert_channels(audio_data, output_channels)
            fin_ch = audio_data.shape[0] if audio_data.ndim > 1 else 1

            if normalize_audio:
                audio_data = self._normalize_audio(audio_data)

            fin_dur = audio_data.shape[-1] / cur_sr

            # Ensure [B,C,S]
            if audio_data.ndim == 1:
                audio_data = audio_data.unsqueeze(0).unsqueeze(0)
            elif audio_data.ndim == 2:
                audio_data = audio_data.unsqueeze(0)

            comfy_audio = {"waveform": audio_data, "sample_rate": cur_sr}

            audio_info = self._generate_audio_info(
                source_name, orig_sr, cur_sr, orig_dur, fin_dur, orig_ch, fin_ch,
                enable_trim, trim_start, trim_end, volume_adjustment
            )

            waveform_data = self._generate_waveform_data(audio_data, cur_sr) if generate_waveform_data else ""
            analysis = self._analyze_audio(audio_data, cur_sr, analyze_loudness, detect_silence) if (analyze_loudness or detect_silence) else ""

            logger.info(f"Studio42AudioLoader: {fin_dur:.2f}s @ {cur_sr}Hz, {fin_ch} ch (source={source_name})")
            return (comfy_audio, fin_dur, cur_sr, audio_info, waveform_data, analysis)

        except Exception as e:
            logger.error(f"Studio42AudioLoader failed: {e}")
            fallback = torch.zeros(1, 2, int(44100 * 1.0), dtype=torch.float32)
            return ({"waveform": fallback, "sample_rate": 44100}, 1.0, 44100, f"Error loading audio: {e}", "{}", "{}")

    # ------------------- Backends -------------------
    def _load_with_torchaudio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        audio, sr = torchaudio.load(audio_path)
        return audio, int(sr)  # [C,S], sr

    def _load_with_soundfile(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        data, sr = sf.read(audio_path, dtype="float32")
        if data.ndim == 1:
            audio = torch.from_numpy(data).unsqueeze(0)
        else:
            audio = torch.from_numpy(data.T)  # [C,S]
        return audio, int(sr)

    # ------------------- DSP helpers -------------------
    def _apply_trimming(self, audio: torch.Tensor, sr: int, start: float, end: float, max_dur: float) -> torch.Tensor:
        total = audio.shape[-1]
        s = max(0, min(int(start * sr), total))
        if end > 0.0 and end > start:
            e = max(s, min(int(end * sr), total))
        else:
            e = total
        if max_dur > 0.0:
            e = min(e, s + int(max_dur * sr))
        return audio[..., s:e]

    def _apply_fades(self, audio: torch.Tensor, sr: int, fade_in: float, fade_out: float) -> torch.Tensor:
        mono_added = False
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
            mono_added = True
        n = audio.shape[1]
        if fade_in > 0:
            fi = min(int(fade_in * sr), n // 2)
            if fi > 0:
                curve = torch.linspace(0, 1, fi, device=audio.device, dtype=audio.dtype)
                audio[:, :fi] *= curve.unsqueeze(0)
        if fade_out > 0:
            fo = min(int(fade_out * sr), n // 2)
            if fo > 0:
                curve = torch.linspace(1, 0, fo, device=audio.device, dtype=audio.dtype)
                audio[:, -fo:] *= curve.unsqueeze(0)
        return audio.squeeze(0) if mono_added else audio

    def _resample_audio(self, audio: torch.Tensor, orig: int, new: int, method: str) -> torch.Tensor:
        if TORCHAUDIO_AVAILABLE:
            try:
                return TAF.resample(audio, orig_freq=orig, new_freq=new, resampling_method=method)
            except Exception as e:
                logger.warning(f"High-quality resample failed ({e}); falling back to linear.")
        ratio = new / float(orig)
        if audio.ndim == 1:
            new_len = int(audio.shape[0] * ratio)
            return torch.nn.functional.interpolate(audio[None, None, :], size=new_len, mode="linear", align_corners=False).squeeze(0).squeeze(0)
        else:
            new_len = int(audio.shape[1] * ratio)
            return torch.nn.functional.interpolate(audio[None, :, :], size=new_len, mode="linear", align_corners=False).squeeze(0)

    def _convert_channels(self, audio: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "keep_original":
            return audio
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        ch = audio.shape[0]
        if mode == "mono":
            if ch > 1:
                return torch.mean(audio, dim=0, keepdim=True)
            return audio
        if mode == "stereo":
            if ch == 1:
                return audio.repeat(2, 1)
            if ch > 2:
                return audio[:2, :]
            return audio
        return audio

    def _normalize_audio(self, audio: torch.Tensor) -> torch.Tensor:
        maxv = torch.max(torch.abs(audio))
        return audio * (0.95 / maxv) if maxv > 0 else audio

    def _generate_audio_info(self, filename: str, orig_sr: int, final_sr: int,
                             orig_dur: float, final_dur: float,
                             orig_ch: int, final_ch: int,
                             trimmed: bool, trim_start: float, trim_end: float,
                             volume: float) -> str:
        info = {
            "filename": filename,
            "source_type": "audio",
            "original": {"sample_rate": orig_sr, "duration": round(orig_dur, 2), "channels": orig_ch},
            "processed": {"sample_rate": final_sr, "duration": round(final_dur, 2),
                          "channels": final_ch, "volume_adjustment": volume},
            "processing": {"trimmed": trimmed, "trim_start": trim_start if trimmed else 0,
                           "trim_end": trim_end if trimmed else 0,
                           "resampled": abs(orig_sr - final_sr) > 100}
        }
        return json.dumps(info, indent=2)

    def _generate_waveform_data(self, audio_bcs: torch.Tensor, sr: int, num_points: int = 1000) -> str:
        try:
            if audio_bcs.ndim == 3:   # [B, C, S]
                wf = audio_bcs[0, 0, :]
            elif audio_bcs.ndim == 2: # [C, S]
                wf = audio_bcs[0, :]
            else:
                wf = audio_bcs
            n = wf.numel()
            if n == 0:
                return json.dumps({"error": "Empty audio"})
            if n > num_points:
                win = n // num_points
                m = num_points * win
                trimmed = wf[:m].view(num_points, win)
                amax = torch.max(trimmed, dim=1)[0]
                amin = torch.min(trimmed, dim=1)[0]
                import numpy as np
                t = np.linspace(0, n / sr, num_points)
            else:
                amax = wf
                amin = wf
                import numpy as np
                t = np.linspace(0, n / sr, n)
            return json.dumps({
                "time": t.tolist(),
                "amplitude_max": amax.detach().cpu().numpy().tolist(),
                "amplitude_min": amin.detach().cpu().numpy().tolist(),
                "sample_rate": sr,
                "duration": n / sr,
                "num_points": len(t),
            })
        except Exception as e:
            logger.error(f"Waveform generation failed: {e}")
            return json.dumps({"error": str(e)})

    def _analyze_audio(self, audio_bcs: torch.Tensor, sr: int, do_loud: bool, do_silence: bool) -> str:
        try:
            analysis = {}
            audio = audio_bcs[0] if audio_bcs.ndim == 3 else audio_bcs
            if do_loud:
                if audio.ndim == 2:
                    l, r = audio[0], audio[1] if audio.shape[0] > 1 else audio[0]
                    rms_l = torch.sqrt(torch.mean(l**2)); rms_r = torch.sqrt(torch.mean(r**2))
                    peak_l = torch.max(torch.abs(l));      peak_r = torch.max(torch.abs(r))
                    analysis["loudness"] = {
                        "rms_left_db": float(20 * torch.log10(rms_l + 1e-10)),
                        "rms_right_db": float(20 * torch.log10(rms_r + 1e-10)),
                        "peak_left_db": float(20 * torch.log10(peak_l + 1e-10)),
                        "peak_right_db": float(20 * torch.log10(peak_r + 1e-10)),
                        "stereo_balance": "balanced" if abs(rms_l - rms_r) < 0.1 else "unbalanced",
                    }
                else:
                    rms = torch.sqrt(torch.mean(audio**2)); peak = torch.max(torch.abs(audio))
                    analysis["loudness"] = {
                        "rms_db": float(20 * torch.log10(rms + 1e-10)),
                        "peak_db": float(20 * torch.log10(peak + 1e-10)),
                        "dynamic_range": float(20 * torch.log10((peak + 1e-10)/(rms + 1e-10))),
                    }
            if do_silence:
                thr = 10 ** (-50 / 20)  # -50 dBFS in linear
                energy = torch.max(torch.abs(audio), dim=0)[0] if audio.ndim == 2 else torch.abs(audio)
                silent = energy < thr
                regions = []
                in_sil = False
                start = 0
                for i, s in enumerate(silent):
                    if bool(s) and not in_sil:
                        start = i; in_sil = True
                    elif not bool(s) and in_sil:
                        dur = (i - start)/sr
                        if dur > 0.1:
                            regions.append({"start": start/sr, "duration": dur})
                        in_sil = False
                analysis["silence"] = {
                    "regions": regions,
                    "total_silence": sum(r["duration"] for r in regions),
                    "silence_percentage": float((silent.float().sum()/silent.numel())*100.0),
                }
            return json.dumps(analysis, indent=2)
        except Exception as e:
            logger.error(f"Audio analysis failed: {e}")
            return json.dumps({"error": str(e)})


# Node mapping for ComfyUI
NODE_CLASS_MAPPINGS = {
    "Studio42AudioLoader": Studio42AudioLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Studio42AudioLoader": "Studio42 Audio Loader"
}
