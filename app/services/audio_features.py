from pathlib import Path

import librosa
import numpy as np


TARGET_SR = 16000
N_MFCC = 40
N_MELS = 128
HOP_LENGTH = 512
FRAME_LENGTH = 2048


def summarize_array(values, prefix):
    values = np.asarray(values).astype(float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_p25": np.nan,
            f"{prefix}_p75": np.nan,
            f"{prefix}_iqr": np.nan,
        }

    p25 = float(np.percentile(values, 25))
    p75 = float(np.percentile(values, 75))

    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p25": p25,
        f"{prefix}_p75": p75,
        f"{prefix}_iqr": float(p75 - p25),
    }


def estimate_pause_features(y, sr, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH, top_db=30):
    duration = librosa.get_duration(y=y, sr=sr)

    if duration <= 0:
        return {
            "non_silent_duration_sec": 0,
            "silent_duration_sec": 0,
            "speech_ratio": 0,
            "pause_ratio": 1,
            "num_non_silent_segments": 0,
            "avg_non_silent_segment_sec": 0,
            "num_pauses": 0,
            "avg_pause_duration_sec": 0,
            "pause_duration_std": 0,
            "max_pause_duration_sec": 0,
            "speaking_rate_segments_per_min": 0,
        }

    intervals = librosa.effects.split(
        y,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    if len(intervals) == 0:
        segment_lengths = np.array([], dtype=float)
        non_silent_duration = 0.0
        num_segments = 0
        avg_segment = 0.0
        pause_durations = []
    else:
        segment_lengths = (intervals[:, 1] - intervals[:, 0]) / sr
        non_silent_duration = float(np.sum(segment_lengths))
        num_segments = int(len(segment_lengths))
        avg_segment = float(np.mean(segment_lengths))

        pause_durations = []
        for i in range(1, len(intervals)):
            gap = (intervals[i, 0] - intervals[i - 1, 1]) / sr
            if gap > 0.05:
                pause_durations.append(float(gap))

    silent_duration = max(0.0, duration - non_silent_duration)
    num_pauses = len(pause_durations)

    return {
        "non_silent_duration_sec": float(non_silent_duration),
        "silent_duration_sec": float(silent_duration),
        "speech_ratio": float(non_silent_duration / duration),
        "pause_ratio": float(silent_duration / duration),
        "num_non_silent_segments": num_segments,
        "avg_non_silent_segment_sec": float(avg_segment),
        "num_pauses": int(num_pauses),
        "avg_pause_duration_sec": float(np.mean(pause_durations)) if pause_durations else 0.0,
        "pause_duration_std": float(np.std(pause_durations)) if pause_durations else 0.0,
        "max_pause_duration_sec": float(np.max(pause_durations)) if pause_durations else 0.0,
        "speaking_rate_segments_per_min": float(len(segment_lengths) / duration * 60) if duration > 0 else 0.0,
    }


def extract_pitch_features(y, sr, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH):
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length,
        )

        voiced_f0 = f0[np.isfinite(f0)]
        pitch_stats = summarize_array(voiced_f0, "pitch_f0")
        pitch_stats["voiced_frame_ratio"] = float(np.mean(voiced_flag)) if voiced_flag is not None else np.nan
        pitch_stats["voiced_probability_mean"] = float(np.nanmean(voiced_probs)) if voiced_probs is not None else np.nan

        if len(voiced_f0) > 1 and np.mean(voiced_f0) > 0:
            pitch_stats["pitch_f0_cv"] = float(np.std(voiced_f0) / np.mean(voiced_f0))
        else:
            pitch_stats["pitch_f0_cv"] = np.nan

        return pitch_stats
    except Exception:
        return {
            "pitch_f0_mean": np.nan,
            "pitch_f0_std": np.nan,
            "pitch_f0_min": np.nan,
            "pitch_f0_max": np.nan,
            "pitch_f0_median": np.nan,
            "pitch_f0_p25": np.nan,
            "pitch_f0_p75": np.nan,
            "pitch_f0_iqr": np.nan,
            "voiced_frame_ratio": np.nan,
            "voiced_probability_mean": np.nan,
            "pitch_f0_cv": np.nan,
        }


def estimate_jitter_shimmer(y, hop_length=HOP_LENGTH):
    try:
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        rms_nz = rms[rms > 1e-6]
        shimmer_approx = float(np.mean(np.abs(np.diff(rms_nz))) / np.mean(rms_nz)) if len(rms_nz) > 1 else np.nan

        zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
        jitter_approx = float(np.std(zcr) / np.mean(zcr)) if np.mean(zcr) > 0 else np.nan

        return {"shimmer_approx": shimmer_approx, "jitter_approx": jitter_approx}
    except Exception:
        return {"shimmer_approx": np.nan, "jitter_approx": np.nan}


def extract_spectral_flux(y, hop_length=HOP_LENGTH):
    try:
        stft = np.abs(librosa.stft(y, hop_length=hop_length))
        flux = np.sqrt(np.sum(np.diff(stft, axis=1) ** 2, axis=0))
        flux = flux[np.isfinite(flux)]

        if len(flux) == 0:
            return {"spectral_flux_min": np.nan}

        return {"spectral_flux_min": float(np.min(flux))}
    except Exception:
        return {"spectral_flux_min": np.nan}


def _fill_feature_defaults(features: dict) -> dict:
    features.setdefault("audio_load_success", False)
    features.setdefault("audio_error", None)
    features.setdefault("sample_rate", TARGET_SR)
    features.setdefault("duration_sec", 0.0)
    features.setdefault("num_samples", 0)
    return features


def extract_audio_features(audio_path: str) -> dict:
    audio_path = Path(audio_path)
    features = _fill_feature_defaults({})

    try:
        y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)

        y_trimmed, _ = librosa.effects.trim(y, top_db=35)
        if len(y_trimmed) > sr * 0.5:
            y = y_trimmed

        duration = librosa.get_duration(y=y, sr=sr)
        features.update(
            {
                "audio_load_success": True,
                "sample_rate": sr,
                "duration_sec": float(duration),
                "num_samples": int(len(y)),
            }
        )

        rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
        features.update(summarize_array(rms, "rms_energy"))
        features["rms_energy_p10"] = float(np.percentile(rms[rms > 1e-8], 10)) if np.any(rms > 1e-8) else 0.0
        features["rms_energy_p90"] = float(np.percentile(rms, 90)) if len(rms) else np.nan
        features["rms_dynamic_range"] = features["rms_energy_p90"] - features["rms_energy_p10"] if np.isfinite(features["rms_energy_p90"]) else np.nan

        zcr = librosa.feature.zero_crossing_rate(y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
        features.update(summarize_array(zcr, "zero_crossing_rate"))

        sc = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
        features.update(summarize_array(sc, "spectral_centroid"))

        sb = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
        features.update(summarize_array(sb, "spectral_bandwidth"))

        sr_ = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
        features.update(summarize_array(sr_, "spectral_rolloff"))

        sf = librosa.feature.spectral_flatness(y=y, hop_length=HOP_LENGTH)[0]
        features.update(summarize_array(sf, "spectral_flatness"))
        features.update(extract_spectral_flux(y, HOP_LENGTH))

        try:
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=HOP_LENGTH)
            for i in range(contrast.shape[0]):
                features[f"spectral_contrast_{i+1}_mean"] = float(np.mean(contrast[i]))
                features[f"spectral_contrast_{i+1}_std"] = float(np.std(contrast[i]))
        except Exception:
            for i in range(7):
                features[f"spectral_contrast_{i+1}_mean"] = np.nan
                features[f"spectral_contrast_{i+1}_std"] = np.nan

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
        for i in range(N_MFCC):
            features[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
            features[f"mfcc_{i+1}_std"] = float(np.std(mfcc[i]))
            features[f"mfcc_{i+1}_median"] = float(np.median(mfcc[i]))

        mfcc_delta = librosa.feature.delta(mfcc)
        for i in range(N_MFCC):
            features[f"mfcc_delta_{i+1}_mean"] = float(np.mean(mfcc_delta[i]))
            features[f"mfcc_delta_{i+1}_std"] = float(np.std(mfcc_delta[i]))

        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        for i in range(N_MFCC):
            features[f"mfcc_delta2_{i+1}_mean"] = float(np.mean(mfcc_delta2[i]))
            features[f"mfcc_delta2_{i+1}_std"] = float(np.std(mfcc_delta2[i]))

        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        group_size = N_MELS // 4
        for g in range(4):
            band = mel_db[g * group_size : (g + 1) * group_size, :].ravel()
            features[f"mel_band_g{g+1}_mean"] = float(np.mean(band)) if len(band) else np.nan
            features[f"mel_band_g{g+1}_std"] = float(np.std(band)) if len(band) else np.nan
        mel_band_means = mel_db.mean(axis=1)
        features["mel_band_profile_min"] = float(np.min(mel_band_means)) if len(mel_band_means) else np.nan

        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP_LENGTH)
        for i in range(chroma.shape[0]):
            features[f"chroma_{i+1}_mean"] = float(np.mean(chroma[i]))
            features[f"chroma_{i+1}_std"] = float(np.std(chroma[i]))

        try:
            y_harmonic = librosa.effects.harmonic(y)
            tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
            for i in range(tonnetz.shape[0]):
                features[f"tonnetz_{i+1}_mean"] = float(np.mean(tonnetz[i]))
                features[f"tonnetz_{i+1}_std"] = float(np.std(tonnetz[i]))
        except Exception:
            for i in range(6):
                features[f"tonnetz_{i+1}_mean"] = np.nan
                features[f"tonnetz_{i+1}_std"] = np.nan

        features.update(estimate_pause_features(y, sr, FRAME_LENGTH, HOP_LENGTH, top_db=30))
        features.update(estimate_jitter_shimmer(y, HOP_LENGTH))
        features.update(extract_pitch_features(y, sr, FRAME_LENGTH, HOP_LENGTH))

    except Exception as exc:
        features["audio_error"] = str(exc)

    return features