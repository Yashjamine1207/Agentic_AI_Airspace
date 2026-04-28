"""
core/surrogate_loader.py
Loads the TF Keras 8-layer Transformer surrogate fuel model.
Custom layers (PositionalEncoding, TransformerBlock) are re-defined
here so the .keras file can be deserialised without the notebook.
Uses st.cache_resource so the model loads once per session.
"""

import numpy as np
import pickle
import time
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

AIRCRAFT_PROFILES = {
    "B737": {"type_id": 0, "cruise_spd": 450, "fuel_rate_base": 2.5},
    "A320": {"type_id": 1, "cruise_spd": 447, "fuel_rate_base": 2.3},
    "B777": {"type_id": 2, "cruise_spd": 490, "fuel_rate_base": 6.8},
    "A380": {"type_id": 3, "cruise_spd": 488, "fuel_rate_base": 9.2},
    "CRJ9": {"type_id": 4, "cruise_spd": 410, "fuel_rate_base": 1.8},
}


def _build_custom_objects():
    """Build custom Keras layer classes needed to load the surrogate model."""
    import tensorflow as tf
    from tensorflow import keras

    class PositionalEncoding(keras.layers.Layer):
        def __init__(self, seq_len, d_model, **kwargs):
            super().__init__(**kwargs)
            positions = np.arange(seq_len)[:, np.newaxis]
            dims      = np.arange(d_model)[np.newaxis, :]
            angles    = positions / np.power(10000, (2 * (dims // 2)) / d_model)
            angles[:, 0::2] = np.sin(angles[:, 0::2])
            angles[:, 1::2] = np.cos(angles[:, 1::2])
            self.pos_encoding = tf.cast(angles[np.newaxis, :, :], dtype=tf.float32)

        def call(self, x):
            return x + self.pos_encoding[:, :tf.shape(x)[1], :]

        def get_config(self):
            cfg = super().get_config()
            return cfg

    class TransformerBlock(keras.layers.Layer):
        def __init__(self, d_model, num_heads, ff_dim, dropout=0.1, **kwargs):
            super().__init__(**kwargs)
            self.attn  = keras.layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=d_model // num_heads)
            self.ffn   = keras.Sequential([
                keras.layers.Dense(ff_dim, activation="relu"),
                keras.layers.Dense(d_model),
            ])
            self.norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
            self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6)
            self.drop1 = keras.layers.Dropout(dropout)
            self.drop2 = keras.layers.Dropout(dropout)

        def call(self, x, training=False):
            attn_out = self.attn(x, x, training=training)
            x = self.norm1(x + self.drop1(attn_out, training=training))
            return self.norm2(x + self.drop2(self.ffn(x), training=training))

        def get_config(self):
            cfg = super().get_config()
            return cfg

    return {
        "PositionalEncoding": PositionalEncoding,
        "TransformerBlock":   TransformerBlock,
    }


def load_surrogate_model():
    """
    Load transformer surrogate + scalers.
    Returns (model, scaler_X, scaler_y) or (None, None, None) on failure.
    """
    try:
        from tensorflow import keras
        model_path   = os.path.join(BASE_DIR, "transformer_surrogate_best.keras")
        scaler_x_path = os.path.join(BASE_DIR, "scaler_X.pkl")
        scaler_y_path = os.path.join(BASE_DIR, "scaler_y.pkl")

        if not all(os.path.exists(p) for p in [model_path, scaler_x_path, scaler_y_path]):
            return None, None, None

        custom_objects = _build_custom_objects()
        model = keras.models.load_model(model_path, custom_objects=custom_objects)

        with open(scaler_x_path, "rb") as f:
            scaler_X = pickle.load(f)
        with open(scaler_y_path, "rb") as f:
            scaler_y = pickle.load(f)

        return model, scaler_X, scaler_y
    except Exception:
        return None, None, None


def predict_fuel_burn(model, scaler_X, scaler_y,
                      altitude_ft: float, ground_speed_kts: float,
                      vertical_rate_fpm: float, aircraft_type_id: int,
                      track_deg: float = 90.0) -> tuple:
    """
    Run a single surrogate inference for the given flight state.
    Returns (fuel_kg_per_s, latency_ms).
    """
    # Build a 20-step synthetic lookback sequence
    seq = []
    for i in range(20):
        alt_v = altitude_ft + np.random.normal(0, 50)
        spd_v = ground_speed_kts + np.random.normal(0, 5)
        vr_v  = vertical_rate_fpm + np.random.normal(0, 20)
        seq.append([37.5, -122.3, alt_v, spd_v, vr_v, track_deg,
                    float(aircraft_type_id)])

    seq = np.array(seq, dtype=np.float32)
    seq_scaled = scaler_X.transform(seq)
    seq_input  = seq_scaled[np.newaxis, :, :]

    t0 = time.perf_counter()
    pred_scaled = model(seq_input, training=False).numpy()[0, 0]
    latency_ms  = (time.perf_counter() - t0) * 1000

    fuel_kgs = float(scaler_y.inverse_transform([[pred_scaled]])[0, 0])
    return max(fuel_kgs, 0.1), latency_ms


def physics_fuel_burn(altitude_ft: float, ground_speed_kts: float,
                      vertical_rate_fpm: float, aircraft_type_id: int) -> float:
    """
    Fallback first-principles fuel burn (kg/s).
    Used when the surrogate model is unavailable.
    """
    profiles = list(AIRCRAFT_PROFILES.values())
    profile  = profiles[min(aircraft_type_id, len(profiles) - 1)]
    base     = profile["fuel_rate_base"]
    speed_f  = (ground_speed_kts / profile["cruise_spd"]) ** 2.5
    alt_km   = altitude_ft * 0.0003048
    density  = np.exp(-alt_km / 8.5)
    alt_eff  = 1.0 / (density ** 0.5 + 0.01)
    climb_f  = 1.0 + abs(vertical_rate_fpm) / 3000.0
    return float(np.clip(base * speed_f * alt_eff * climb_f, 0.5, 20.0))
