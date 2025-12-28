from features import build_features
from patterns import detect_patterns
from trend import trend_signal, market_regime
from confidence import confidence_from_probs
from model_registry import get_model
from binance_data import get_candles
import numpy as np

def analyze(image_bytes=None, tf=None, symbol=None):
    candles = get_candles(symbol, interval=f"{tf}m")
    features = build_features(candles)
    X = np.array([features])

    model = get_model()
    ml_prob = model.predict_proba(X)[0][1]

    pattern_prob, patterns = detect_patterns(candles)
    trend_prob = trend_signal(candles)

    regime = market_regime(candles)

    # 🔥 regime-aware weighting
    if regime == "trend":
        weights = [0.5, 0.2, 0.3]
    elif regime == "flat":
        weights = [0.2, 0.5, 0.3]
    else:
        weights = [0.3, 0.3, 0.4]

    final_prob = (
        weights[0] * ml_prob +
        weights[1] * pattern_prob +
        weights[2] * trend_prob
    )

    conf_label, conf_score = confidence_from_probs(
        [ml_prob, pattern_prob, trend_prob]
    )

    return {
        "prob": round(final_prob, 3),
        "confidence": conf_label,
        "confidence_score": conf_score,
        "regime": regime,
        "patterns": patterns,
        "tf": tf,
        "symbol": symbol
    }, None
