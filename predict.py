"""Evaluate one draft using a portable model exported by TRAINING LAB."""

import argparse
import json
from pathlib import Path

from lab import FORMAT, MAX_FEATURES, MAX_TEXT, MIN_TEXT, normalize, score_rows


def predict(text, model):
    if not isinstance(model, dict) or model.get("format") != FORMAT:
        raise ValueError("対応するTRAINING LABのモデルJSONを指定してください。")
    bias, threshold, entries = (model.get(key) for key in ("bias", "threshold", "features"))
    if (type(bias) not in (int, float) or not -100 <= bias <= 100
            or type(threshold) not in (int, float) or not 0 < threshold < 1
            or not isinstance(entries, dict) or not 1 <= len(entries) <= MAX_FEATURES):
        raise ValueError("モデルのパラメーターが無効です。")
    features, weights = {}, []
    for term, pair in entries.items():
        if (len(term) not in (2, 3, 4) or not isinstance(pair, list) or len(pair) != 2
                or any(type(value) not in (int, float) for value in pair)):
            raise ValueError("モデルの特徴量が無効です。")
        idf, weight = pair
        if not 0 < idf <= 100 or not -100 <= weight <= 100:
            raise ValueError("モデルの特徴量が無効です。")
        features[term] = len(weights), idf
        weights.append(weight)
    normalized = normalize(text)
    if len(normalized) > MAX_TEXT:
        raise ValueError(f"正規化後の本文は最大{MAX_TEXT:,}字です。")
    if len(normalized) < MIN_TEXT:
        return {"state": "short"}
    score, = score_rows([{"normalized": normalized}], (features, weights, bias))
    if score is None:
        return {"state": "unmatched"}
    return {"state": "scored", "version": model.get("version"), "score": score,
            "threshold": threshold, "ai_like": score >= threshold}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("text", type=Path, help="UTF-8のテキストファイル")
    args = parser.parse_args()
    try:
        if args.model.stat().st_size > 3_000_000 or args.text.stat().st_size > MAX_TEXT * 4 + 3:
            raise ValueError("ファイルが大きすぎます。")
        model = json.loads(args.model.read_text(encoding="utf-8-sig"))
        text = args.text.read_text(encoding="utf-8-sig")
        result = predict(text, model)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
