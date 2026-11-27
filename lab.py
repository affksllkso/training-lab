"""Dataset storage, group-aware evaluation, and portable Japanese text models."""

import hashlib
import json
import math
import random
import re
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

MIN_TEXT = 100
MAX_TEXT = 20_000
MAX_FEATURES = 12_000
FORMAT = "script-lab-char-tfidf-v1"
PARTITIONS = ("train", "validation", "test")
SPACE = re.compile("[\x09-\x0d\x20\x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff]+")


@contextmanager
def database(path):
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS documents(
                id TEXT PRIMARY KEY, digest TEXT NOT NULL UNIQUE, normalized TEXT NOT NULL,
                label INTEGER NOT NULL CHECK(label IN (0, 1)), record TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS experiments(
                version TEXT PRIMARY KEY, report TEXT NOT NULL, model TEXT NOT NULL
            );
        """)
        with connection:
            yield connection
    finally:
        connection.close()


def normalize(text):
    return SPACE.sub(" ", unicodedata.normalize("NFKC", text)).strip(" ").lower()


def fingerprint(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def get_field(record, name):
    value = record.get(name, "")
    if not isinstance(value, str) or len(value) > 500:
        raise ValueError(f"{name} は500字以内の文字列で指定してください。")
    return value.strip()


def parse_record(raw):
    if not isinstance(raw, dict):
        raise ValueError("各行をJSONオブジェクトにしてください。")
    record = dict(raw)
    for name in ("id", "source", "genre", "category", "topic", "author_id", "work_id",
                 "group_id", "prompt_id", "model", "license", "split", "style",
                 "difficulty", "generated_at"):
        record[name] = get_field(record, name)
    if not record["genre"]:
        record["genre"] = record["category"]
    if any(not record[key] for key in ("id", "source", "genre")):
        raise ValueError("id・source・genre（または category）が必要です。")
    label = record.get("label")
    if type(label) is not int or label not in (0, 1):
        raise ValueError("label は人間執筆の0、またはAI生成の1を指定してください。")
    if label == 0 and (not record["license"] or not record["author_id"]):
        raise ValueError("人間執筆の文章には利用許諾 license と著者ID author_id が必要です。")
    if label == 1:
        record["model"] = record["model"] or "unknown"
    if not any(record[key] for key in ("topic", "work_id", "group_id", "prompt_id", "author_id")):
        raise ValueError("分割用に topic、work_id、group_id、prompt_id、author_id のいずれかが必要です。")
    if record["split"] and record["split"] not in PARTITIONS:
        raise ValueError("split は train・validation・test のいずれかです。")
    text = record.get("text")
    if not isinstance(text, str) or len(text) > MAX_TEXT:
        raise ValueError(f"text は最大{MAX_TEXT:,}字の文字列にしてください。")
    normalized = normalize(text)
    if not MIN_TEXT <= len(normalized) <= MAX_TEXT:
        raise ValueError(f"正規化後の本文は{MIN_TEXT}〜{MAX_TEXT:,}字が必要です。")
    record["character_count"] = len(text)
    prompt = record.get("prompt", "")
    if not isinstance(prompt, str) or len(prompt) > MAX_TEXT:
        raise ValueError("prompt は20,000字以内の文字列で指定してください。")
    return record, normalized


def import_jsonl(path, raw):
    counts = Counter()
    errors = []
    with database(path) as db:
        db.execute("BEGIN IMMEDIATE")
        for number, line in enumerate(raw.decode("utf-8-sig").split("\n"), 1):
            if not line.strip():
                continue
            try:
                record, normalized = parse_record(json.loads(line))
                digest = fingerprint(normalized)
                matches = db.execute(
                    "SELECT id, digest, label FROM documents WHERE id = ? OR digest = ?",
                    (record["id"], digest),
                ).fetchall()
                if matches:
                    if any(row["label"] != record["label"] or
                           (row["id"] == record["id"] and row["digest"] != digest) for row in matches):
                        raise ValueError("既存IDの本文、または同じ本文のラベルが一致しません。")
                    counts["duplicate"] += 1
                    continue
                db.execute("INSERT INTO documents VALUES (?, ?, ?, ?, ?)",
                           (record["id"], digest, normalized, record["label"],
                            json.dumps(record, ensure_ascii=False, allow_nan=False)))
                counts["added"] += 1
            except (ValueError, RecursionError) as error:
                counts["invalid"] += 1
                if len(errors) < 30:
                    errors.append(f"{number}行目: {error}")
    return counts, errors


def dataset(path):
    with database(path) as db:
        return [
            {**json.loads(row["record"]), "digest": row["digest"], "normalized": row["normalized"]}
            for row in db.execute("SELECT record, digest, normalized FROM documents ORDER BY id")
        ]


def summarize(path, baseline_available=False):
    rows = dataset(path)
    labels = Counter(row["label"] for row in rows)
    return {"human": labels[0], "ai": labels[1],
            "sources": Counter(row["source"] for row in rows).most_common(),
            "genres": Counter(row["genre"] for row in rows).most_common(),
            "models": Counter(row["model"] for row in rows if row["label"] == 1).most_common(),
            "versions": versions(path), "baseline_available": baseline_available}


def group_documents(rows):
    parents = list(range(len(rows)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def merge(first, second):
        a, b = sorted((root(first), root(second)))
        parents[b] = a

    seen = {}
    for index, row in enumerate(rows):
        for field in ("author_id", "work_id", "group_id", "prompt_id", "topic"):
            key = (field, normalize(row[field]))
            if key[1]:
                merge(index, seen.setdefault(key, index))
        if row["label"] == 0:
            key = ("human_source", normalize(row["source"]))
            merge(index, seen.setdefault(key, index))
        if row.get("prompt"):
            key = ("prompt", normalize(row["prompt"]))
            merge(index, seen.setdefault(key, index))

    # A prefix index finds every pair at or above 85% Jaccard similarity.
    shingles = [{text[i:i + 5] for i in range(len(text) - 4)}
                for text in (row["normalized"] for row in rows)]
    frequency = Counter(term for terms in shingles for term in terms)
    index_by_term = defaultdict(list)
    for index, terms in enumerate(shingles):
        size = len(terms)
        prefix = sorted(terms, key=lambda term: (frequency[term], term))[:size - math.ceil(.85 * size) + 1]
        candidates = {other for term in prefix for other in index_by_term[term]}
        for other in candidates:
            other_size = len(shingles[other])
            if root(index) == root(other) or min(size, other_size) * 100 < 85 * max(size, other_size):
                continue
            shared = len(terms & shingles[other])
            if shared * 100 >= 85 * (size + other_size - shared):
                merge(index, other)
        for term in prefix:
            index_by_term[term].append(index)

    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[root(index)].append(row)
    return list(groups.values())


def split_documents(groups, hold_model="", hold_genre=""):
    locked = {part: [] for part in (*PARTITIONS, "audit")}
    free = []
    for group in groups:
        tags = {row["split"] for row in group if row["split"]}
        if len(tags) > 1:
            raise ValueError("著者・資料・テーマ・近似本文が共通する文章に、異なる split が指定されています。")
        hold = any((hold_model and row["label"] == 1 and normalize(row["model"]) == normalize(hold_model))
                   or (hold_genre and normalize(row["genre"]) == normalize(hold_genre)) for row in group)
        if hold and tags:
            raise ValueError("独立評価に移すグループに固定 split があります。固定テストと評価対象を分けてください。")
        if hold:
            locked["audit"].extend(group)
        elif tags:
            locked[tags.pop()].extend(group)
        else:
            free.append(group)

    if (hold_model or hold_genre) and {row["label"] for row in locked["audit"]} != {0, 1}:
        raise ValueError("独立評価には指定条件に結び付いた人間執筆文とAI生成文の両方が必要です。")
    totals = Counter(row["label"] for group in groups for row in group)
    totals.subtract(row["label"] for row in locked["audit"])
    ratios = dict(zip(PARTITIONS, (.7, .15, .15)))
    available = tuple(part for part in PARTITIONS if part != "test" or not locked["test"])
    weights = tuple(ratios[part] for part in available)
    best = None
    for seed in range(96):
        rng = random.Random(seed)
        assignment = {part: list(locked[part]) for part in PARTITIONS}
        for group in free:
            choice = rng.choices(available, weights)[0]
            assignment[choice].extend(group)
        counts = {part: Counter(row["label"] for row in assignment[part]) for part in PARTITIONS}
        if any(counts[part][label] < (2 if part == "train" else 1)
               for part in PARTITIONS for label in (0, 1)):
            continue
        error = sum(abs(counts[part][label] / totals[label] - ratios[part])
                    for part in PARTITIONS for label in (0, 1))
        if best is None or error < best[0]:
            best = error, assignment
    if best is None:
        raise ValueError("著者・資料・テーマが分離された両ラベルの文書を追加してください。学習に各2件以上、検証・テストに各1件以上必要です。")
    return {**best[1], "audit": locked["audit"]}


def grams(text):
    return Counter(text[i:i + size] for size in (2, 3, 4) for i in range(len(text) - size + 1))


def vector(counts, features):
    values = [(index, (1 + math.log(count)) * idf)
              for term, count in counts.items()
              if (term_info := features.get(term)) is not None
              for index, idf in (term_info,)]
    norm = math.sqrt(sum(value * value for _, value in values))
    return [(index, value / norm) for index, value in values] if norm else []


def sigmoid(value):
    return 1 / (1 + math.exp(-max(-60, min(60, value))))


def fit(rows):
    frequencies = Counter()
    counts = [grams(row["normalized"]) for row in rows]
    for terms in counts:
        frequencies.update(terms.keys())
    ordered = sorted(frequencies, key=lambda term: (-frequencies[term], term))
    chosen = [term for term in ordered if frequencies[term] >= 2][:MAX_FEATURES]
    if not chosen:
        raise ValueError("学習文書で共通する文字列が足りません。データを追加してください。")
    features = {term: (index, 1 + math.log((1 + len(rows)) / (1 + frequencies[term])))
                for index, term in enumerate(chosen)}
    vectors = [vector(terms, features) for terms in counts]
    weights = [0.0] * len(features)
    bias = 0.0
    scale = 1.0
    distribution = Counter(row["label"] for row in rows)
    order = list(range(len(rows)))
    rng = random.Random(41)
    for epoch in range(20):
        rate = .45 / math.sqrt(1 + epoch)
        rng.shuffle(order)
        for index in order:
            label = rows[index]["label"]
            values = vectors[index]
            prediction = sigmoid(bias + scale * sum(weights[key] * value for key, value in values))
            step = rate * (label - prediction) * len(rows) / (2 * distribution[label])
            scale *= 1 - rate * .001
            if scale < .1:
                weights = [weight * scale for weight in weights]
                scale = 1.0
            for key, value in values:
                weights[key] += step * value / scale
            bias += step
    return features, [value * scale for value in weights], bias


def score_rows(rows, model):
    features, weights, bias = model
    scores = []
    for row in rows:
        values = vector(grams(row["normalized"]), features)
        scores.append(sigmoid(bias + sum(weights[key] * value for key, value in values)) if values else None)
    return scores


def metrics(rows, scores, threshold):
    outcomes = Counter((row["label"], None if score is None else int(score >= threshold))
                       for row, score in zip(rows, scores, strict=True))
    positives = sum(row["label"] == 1 for row in rows)
    negatives = len(rows) - positives
    tp, tn, fp, fn = (outcomes[key] for key in ((1, 1), (0, 0), (0, 1), (1, 0)))
    no_human, no_ai = outcomes[0, None], outcomes[1, None]
    f1_ai = 2 * tp / max(1, 2 * tp + fp + fn + no_ai)
    f1_human = 2 * tn / max(1, 2 * tn + fp + fn + no_human)
    return {"human": negatives, "ai": positives, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "abstained": no_human + no_ai, "coverage": (tp + tn + fp + fn) / len(rows),
            "f1": f1_ai, "macro_f1": (f1_ai + f1_human) / 2,
            "human_fpr": fp / negatives if negatives else None}


def tune(rows, scores):
    def rank(threshold):
        result = metrics(rows, scores, threshold)
        return result["macro_f1"], -result["human_fpr"], -abs(threshold - .5)
    return max((i / 100 for i in range(5, 96)), key=rank)


def evaluate(path, hold_model="", hold_genre="", baseline=None, baseline_id=None):
    rows = dataset(path)
    if len(rows) < 8:
        raise ValueError("学習・検証・テストに使う文書を最低8件取り込んでください。")
    groups = group_documents(rows)
    parts = split_documents(groups, hold_model, hold_genre)
    model = fit(parts["train"])
    validation = score_rows(parts["validation"], model)
    threshold = tune(parts["validation"], validation)
    rule_scores = baseline(rows) if baseline else {}
    rule_threshold = tune(parts["validation"], [rule_scores[row["id"]] for row in parts["validation"]]) if rule_scores else None
    report = {"partitions": {}, "groups": len(groups), "threshold": threshold, "rule_threshold": rule_threshold,
              "hold_model": hold_model, "hold_genre": hold_genre, "baseline_id": baseline_id}

    for label, members in parts.items():
        if not members:
            continue
        stats = metrics(members, score_rows(members, model), threshold)
        ids = sorted(f'{row["id"]}:{row["digest"]}:{row["label"]}' for row in members)
        section = {"documents": len(members), "fingerprint": fingerprint("\n".join(ids)), "model": stats}
        if rule_threshold is not None:
            section["rules"] = metrics(members, [rule_scores[row["id"]] for row in members], rule_threshold)
        report["partitions"][label] = section

    features, weights, bias = model
    now = datetime.now(timezone.utc)
    identity = "-".join(report["partitions"][part]["fingerprint"][:8] for part in report["partitions"])
    version = f"sl-{now:%Y%m%d-%H%M%S-%f}-{identity}"
    portable = {"format": FORMAT, "version": version, "threshold": threshold, "bias": bias,
                "features": {term: [idf, weights[index]] for term, (index, idf) in features.items()}}
    report.update(version=version, created_at=now.isoformat(), features=len(features))
    with database(path) as db:
        db.execute("INSERT INTO experiments VALUES (?, ?, ?)",
                   (version, json.dumps(report, ensure_ascii=False), json.dumps(portable, ensure_ascii=False)))
    return report


def versions(path):
    with database(path) as db:
        return [json.loads(row["report"]) for row in db.execute(
            "SELECT report FROM experiments ORDER BY version DESC LIMIT 40")]


def model_file(path, version):
    with database(path) as db:
        row = db.execute("SELECT model FROM experiments WHERE version = ?", (version,)).fetchone()
        return row["model"] if row else None
