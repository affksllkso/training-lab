import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import HTTPError

from generate import generate, normalize_model_id, request_document
from lab import dataset, evaluate, group_documents, import_jsonl, model_file, split_documents
from predict import predict


def fixture(count=15):
    records = []
    for index in range(count):
        for label in (0, 1):
            variation = "".join(chr(0x4E00 + index * 83 + (offset * 17 + label * 11) % 83)
                                for offset in range(68))
            suffix = ("昨日は靴が濡れたので、帰りに近所の店に立ち寄って夕食を買った。"
                      if label == 0 else "この事例は、多角的な視点から検討する必要があると考えられる。")
            records.append({
                "id": f"{index}-{label}", "label": label, "topic": f"テーマ{index}",
                "source": f"fixture-{index}" if label == 0 else "fixture-ai",
                "genre": "留保ジャンル" if index >= 13 else "通常ジャンル",
                "author_id": f"author-{index}" if label == 0 else "",
                "license": "合成テストデータ" if label == 0 else "",
                "model": ("held" if index >= 13 else "base") if label == 1 else "",
                "split": "test" if index < 2 else "", "text": f"ここではテーマ{index}について記述する。" + variation + suffix,
            })
    return records


def jsonl(records):
    return "\n".join(json.dumps(item, ensure_ascii=False) for item in records).encode("utf-8")


class LabTest(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.db = self.root / "test.sqlite3"

    def test_import_export_predict(self):
        records = fixture()
        counts, errors = import_jsonl(self.db, jsonl([*records, records[0], {"id": "invalid"}]))
        self.assertEqual(dict(counts), {"added": 30, "duplicate": 1, "invalid": 1})
        self.assertEqual(len(errors), 1)
        rule = lambda rows: {row["id"]: float(row["label"]) for row in rows}
        report = evaluate(self.db, hold_model="held", baseline=rule, baseline_id="fixture-rules")
        self.assertEqual(report["partitions"]["test"]["documents"], 4)
        self.assertEqual(report["partitions"]["audit"]["documents"], 4)
        self.assertEqual(report["partitions"]["test"]["rules"]["macro_f1"], 1)
        model = json.loads(model_file(self.db, report["version"]))
        self.assertEqual(predict(records[0]["text"], model)["state"], "scored")
        self.assertEqual(predict("短文", model), {"state": "short"})
        with self.assertRaises(ValueError):
            predict(records[0]["text"], {**model, "threshold": -1})

    def test_fixed_test_stays_fixed_after_additions(self):
        records = fixture()
        import_jsonl(self.db, jsonl(records))
        before = evaluate(self.db)["partitions"]["test"]["fingerprint"]
        added = fixture(18)[30:]
        import_jsonl(self.db, jsonl(added))
        after = evaluate(self.db)["partitions"]["test"]["fingerprint"]
        self.assertEqual(before, after)

    def test_similar_documents_cannot_cross_explicit_partitions(self):
        first = fixture()[0]
        second = {**first, "id": "duplicate-edit", "author_id": "other", "source": "other",
                  "topic": "新規テーマ", "text": first["text"] + "追記。", "split": "train"}
        import_jsonl(self.db, jsonl([first, second]))
        groups = group_documents(dataset(self.db))
        self.assertEqual(len(groups), 1)
        with self.assertRaisesRegex(ValueError, "異なる split"):
            split_documents(groups)

    def test_generation_persists_before_failure_and_resumes(self):
        plan = self.root / "plan.jsonl"
        plan.write_bytes(jsonl([{"id": f"plan-{index}", "genre": "解説", "topic": f"{index}の説明",
                                 "prompt": f"題材{index}を説明してください。"} for index in range(3)]))
        text = "検証用に模擬結果として作成した日本語の文章です。" * 40
        first = Mock(side_effect=[text, ValueError("上限")])
        with self.assertRaisesRegex(ValueError, "2行目"):
            generate(plan, self.db, "ollama", "example", 3, provider=first, notify=Mock())
        self.assertEqual([row["id"] for row in dataset(self.db)], ["plan-0"])
        later = Mock(return_value=text + "追記。")
        self.assertEqual(generate(plan, self.db, "ollama", "example", 1, provider=later, notify=Mock()), 1)
        self.assertEqual([row["id"] for row in dataset(self.db)], ["plan-0", "plan-1"])
        later.assert_called_once_with("ollama", "example", "題材1を説明してください。")

    def test_provider_rate_limit_stops(self):
        opener = Mock()
        opener.open.side_effect = HTTPError("http://127.0.0.1:11434/api/generate", 429, "limited", {}, None)
        with patch("generate.build_opener", return_value=opener), self.assertRaisesRegex(ValueError, "停止"):
            request_document("ollama", "example", "例")
        self.assertEqual(opener.open.call_count, 1)

    def test_gemini_4_models_normalize_and_strip_thought_parts(self):
        self.assertEqual(normalize_model_id("gemini", "Gemini 4 Hydrogen"), "gemini-4-hydrogen")
        self.assertEqual(normalize_model_id("gemini", "gemini4neon"), "gemini-4-neon")
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.return_value = json.dumps({
            "candidates": [{"content": {"parts": [
                {"text": "思考過程", "thought": True},
                {"text": "生成された日本語本文"},
            ]}}]
        }, ensure_ascii=False).encode("utf-8")
        opener = Mock()
        opener.open.return_value = response
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}), patch("generate.build_opener", return_value=opener):
            output = request_document("gemini", "Gemini 4 Neon", "テスト題材")
        self.assertEqual(output, "生成された日本語本文")
        called_request = opener.open.call_args.args[0]
        self.assertEqual(
            called_request.full_url,
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-4-neon:generateContent",
        )


if __name__ == "__main__":
    unittest.main()
