# SPDX-License-Identifier: Apache-2.0

# Standard
import argparse
import importlib.util
from pathlib import Path
import sys


SCRIPT = Path(__file__).parents[2] / "benchmarks" / "longbench_makv_cachegen.py"
SPEC = importlib.util.spec_from_file_location("longbench_makv_cachegen", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_longbench_answer_hit_is_normalized():
    assert MODULE.score("The answer is Canberra.", ("Canberra",), "triviaqa")
    assert MODULE.score("C2", ("c2",), "trec")
    assert not MODULE.score("The answer is Paris.", ("Canberra",), "triviaqa")


def test_extract_answer_text_drops_qwen_thinking_trace():
    assert MODULE.extract_answer_text("<think>work</think> Canberra") == "Canberra"
    assert MODULE.extract_answer_text("<think>unfinished reasoning") == ""
    assert MODULE.extract_answer_text("Canberra") == "Canberra"


def test_prompt_ids_passes_answer_only_to_chat_template():
    calls = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls.append(kwargs)
            return {"input_ids": [1, 2, 3]}

    example = MODULE.LongBenchExample(
        "id", "context", "question", ("answer",), "hotpotqa"
    )
    assert MODULE.prompt_ids(Tokenizer(), example, "run") == [1, 2, 3]
    assert calls[0]["enable_thinking"] is False


def test_prompt_ids_with_last_user_span_requires_exact_template_prefix():
    class Tokenizer:
        def apply_chat_template(self, _messages, **kwargs):
            return {
                "input_ids": [1, 2, 3, 4]
                if kwargs["add_generation_prompt"]
                else [1, 2]
            }

    example = MODULE.LongBenchExample(
        "id", "context", "question", ("answer",), "hotpotqa"
    )
    assert MODULE.prompt_ids_with_last_user_span(Tokenizer(), example, "run") == (
        [1, 2, 3, 4],
        (0, 2),
    )


def test_importance_file_is_keyed_by_prompt_hash(tmp_path):
    path = tmp_path / "importance.json"
    path.write_text('{"scores":{"abc":[0.1,0.2]}}', encoding="utf-8")
    assert MODULE.load_importance_file(str(path)) == {"abc": [0.1, 0.2]}


def test_nonfinite_importance_is_json_safe_and_fail_closed(tmp_path):
    path = tmp_path / "importance.json"
    path.write_text(
        '{"scores":{"abc":[0.1,Infinity,NaN,-Infinity]}}',
        encoding="utf-8",
    )
    values = MODULE.load_importance_file(str(path))["abc"]
    assert all(MODULE.math.isfinite(value) for value in values)
    status = MODULE.load_importance_status_file(str(path))["abc"]
    assert status[0]["forced_precision"] is None
    assert [item["forced_precision"] for item in status[1:]] == [
        "BF16",
        "BF16",
        "BF16",
    ]


def test_importance_status_file_reads_v32_attention_artifact(tmp_path):
    path = tmp_path / "importance.json"
    path.write_text(
        '{"scores":{"abc":[0.1,0.2]},"attention_results":{"abc":'
        '{"attention":{"valid_mask":[true,false],'
        '"forced_precision_by_token":[null,"BF16"],'
        '"reason_by_token":[null,"NO_FUTURE_PROBE"]}}}}',
        encoding="utf-8",
    )
    assert MODULE.load_importance_status_file(str(path)) == {
        "abc": [
            {"valid_mask": True, "forced_precision": None, "reason": None},
            {
                "valid_mask": False,
                "forced_precision": "BF16",
                "reason": "NO_FUTURE_PROBE",
            },
        ]
    }


def test_precision_plan_file_is_keyed_by_prompt_hash(tmp_path):
    path = tmp_path / "plans.json"
    path.write_text(
        '{"precision_plans":{"abc":{"token_count":2,"precision_by_token":['
        '"K2V2","BF16"]}}}',
        encoding="utf-8",
    )
    assert MODULE.load_precision_plan_file(str(path)) == {
        "abc": {"token_count": 2, "precision_by_token": ["K2V2", "BF16"]}
    }


def test_request_completion_sends_explicit_precision_plan_without_importance():
    captured = {}

    class Response:
        ok = True

        def json(self):
            return {"choices": [{"text": "ok"}]}

    class Session:
        def post(self, _url, *, json, **_kwargs):
            captured.update(json)
            return Response()

    args = argparse.Namespace(
        mode="makv",
        scout_overlap=False,
        require_importance_file=True,
        risk_source="synthetic",
        url="http://unit.test/v1/completions",
        model="unit-test",
        max_tokens=1,
        generation_seed=0,
        timeout=1.0,
    )
    plan = {"token_count": 2, "precision_by_token": ["K2V2", "BF16"]}
    result = MODULE.request_completion(
        Session(),
        args=args,
        ids=[1, 2],
        stream=False,
        precision_plan=plan,
    )
    assert result["text"] == "ok"
    transfer = captured["kv_transfer_params"]
    assert transfer["lmcache.makv_precision_plan"] == plan
    assert transfer["prompt_token_hash"] == MODULE.prompt_token_hash([1, 2])
    assert transfer["lmcache.prompt_token_hash"] == MODULE.prompt_token_hash([1, 2])
    assert transfer["request_token_count"] == 2
    assert "lmcache.makv_importance" not in transfer


def test_request_completion_sends_attention_token_status():
    captured = {}

    class Response:
        ok = True

        def json(self):
            return {"choices": [{"text": "ok"}]}

    class Session:
        def post(self, _url, *, json, **_kwargs):
            captured.update(json)
            return Response()

    args = argparse.Namespace(
        mode="makv",
        scout_overlap=False,
        require_importance_file=True,
        risk_source="synthetic",
        url="http://unit.test/v1/completions",
        model="unit-test",
        max_tokens=1,
        generation_seed=0,
        timeout=1.0,
    )
    status = [
        {"valid_mask": True, "forced_precision": None, "reason": None},
        {
            "valid_mask": False,
            "forced_precision": "BF16",
            "reason": "NO_FUTURE_PROBE",
        },
    ]
    MODULE.request_completion(
        Session(),
        args=args,
        ids=[1, 2],
        stream=False,
        importance_values=[0.1, 0.0],
        importance_status=status,
    )
    assert captured["kv_transfer_params"]["lmcache.makv_importance_status"] == status


def test_load_importance_timing_reads_per_prompt_metadata(tmp_path):
    path = tmp_path / "importance.json"
    path.write_text(
        '{"scores":{"abc":[0.1,0.2]},'
        '"metadata":{"abc":{"scoutrank_time_ms":12.5}}}',
        encoding="utf-8",
    )
    assert MODULE.load_importance_timing(str(path)) == {"abc": 12.5}


def test_longbench_summarization_has_no_fake_exact_accuracy():
    assert MODULE.score("a summary", ("reference",), "gov_report") is None


def test_longbench_loader_reads_official_fields(tmp_path):
    path = tmp_path / "hotpotqa.jsonl"
    path.write_text(
        '{"_id":"x","context":"doc","input":"question",'
        '"answers":["answer"],"length":17,"all_classes":["A","B"]}\n',
        encoding="utf-8",
    )
    rows = MODULE.load_examples(path, "hotpotqa", 1, 0)
    assert rows[0].example_id == "x"
    assert rows[0].context == "doc"
    assert rows[0].answers == ("answer",)
    assert rows[0].length == 17
    assert rows[0].all_classes == ("A", "B")


def test_extract_cached_tokens_accepts_all_response_shapes():
    assert MODULE.extract_cached_tokens(
        {
            "kv_transfer_params": {
                "cached_token_stats": {"num_lmcache_cached_tokens": 256}
            }
        }
    ) == (256, "lmcache")
    assert MODULE.extract_cached_tokens(
        {"kv_transfer_params": {"num_lmcache_cached_tokens": 128}}
    ) == (128, "lmcache")
    assert MODULE.extract_cached_tokens(
        {"usage": {"prompt_tokens_details": {"cached_tokens": 64}}}
    ) == (64, "vllm_usage")
    assert MODULE.extract_cached_tokens(
        {"kv_transfer_params": {"num_lmcache_cached_tokens": -1}}
    ) == (None, None)
    assert MODULE.extract_cached_tokens({}) == (None, None)


def test_makv_manager_timing_summary_reports_quantizer_share():
    summary = MODULE.makv_manager_timing_summary(
        {
            "metrics": {
                "makv_remote_put_requests": 3,
                "makv_raw_input_bytes": 1200,
                "makv_stored_bytes": 400,
                "makv_remote_put_total_time_ms": 100.0,
                "makv_remote_put_decode_time_ms": 5.0,
                "makv_remote_quantize_time_ms": 60.0,
                "makv_remote_quantize_kernel_time_ms": 50.0,
                "makv_remote_encode_validate_time_ms": 20.0,
                "makv_remote_storage_put_time_ms": 15.0,
                "makv_remote_get_requests": 4,
                "makv_remote_get_total_time_ms": 12.0,
                "makv_remote_get_storage_time_ms": 4.0,
                "makv_remote_get_validate_time_ms": 2.0,
            }
        }
    )
    assert summary["put_requests"] == 3
    assert summary["compression_ratio"] == 3.0
    assert summary["remote_quantize_core_share_of_put_pct"] == 50.0


def test_summarize_exposes_official_scores_without_removing_legacy_scores():
    records = [
        {
            "valid": True,
            "answers": ["Canberra"],
            "all_classes": [],
            "prompt_tokens": 4,
            "scoutrank_time_ms": 2.0,
            "cold": {
                "correct": True,
                "official_score": 2.0 / 3.0,
                "text": "Canberra city",
                "ttft_ms": 1.0,
                "ttft_with_scoutrank_ms": 3.0,
                "latency_ms": 2.0,
            },
            "hit": {
                "correct": True,
                "official_score": 1.0,
                "text": "Canberra",
                "ttft_ms": 0.5,
                "ttft_with_scoutrank_ms": 2.5,
                "latency_ms": 1.0,
            },
        }
    ]
    args = argparse.Namespace(
        mode="cachegen",
        task="hotpotqa",
        evaluator="official",
        model_layers=1,
        model_kv_heads=1,
        model_head_dim=1,
        model_dtype_bytes=2,
        redis_url=None,
        storage_dir=None,
    )
    summary = MODULE.summarize(records, args)
    assert summary["cold_accuracy"] == 2.0 / 3.0
    assert summary["hit_accuracy"] == 1.0
    assert summary["legacy_cold_accuracy"] == 1.0
    assert summary["official_metric"] == "qa_f1"
    assert summary["hit_official_score_percent"] == 100.0
    assert summary["cold_ttft_mean_ms"] == 1.0
    assert summary["cold_ttft_p95_ms"] == 1.0
    assert summary["hit_ttft_mean_ms"] == 0.5
    assert summary["scoutrank_time_mean_ms"] == 2.0
    assert summary["cold_ttft_with_scoutrank_mean_ms"] == 3.0
    assert summary["hit_ttft_with_scoutrank_mean_ms"] == 2.5
    assert round(summary["scoutrank_share_of_cold_ttft_pct"], 6) == 66.666667
    assert summary["hit_latency_mean_ms"] == 1.0
    assert summary["hit_latency_p95_ms"] == 1.0
