import json
import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

from experiments.config import (
    ExperimentConfig,
    ModelConfig,
    ModelProvider,
    Modality,
    load_benchmark_problems,
    resolve_model_config,
)
from experiments.conversation import ConversationRunner
from experiments.evaluation.llm_judge import judge_dialogue
from experiments.kg_modalities import create_modality
from experiments.knowledge_graph_registry import (
    get_graph_dot_text,
    get_graph_image_path,
    get_graph_mermaid_text,
    list_graph_topics,
)
from experiments.models import ModelResponse, OpenAICompatibleBackend
from experiments.run_experiment import _dry_run_report, _serialise_dialogue
from experiments.report_results import load_shards
from knowledge_graph import KnowledgeGraph


PAPER_MODELS = [
    "gpt-5.4-nano-medium",
    "gpt-5.4-mini-medium",
    "gemma-4-E4B-it",
    "gemma-4-31b-it",
]


class SequenceBackend:
    def __init__(self, responses):
        self._responses = iter(responses)

    def chat(self, **kwargs):
        return next(self._responses)


class FakeUser:
    def __init__(self):
        self.usage_log = []
        self.last_usage = {}

    def respond(self, modeller_message):
        self.last_usage = {"total_tokens": 3}
        self.usage_log.append({"usage": self.last_usage})
        return "The route must visit every city once and return to its start."


class BenchmarkTests(unittest.TestCase):
    def test_exact_benchmark_shape(self):
        problems = load_benchmark_problems()
        self.assertEqual(len(problems), 30)
        self.assertEqual(list(problems), sorted(problems))
        self.assertEqual(len(set(problems)), 30)

        counts = {}
        for problem in problems.values():
            counts[problem["kg_topic"]] = counts.get(problem["kg_topic"], 0) + 1
            self.assertIn(problem["mamo_difficulty"], {"easy", "complex"})
            self.assertIsInstance(problem["mamo_row_index"], int)
            self.assertTrue(problem["selection_reason"])

        self.assertEqual(
            counts,
            {
                "Production Planning": 15,
                "Travel Salesman Problem": 15,
            },
        )

    def test_historical_model_alias_is_disclosed(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            historical = resolve_model_config("gpt-5.4-nano-medium")
        canonical = resolve_model_config("gpt-5.4-mini-low", warn_alias=False)

        self.assertIs(historical, canonical)
        self.assertEqual(historical.model_id, "gpt-5.4-mini")
        self.assertEqual(historical.reasoning_effort, "low")
        self.assertIn("historical paper-run label", str(caught[0].message))

    def test_gemma_models_use_generic_openai_compatible_provider(self):
        for name in ["gemma-4-31b-it", "gemma-4-E4B-it"]:
            config = resolve_model_config(name, warn_alias=False)
            self.assertEqual(config.provider, ModelProvider.OPENAI_COMPATIBLE)
            self.assertTrue(config.model_id)

    def test_openai_compatible_backend_uses_configured_url_key_and_model(self):
        config = ModelConfig(
            provider=ModelProvider.OPENAI_COMPATIBLE,
            model_id="provider/gemma-test",
            base_url="https://provider.example/v1",
            api_key_env="TEST_COMPATIBLE_API_KEY",
        )
        with patch.dict(
            os.environ, {"TEST_COMPATIBLE_API_KEY": "test-key"}, clear=False
        ), patch("openai.OpenAI") as client_class:
            backend = OpenAICompatibleBackend(config)

        client_class.assert_called_once()
        self.assertEqual(client_class.call_args.kwargs["api_key"], "test-key")
        self.assertEqual(
            client_class.call_args.kwargs["base_url"],
            "https://provider.example/v1",
        )
        _, request = backend._build_chat_completion_kwargs(
            [{"role": "user", "content": "hello"}]
        )
        self.assertEqual(request["model"], "provider/gemma-test")


class GraphAndModalityTests(unittest.TestCase):
    def test_two_graphs_have_all_representations(self):
        self.assertEqual(
            list_graph_topics(),
            ["Production Planning", "Travel Salesman Problem"],
        )
        for topic in list_graph_topics():
            self.assertTrue(get_graph_dot_text(topic).strip())
            self.assertTrue(get_graph_mermaid_text(topic).strip())
            self.assertTrue(Path(get_graph_image_path(topic)).is_file())

    def test_graph_parser_and_tool_navigation(self):
        topic = "Production Planning"
        graph_text = get_graph_dot_text(topic)
        graph = KnowledgeGraph(graph_text)
        self.assertTrue(graph.get_children(topic)["children"])
        self.assertTrue(graph.search_nodes("cost"))

        modality = create_modality("tools", graph_text, topic)
        self.assertEqual(
            modality.get_tools()[0]["function"]["name"],
            "load_knowledge_graph",
        )
        loaded = json.loads(modality.handle_tool_call("load_knowledge_graph", {}))
        self.assertEqual(loaded["loaded_topic"], topic)
        self.assertTrue(loaded["top_level_categories"])

        search = json.loads(
            modality.handle_tool_call("search_nodes", {"query": "cost"})
        )
        self.assertTrue(search)
        self.assertEqual(modality.loaded_topic, topic)

    def test_modality_prompts_and_image_asset(self):
        topic = "Travel Salesman Problem"
        dot_text = get_graph_dot_text(topic)
        mermaid_text = get_graph_mermaid_text(topic)

        text_modality = create_modality("text", mermaid_text, topic)
        self.assertIn(mermaid_text.strip(), text_modality.get_system_prompt())

        no_graph = create_modality("no_graph", dot_text, topic)
        self.assertNotIn(dot_text.strip(), no_graph.get_system_prompt())

        image_path = get_graph_image_path(topic)
        image = create_modality(
            "image", dot_text, topic, image_path=image_path
        )
        self.assertEqual(image.get_images(), [image_path])


class OfflinePipelineTests(unittest.TestCase):
    def test_reporting_accepts_direct_runner_output(self):
        rows = [{"modality": "no_graph", "model": "test", "problem_id": "p"}]
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory)
            (run_dir / "raw_results.json").write_text(json.dumps(rows))
            loaded, groups = load_shards(run_dir)

        self.assertEqual(loaded, rows)
        self.assertEqual(groups, {"direct": rows})

    def test_mocked_conversation_terminates_and_serialises(self):
        backend = SequenceBackend(
            [
                ModelResponse("Which cities must the route visit?", usage={"total_tokens": 5}),
                ModelResponse(
                    "```markdown\nVisit every city once and return to the start.\n```",
                    usage={"total_tokens": 7},
                ),
            ]
        )
        topic = "Travel Salesman Problem"
        modality = create_modality("no_graph", get_graph_dot_text(topic), topic)
        with patch("experiments.conversation.create_backend", return_value=backend):
            runner = ConversationRunner(
                modality=modality,
                modeller_config=resolve_model_config(
                    "gpt-5.4-mini-medium", warn_alias=False
                ),
                user=FakeUser(),
                max_turns=2,
                problem_id="offline-test",
            )
            conversation = runner.run()

        self.assertTrue(conversation.metadata["completed"])
        self.assertEqual(conversation.metadata["completion_reason"], "summary_produced")
        self.assertEqual(conversation.num_turns, 2)
        serialised = _serialise_dialogue(conversation)
        self.assertEqual([turn["role"] for turn in serialised], ["modeller", "user", "modeller"])
        self.assertIn("usage", serialised[0])

    def test_mocked_judge_parses_scores(self):
        problems = load_benchmark_problems()
        problem = next(iter(problems.values()))
        conversation_backend = SequenceBackend(
            [
                ModelResponse(
                    '{"information_recall_score":"4",'
                    '"information_precision_score":5,'
                    '"information_redundancy_score":"3"}',
                    usage={"total_tokens": 11},
                )
            ]
        )

        from experiments.config import ConversationLog, ConversationTurn

        conversation = ConversationLog(
            experiment_id="judge-test",
            modality=Modality.NO_GRAPH,
            model="openai/gpt-5.4-mini",
            problem_id=problem["id"],
            turns=[
                ConversationTurn(
                    role="modeller",
                    content="```markdown\nA concise problem summary.\n```",
                )
            ],
        )
        with patch(
            "experiments.evaluation.llm_judge.create_backend",
            return_value=conversation_backend,
        ):
            result = judge_dialogue(
                conversation,
                problem["description"],
                judge_model="gpt-5.4-medium",
                eval_type="summary",
            )

        self.assertEqual(result["information_recall_score"], 4)
        self.assertEqual(result["information_precision_score"], 5)
        self.assertEqual(result["information_redundancy_score"], 3)
        self.assertEqual(result["eval_type"], "summary")

    def test_full_paper_dry_run_plans_2400_conversations(self):
        config = ExperimentConfig(
            modalities=list(Modality),
            models=PAPER_MODELS,
            num_runs=5,
            max_turns=40,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = _dry_run_report(config, load_benchmark_problems())

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["valid_model_modality_pairs"], 16)
        self.assertEqual(report["planned_conversations"], 2400)
        self.assertEqual(report["manifest"]["problem_count"], 30)
        self.assertFalse(report["skipped_model_modality_pairs"])


if __name__ == "__main__":
    unittest.main()
