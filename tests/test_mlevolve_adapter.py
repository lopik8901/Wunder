import io
import json
import unittest
from types import SimpleNamespace

from connectome.mlevolve_adapter import (
    RESULT_PREFIX,
    adapt_execution_result,
    emit_result,
    make_exec_callback,
    parse_result_output,
    to_mlevolve_feedback,
)


class MlevolveAdapterTests(unittest.TestCase):
    def setUp(self):
        self.result = {
            "status": "success",
            "primary_metric": 0.2,
            "metric_name": "combined_WP",
            "maximize": True,
            "metrics": {"WP_t0": 0.1, "WP_t1": 0.3, "combined_WP": 0.2},
            "runtime_seconds": 1.5,
            "artifact_bytes": 10,
        }

    def test_result_marker_round_trips_all_official_metrics(self):
        stream = io.StringIO()
        emit_result(self.result, stream=stream)

        parsed = parse_result_output(["training complete", stream.getvalue()])

        self.assertEqual(parsed, self.result)
        self.assertTrue(stream.getvalue().startswith(RESULT_PREFIX))

    def test_rejects_inconsistent_combined_score(self):
        invalid = dict(self.result)
        invalid["metrics"] = dict(self.result["metrics"], combined_WP=0.9)
        line = RESULT_PREFIX + json.dumps(invalid)

        with self.assertRaises(ValueError):
            parse_result_output(line)

    def test_maps_result_to_existing_mlevolve_parser_schema(self):
        feedback = to_mlevolve_feedback(self.result)

        self.assertFalse(feedback["is_bug"])
        self.assertFalse(feedback["lower_is_better"])
        self.assertEqual(feedback["metric"], 0.2)
        self.assertEqual(feedback["wp_t0"], 0.1)
        self.assertEqual(feedback["wp_t1"], 0.3)
        self.assertEqual(feedback["combined_wp"], 0.2)
        self.assertEqual(feedback["status"], "success")
        self.assertIn("WP_t0=0.1", feedback["summary"])
        self.assertIn("WP_t1=0.3", feedback["summary"])

    def test_exec_callback_preserves_execution_result_and_appends_feedback(self):
        line = RESULT_PREFIX + json.dumps(self.result)
        execution_result = SimpleNamespace(
            term_out=[line], exec_time=1.5, exc_type=None, exc_info=None, exc_stack=None
        )
        interpreter = FakeInterpreter(execution_result)

        returned = make_exec_callback(interpreter)("candidate code", "node-7", True)

        self.assertIs(returned, execution_result)
        self.assertEqual(interpreter.call, ("candidate code", "node-7", True))
        feedback = json.loads(returned.term_out[-1].removeprefix("MLEVOLVE_FEEDBACK_JSON="))
        self.assertEqual(feedback["metric"], 0.2)
        self.assertFalse(feedback["lower_is_better"])

    def test_execution_failure_is_preserved_as_bug_feedback(self):
        execution_result = SimpleNamespace(
            term_out=["candidate failed"], exec_time=0.4,
            exc_type="RuntimeError", exc_info={"message": "boom"}, exc_stack=None,
        )

        returned = adapt_execution_result(execution_result)

        self.assertIs(returned, execution_result)
        feedback = json.loads(returned.term_out[-1].removeprefix("MLEVOLVE_FEEDBACK_JSON="))
        self.assertTrue(feedback["is_bug"])
        self.assertIsNone(feedback["metric"])
        self.assertEqual(feedback["status"], "failure")
        self.assertIn("RuntimeError", feedback["summary"])


class FakeInterpreter:
    def __init__(self, result):
        self.result = result
        self.call = None

    def run(self, code, node_id, reset_session=True):
        self.call = (code, node_id, reset_session)
        return self.result


if __name__ == "__main__":
    unittest.main()
    adapt_execution_result,
