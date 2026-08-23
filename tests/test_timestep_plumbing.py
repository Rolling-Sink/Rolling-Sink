import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TimestepPlumbingTest(unittest.TestCase):
    def test_causal_pipeline_model_calls_pass_host_decision(self):
        for relative_path in (
            "pipeline/causal_inference.py",
            "pipeline/causal_diffusion_inference.py",
        ):
            tree = ast.parse((ROOT / relative_path).read_text())
            generator_calls = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"
                and node.func.attr == "generator"
            ]

            self.assertTrue(generator_calls, relative_path)
            for call in generator_calls:
                keywords = {keyword.arg for keyword in call.keywords}
                self.assertIn("timestep_is_zero", keywords, relative_path)

    def test_compiled_model_path_has_no_unconditional_item_call(self):
        tree = ast.parse((ROOT / "wan/modules/causal_model.py").read_text())
        inference = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_forward_inference"
        )
        item_calls = [
            node
            for node in ast.walk(inference)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "item"
        ]

        self.assertEqual(len(item_calls), 1)
        fallback = next(
            node
            for node in inference.body
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "timestep_is_zero"
        )
        self.assertIn(item_calls[0], list(ast.walk(fallback)))


if __name__ == "__main__":
    unittest.main()
