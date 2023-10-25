import unittest
from unittest.mock import patch

from src.pipelines import ShotSamplingType
from src.prompt_generation.robustness_checks.continuation_prompt import (
    create_continuation_prompt,
)
from src.prompt_generation.robustness_checks.explanation_prompt import (
    create_explanation_prompt,
)


class TestQ11(unittest.TestCase):
    @patch("src.prompt_generation.prompt_loader.get_original_cwd")
    def test_completion_prompt(self, cwd_patch):

        cwd_patch.return_value = ""

        sequence = [1, 2, 3, 4, 5]
        task_prompt = "self-consistency"
        model_name = "gpt-3.5-turbo-0301"
        base = 2
        prompt = create_continuation_prompt(
            sequence=sequence,
            task_prompt=task_prompt,
            model_name=model_name,
            base=base,
        )
        print(prompt)

    @patch("src.prompt_generation.prompt_loader.get_original_cwd")
    def test_create_continuation_prompt(self, cwd_patch):

        cwd_patch.return_value = ""

        sequence = [1, 2, 3, 4, 5, 6]
        task_prompt = "max-probability"
        role_prompt = "gpt-og"
        model_name = "gpt-3.5-turbo-0301"
        base = 2
        shots = 2
        shot_method = ShotSamplingType.RANDOM
        prompt = create_continuation_prompt(
            sequence=sequence,
            task_prompt=task_prompt,
            role_prompt=role_prompt,
            model_name=model_name,
            base=base,
            shots=shots,
            shot_method=shot_method,
        )
        print(prompt)

    @patch("src.prompt_generation.prompt_loader.get_original_cwd")
    def test_create_explanation_prompt(self, cwd_patch):

        cwd_patch.return_value = ""

        sequence = [1, 2, 3, 4, 5, 6]
        task_prompt = "max-probability"
        role_prompt = "gpt-og"
        model_name = "gpt-3.5-turbo-0301"
        base = 2
        shots = 2
        shot_method = ShotSamplingType.RANDOM
        prompt = create_explanation_prompt(
            sequence=sequence,
            task_prompt=task_prompt,
            role_prompt=role_prompt,
            model_name=model_name,
            base=base,
            shots=shots,
            shot_method=shot_method,
        )
        print(prompt)

    # Note: commented out because it call the model which is not a unittest
    # @patch("src.prompt_generation.prompt_loader.get_original_cwd")
    # def test_self_consistency_evaluation(self, cwd_patch):

    #     cwd_patch.return_value = ""

    #     model_name = "text-davinci-003"
    #     sequence = [1, 2, 3]
    #     task_prompt = "self-consistency"
    #     base = 2
    #     shots = 4
    #     shot_method = "random"
    #     temperature = 0
    #     samples = 4

    #     outputs = self_consistency_evaluation(
    #         model_name=model_name,
    #         sequence=sequence,
    #         task_prompt=task_prompt,
    #         base=base,
    #         shots=shots,
    #         shot_method=shot_method,
    #         temperature=temperature,
    #         samples=samples,
    #         seed=0,
    #     )

    #     print(outputs)


if __name__ == "__main__":
    unittest.main()
