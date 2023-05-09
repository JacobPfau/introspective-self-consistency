"""
Conducting a self-consistency check, using integer sequences.
"""
import argparse

from pipelines.sequence_completions import find_ambiguous_integer_sequences
from pipelines.sequence_completions import sequence_functions as all_sequence_functions

from q11.evals.check_self_consistency import self_consistency_evaluation

# Removing this class of function as they cause errors
all_sequence_functions.pop("indexing_criteria_progression")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--sequence-type",
    default="integer",
    type=str,
    choices=["binary", "integer"],
)

parser.add_argument("--on-ambiguous-sequences", default="True", type=str2bool)
parser.add_argument(
    "--model",
    default="DAVINCI",
    type=str,
    choices=["CHAT", "DAVINCI"],
)
parser.add_argument("--num-shots", default=4, type=int)
parser.add_argument("--use-cot", default=False, type=str2bool, nargs="?", const=True)
parser.add_argument("--num-samples", default=5, type=int)


args = parser.parse_args()
if __name__ == "__main__":
    sequence_functions = None
    if args.on_ambiguous_sequences:
        if args.sequence_type == "integer":
            sequence_functions = all_sequence_functions
            # Get the ambiguous sequences
            # Use default parameters for now
            results = {}
            ambiguous_sequences = find_ambiguous_integer_sequences()
            for sequence in ambiguous_sequences:
                print(f"Sequence: {sequence}")
                (
                    correct_choices,
                    incorrect_choices,
                    invalid_outputs,
                ) = self_consistency_evaluation(
                    model_name=args.model,
                    sequence=sequence,
                    distribution="default",
                    shots=args.num_shots,
                    shot_method="random",
                    temperature=0.0,
                    samples=args.num_samples,
                )
                results[sequence] = {
                    "correct": correct_choices,
                    "incorrect": incorrect_choices,
                    "invalid": invalid_outputs,
                }
        else:
            pass
            # TODO: have support for general base sequences here

    print(f"Correct: {correct_choices}")
    print(f"Incorrect: {incorrect_choices}")
    print(f"Invalid: {invalid_outputs}")

    # Save the results
    import datetime
    import json
    import os

    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d-%H-%M-%S")
    results_dir = os.path.join(
        "evals/results", "ambiguous_sequences_function_selection_evaluation"
    )
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results_path = os.path.join(results_dir, f"{now_str}.json")
    with open(results_path, "w") as f:
        json.dump(results, f)

    print(f"Results saved to {results_path}")