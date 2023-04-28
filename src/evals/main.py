"""
Script for seeing if the model can select the correct function from a list of
functions. Looking directly at ambiguous functions generated in pipelines.sequence_completions
"""

import argparse
import logging

from evals.function_selection_evaluation import (  # function_class_selection_evaluation,
    function_selection_evaluation,
)
from evals.utils import reformat_ambiguous_sequences, reformat_results
from pipelines.sequence_completions import find_ambiguous_integer_sequences
from pipelines.sequence_completions import sequence_functions as all_sequence_functions

# Note: sometimes the "indexing_criteria_progression" function class
# Raises an error, as we may index a list which is too small.
# all_sequence_functions.pop("indexing_criteria_progression")

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("This message will be displayed in the console")

string_to_base = {
    "binary": 2,
    "integer": 10,
}


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
    default="binary",
    type=str,
    choices=["binary", "integer"],
)
parser.add_argument(
    "--evaluation",
    default="function_selection",
    type=str,
    choices=["function_class_selection", "function_selection"],
)
parser.add_argument("--on-ambiguous-sequences", default="True", type=str2bool)
parser.add_argument(
    "--model",
    default="CHAT",
    type=str,
    choices=["CHAT", "DAVINCI"],
)
parser.add_argument("--num-shots", default=4, type=int)
parser.add_argument("--use-cot", default=False, type=str2bool, nargs="?", const=True)
parser.add_argument("--num-samples", default=5, type=int)
parser.add_argument("--num-functions", default=4, type=int)


args = parser.parse_args()
if __name__ == "__main__":
    total = 0
    sequence_functions = None
    if args.on_ambiguous_sequences:
        if args.sequence_type == "integer" or args.sequence_type == "binary":
            # Get inetegr value for base
            base = string_to_base[args.sequence_type]
            sequence_functions = all_sequence_functions
            # Get the ambiguous sequences
            # Use default parameters for now
            results = {}
            ambiguous_sequences = find_ambiguous_integer_sequences()
            if args.sequence_type != "integer":
                # Change the base of the ambiguous sequences
                ambiguous_sequences = reformat_ambiguous_sequences(ambiguous_sequences)
            for sequence in ambiguous_sequences:
                print(f"Sequence: {sequence}")
                # Go through each function and see if the model can select it
                for fn in ambiguous_sequences[sequence]:
                    total += 1
                    print("total is: ", total)
                    func = fn["fn"]
                    print("func is: ", func)
                    offset = fn["offset"]
                    # Try multiple times (in case the openai api fails)
                    # May be another error with certain function, mentioned at top
                    for _ in range(2):
                        try:
                            (
                                correct_choices,
                                incorrect_choices,
                                invalid_outputs,
                            ) = function_selection_evaluation(
                                model_name=args.model,
                                target_sequence=sequence,
                                temperature=0.0,
                                num_shots=args.num_shots,
                                use_cot=args.use_cot,
                                num_samples=args.num_samples,
                                num_functions=args.num_functions,
                                generate_functions=True,
                                incorrect_functions=None,
                                correct_functions=[func],
                                base=base,
                            )
                        except Exception as e:
                            print("oopies")
                            print(e)
                        else:
                            # If the function already exists, add to the results
                            if func in results:
                                results[func]["correct"] += correct_choices
                                results[func]["incorrect"] += incorrect_choices
                                results[func]["invalid"] += invalid_outputs
                            else:
                                results[str(fn["fn"])] = {
                                    "correct": correct_choices,
                                    "incorrect": incorrect_choices,
                                    "invalid": invalid_outputs,
                                }
                            break
        else:
            pass
            # TODO: have support for general base sequences here

    print(total)

    # Reformat results
    results = reformat_results(results)

    # Save the results
    import datetime
    import json
    import os

    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d-%H-%M-%S")
    results_dir = os.path.join(
        "evals/results/ambiguous_sequences_function_selection_evaluation", f"{now_str}"
    )
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f)

    # Save command line arguments
    args_path = os.path.join(results_dir, "args.json")
    args_dict = vars(args)
    args_dict["sequence_functions"] = all_sequence_functions
    with open(args_path, "w") as f:
        json.dump(args_dict, f)

    print(f"Results saved to {results_path}")
