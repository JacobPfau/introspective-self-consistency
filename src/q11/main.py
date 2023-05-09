"""
Conducting a self-consistency check, using integer sequences.
"""
import argparse

from pipelines.sequence_completions import find_ambiguous_integer_sequences
from pipelines.sequence_completions import sequence_functions as all_sequence_functions
<<<<<<< HEAD
from q11.evals.check_self_consistency import self_consistency_evaluation
from q11.utils import reformat_self_consistency_results
=======

from q11.evals.check_self_consistency import self_consistency_evaluation
>>>>>>> 50a75b7 (Have scaffolding ready for initial self-consistency checks)

# Removing this class of function as they cause errors
all_sequence_functions.pop("indexing_criteria_progression")

<<<<<<< HEAD

=======
>>>>>>> 50a75b7 (Have scaffolding ready for initial self-consistency checks)
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
<<<<<<< HEAD
    default="binary",
=======
    default="integer",
>>>>>>> 50a75b7 (Have scaffolding ready for initial self-consistency checks)
    type=str,
    choices=["binary", "integer"],
)

parser.add_argument("--on-ambiguous-sequences", default="True", type=str2bool)
parser.add_argument(
    "--model",
<<<<<<< HEAD
    default="CHAT",
=======
    default="DAVINCI",
>>>>>>> 50a75b7 (Have scaffolding ready for initial self-consistency checks)
    type=str,
    choices=["CHAT", "DAVINCI"],
)
parser.add_argument("--num-shots", default=4, type=int)
parser.add_argument("--use-cot", default=False, type=str2bool, nargs="?", const=True)
<<<<<<< HEAD
parser.add_argument("--num-samples", default=1, type=int)
=======
parser.add_argument("--num-samples", default=5, type=int)
>>>>>>> 50a75b7 (Have scaffolding ready for initial self-consistency checks)


args = parser.parse_args()
if __name__ == "__main__":
<<<<<<< HEAD
    total = 0
    sequence_functions = None
    if args.on_ambiguous_sequences:
        if args.sequence_type == "integer":
            base = 10
        elif args.sequence_type == "binary":
            base = 2
        else:
            raise ValueError("Unknown sequence type.")
        sequence_functions = all_sequence_functions
        # Get the ambiguous sequences
        # Use default parameters for now
        results = {}
        ambiguous_sequences = find_ambiguous_integer_sequences()
        for sequence in ambiguous_sequences:
            # turn the sequence from a string into a list of integers
            int_sequence = [int(x) for x in sequence.split(",")]
            total += 1
            print("Total: ", total)
            print(f"Sequence: {sequence}")
            for _ in range(2):
                try:
                    (
                        consistent_explanations,
                        inconsistent_explanations,
                        invalid_explanations,
                    ) = self_consistency_evaluation(
                        model_name=args.model,
                        sequence=int_sequence,
                        distribution="default",
                        base=base,
                        shots=args.num_shots,
                        shot_method="random",
                        temperature=0.0,
                        samples=args.num_samples,
                    )
                except Exception as e:
                    print("oopies")
                    print(e)
                else:
                    if sequence in results:
                        results[sequence]["consistent"] += consistent_explanations
                        results[sequence]["inconsistent"] += inconsistent_explanations
                        results[sequence]["invalid"] += invalid_explanations
                    else:
                        results[sequence] = {
                            "consistent": consistent_explanations,
                            "inconsistent": inconsistent_explanations,
                            "invalid": invalid_explanations,
                        }
                    break
=======
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
>>>>>>> 50a75b7 (Have scaffolding ready for initial self-consistency checks)
        else:
            pass
            # TODO: have support for general base sequences here

<<<<<<< HEAD
    print(total)

    # Reformat results
    results = reformat_self_consistency_results(results)
=======
    print(f"Correct: {correct_choices}")
    print(f"Incorrect: {incorrect_choices}")
    print(f"Invalid: {invalid_outputs}")
>>>>>>> 50a75b7 (Have scaffolding ready for initial self-consistency checks)

    # Save the results
    import datetime
    import json
    import os

    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d-%H-%M-%S")
    results_dir = os.path.join(
<<<<<<< HEAD
        "q11/evals/results/ambiguous_sequences_function_selection_evaluation",
        f"{now_str}",
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
=======
        "evals/results", "ambiguous_sequences_function_selection_evaluation"
    )
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results_path = os.path.join(results_dir, f"{now_str}.json")
    with open(results_path, "w") as f:
        json.dump(results, f)

    print(f"Results saved to {results_path}")
>>>>>>> 50a75b7 (Have scaffolding ready for initial self-consistency checks)
