import argparse

from evals.function_selection_evaluation import (
    function_class_selection_evaluation,
    function_selection_evaluation,
)
from pipelines.sequence_completions import sequence_functions


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
    "--evaluation",
    default="function_selection_evaluation",
    type=str,
    choices=[
        "function_selection_evaluation",
        "function_class_selection_evaluation",
    ],
)

# Function selection evaluation arguments
parser.add_argument("--target-sequence", default="5,10,15,20", type=str)
parser.add_argument(
    "--incorrect-functions",
    default='["lambda x: x", "lambda x: x + 7", "lambda x: x ** 2", "lambda x: x ** 3"]',
    type=list,
)
parser.add_argument(
    "--correct-functions", default=["lambda x: x*5"], type=list, nargs="+"
)

# Function class selection evaluation arguments
parser.add_argument(
    "--function_class",
    default="arithmetic_progression",
    type=str,
    choices=sequence_functions.keys(),
)
parser.add_argument("--sequence_length", default="5", type=int)

# Common arguments
parser.add_argument(
    "--model",
    default="DAVINCI",
    type=str,
    choices=["CHAT", "DAVINCI"],
)
parser.add_argument("--num-shots", default=6, type=int)
parser.add_argument("--use-cot", default=False, type=str2bool, nargs="?", const=True)
parser.add_argument("--num-samples", default=5, type=int)
parser.add_argument("--num-functions", default=4, type=int)

args = parser.parse_args()
if __name__ == "__main__":
    if args.evaluation == "function_selection_evaluation":
        print(args.correct_functions)
        (
            correct_choices,
            incorrect_choices,
            invalid_outputs,
        ) = function_selection_evaluation(
            model_name=args.model,
            target_sequence=args.target_sequence,
            temperature=0.0,
            num_shots=args.num_shots,
            use_cot=args.use_cot,
            num_samples=args.num_samples,
            num_functions=args.num_functions,
            generate_functions=True,
            incorrect_functions=args.incorrect_functions,
            correct_functions=args.correct_functions,
        )
    elif args.evaluation == "function_class_selection_evaluation":
        (
            correct_choices,
            incorrect_choices,
            invalid_outputs,
        ) = function_class_selection_evaluation(
            model_name=args.model,
            function_class=args.function_class,
            sequence_functions=sequence_functions,
            sequence_length=args.sequence_length,
            temperature=0.0,
            num_shots=args.num_shots,
            use_cot=args.use_cot,
            num_samples=args.num_samples,
            num_functions=args.num_functions,
        )

    print(f"Correct choices: {correct_choices}")
    print(f"Incorrect choices: {incorrect_choices}")
    print(f"Invalid outputs: {invalid_outputs}")
