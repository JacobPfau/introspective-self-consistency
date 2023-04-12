import argparse

from src.evals.sequence_completion import evaluate_sequence_completion_equality
from src.evals.string_transformation import evaluate_string_transformation_equality


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
    "--model",
    default="gpt-3.5-turbo",
    type=str,
    choices=["gpt-3.5-turbo", "text-davinci-003"],
)
parser.add_argument(
    "--sequence-type",
    default="integer",
    type=str,
    choices=["binary", "integer"],
)
parser.add_argument(
    "--evaluation",
    default="sequence_completion_equality",
    type=str,
    choices=[
        "sequence_completion_equality",
        "string_transformation_completion_equality"
    ],
)
parser.add_argument("--num-shots", default=8, type=int)
parser.add_argument("--use-cot", default=True, type=str2bool, nargs="?", const=True)

args = parser.parse_args()
if __name__ == "__main__":
    if args.evaluation == "string_transformation_completion_equality":
        evaluate_string_transformation_equality(
            args.model, num_shots=args.num_shots, cot=args.use_cot
        )
    if args.evaluation == "sequence_completion_equality":
        evaluate_sequence_completion_equality(
            args.model, num_shots=args.num_shots, cot=args.use_cot
        )
