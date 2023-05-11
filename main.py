import argparse

from src.evals.sequence_completion import evaluate_sequence_completion_equality
from src.evals.sequence_completion_with_base_change import evaluate_compute_dependence_with_base_changes
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
    "--sequence-type",
    default="binary",
    type=str,
    choices=["binary", "integer"],
)

parser.add_argument("--on-ambiguous-sequences", default="True", type=str2bool)
parser.add_argument(
    "--model",
    default="gpt-3.5-turbo",
    type=str,
    choices=["gpt-3.5-turbo", "text-davinci-003"],
)
parser.add_argument("--num-shots", default=4, type=int)
parser.add_argument("--use-cot", default=False, type=str2bool, nargs="?", const=True)
parser.add_argument("--num-samples", default=1, type=int)
parser.add_argument(
    "--task",
    default="sequence_completion_equality",
    type=str,
    choices=[
        "compute_dependence_with_base_changes",
        "sequence_completion_equality",
        "string_transformation_completion_equality",
    ],
)


args = parser.parse_args()
if __name__ == "__main__":
    if args.task == "string_transformation_completion_equality":
        evaluate_string_transformation_equality(
            args.model, num_shots=args.num_shots, cot=args.use_cot
        )
    if args.task == "sequence_completion_equality":
        evaluate_sequence_completion_equality(
            args.model,
            num_shots=args.num_shots,
            cot=args.use_cot,
            few_shot_prompt_type=args.few_shot_prompt_type,
        )
    if args.task == "compute_dependence_with_base_changes":
        evaluate_compute_dependence_with_base_changes(
            args.sequence_type, args.model, args.num_shots, args.on_ambiguous_sequences, args.num_samples
        )
