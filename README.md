# "Self-Consistency of Large Language Models under Ambiguity" - Repository for the paper resulting from AI safety camp 2023

# Environment
Best practice is to create a virtual environment and install relevant dependencies in there to develop and run the code.

```
conda create -n <env_name> python=3.10
conda activate <env_name>
pip install -r requirements.txt -r requirements_precommit.txt
```

## Pre-commit
Install `pre-commit` in your local dev environment.
```
pip install pre-commit
pre-commit install
```
This makes applies various hooks before when you commit code. These hooks check the linting and formatting of your code and ensure that code is formatted properly following `flake8` and `black` standards.

You can read more [here](https://pre-commit.com/).

When you get stuck/annoyed by pre-commit rejecting your commit, you may choose to run `git commit -m "your message" --no-verify` or `-n` to skip the hooks. This is not recommended because it bypasses the linting and can introduce trouble for other devs.

# Experiments

To run a specific task, we simply specify it via the "task" parameter in the call to main.py:
```sh
python main.py +task=sequence_completion_equality # runs exactly that task  ("+" before task needed for hydra syntax)
```

Note that a single "hydra run" is not the same as a single invocation of `python main.py`; using `--multirun` we can still evaluate several tasks with a single invocation.


## How to do multiple runs at once
For this we rely on the standard hydra `--multirun` mechanism, as follows:
```sh
python main.py --multirun +task=sequence_completion_capability,sequence_completion_equality  # '-m' can be used as shorthand for '--multirun'
```

## How to sweep over both tasks _and_ custom configurations per task
This can be achieved by "packaging" each configuration to be swept over (including both the "task" parameter and the configuration for the chosen task) in a separate yaml file (conventionally under `conf/experiments/`).

See `conf/experiment/demo_1.yaml` and `conf/experiment/demo_2.yaml` for extensively commented example files.

Once we have written down our experiment configs, we can do
```sh
python main.py -m +experiment=demo_1,demo_2
```

## Q0.1: Sequence Completion Capability
```sh
python main.py --multirun +task=sequence_completion_capability model=davinci,text-davinci-003,gpt-3.5-turbo-0301,gpt-4-0314

```

## Q0.2: Sequence Completion Equality

```sh
python main.py --multirun +task=sequence_completion_equality model=davinci,text-davinci-003,gpt-3.5-turbo-0301,gpt-4-0314,claude-v1
```

## Q1.2: Self-Consistency under

The goal of Q1.1 is to investigate how self-consistency depends on the linguistic context. We vary both _what_ precisely we ask for, as well as _who_ (which simulacrum) we ask it of.

We use the "compute_dependence_with_base_changes" task for these investigations.

a. Task / What we are asking for
    1. **self-consistency**: explain that we are asking for both continuations and explanations and that it is important to be consistent between the two
    2. **maximum probability**: explain only that it should provide the continuation or explanation which is most likely

b. Role / Who we are asking it of
    1. **self**: "continue how you, <MODEL_NAME> would continue"
    2. **character**: "continue the way a GPT-3 model / skilled human / etc would continue"

Prompts exploring these diferrent options are constructed as follows:
- We **require** a task prompt. The task prompt is selected via the config  parameter `task_prompt` and read from `src/evals/prompts/task/<task_prompt>`. In each such directory, we have two files `continuation.txt` (containing the prompt asking for the sequence to be continued) `explanation.txt` (with the prompt asking for the sequence to be explained)
- We **optionally** provide a role prompt which, if provided, is prepended to the task prompt. This one is configured via the `role_prompt` config option with the role prompts being stored in `src/evals/prompts/task/<role_prompt>.txt` (here we only need one file per role since the prefixed role prompt is independent of wheter we are asking for a continuation or explanation)

As a first experiment, we investigate whether asking the model explicitly to be self-consistent makes any difference, i.e. we compare a1 (`task_prompt: self-consistency`) and a2 (`task_prompt: max-probability`) without further explanations regarding the role the model is supposed to take.
```sh
python main.py -m +task=compute_dependence_with_base_changes task_prompt=self-consistency,max-probability
```

## Q2.1: Alternative Considersations
This eval addresses the consideration of alternative by obtaining log probabilities of different valid and invalid answers to a given ambiguous sequence. We wish to determine whether the model consistently allocates significant probability mass to valid options and what distribution over log probabilities of alternative answers can be observed.

```sh
python main.py -m +task=q2_1_logprob_inequality num_shots=4,6,8,10 seed=41,42,43
```

## Q2.2: Verbalization of Alternatives
```sh
python main.py -m +task=q2_2_alternative_verbalization num_shots=4,6,8,10 model=text-davinci-003,gpt-3.5-turbo-0301,gpt-4-0314 seed=41,42,43
```

# Tests
Tests are run using `pytest`.
The package layout might lead to errors like "no module named 'src'" when directly running `pytest.`
To work around this invoke pytest as a python module or update Python path:
```sh
python -m pytest tests
```
