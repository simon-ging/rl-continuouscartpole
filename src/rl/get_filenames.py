"""Utilities to get folders / filenames / find last episode / find last runs,
etc.
"""

from pathlib import Path


def get_exp_id(exp_name, run_name, run_number):
    """Given experiment, get unique experiment id (full experiment name)."""
    return "{}_{}_{}".format(
        exp_name, run_name, run_number)


def get_exp_dir(exp_name, run_name, run_number, log_dir="runs"):
    """Given experiment, get experiment directory."""
    return Path(log_dir) / get_exp_id(exp_name, run_name, run_number)


def get_filename_infos(
        exp_name, run_name, run_number, step, log_dir="runs"):
    """Given experiment and step, get filename of the train infos.
    These also contain evaluation done during training."""
    exp_dir = get_exp_dir(exp_name, run_name, run_number, log_dir=log_dir)
    return exp_dir / "scores_{}.json".format(step)


def get_filename_eval(
        exp_name, run_name, run_number, log_dir="runs"):
    """Given experiment and episode, get filename of the evaluation infos.
    These only contain the evaluations done at the end of training, with the
    eval.py script.
    """
    exp_dir = get_exp_dir(exp_name, run_name, run_number, log_dir=log_dir)
    return exp_dir / "eval.json".format()


def get_filename_agent(
        exp_name, run_name, run_number, step, log_dir="runs"):
    """Given experiment and step, get filename of the agent weights."""
    exp_dir = get_exp_dir(exp_name, run_name, run_number, log_dir=log_dir)
    return exp_dir / "agent_{}.pth".format(step)


def get_filename_config(
        exp_name, run_name, run_number, log_dir="runs"):
    """Given experiment, get name of experiment config file."""
    exp_dir = get_exp_dir(exp_name, run_name, run_number, log_dir=log_dir)
    return exp_dir / "config.json"
