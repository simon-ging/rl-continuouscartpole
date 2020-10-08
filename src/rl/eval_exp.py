"""Utilities to get folders / filenames / find last episode / find last runs,
check experiment status, ..."""

import copy
import glob
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

from src.rl.get_filenames import get_exp_dir, get_filename_infos, \
    get_filename_config
from src.utils.config_mapping import load_experiment_config


def find_last_step(exp_name, run_name, run_number, log_dir="runs"):
    """ Given experiment, find last saved episode. Return -1 if no episode
    is found."""
    exp_dir = get_exp_dir(exp_name, run_name, run_number, log_dir=log_dir)
    files = glob.glob(str(exp_dir / "scores*.json"))
    if len(files) == 0:
        return -1
    else:
        return sorted([
            int(file.replace(".json", "").split("_")[-1])
            for file in files])[-1]


def load_infos(exp_name, run_name, run_number, step, log_dir="runs"):
    result_file = get_filename_infos(
        exp_name, run_name, run_number, step, log_dir=log_dir)
    infos = json.load(result_file.open("rt", encoding="utf8"))
    return infos


def find_best_step_eval(exp_name, run_name, run_number, log_dir="runs",
                        verbose=False, config_dir="config"):
    last_step = find_last_step(exp_name, run_name, run_number, log_dir=log_dir)
    infos = load_infos(
        exp_name, run_name, run_number, last_step, log_dir=log_dir)
    avg_scores = infos["metrics_eval"]["avg_scores"]
    best_score = infos["best_eval_score"]
    best_idx = np.argsort(avg_scores)[-1]
    best_step = infos["metrics_eval"]["steps"][best_idx]
    if verbose:
        print("step {} scores {} best {} / {}".format(
            last_step, avg_scores, best_score, best_idx))
    return best_step, best_idx, best_score


def get_available_runs(exp_name, run_name, log_dir="runs"):
    """Given experiment name and run name, find available run numbers."""
    glob_str = str(Path(log_dir) / "{}_{}_*".format(exp_name, run_name))
    dirs = glob.glob(glob_str)
    run_numbers = sorted([int(a.split(run_name)[-1][1:]) for a in dirs])
    if len(run_numbers) == 0:
        print("glob return nothing: {}".format(glob_str))
    return run_numbers


def check_experiment(
        exp_name, run_name, run_number, log_dir="runs", config_dir="config",
        verbose=True, use_unfinished=False, ignore_agent=False):
    if verbose:
        print("***** Checking exp {} run {}".format(exp_name, run_number))

    # check experiment already exists
    exp_dir = get_exp_dir(
        exp_name, run_name, run_number, log_dir=log_dir)
    is_done = False
    if exp_dir.is_dir():
        # load experiment config
        config_file = (Path(config_dir) / "experiments" /
                       "{}.json".format(exp_name))
        config = load_experiment_config(config_file)
        max_steps = config["training"]["max_steps"]

        # check last info file
        last_step = find_last_step(
            exp_name, run_name, run_number, log_dir=log_dir)
        result_file = exp_dir / "scores_{}.json".format(last_step)
        agent_file = exp_dir / "agent_{}.pth".format(last_step)
        if last_step == -1:
            print("Did not find any experiments")
        elif (agent_file.is_file() or ignore_agent) and result_file.is_file():
            infos = json.load(result_file.open("rt", encoding="utf8"))
            if last_step >= max_steps:
                is_done = True
            else:
                if verbose:
                    print("Not enough steps, checking if it is done")
                if infos["is_solved"]:
                    is_done = True
                else:
                    if use_unfinished:
                        is_done = True
                    if verbose:
                        print(
                            "Experiment {}_{}_{} trained {}/{} steps and "
                            "is not solved.".format(
                                exp_name, run_name, run_number, last_step,
                                max_steps))
        else:
            if verbose:
                print("Agent {} / results {} missing".format(
                    agent_file, result_file))
    else:
        if verbose:
            print("Directory does not exist: {}".format(exp_dir))
    return is_done


def check_all_experiments(
        exp_list, run_name, log_dir="runs", use_unfinished=False,
        ignore_agent=False, config_dir="config", sort=False, topk=-1, lowk=-1):
    exps = OrderedDict()
    print("Checking experiments...")
    sort_scores = []
    sort_keys = []
    for n_exp, exp_name in enumerate(exp_list):
        print("{}_{} ".format(exp_name, run_name), end=" ")
        # find runs that have this name
        runs = get_available_runs(
            exp_name, run_name, log_dir=log_dir)
        best_scores = []
        print("runs", runs)
        for n in runs:
            # print("---------- {}_{}_{}".format(exp_name, multi_run_name, n),
            #       end=" ")
            is_done = check_experiment(
                exp_name, run_name, n, log_dir=log_dir,
                verbose=False, use_unfinished=use_unfinished,
                ignore_agent=ignore_agent, config_dir=config_dir)
            # print(is_done)
            if not is_done:
                continue
            if exps.get(exp_name) is None:
                exps[exp_name] = OrderedDict()
            last_step = find_last_step(
                exp_name, run_name, n, log_dir=log_dir)
            best_step, best_idx, best_score = find_best_step_eval(
                exp_name, run_name, n, log_dir=log_dir, config_dir=config_dir)
            infos = load_infos(
                exp_name, run_name, n, last_step, log_dir=log_dir)
            # print("best step {} with {}".format(best_step, best_score))
            exps[exp_name][n] = {
                "last_step": last_step,
                "best_step": best_step,
                "best_idx": best_idx,
                "best_score": best_score,
                "infos": infos,
                "config": load_experiment_config(get_filename_config(
                    exp_name, run_name, n, log_dir=log_dir))
            }
            best_scores.append(best_step)
        try:
            print("found {} completed runs".format(len(exps[exp_name])))
        except KeyError:
            print("found nothing. if the experiment is not converged, consider"
                  " using flag --use_unfinished")
        mean_score = np.mean(best_scores)
        sort_scores.append(float(mean_score))
        sort_keys.append(exp_name)
    print()
    if sort:
        new_dict = OrderedDict()
        new_index = np.argsort(sort_scores)
        maxi = len(new_index)
        for i, idx in enumerate(new_index):
            if topk != -1 and lowk == -1:
                if i >= topk:
                    continue
            elif topk == -1 and lowk != -1:
                if maxi - i - 1 >= lowk:
                    continue
            elif topk != -1 and lowk != -1:
                if not (i < topk or maxi - i - 1 < lowk):
                    continue
            key = sort_keys[idx]
            new_dict[key] = copy.deepcopy(exps[key])
        return new_dict
    return exps


def read_experiment_list(exp_list, config_dir="config"):
    exp_list_file = Path(config_dir) / "lists" / "{}.txt".format(
        exp_list)
    file_content = exp_list_file.open("rt",
                                      encoding="utf8").read().splitlines()
    experiments = []
    for a in file_content:
        if a.strip() == "":
            continue
        if a[0] == "#":
            continue
        experiments.append(a)
    return experiments
