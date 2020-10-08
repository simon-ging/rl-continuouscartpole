"""Evaluate single experiment with multiple runs.
Features:
- render agent / create video
- evaluate agent for N episodes, save results, create plot
"""
import argparse
import json
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.rl.eval_exp import get_available_runs
from src.rl.experiment import Experiment
from src.rl.get_filenames import get_filename_eval
from src.rl.render import render_agent
from src.utils.config_mapping import load_experiment_config
from src.utils.jsonencoder import to_json
from src.utils.logger import get_logger
from src.utils.plots import setup_latex_return_str_formatter, FIGSIZE, \
    get_color_adaptive, COLOR_SUCCESS, NAME_MAP, ALPHA_LEGEND
from src.utils.stats import Stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name")
    parser.add_argument("run_name")
    parser.add_argument("--config_dir", type=str, default="config")
    parser.add_argument("--log_dir", type=str, default="runs")
    parser.add_argument("--test_dir", type=str, default="test")
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--run_number", type=int, default=-1,
                        help="number of single run "
                             "(default -1 = all available runs)")
    parser.add_argument("--reload_step", type=int, default=-1,
                        help="which step to reload "
                             "(default -1=last step)")
    parser.add_argument("--reload_last", action="store_true",
                        help="reload last instead of best step")
    parser.add_argument("--render", action="store_true",
                        help="render agent")
    parser.add_argument("--video", action="store_true",
                        help="capture rendering on video")
    parser.add_argument("--render_num", type=int, default=1,
                        help="how many times to render agent")
    parser.add_argument("--overwrite", action="store_true",
                        help="force overwriting old eval results")
    parser.add_argument("--eval_episodes", type=int, default=200)
    parser.add_argument("--num_steps", type=int, default=500)
    parser.add_argument("--video_dir", type=str, default="test_videos")
    parser.add_argument("--disable_tex", action="store_true", help="disable latex")
    args = parser.parse_args()

    exp_name = args.exp_name
    run_name = args.run_name
    # disable logging to file for notebooks
    logger = get_logger(".", "stats", with_file=False)
    # ä logger.setLevel(logging.ERROR)

    # find runs
    if args.run_number != -1:
        run_start = args.run_number
        run_end = args.run_number + 1
        runs = list(range(run_start, run_end))
    else:
        runs = get_available_runs(
            exp_name, run_name, log_dir=args.log_dir)
    print("experiment: {}_{}, runs to be evaluated: {}".format(
        exp_name, run_name, runs))

    # ---------- rendering ----------
    if args.render:
        for n in runs:
            # create experiment
            config_file = Path(
                args.config_dir) / "experiments" / "{}.json".format(
                exp_name)
            config = load_experiment_config(config_file)
            exp = Experiment(
                config, exp_name, run_name, logger=logger,
                log_dir=args.log_dir, run_number=n, reset=False, reload=True,
                reload_step=args.reload_step, reload_best=not args.reload_last,
                use_cuda=args.cuda, verbose=False)
            logger.info("Reloaded step {}".format(exp.current_step))

            # render agent
            for m in range(args.render_num):
                video_id = "vid_{}_{}_{}".format(n, m, exp.current_step)
                video_dir = ""
                if args.video:
                    video_dir = Path(args.video_dir) / "{}_{}".format(
                        exp_name, run_name)

                logger.info(
                    "Run {:2} trial {:2} rendering agent...".format(
                        n, m))
                render_agent(
                    exp.env, exp.agent, exp.observ_transformer,
                    num_steps=args.num_steps, video_dir=video_dir,
                    video_name=video_id)
        # done rendering, exit
        return

    # ---------- evaluation ----------
    eval_scores, steps = evaluate_all_runs(
        exp_name, run_name, run_number=args.run_number, log_dir=args.log_dir,
        config_dir=args.config_dir, use_cuda=args.cuda,
        reload_best=not args.reload_last, reload_step=args.reload_step,
        eval_episodes=args.eval_episodes, overwrite=args.overwrite,
        num_steps=args.num_steps)

    # plot
    format_text = setup_latex_return_str_formatter(disable_tex=args.disable_tex)
    plt.figure(figsize=FIGSIZE)
    for i, n in enumerate(runs):
        run_scores = eval_scores[i]
        step = steps[i]
        stats = Stats(run_scores)
        color = get_color_adaptive(i, len(runs))
        # capsize=CAPSIZE_ERRORBAR
        capsize = 5
        ms = 20
        plt.errorbar(
            i, stats.mean, stats.stddev, color=color,
            label=format_text("#{} step {:.0f}K mean {:.3f}".format(
                n, step / 1000, stats.mean)),
            capsize=capsize, marker='x', lw=1, ms=10)

        plt.plot(
            [i], [stats.min_], color=color,
            ms=ms, marker=6, fillstyle='none')
        plt.plot([i], [stats.max_], marker=7, color=color,
                 ms=ms, fillstyle='none')
        plt.xlabel("Agent Number")
        plt.ylabel(format_text(
            "Eval Score ($\epsilon=0$, {} episodes)".format(
                args.eval_episodes)))
    name = NAME_MAP.get(exp_name)
    plt.title(
        format_text("{} Agent Quality: Mean, stddev, min, max of eval "
                    "score".format(name if name else exp_name)))
    plt.plot([-.5, len(runs) - .5], [400, 400], "--", c=COLOR_SUCCESS,
             label="success")
    plt.xlim([-.5, len(runs) - .5])
    plt.grid()
    plt.legend(framealpha=ALPHA_LEGEND)

    dir_ = Path(args.test_dir) / "eval"
    os.makedirs(str(dir_), exist_ok=True)
    fn = str(dir_ / "{}_{}_eval{}ep.png".format(
        exp_name, run_name, args.eval_episodes))
    plt.savefig(fn)
    plt.close()
    print("Plot saved to {}".format(fn))


def evaluate_all_runs(
        exp_name, run_name, run_number=-1, log_dir="runs", config_dir="config",
        use_cuda=False, reload_best=True, reload_step=-1, eval_episodes=200,
        overwrite=False, num_steps=500):
    scores_collector = []
    steps_collector = []
    if run_number >= 0:
        runs = [run_number]
    else:
        runs = get_available_runs(
            exp_name, run_name, log_dir=log_dir)
    for n in runs:
        scores, step = evaluate_run(
            exp_name, run_name, n, log_dir=log_dir, config_dir=config_dir,
            use_cuda=use_cuda, reload_best=reload_best,
            reload_step=reload_step, eval_episodes=eval_episodes,
            overwrite=overwrite, num_steps=num_steps)
        scores_collector.append(scores)
        single_stats = Stats(scores)
        print("{}_{}_{} x{}: {:8.3f} +- {:8.3f} range [{} {}]".format(
            exp_name, run_name, n, len(scores), single_stats.mean,
            single_stats.stddev, single_stats.min_, single_stats.max_))
        steps_collector.append(step)
    if len(scores_collector) > 1:
        s = Stats(np.mean(scores_collector, axis=1))
        print("Stats of mean results per run:\n    {}".format(s))
        s2 = Stats(np.reshape(scores_collector, -1))
        print("Stats of all concatenated results:\n    {}".format(s2))
    else:
        print("Need at least 2 runs to compute stddev etc.")
    return scores_collector, steps_collector


def evaluate_run(
        exp_name, run_name, run_number, log_dir="runs", config_dir="config",
        use_cuda=False, reload_best=True, reload_step=-1, eval_episodes=200,
        overwrite=False, num_steps=500):
    """Evaluate a run. returns list of eval scores"""
    # disable logging
    logger = get_logger(".", "stats_{}_{}_{}".format(
        exp_name, run_name, run_number), with_file=False)
    logger.setLevel(logging.ERROR)

    # load experiment
    config_file = Path(config_dir) / "experiments" / "{}.json".format(
        exp_name)
    config = load_experiment_config(config_file)
    exp = Experiment(
        config, exp_name, run_name, logger=logger,
        log_dir=log_dir, run_number=run_number, reset=False, reload=True,
        reload_step=reload_step, reload_best=reload_best,
        use_cuda=use_cuda, verbose=False)

    # read old evaluation file if exists
    eval_dict = {}
    eval_file = get_filename_eval(
        exp_name, run_name, run_number, log_dir=log_dir)
    if eval_file.is_file():
        with eval_file.open("rt", encoding="utf8") as fh:
            eval_dict = json.load(fh)
    else:
        print("Old eval file {} not found".format(eval_file))

    # check if there already is evaluation for this episode and
    # print a warning that we will overwrite it now
    current_step = exp.current_step
    current_step_str = str(current_step)
    old_eval = eval_dict.get(current_step_str)
    eval_episodes_str = str(eval_episodes)
    if old_eval is not None:
        old_scores = old_eval.get(eval_episodes_str)
        if old_scores is None:
            print(
                "Old scores for step {} episodes {} not found, creating new"
                "".format(current_step_str, eval_episodes))
        elif overwrite:
            print(
                "Run {:2} Overwriting evaluation for step "
                "{} because of --overwrite flag".format(
                    run_number, current_step))
        else:
            print("Reloading existing eval:", end=" ")
            # logger.info("Reloading existing evaluation")
            eval_dict = json.load(
                eval_file.open("rt", encoding="utf8"))
            return eval_dict[current_step_str][eval_episodes_str], current_step
    else:
        print("Current step {} not found in eval dict: {}".format(
            current_step_str, eval_dict.keys()
        ))
        eval_dict[current_step_str] = {}

    # evaluate
    exp.verbose = True
    scores = exp.run_evaluation(
        num_ep=eval_episodes, num_steps=num_steps,
        print_steps=num_steps > 10000)
    # print("{} tries, mean: {}".format(len(scores), np.mean(scores)))

    # update evaluation dictionary
    eval_dict[current_step_str][eval_episodes_str] = scores

    # write evaluation to file
    with eval_file.open("wt", encoding="utf8") as fh:
        fh.write(to_json(eval_dict))

    return scores, current_step


if __name__ == '__main__':
    main()
