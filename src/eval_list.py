"""Evaluate list of experiments. Creates plots and CSV/txt data."""
import argparse
import json
import os
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from src.rl.eval_exp import check_all_experiments, read_experiment_list
from src.utils.plots import setup_latex_return_str_formatter, FIGSIZE, \
    get_color_adaptive, FONTSIZE_LEGEND, COLOR_MEAN, COLOR_SUCCESS, \
    CAPSIZE_ERRORBAR, NAME_MAP, ALPHA_LEGEND
from src.utils.stats import Stats


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "multi_file_name",
        help="file with list of experiments in config/lists/*.txt")
    parser.add_argument("multi_run_name", help="name of multi run")
    parser.add_argument("--config_dir", type=str, default="config")
    parser.add_argument("--log_dir", type=str, default="runs")
    parser.add_argument("--test_dir", type=str, default="test")
    parser.add_argument("--performance_dir", type=str, default="performance")
    parser.add_argument(
        "--use_unfinished", action="store_true",
        help="include unfinished runs in the analysis, i.e. not converged and "
             "step < max_step")
    parser.add_argument(
        "--no_plots", action="store_true",
        help="dont plot (only create text analysis)")
    parser.add_argument(
        "--no_single_plots", action="store_true",
        help="dont create single plots (only joined eval plot and text data)")
    parser.add_argument(
        "--sort", action="store_true",
        help="sort experiment by best step (only works correctly for "
             "eval metric)")
    parser.add_argument(
        "--topk", type=int, default=-1,
        help="only K best (default -1=all)")
    parser.add_argument(
        "--lowk", type=int, default=-1,
        help="only K worst (default -1=all)")
    parser.add_argument("--disable_tex", action="store_true", help="disable latex")

    args = parser.parse_args()
    log_dir = args.log_dir

    # ---------- setup plotting ----------
    format_text = setup_latex_return_str_formatter(disable_tex=args.disable_tex)
    ms = 10

    # setup different metrics to plot over for all plots
    x_metrics = "step", "time_cpu", "time_gpu"
    x_metrics_labels = (
        "K Steps", "Approx Time on CPU (sec)", "Approx Time on GPU (sec)")
    y_metric_label_train = "Train total reward (avg last 100 epochs)"
    y_metric_label_eval = r"Eval total reward ($\epsilon$=0, " \
                          "evaluated for 25 epochs)"

    # ---------- setup experiments ----------
    multi_run_name = args.multi_run_name
    experiments = read_experiment_list(
        args.multi_file_name, config_dir=args.config_dir)

    # find all relevant experiments (they have files, are converged etc.)
    exps = check_all_experiments(
        experiments, multi_run_name, log_dir=log_dir,
        use_unfinished=args.use_unfinished, ignore_agent=True,
        config_dir=args.config_dir, sort=args.sort,
        topk=args.topk, lowk=args.lowk)

    # directories
    plots_dir = Path(args.test_dir) / "_{}_{}".format(
        args.multi_file_name, multi_run_name)
    plots_dir_base = Path(args.test_dir)
    os.makedirs(str(plots_dir), exist_ok=True)

    # ---------- plot final evaluation results best epoch ----------
    print("Plotting all experiments in one figure...")

    # this is an example how to create multiple plots at once
    # 6 plots: eval score of all / only solved runs over steps /
    # approx time CPU / approx time GPU

    # setup figures and axes
    figs = dict()  # type: Dict[str, Dict[str, plt.Figure]]
    axs = dict()  # type: Dict[str, Dict[str, plt.Axes]]
    for metric, metric_label in zip(x_metrics, x_metrics_labels):
        figs[metric] = dict()  # type: Dict[str, plt.Figure]
        axs[metric] = dict()  # type: Dict[str, plt.Axes]
        for p_name in "eval_all", "eval_solved", "train_all", "train_solved":
            fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
            if p_name.find("eval") > -1:
                ax.set_ylabel(y_metric_label_eval)
            else:
                ax.set_ylabel(y_metric_label_train)
            ax.set_xlabel(metric_label)
            figs[metric][p_name] = fig
            axs[metric][p_name] = ax

    # setup csv creation
    field_names = ["EvalScore", "TrainScore", "K Steps", "Time CPU",
                   "Time GPU", "Episodes"]
    csv_dir = Path(args.test_dir) / "csv"
    txt_dir = Path(args.test_dir) / "txt"
    os.makedirs(str(csv_dir), exist_ok=True)
    os.makedirs(str(txt_dir), exist_ok=True)
    txt_file = txt_dir / "{}_{}.txt".format(
        args.multi_file_name, multi_run_name)
    csv_file = csv_dir / "{}_{}.csv".format(
        args.multi_file_name, multi_run_name)
    csv_file_means = csv_dir / "{}_{}_means.csv".format(
        args.multi_file_name, multi_run_name)
    csv_file_stddev = csv_dir / "{}_{}_stddev.csv".format(
        args.multi_file_name, multi_run_name)
    if args.topk != -1 or args.lowk != -1 or args.sort:
        # disable writing if not all data is used
        txt_file = os.devnull
        csv_file = os.devnull
        csv_file_means = os.devnull
        csv_file_stddev = os.devnull

    csv_handler = open(str(csv_file), "wt", encoding="utf8")
    csv_handler.write(";".join(
        ["ExpName", "RunNumber"] + field_names) + "\n")
    csv_handler_means = open(str(csv_file_means), "wt", encoding="utf8")
    csv_handler_means.write(
        ";".join(["ExpName", "RunAmount"] + field_names) + "\n")
    csv_handler_stddev = open(str(csv_file_stddev), "wt", encoding="utf8")
    csv_handler_stddev.write(
        ";".join(["ExpName", "RunAmount"] + field_names) + "\n")
    txt_file_handler = open(str(txt_file), "wt", encoding="utf8")

    # loop found experiments and infos
    exp_performance_cpu, exp_performance_gpu = dict(), dict()
    num_eval_ep = 0
    for n_exp, (exp_name, content) in enumerate(exps.items()):
        s = "---------- {}: {} runs".format(exp_name, len(content))
        print(s)
        print(s, file=txt_file_handler)
        (best_scores, best_steps, best_episodes, best_train_scores) = \
            [], [], [], []
        (best_scores_solved, best_steps_solved, best_episodes_solved,
         best_train_scores_solved) = [], [], [], []
        color = get_color_adaptive(n_exp, len(exps))

        # load performance metrics
        file_cpu = Path(args.performance_dir) / "cpu_{}.json".format(exp_name)
        file_gpu = Path(args.performance_dir) / "gpu_{}.json".format(exp_name)
        try:
            perf_dict_cpu = json.load(file_cpu.open("rt", encoding="utf8"))
        except FileNotFoundError as e:
            print(e, "CPU time plot will not work")
            perf_dict_cpu = dict([("ms_mean", 0)])
        try:
            perf_dict_gpu = json.load(file_gpu.open("rt", encoding="utf8"))
        except FileNotFoundError as e:
            print(e, "GPU time plot will not work")
            perf_dict_gpu = dict([("ms_mean", 0)])

        # # save to the experiment dict we are looping over for later
        exp_performance_cpu[exp_name] = perf_dict_cpu
        exp_performance_gpu[exp_name] = perf_dict_gpu

        best_times_cpu, best_times_gpu = [], []
        best_times_cpu_solved, best_times_gpu_solved = [], []

        for run_num, run_info in content.items():
            infos = run_info["infos"]
            metrics = infos["metrics"]
            eval_episodes = infos["metrics_eval"]["episodes"]
            train_solve_score = run_info["config"]["evaluation"].get(
                "train_solve_score")
            if train_solve_score is None:
                # standard procedure
                best_step = run_info["best_step"] / 1000
                best_idx = run_info["best_idx"]
                best_score = run_info["best_score"]
                best_episode = eval_episodes[best_idx]
                # find train avg scores
                try:
                    best_train_score = metrics["avg_scores"][best_episode]
                except IndexError:
                    best_train_score = metrics["avg_scores"][best_episode - 1]
            else:
                # training avg score metric
                # last step is always the best? because then the experiment
                # ends, unless not converged
                # if not converged we do not care about best step really
                # so always take last
                best_step = metrics["step"][-1] / 1000
                # train scores are indexed by episodes
                best_episode = np.argmax(metrics["avg_scores"])
                best_train_score = metrics["avg_scores"][best_episode]
                best_score = infos["metrics_eval"]["avg_scores"][-1]

            time_cpu = best_step * perf_dict_cpu["ms_mean"]
            time_gpu = best_step * perf_dict_gpu["ms_mean"]

            # append data for all runs
            best_scores.append(best_score)
            best_train_scores.append(best_train_score)
            best_steps.append(best_step)
            best_episodes.append(best_episode)
            best_times_cpu.append(time_cpu)
            best_times_gpu.append(time_gpu)

            # plot single run results for all runs
            alpha = 1.
            how = 'x'
            axs["step"]["eval_all"].plot(
                best_step, best_score, how, c=color, ms=ms, alpha=alpha)
            axs["time_cpu"]["eval_all"].plot(
                time_cpu, best_score, how, c=color, ms=ms, alpha=alpha)
            axs["time_gpu"]["eval_all"].plot(
                time_gpu, best_score, how, c=color, ms=ms, alpha=alpha)

            axs["step"]["train_all"].plot(
                best_step, best_train_score, how, c=color, ms=ms, alpha=alpha)
            axs["time_cpu"]["train_all"].plot(
                time_cpu, best_train_score, how, c=color, ms=ms, alpha=alpha)
            axs["time_gpu"]["train_all"].plot(
                time_gpu, best_train_score, how, c=color, ms=ms, alpha=alpha)

            # append data / plot only for solved runs
            if infos["is_solved"]:
                best_scores_solved.append(best_score)
                best_train_scores_solved.append(best_train_score)
                best_steps_solved.append(best_step)
                best_episodes_solved.append(best_episode)
                best_times_cpu_solved.append(time_cpu)
                best_times_gpu_solved.append(time_gpu)

            # plot only solved runs
            if infos["is_solved"]:
                axs["step"]["eval_solved"].plot(
                    best_step, best_score, how, c=color, ms=ms, alpha=alpha)
                axs["time_cpu"]["eval_solved"].plot(
                    time_cpu, best_score, how, c=color, ms=ms, alpha=alpha)
                axs["time_gpu"]["eval_solved"].plot(
                    time_gpu, best_score, how, c=color, ms=ms, alpha=alpha)
                axs["step"]["train_solved"].plot(
                    best_step, best_train_score, how, c=color, ms=ms,
                    alpha=alpha)
                axs["time_cpu"]["train_solved"].plot(
                    time_cpu, best_train_score, how, c=color, ms=ms,
                    alpha=alpha)
                axs["time_gpu"]["train_solved"].plot(
                    time_gpu, best_train_score, how, c=color, ms=ms,
                    alpha=alpha)

            # find num eval episodes
            metrics_eval = infos["metrics_eval"]
            eval_scores_list = metrics_eval["scores_list"]
            for scores in eval_scores_list:
                num_eval_ep_new = len(scores)
                if num_eval_ep == 0:
                    num_eval_ep = num_eval_ep_new
                assert num_eval_ep_new == num_eval_ep, "#eval_episodes unequal"

        # setup errorbar plot
        def plot_errbar(ax_, x_stat, y_stat, label_):
            ax_.errorbar(
                x_stat.mean, y_stat.mean, yerr=y_stat.stddev,
                xerr=x_stat.stddev, c=color, marker='^', linestyle='-',
                capsize=CAPSIZE_ERRORBAR, alpha=1, label=label_)

        # plot mean + stddev for all runs
        stats_scores = Stats(best_scores)
        stats_train_scores = Stats(best_train_scores)
        stats_steps = Stats(best_steps)
        stats_times_cpu = Stats(best_times_cpu)
        stats_times_gpu = Stats(best_times_gpu)
        stats_episodes = Stats(best_episodes)
        exp_nice_name = NAME_MAP.get(exp_name)
        if exp_nice_name is None:
            exp_nice_name = exp_name
        label = format_text("{} (#{})".format(exp_nice_name, len(best_scores)))

        plot_errbar(
            axs["step"]["eval_all"], stats_steps, stats_scores, label)
        plot_errbar(
            axs["time_cpu"]["eval_all"], stats_times_cpu, stats_scores, label)
        plot_errbar(
            axs["time_gpu"]["eval_all"], stats_times_gpu, stats_scores, label)
        plot_errbar(
            axs["step"]["train_all"], stats_steps, stats_train_scores, label)
        plot_errbar(
            axs["time_cpu"]["train_all"], stats_times_cpu, stats_train_scores,
            label)
        plot_errbar(
            axs["time_gpu"]["train_all"], stats_times_gpu, stats_train_scores,
            label)

        # plot mean + stddev only for solved runs
        if len(best_scores_solved) > 0:
            stats_scores_solved = Stats(best_scores_solved)
            stats_train_scores_solved = Stats(best_train_scores_solved)
            stats_steps_solved = Stats(best_steps_solved)
            stats_times_cpu_solved = Stats(best_times_cpu_solved)
            stats_times_gpu_solved = Stats(best_times_gpu_solved)
            if len(best_scores_solved) == len(best_scores):
                label = format_text("{} (#{})".format(
                    exp_nice_name, len(best_scores)))
            else:
                label = format_text("{} (#{}/{})".format(
                    exp_nice_name, len(best_scores_solved),
                    len(best_scores)))

            plot_errbar(axs["step"]["eval_solved"],
                        stats_steps_solved, stats_scores_solved, label)
            plot_errbar(axs["time_cpu"]["eval_solved"],
                        stats_times_cpu_solved, stats_scores_solved, label)
            plot_errbar(axs["time_gpu"]["eval_solved"],
                        stats_times_gpu_solved, stats_scores_solved, label)
            plot_errbar(axs["step"]["train_solved"],
                        stats_steps_solved, stats_train_scores_solved, label)
            plot_errbar(axs["time_cpu"]["train_solved"],
                        stats_times_cpu_solved, stats_train_scores_solved,
                        label)
            plot_errbar(axs["time_gpu"]["train_solved"],
                        stats_times_gpu_solved, stats_train_scores_solved,
                        label)

        # ---------- print stats ----------

        # write mean/stddev csv
        csv_handler_means.write(";".join([exp_name, str(len(content))]))
        csv_handler_stddev.write(";".join([exp_name, str(len(content))]))

        # print infos
        lines = []
        for k in range(len(content)):
            lines.append("{};{}".format(exp_name, k))

        for name, stats in zip(
                field_names,
                [stats_scores, stats_train_scores, stats_steps,
                 stats_times_cpu, stats_times_gpu, stats_episodes]):
            s = ("{:10} range [{:8.2f}, {:8.2f}] mean {:8.2f} +- {:9.3f} "
                 "values [{}]").format(
                name, stats.min_, stats.max_, stats.mean, stats.stddev,
                ", ".join(["{:8.2f}".format(a) for a in stats.array.tolist(
                )]))
            print(s)
            print(s, file=txt_file_handler)
            for k, v in enumerate(stats.array.tolist()):
                lines[k] += ";" + str(v)

            csv_handler_means.write(";" + str(stats.mean))
            csv_handler_stddev.write(";" + str(stats.stddev))
        csv_handler_means.write("\n")
        csv_handler_stddev.write("\n")
        for l in lines:
            csv_handler.write(l + "\n")
    csv_handler.close()
    csv_handler_means.close()
    csv_handler_stddev.close()
    txt_file_handler.close()

    if args.no_plots:
        return

    # finalize plots
    for score_name in "train", "eval":
        title_add2 = ""
        if args.topk != -1 and args.lowk != -1:
            title_add2 = ", only best {} & worst {}".format(
                args.topk, args.lowk)
        elif args.lowk != -1:
            title_add2 = ", only worst {}".format(args.lowk)
        elif args.topk != -1:
            title_add2 = ", only best {}".format(args.topk)
        if score_name == "eval":
            title_str = "Mean+stddev of final evaluation score over {}{}{}"
        else:
            title_str = "Mean+stddev of final training score over {}{}{}"
        for metric, metric_name in zip(x_metrics, x_metrics_labels):
            for title_add in "all", "solved":
                idx1, idx2 = metric, "{}_{}".format(score_name, title_add)
                ax = axs[idx1][idx2]
                title_add3 = ""
                if title_add == "solved":
                    title_add3 = ", only solved"

                ax.set_title(format_text(title_str.format(
                    metric_name, title_add2, title_add3)))
                ax.legend(fontsize=FONTSIZE_LEGEND, framealpha=ALPHA_LEGEND)
                ax.grid()
                fig = figs[idx1][idx2]
                fig.savefig(str(
                    plots_dir / "{}{}{}{}_{}_scores_over_{}.png".format(
                        "sorted_" if (
                                args.sort and args.topk == -1 and
                                args.lowk == -1) else "",
                        "top{}_".format(args.topk) if args.topk != -1 else "",
                        "bot{}_".format(args.lowk) if args.lowk != -1 else "",
                        score_name, title_add, metric)))
                plt.close(fig)

    if args.no_single_plots:
        return

    # ---------- plot eval score over metrics per run ----------
    print("\nPlotting run eval scores...")

    # loop found experiments and infos
    for n_exp, (exp_name, content) in enumerate(exps.items()):
        print("{:2} {}".format(n_exp, exp_name))
        # loop metrics
        for metric, metric_label in zip(x_metrics, x_metrics_labels):
            # setup plot
            plt.figure(figsize=FIGSIZE)
            plt.xlabel(metric_label)
            plt.ylabel(y_metric_label_eval)

            # for mean calculation, save the calculated x spaces
            x_spaces, y_spaces = [], []
            min_x, max_x = 1e20, 0
            has_metric = True
            # final plot will be in K steps / seconds not step / ms
            metric_correction = 1000
            for i, (run_num, run_info) in enumerate(content.items()):
                infos = run_info["infos"]

                metrics_eval = infos["metrics_eval"]
                eval_scores = metrics_eval["avg_scores"]  # [:best_idx+1]
                eval_steps = metrics_eval["steps"]  # [:best_idx+1]
                best_idx = run_info["best_idx"]

                if infos["is_solved"]:
                    train_solve_score = run_info["config"]["evaluation"].get(
                        "train_solve_score")
                    if train_solve_score is None:
                        # regular procedure
                        # skip all after best idx since we dont care about
                        # those
                        eval_scores = eval_scores[:best_idx + 1]
                        eval_steps = eval_steps[:best_idx + 1]
                    else:
                        # don't skip anything
                        pass

                # calculate x space for x metric
                perf = 1
                x_space = np.array(eval_steps, dtype=np.float64)
                if metric == "time_cpu":
                    perf = exp_performance_cpu[exp_name]["ms_mean"]
                    x_space *= perf
                elif metric == "time_gpu":
                    perf = exp_performance_gpu[exp_name]["ms_mean"]
                    x_space *= perf
                min_x = min(min_x, x_space[0])
                max_x = max(max_x, x_space[-1])
                x_spaces.append(x_space)

                if perf == 0:
                    # this means there is no performance evaluation for this
                    # metric and experiment - skip
                    has_metric = False
                    break

                # calculate y space for y metric
                y_space = np.array(eval_scores)
                y_spaces.append(y_space)

                # plot single run results - eval
                color = get_color_adaptive(run_num, len(content))
                label = format_text("#{}".format(run_num))
                plt.plot(
                    x_space / metric_correction, y_space, "x--", c=color,
                    ms=15, alpha=1.0,
                    label=label)

            # if there is no performance evaluation for this experiment and
            # metric, skip plot
            if not has_metric:
                print(
                    "Skipping plot {} metric {} - no performance eval".format(
                        exp_name, metric))
                plt.close()
                continue

            # ----- calculate mean
            interval = 10  # accuracy, 1 = best and slowest
            start_point = int(min_x) - int(min_x) % interval
            end_point = int(max_x) - int(max_x) % interval + interval

            # create x space for mean and mask
            new_x_space = np.arange(
                start_point, end_point + 1, interval, dtype=np.float64)
            num_points = len(new_x_space)
            num_runs = len(x_spaces)
            mask = np.zeros_like(new_x_space)
            all_data = np.zeros((num_points, num_runs))
            none_value = np.finfo(np.float64).min
            new_y_spaces = []
            for i, (x_space, y_space) in enumerate(
                    zip(x_spaces, y_spaces)):
                # interpolate the run eval data
                new_y_space = np.interp(new_x_space, x_space, y_space)
                # # this with all the mask code: ended runs are not taken into
                # # consideration for current mean/stddev
                # left=none_value, right=none_value)
                all_data[:, i] = new_y_space

                # add a 1 to the mask where the data is valid
                mask += new_y_space != none_value
                new_y_spaces.append(new_y_space)
            new_y_spaces = np.array(new_y_spaces)

            # check if there is invalid data at the end (where no more runs
            # where running)
            is_invalid = np.where(mask == 0)[0]
            if len(is_invalid > 0):
                # found invalid points - cut all data
                first_invalid_point = is_invalid[-1]
                all_data = all_data[:first_invalid_point]
                mask = mask[:first_invalid_point]
                new_x_space = new_x_space[:first_invalid_point]

            # set data to 0 where the value is invalid
            all_data[all_data == none_value] = 0

            # get mean and stdev denominator and set it to 1 to avoid div by 0
            mean_denom = np.maximum(mask, np.ones_like(mask))
            stddev_denom = np.maximum(mask - 1, np.ones_like(mask))

            # calculate mean and stddev finally
            mean = all_data.sum(axis=-1) / mean_denom
            # mean is correct

            stddev = all_data.std(axis=-1)
            mean_exp = np.expand_dims(mean, axis=-1)
            stddev2 = np.sqrt((np.sum(
                (all_data - mean_exp) ** 2, axis=-1)) / stddev_denom)

            # plot mean
            plt.plot(
                new_x_space / metric_correction, mean,
                label="mean".format(num_runs), lw=3, c=COLOR_MEAN)
            plt.fill_between(
                new_x_space / metric_correction, mean - stddev,
                mean + stddev, alpha=0.3, label="stddev", color=COLOR_MEAN)

            # stddev explodes the plot - yrange = datarange + 10%
            try:
                # noinspection PyArgumentList
                data_min = all_data.min()
                # noinspection PyArgumentList
                data_max = all_data.max()
            except ValueError:
                # happens when there are no performance measures for the metric
                # and the plot is invalid anyway
                data_min = 0
                data_max = 1
            data_range = data_max - data_min
            data_add = data_range * 0.02
            plt.ylim(data_min - data_add, data_max + data_add)

            # x-y range is a little high
            x_min = new_x_space[0] / metric_correction
            x_max = new_x_space[-1] / metric_correction
            x_diff = x_max - x_min
            x_add = x_diff * 0.02
            plt.xlim(x_min - x_add, x_max + x_add)

            # plot success line
            success = 400
            plt.plot([x_min, x_max], [success, success], "-",
                     c=COLOR_SUCCESS, lw=2, label="success")

            # finalize plot
            exp_nice_name = NAME_MAP.get(exp_name)
            if exp_nice_name is None:
                exp_nice_name = exp_name
            plt.title(
                "Experiment {} - Evaluation score over {} ".format(
                    format_text(exp_nice_name), metric_label))
            plt.legend(fontsize=FONTSIZE_LEGEND, framealpha=ALPHA_LEGEND)
            plt.grid()
            dir_ = plots_dir_base / metric
            os.makedirs(str(dir_), exist_ok=True)
            plt.savefig(str(dir_ / "run_eval_{}_over_{}.png".format(
                exp_name, metric)))
            plt.close()

    # ---------- plot train score over steps per run ----------
    print("\nPlotting run train scores...")

    # loop found experiments and infos
    for max_k in [None, 100]:
        print("\nLimiting steps to {}k".format(max_k))
        for n_exp, (exp_name, content) in enumerate(exps.items()):
            print("{:2} {}".format(n_exp, exp_name))
            # loop metrics
            for metric, metric_label in zip(x_metrics, x_metrics_labels):
                # setup plot
                plt.figure(figsize=FIGSIZE)
                plt.xlabel(metric_label)
                plt.ylabel(y_metric_label_train)

                # for mean calculation, save the calculated x spaces
                x_spaces, y_spaces = [], []
                min_x, max_x = 1e20, 0
                has_metric = True
                # final plot will be in K steps / seconds not step / ms
                metric_correction = 1000
                for i, (run_num, run_info) in enumerate(content.items()):
                    infos = run_info["infos"]

                    metrics = infos["metrics"]
                    train_scores = metrics["avg_scores"]
                    train_steps = metrics["step"]
                    best_step = run_info["best_step"]

                    if infos["is_solved"]:
                        train_solve_score = run_info["config"][
                            "evaluation"].get("train_solve_score")
                        if train_solve_score is None:
                            # standard procedure
                            pass
                        else:
                            # train metric - dont skip anything
                            # overwrite best step with last step
                            best_step = run_info["last_step"]

                        if max_k is not None:
                            best_step = min(max_k * 1000, best_step)

                        # find index where step is > best step for the first
                        # time
                        where_arr = np.where(np.array(
                            train_steps) > best_step)[0]
                        if len(where_arr) > 0:
                            best_idx = where_arr[0]

                            # skip all after best idx since we dont care about
                            # those
                            train_scores = train_scores[:best_idx + 1]
                            train_steps = train_steps[:best_idx + 1]
                    else:
                        # find index where step is > maxk for the first
                        # time
                        if max_k is not None:
                            where_arr = np.where(np.array(
                                train_steps) > max_k * 1000)[0]
                            best_idx = where_arr[0]

                            # skip all after best idx since we dont care about
                            # those
                            train_scores = train_scores[:best_idx + 1]
                            train_steps = train_steps[:best_idx + 1]

                    # calculate x space for x metric
                    perf = 1
                    x_space = np.array(train_steps, dtype=np.float64)
                    if metric == "time_cpu":
                        perf = exp_performance_cpu[exp_name]["ms_mean"]
                        x_space *= perf
                    elif metric == "time_gpu":
                        perf = exp_performance_gpu[exp_name]["ms_mean"]
                        x_space *= perf
                    min_x = min(min_x, x_space[0])
                    max_x = max(max_x, x_space[-1])
                    x_spaces.append(x_space)

                    if perf == 0:
                        # this means there is no performance evaluation for
                        # this metric and experiment - skip
                        has_metric = False
                        break

                    # calculate y space for y metric
                    y_space = np.array(train_scores)
                    y_spaces.append(y_space)

                    # plot single run results - eval
                    color = get_color_adaptive(run_num, len(content))
                    label = format_text("#{}".format(run_num))
                    plt.plot(
                        x_space / metric_correction, y_space, "-", c=color,
                        alpha=1.0, label=label)
                    # plot finish indicator
                    plt.plot(
                        (x_space / metric_correction)[-1],
                        y_space[-1], "|", c=color, alpha=1.0, ms=50)

                # if there is no performance evaluation for this experiment and
                # metric, skip plot
                if not has_metric:
                    print(
                        "Skip plot {} metric {} - no performance eval".format(
                            exp_name, metric))
                    plt.close()
                    continue

                # ----- calculate mean
                interval = 10  # accuracy, 1 = best and slowest
                start_point = int(min_x) - int(min_x) % interval
                end_point = int(max_x) - int(max_x) % interval + interval

                # create x space for mean and mask
                new_x_space = np.arange(
                    start_point, end_point + 1, interval, dtype=np.float64)
                num_points = len(new_x_space)
                num_runs = len(x_spaces)
                mask = np.zeros_like(new_x_space)
                all_data = np.zeros((num_points, num_runs))
                none_value = np.finfo(np.float64).min
                for i, (x_space, y_space) in enumerate(
                        zip(x_spaces, y_spaces)):
                    # interpolate the run eval data
                    # for train, remove ended runs
                    new_y_space = np.interp(new_x_space, x_space, y_space,
                                            left=none_value, right=none_value)
                    all_data[:, i] = new_y_space

                    # add a 1 to the mask where the data is valid
                    mask += new_y_space != none_value

                # check if there is invalid data at the end (where no more runs
                # where running)
                is_invalid = np.where(mask == 0)[0]
                if len(is_invalid) > 0:
                    # found invalid points - cut all data
                    first_invalid_point = is_invalid[-1]
                    all_data = all_data[:first_invalid_point]
                    mask = mask[:first_invalid_point]
                    new_x_space = new_x_space[:first_invalid_point]

                # dont plot mean sttdev if its only one run left
                is_only_one = np.where(np.logical_and(
                    mask == 1, new_x_space > 100000))[0]
                if len(is_only_one) > 0:
                    first_only_one_point = is_only_one[0]
                    all_data = all_data[:first_only_one_point]
                    mask = mask[:first_only_one_point]
                    new_x_space = new_x_space[:first_only_one_point]

                # get mean and stdev denominator and set it to 1 to avoid
                # div by 0
                mean_denom = np.maximum(mask, np.ones_like(mask))
                stddev_denom = np.maximum(mask - 1, np.ones_like(mask))

                # calculate mean and stddev finally
                # stddev is WRONG where there is less than full runs
                # because sum of differences is not 0 for done runs

                # calculate mean
                # set data to 0 where the value is invalid for mean calc
                all_data_mean = np.copy(all_data)
                all_data_mean[all_data == none_value] = 0
                mean = all_data_mean.sum(axis=-1) / mean_denom

                # calculate stddev
                # set data to mean where the value is invalid for stddev calc
                # with the stddev_denom the math then works out
                mean_exp = np.expand_dims(mean, axis=-1).repeat(
                    num_runs, axis=-1)
                all_data[all_data == none_value] = \
                    mean_exp[all_data == none_value]

                stddev = np.sqrt(np.sum((all_data - np.expand_dims(
                    mean, axis=-1)) ** 2, axis=-1) / stddev_denom)

                # plot mean
                plt.plot(
                    new_x_space / metric_correction, mean,
                    label="mean", lw=3, c=COLOR_MEAN)
                plt.fill_between(
                    new_x_space / metric_correction, mean - stddev,
                    mean + stddev, alpha=0.3, label="stddev", color=COLOR_MEAN)

                # fix y limites
                y_spaces_flat = np.concatenate(y_spaces, axis=-1)
                # noinspection PyArgumentList
                data_min = y_spaces_flat.min()
                # noinspection PyArgumentList
                data_max = y_spaces_flat.max()
                data_range = data_max - data_min
                data_add = data_range * 0.02
                plt.ylim(data_min - data_add, data_max + data_add)

                # x-y range is a little high
                x_min = min_x / metric_correction
                x_max = max_x / metric_correction
                x_diff = x_max - x_min
                x_add = x_diff * 0.02
                plt.xlim(x_min - x_add, x_max + x_add)

                # finalize plot
                exp_nice_name = NAME_MAP.get(exp_name)
                if exp_nice_name is None:
                    exp_nice_name = exp_name
                plt.title(
                    "Experiment {} - Train score over {}{}".format(
                        format_text(exp_nice_name), metric_label,
                        ", max {}k steps".format(max_k) if
                        max_k is not None else ""))
                plt.legend(fontsize=FONTSIZE_LEGEND, framealpha=ALPHA_LEGEND)
                plt.grid()
                dir_ = plots_dir_base / metric
                os.makedirs(str(dir_), exist_ok=True)
                plt.savefig(str(dir_ / "run_train_{}{}_over_{}.png".format(
                    "max{}k_".format(max_k) if max_k is not None else "",
                    exp_name, metric)))
                plt.close()


if __name__ == '__main__':
    main()
