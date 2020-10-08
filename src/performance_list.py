"""Evaluate performance for a list of experiments
(mean CPU/GPU time per step)"""
import argparse
import logging
import os
import shutil
from collections import OrderedDict
from pathlib import Path

from src.rl.eval_exp import read_experiment_list
from src.rl.experiment import Experiment
from src.rl.get_filenames import get_exp_dir
from src.utils.config_mapping import load_experiment_config
from src.utils.jsonencoder import to_json
from src.utils.logger import get_logger
from src.utils.stats import Stats


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "multi_file_name",
        help="file with list of experiments in config/lists/*.txt")
    parser.add_argument("--config_dir", type=str, default="config")
    parser.add_argument("--log_dir", type=str, default="runs")
    parser.add_argument("--performance_dir", type=str, default="performance")
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument(
        "--save", action="store_true", help="save performance list to file")
    parser.add_argument(
        "--cleanup", action="store_true", help="delete performance runs")

    args = parser.parse_args()

    # read list of experiments
    run_name = "test-perf"
    experiments = read_experiment_list(
        args.multi_file_name, config_dir=args.config_dir)

    # disable logging
    logger = get_logger(".", "perf", with_file=False)
    logger.setLevel(logging.ERROR)

    # loop experiments
    steps = 10000
    trials = 5
    for exp_name in experiments:
        print("{:30}...".format(exp_name), end=" ")
        config_file = Path(args.config_dir) / "experiments" / "{}.json".format(
            exp_name)
        config = load_experiment_config(config_file)
        # disable eval (1 eval done at the end for all experiments)
        config["evaluation"]["eval_every_n_steps"] = steps * 2
        ms_per_step = []
        for t in range(trials):
            # run experiment
            exp = Experiment(
                config, exp_name, run_name, logger=logger,
                log_dir=args.log_dir, run_number=t, reset=True, reload=False,
                use_cuda=args.cuda, verbose=False, max_steps=steps)
            exp.run_training()

            # calculate time spent
            time_total = exp.current_time_total
            ms_per_step.append(1000 * time_total / steps)

            # cleanup
            if args.cleanup:
                exp_dir = get_exp_dir(
                    exp_name, run_name, t, log_dir=args.log_dir)
                shutil.rmtree(exp_dir, ignore_errors=True)

        # calculate stats over N trials
        s = Stats(ms_per_step)
        print("{:7.3f} +- {:7.3f} ms/step".format(
            s.mean, s.stddev))

        # save to file
        if args.save:
            infos = OrderedDict()
            infos["ms"] = ms_per_step
            infos["ms_mean"] = s.mean
            infos["ms_std"] = s.stddev
            infos["cuda"] = args.cuda
            infos["steps"] = steps
            infos["trials"] = trials
            os.makedirs(args.performance_dir, exist_ok=True)
            filename = Path(args.performance_dir) / "{}_{}.json".format(
                "gpu" if args.cuda else "cpu", exp_name)
            with filename.open("wt", encoding="utf8") as fh:
                fh.write(to_json(infos))


if __name__ == '__main__':
    main()
