"""Submit Experiments in a multiprocessing queue.

Make sure (num_cores >= num_workers * num_threads) to avoid locks.

Example for 8 Cores (4 workers with 2 threads each),
5 experiments with 5 runs each:

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
python -m src.submit presented run2 --num_workers 4 --num_threads 2
"""

import argparse
from pathlib import Path

from src.rl.eval_exp import check_experiment, read_experiment_list
from src.rl.experiment import Experiment
from src.utils.config_mapping import load_experiment_config
from src.utils.multiproc import MultiProcessor
from src.utils.torchutils import set_seed


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "multi_file_name",
        help="file with list of experiments in config/*.txt")
    parser.add_argument("multi_run_name", help="name of multi run")
    parser.add_argument("--config_dir", type=str, default="config")
    parser.add_argument("--log_dir", type=str, default="runs")
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--run_start", type=int, default=0,
                        help="start of the range (first run)")
    parser.add_argument("--run_end", type=int, default=5,
                        help="end of the range (last run +1)")
    parser.add_argument("--seed", type=int, default=-1,
                        help="seed")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="amount of workers")
    parser.add_argument(
        "--num_threads", type=int, default=0,
        help="number of threads per experiment")
    parser.add_argument(
        "--eval", help="check status only (don't run anything)",
        action="store_true")
    parser.add_argument(
        "--reset", action="store_true",
        help="force reset (delete existing experiments). without this flag "
             "the code is capable of continuing stopped runs.")

    args = parser.parse_args()

    if args.num_threads == 0:
        print("WARNING: No limit on threads (--num_threads=0). This will "
              "probably end in a deadlock unless --num_workers=1. "
              "See --help.")

    if args.seed != -1:
        set_seed(args.seed)

    # read list of experiments
    multi_run_name = args.multi_run_name
    experiments = read_experiment_list(args.multi_file_name, config_dir=args.config_dir)

    # setup multiprofessor
    mp = MultiProcessor(num_workers=args.num_workers, sleep=10)

    # loop experiments
    completed_list = []
    for exp_name in experiments:
        # loop nruns
        print(exp_name)
        for n in range(args.run_start, args.run_end):
            # load config
            config_file = (Path(args.config_dir) / "experiments" /
                           "{}.json".format(exp_name))
            config = load_experiment_config(config_file)
            is_done = check_experiment(
                exp_name, multi_run_name, n, log_dir=args.log_dir)
            if not is_done:
                # submit experiment
                task = TaskExperiment(
                    config, exp_name, multi_run_name, n, use_cuda=args.cuda,
                    log_dir=args.log_dir, reset=args.reset)
                mp.add_task(task)
            else:
                completed_list.append((exp_name, multi_run_name, n))
                print("Is already done")

    print("Tasks: {}".format(mp.get_num_tasks()))
    if args.eval:
        mp.close()
        return

    # run all experiments
    results = mp.run()
    # check results
    print("analyzing results...")
    emptys = 0
    for result in results:
        if result is None:
            # experiment failed (stacktrace is somewhere in the log / stdout)
            emptys += 1
            continue
        # # don't need to process results here, use src/eval_list.py
        # metrics, metrics_eval = result
    print("{} empty results AKA errors".format(emptys))


class TaskExperiment(object):
    """Run experiment in multitask"""

    def __init__(
            self, config, exp_name, run_name, run_number, use_cuda=False,
            log_dir="runs", num_threads=0, reset=False):
        self.use_cuda = use_cuda
        self.run_number = run_number
        self.run_name = run_name
        self.exp_name = exp_name
        self.config = config
        self.log_dir = log_dir
        self.num_threads = num_threads
        self.reset = reset

    def __call__(self):
        exp = Experiment(
            self.config, self.exp_name, self.run_name, logger=None,
            log_dir=self.log_dir,
            run_number=self.run_number, reset=self.reset,
            reload=not self.reset, use_cuda=self.use_cuda, print_every=100,
            verbose=False, num_threads=self.num_threads)
        metrics, metrics_eval = exp.run_training()
        exp.close()
        return metrics, metrics_eval


if __name__ == '__main__':
    main()
