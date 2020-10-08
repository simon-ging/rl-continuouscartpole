"""Run 1 (default) or N runs for a given experiment."""
import argparse
from pathlib import Path
from pprint import pprint

from src.rl.experiment import Experiment
from src.utils.config_mapping import load_experiment_config
from src.utils.torchutils import set_seed


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("exp_name")
    parser.add_argument("run_name")
    parser.add_argument("--config_dir", type=str, default="config")
    parser.add_argument("--log_dir", type=str, default="runs")
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--run_start", type=int, default=0,
                        help="number of the first run")
    parser.add_argument("--run_end", type=int, default=1,
                        help="number of the last run")
    parser.add_argument("--reset", action="store_true",
                        help="delete old experiment")
    parser.add_argument("--reload", action="store_true",
                        help="reload old experiment")
    parser.add_argument("--reload_episode", type=int, default=-1,
                        help="which episode to reload "
                             "(default -1=last episode)")
    parser.add_argument("--base_seed", type=int, default=0,
                        help="offset the seeds")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="overwrite max steps for debugging "
                             "(default -1=don't)")

    args = parser.parse_args()

    exp_name = args.exp_name
    run_name = args.run_name
    config_file = Path(args.config_dir) / "experiments" / "{}.json".format(
        exp_name)
    config = load_experiment_config(config_file)
    print("LOADED CONFIG {}".format(config_file))
    pprint(config)

    # exp = Experiment(
    #     config, reset=args.reset, reload=args.reload,
    #     reload_episode=args.reload_episode, max_steps=args.max_steps,
    #     use_cuda=not args.cpu)

    # run multiple experiments with different seeds
    print("Running {} runs".format(args.run_end - args.run_start))
    for n in range(args.run_start, args.run_end):
        set_seed(n + args.base_seed)
        exp = Experiment(
            config, exp_name, run_name, logger=None, log_dir=args.log_dir,
            run_number=n, reset=args.reset, reload=args.reload,
            reload_step=args.reload_episode,
            use_cuda=args.cuda, verbose=True, max_steps=args.max_steps)
        metrics, metrics_eval = exp.run_training()
        exp.close()

    # watch untrained agent
    # test_agent(exp.env, exp.agent)

    # train DQN
    # exp.train_quantized()


if __name__ == '__main__':
    main()
