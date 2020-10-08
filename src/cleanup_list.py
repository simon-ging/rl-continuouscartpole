"""Clean up experiment folders:
- delete all models.pth/scores.json except best and last model

Will not delete anything unless run with --delete
"""
import argparse
import glob
import os
from pathlib import Path

from src.rl.eval_exp import check_all_experiments, get_exp_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "multi_file_name",
        help="file with list of experiments in config/lists/*.txt")
    parser.add_argument("multi_run_name", help="name of multi run")
    parser.add_argument("--config_dir", type=str, default="config")
    parser.add_argument("--log_dir", type=str, default="runs")
    parser.add_argument("--delete", action="store_true")

    args = parser.parse_args()
    log_dir = args.log_dir

    # read list of experiments
    multi_run_name = args.multi_run_name
    exp_list_file = Path(args.config_dir) / "lists" / "{}.txt".format(
        args.multi_file_name)
    experiments = exp_list_file.open("rt", encoding="utf8").read().splitlines()

    # find all relevant experiments
    exps = check_all_experiments(
        experiments, multi_run_name, log_dir=log_dir,
        use_unfinished=True, ignore_agent=True, config_dir=args.config_dir)

    for n_exp, (exp_name, content) in enumerate(exps.items()):
        print("---------- {}: {} runs".format(exp_name, len(content)))

        for run_num, run_info in content.items():
            last_step = run_info["last_step"]
            best_step = run_info["best_step"]
            # print( "run {} last {} best {}".format(
            #     run_num, last_step, best_step))

            # get score files
            exp_dir = get_exp_dir(
                exp_name, multi_run_name, run_num, log_dir=args.log_dir)
            score_files = sorted(glob.glob(str(exp_dir / "scores_*.json")))

            for score_file in score_files:
                step = int(score_file.split("scores_")[-1].split(".json")[0])
                if step == last_step or step == best_step:
                    # save best and last step infos
                    continue
                if args.delete:
                    print("deleting {}".format(score_file))
                    os.remove(score_file)
                else:
                    print("use --delete to remove {}".format(score_file))

            # get agent files
            agent_files = sorted(glob.glob(str(exp_dir / "agent_*.pth")))
            for agent_file in agent_files:
                step = int(agent_file.split("agent_")[-1].split(".pth")[0])
                if step == last_step or step == best_step:
                    # save both best and last agent
                    continue
                if args.delete:
                    print("deleting {}".format(agent_file))
                    os.remove(agent_file)
                else:
                    print("use --delete to remove {}".format(agent_file))


if __name__ == '__main__':
    main()
