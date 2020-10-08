"""Create jobs for pool"""
import argparse
import os
from pathlib import Path


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
    parser.add_argument("--num_runs", type=int, default=5,
                        help="amount of runs")

    args = parser.parse_args()

    # read list of experiments
    multi_run_name = args.multi_run_name
    exp_list_file = Path(args.config_dir) / "lists" / "{}.txt".format(
        args.multi_file_name)
    experiments = exp_list_file.open("rt", encoding="utf8").read().splitlines()

    os.makedirs("_jobs/logs", exist_ok=True)
    os.makedirs("_jobs/psingle", exist_ok=True)
    os.makedirs("_jobs/pmulti", exist_ok=True)

    # loop experiments
    for exp_name in experiments:
        # create job
        file = "_jobs/{}.sh".format(exp_name)
        content = """#!/bin/bash

#PBS -N {0}
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=8,mem=20gb,walltime=24:00:00
#PBS -q student
#PBS -m aef
#PBS -M youremail@informatik.uni-freiburg.de
#PBS -j oe
#PBS -t 1-1%1
#PBS -o _jobs/logs/{0}
#PBS -e _jobs/logs/{0}

cd /misc/lmbraid19/gings/repos/rlproject
source ~/raid/venvs/venvcv/bin/activate

python -m src.run --reset {1} {2} --run_start 0 --run_end {3}
""".format(exp_name, exp_name, multi_run_name, args.num_runs)
        open(file, "wt").write(content)
        print("created {} with {} runs".format(file, args.num_runs))

        # create another job
        file = "_jobs/psingle/{}.sh".format(exp_name)
        content = """#!/bin/bash

        cd ~/repos/rlproject
        source ~/venvpool/bin/activate

        python -m src.run --reset {1} {2} --run_start 0 --run_end {3}
                """.format(exp_name, exp_name, multi_run_name, args.num_runs)
        open(file, "wt").write(content)
        print("created {} with {} runs".format(file, args.num_runs))

        # create multi job
        file = "_jobs/pmulti/{}.sh".format(exp_name)
        content = """#!/bin/bash
    
cd ~/repos/rlproject
source ~/venvpool/bin/activate

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

echo "{1}" > config/lists/tmp_{1}.txt 

python -m src.submit tmp_{1} {2} --reset --run_start 0 --run_end {3} \\
    --num_workers 4 --num_threads 2
        """.format(exp_name, exp_name, multi_run_name, args.num_runs)
        open(file, "wt").write(content)
        print("created {} with {} runs".format(file, args.num_runs))


if __name__ == '__main__':
    main()
