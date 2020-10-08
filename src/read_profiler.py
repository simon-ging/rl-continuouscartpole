"""Reads the output from a profiling scripts.

First run the profiler:

python -m cProfile -s cumtime -o profiling/dqnprofile.pstats \
    -m src.run dqn-baseline test1 --cpu --reset --max_steps 200

Then this script:

python -m src.read_profiler profiling/dqnprofile.pstats
"""

import argparse
import pstats


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", type=str)
    args = parser.parse_args()

    file = args.file
    p = pstats.Stats(file)
    # p.strip_dirs()
    p.sort_stats("cumtime")
    p.print_stats(20)


if __name__ == '__main__':
    main()
