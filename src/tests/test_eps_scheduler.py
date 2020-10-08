"""Epsilon scheduler test / plot"""
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.rl.epsilon_scheduler import get_eps_scheduler
from src.utils.plots import FIGSIZE, setup_latex_return_str_formatter, \
    NAME_MAP, get_color_adaptive, CAPSIZE_ERRORBAR


def main():
    # test epsilon schedulers
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, default="test")
    parser.add_argument("--disable_tex", action="store_true", help="disable latex")
    args = parser.parse_args()

    test_path = Path(args.test_dir) / "eps_schedulers"
    os.makedirs(str(test_path), exist_ok=True)

    exps = ["vanilla-v6-baseline-step-eval5k",
            "vanilla-v6-baseline-step-eval20k",
            "vanilla-v6-baseline-step-train",
            "vanilla-v6-baseline-time-eval5k"]
    means = [189.02, 197.41, 512.08, 256.03]
    stds = [28.414, 35.292, 162.597, 54.989]

    max_steps = 1000000
    opts = {
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay": 0.99995,
        "warmup": 0.05
    }
    names = ["mult_decay", "cos_decay"]
    pretty_names = ["Exponential Decay", "Cosine Decay"]
    scheds = []
    for n in names:
        scheds.append(get_eps_scheduler({
            "name": n, **opts}, max_steps))

    xr = np.arange(max_steps)
    every = 10
    xss = []
    epsss = []
    format_text = setup_latex_return_str_formatter(disable_tex=args.disable_tex)

    def plot_exps(epss_):
        for k, (exp, mean, std) in enumerate(zip(exps, means, stds)):
            color = get_color_adaptive(k + 1, len(exps) + 1)
            eps_ = int(mean * 1000 / every)
            plt.errorbar(
                mean, epss_[eps_], yerr=0, xerr=std, c=color, marker='x',
                linestyle='-', ms=15, capsize=CAPSIZE_ERRORBAR, alpha=1,
                label=format_text(NAME_MAP[exp]))

    for name, sched, pretty_name in zip(names, scheds, pretty_names):
        sched.reset()
        epss = []
        xs = []
        print("---------- {}".format(name))
        print("Accumulate stats")
        for i in range(max_steps):
            eps = sched.get_epsilon()
            sched.step()
            if i % every == 0:
                epss.append(eps)
                xs.append(xr[i] / 1000)
        xss.append(xs)
        epsss.append(epss)

        print("Plot")
        plt.figure(figsize=FIGSIZE)
        plt.plot(xs, epss, label=format_text(pretty_name))
        plt.grid()
        plt.title(format_text(
            "{} $\epsilon$-scheduler and mean+std "
            "experiment convergence step".format(pretty_name)))
        plt.xlabel("K steps")
        plt.ylabel("$\epsilon$")
        plot_exps(epss)

        plt.legend()
        plt.savefig(str(test_path / "{}.png".format(name)))
        plt.close()

    plt.figure(figsize=FIGSIZE)
    for name, xs, epss in zip(names, xss, epsss):
        plt.plot(xs, epss, label=format_text(name))
    plt.grid()
    plt.legend()
    plt.xlabel("K steps")
    plt.ylabel("$\epsilon$")
    plt.title("$\epsilon$-scheduler comparison")
    filename = str(test_path / "all.png")
    plt.savefig(filename)
    plt.close()
    print("Epsilon scheduler plotted to {}".format(filename))


if __name__ == '__main__':
    main()
