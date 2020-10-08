"""Matplotlib setup / parameters / helper functions"""
import errno
import os
import subprocess

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import style


STYLE = "default"
COLOR_MAP = "hsv"
FIGSIZE = (17, 12)
FONTSIZE = 16
FONTSIZE_LEGEND = 14
ALPHA_LEGEND = 0.2
COLOR_MEAN = 'black'
COLOR_SUCCESS = 'red'
CAPSIZE_ERRORBAR = 10

style.use(STYLE)
mpl.rcParams.update({
    'font.size': FONTSIZE})

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

NAME_MAP = {
    "vanilla-v0": "Baseline",
    "vanilla-v6-baseline-step-eval5k": "Best (eval @5K)",
    "vanilla-v6-baseline-step-eval20k": "Best (eval @20K)",
    "vanilla-v6-baseline-step-train": "Best (train score >= 400)",
    "vanilla-v6-baseline-time-eval5k": "Best + shaped reward (eval @5K)",
}


def get_color(num):
    # cycle colors from standard matplotlib color list
    return color_cycle[num % len(color_cycle)]


def reduce_lightness(color_, lightness=1):
    color_ = colors.to_rgba(color_, alpha=1)
    color_ = [a * lightness if m < 3 else a for m, a in enumerate(color_)]
    return color_


def get_color_gradient(num, max_num):
    # cycle colors uniformly in a gradient colormap
    cmap = plt.get_cmap(COLOR_MAP)
    denom = (max_num - 1) if max_num > 1 else 1
    factor = 0.95  # for repeating color maps to not repeat too much
    lightness = 0.8  # also a little darker
    return reduce_lightness(cmap(num / denom * factor), lightness)


def get_color_adaptive(num, max_num):
    # color cycle if few runs, gradient if more runs. this avoids reusing
    # the same color for different runs
    if max_num < len(color_cycle):
        return get_color(num)
    else:
        return get_color_gradient(num, max_num)


def setup_latex_return_str_formatter(disable_tex=False):
    # check latex available
    if disable_tex:
        print("LATEX deactivated.")
        tex = False
    else:
        tex = True
        try:
            devnull = open(os.devnull, 'w')
            subprocess.call(["latex", "-version"],
                            stdout=devnull, stderr=devnull)
        except OSError as e:
            if e.errno == errno.ENOENT:
                print("Latex not found, deactivating...")
                tex = False
            else:
                # Something else went wrong while trying to run latex
                raise
        if tex:
            mpl.rc('text', usetex=tex)

    def format_text(text: str):
        if tex:
            escapees = ["_", "#", "&"]
            for escapee in escapees:
                text = text.replace(escapee, "\\{}".format(escapee))
            ds = [(">=", "$\\geq$")]
            for d1, d2 in ds:
                text = text.replace(d1, d2)
            return text
        return text

    return format_text
