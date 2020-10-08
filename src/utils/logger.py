"""Easy logging to stdout and file simultanously"""

import datetime
import logging
import sys
from pathlib import Path


def get_logger(logdir, name, filename="run", with_file=True) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        # "%(name)s %(asctime)s %(levelname)s %(message)s",
        # datefmt="%m%d %H%M%S")
        "%(asctime)s %(levelname)s %(message)s", datefmt="%m-%d %H:%M:%S")

    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")

    if with_file:
        file_path = str(Path(logdir) / "{}_{}.log".format(filename, ts))
        file_hdlr = logging.FileHandler(file_path)
        file_hdlr.setFormatter(formatter)
        logger.addHandler(file_hdlr)

    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(formatter)
    logger.addHandler(strm_hdlr)

    return logger


def close_logger(logger: logging.Logger):
    x = list(logger.handlers)
    for i in x:
        logger.removeHandler(i)
        i.flush()
        i.close()


def logger_info(logger: logging.Logger):
    print(logger.name)
    x = list(logger.handlers)
    for i in x:
        handler_str = "Handler {} Type {}".format(i.name, type(i))
        print(handler_str)
