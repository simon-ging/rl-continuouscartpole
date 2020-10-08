"""Experiment class"""

import glob
import json
import os
from timeit import default_timer as timer
from typing import Dict

import numpy as np
import shutil
import time
import torch
from gym import Env
from gym.spaces import Box, Space, Discrete

from src.agent.dqn_agent import create_dqn_agent
from src.rl.epsilon_scheduler import get_eps_scheduler
from src.rl.eval_exp import find_last_step, find_best_step_eval
from src.rl.get_filenames import get_exp_dir, get_filename_config, \
    get_filename_infos, get_exp_id, get_filename_agent
from src.rl.observ_transform import ObservTransformer
from src.utils.action_quantization import Quantizer1D, QuantizerND
from src.utils.config_mapping import get_reward_fn, get_env
from src.utils.jsonencoder import to_json
from src.utils.logger import get_logger, close_logger
from src.utils.stats import Stats


class Config(object):
    """Holds experiment configuration"""

    def __init__(self, config: dict):
        # ***** Load config

        # quantizer
        cfg_quant = config["quantizer"]
        self.quantize_flag = cfg_quant["quantize_flag"]
        self.quantize_num = cfg_quant.get("quantize_num")
        # optional field for lists (at some point all should be)
        self.quantize_nums = cfg_quant.get("quantize_nums")

        # training
        cfg_train = config["training"]
        self.max_steps = cfg_train["max_steps"]
        self.batch_size = cfg_train["batch_size"]
        self.discount_factor = cfg_train["discount_factor"]
        self.reward_function_name = cfg_train["reward_function"]
        self.max_steps_per_episode = cfg_train["max_steps_per_episode"]
        self.max_steps_per_episode_eval = cfg_train[
            "max_steps_per_episode_eval"]
        self.avg_score_episodes = cfg_train["avg_score_episodes"]

        # optimizer
        self.cfg_optimizer = config["optimizer"]

        # epsilon scheduler: allow not existing (optional) for continuous
        # methods
        self.cfg_eps_scheduler = config.get("eps_scheduler")

        # algorithm / agent
        self.cfg_agent = config["algorithm"]

        # evaluation
        cfg_eval = config["evaluation"]
        self.eval_every_n_steps = cfg_eval["eval_every_n_steps"]
        self.eval_n_episodes = cfg_eval["eval_n_episodes"]
        self.solve_score = cfg_eval["solve_score"]
        # special case: this field overrides solve_score but is optional
        # for convenience so not all configs have to be modified.
        self.solve_score_train = cfg_eval.get("train_solve_score")

        # saving
        cfg_saving = config["saving"]
        self.save_every_n_steps = cfg_saving["save_every_n_steps"]
        self.save_last = cfg_saving["save_last"]
        self.save_best = cfg_saving["save_best"]

        # environment
        cfg_env = config["environment"]
        self.env_name = cfg_env["env"]
        self.env_observ_transform = cfg_env["env_observ_transform"]


class Experiment(object):
    def __init__(
            self, config: Dict, exp_name, run_name, logger=None,
            log_dir="runs",
            run_number=0, reset=False, reload=False, reload_step=-1,
            reload_best=False, use_cuda=True, verbose=True, print_every=1,
            num_threads=0, max_steps=-1):
        """
        Args:
            config (dict): configuration (see
                config/experiments/dqn_baseline.json) for an example
            run_name (str): name the run
            logger (logging.Logger, optional): pass a logger object or if None,
                will create a new logger
            log_dir (str, optional): base directory to save logs, models etc.
            run_number (int, optional): run number to be able to run the same
                experiment multiple times
            reset (bool, optional): reset and delete the old experiment
            reload (bool, optional): reload stats and models from checkpoint
            reload_step (int, optional): which step to reload (default
                -1 = reload latest step)
            reload_best (bool, optional): if reload == True and
                reload_step == -1, loads the best instead of last episode
            use_cuda (bool, optional): use cuda if available (default True)
            verbose (bool, optional): disable some of the logging
            print_every (int, optional): only print some of the training steps
                (default 1 = print all steps)
            num_threads (int, optional): how many threads to use, (default 0=
                let torch decide)
            max_steps (int, optional): overwrite max steps manually for e.g.
                performance test. default -1 = dont overwrite
        """
        # ---------- initalize blank variables ----------
        self.current_episode = 0
        self.current_time_eval = 0  # total time (sec) spent evaluating
        self.current_time_total = 0  # total time for everything
        self.current_step = 0  # total step counter
        self.is_solved = False  # is already solved
        self.metrics = {
            "episodes": [],  # list of int: episodes trained
            "scores": [],  # list of float: total rewards
            "avg_scores": [],  # list of float: average total rewards over last
            # X episodes
            "time_total": [],  # list of float: total time spent
            "time_eval": [],  # list of float: eval time spent
            "step": [],  # current total step
            "other": []  # list of tuple: with other info (currently epsilon)
        }
        self.metrics_eval = {
            "episodes": [],  # list of int: episodes evaluated
            "steps": [],  # list of int: steps evaluated
            "scores_list": [],  # list per episode evaluated:
            # list of total rewards gained over N evaluations in this episode
            "avg_scores": []  # list of float: average over the N evaluations
            # done this episode
        }
        self.best_eval_score = -np.finfo(np.float32).max
        self.best_train_avg_score = -np.finfo(np.float32).max

        # ---------- read arguments, setup directories ----------
        self.exp_name = exp_name
        self.run_name = run_name
        self.run_number = run_number
        self.log_dir = log_dir
        self.exp_id = get_exp_id(self.exp_name, self.run_name, self.run_number)
        self.exp_dir = get_exp_dir(
            self.exp_name, self.run_name, self.run_number,
            log_dir=self.log_dir)
        self.use_cuda = use_cuda
        self.verbose = verbose
        self.print_every = print_every

        # make sure folder exists
        if reset or not self.exp_dir.is_dir():
            # on reset, delete previous experiment
            shutil.rmtree(str(self.exp_dir), ignore_errors=True)
            time.sleep(.1)
            os.makedirs(str(self.exp_dir), exist_ok=False)
        elif self.exp_dir.is_dir() and not reload:
            raise ValueError(
                "Experiment already exists in {}. "
                "Use either --reset or --reload".format(
                    self.exp_dir
                ))

        # create logger
        if logger is None:
            self.logger = get_logger(self.exp_dir, self.exp_id)
            self.logger_is_external = False
        else:
            self.logger = logger
            self.logger_is_external = True

        # output device info
        if self.verbose:
            self.logger.info("cuda available: {} use it: {}".format(
                torch.cuda.is_available(), use_cuda))

        # load config
        self.cfg = Config(config)

        # overwrite max steps
        if max_steps > -1:
            self.cfg.max_steps = max_steps

        # save config to file
        with open(str(get_filename_config(
                self.exp_name, self.run_name, self.run_number,
                log_dir=self.log_dir)), "wt", encoding="utf8") as fh:
            json.dump(config, fh, indent=4, sort_keys=True)

        # load reward function
        self.reward_function = get_reward_fn(self.cfg.reward_function_name)

        if num_threads > 0:
            torch.set_num_threads(num_threads)
            self.logger.info("Set num threads to {}".format(num_threads))

        # ---------- setup environment, quantizer ----------

        # get environment class
        env_class = get_env(self.cfg.env_name)
        if self.cfg.env_name == "default_cartpole":
            # create training environment with shaped reward (if requested)
            self.env = env_class(
                reward_function=self.reward_function)  # type: Env
            # create evaluation environment, always with sparse reward
            self.env_eval = env_class(reward_function=None)
        else:
            # gym envs don't have this reward function keyword
            self.env = env_class
            self.env_eval = env_class
        self.action_space = self.env.action_space

        # create input transformer
        self.observ_transformer = ObservTransformer(
            self.env.observation_space,
            transform_type=self.cfg.env_observ_transform)
        self.observ_space = self.observ_transformer.observ_space  # type: Box
        self.observ_size = self.observ_space.shape[0]

        # check action space type
        if type(self.action_space) is Box:
            # box - continuous actions
            if self.cfg.quantize_flag:
                # check action space is 1D
                if self.action_space.shape == (1,):
                    # regular behaviour, quantize actions
                    self.action_size = self.cfg.quantize_num
                    self.quantizer = Quantizer1D(
                        self.action_space, self.action_size)
                    self.action_dim = 1
                else:
                    # multiple dim actions
                    self.action_size = self.cfg.quantize_num
                    self.action_dim = self.action_space.shape[0]
                    self.quantizer = QuantizerND(
                        self.action_space, self.cfg.quantize_num)

                self.eps_scheduler = get_eps_scheduler(
                    config["eps_scheduler"], self.cfg.max_steps)
                self.do_quantization = True
            else:
                raise NotImplementedError(
                    "Continuous actions without quantizer not implemented")
        elif type(self.action_space) is Discrete:
            # discrete action space, no quantizer
            self.action_size = self.action_space.n
            self.action_dim = 1
            self.quantizer = None
            self.do_quantization = False
            self.eps_scheduler = get_eps_scheduler(
                config["eps_scheduler"], self.cfg.max_steps)
        else:
            # unknown action space
            raise NotImplementedError(
                "unknown gym.Space: {}".format(type(self.action_space)))

        if self.verbose:
            self.logger.info(
                "Created experiment {} #{} environment {} {} "
                "observs {} actions {}".format(
                    self.exp_id, self.run_number,
                    self.cfg.env_name, self.env, self.observ_space,
                    self.action_space))

        # ---------- create agent ----------

        # create agent
        self.algo_name = self.cfg.cfg_agent["name"]
        if self.algo_name == "dqn":
            # create DQN agent
            self.agent = create_dqn_agent(
                self.cfg.cfg_agent, self.cfg.cfg_optimizer,
                self.observ_size, self.action_size, self.action_dim,
                self.cfg.batch_size, self.cfg.discount_factor,
                self.quantizer, use_cuda=self.use_cuda, verbose=self.verbose)
            if self.verbose:
                self.logger.info("---------- Optimizer: {}".format(
                    self.agent.optimizer))
        else:
            raise NotImplementedError(
                "algorithm {} not implemented".format(
                    self.algo_name))

        # reload agent state dict if needed
        if reload:
            if reload_step == -1:
                # find last episode
                reload_step = find_last_step(
                    self.exp_name, self.run_name, self.run_number,
                    log_dir=self.log_dir)
                if reload_step == -1:
                    # nothing found - start from scratch
                    self.logger.warning("*" * 20)
                    self.logger.warning(
                        "Reloading experiment but no files found in {} - "
                        "Starting from scratch".format(
                            self.exp_dir))
                    self.logger.warning("*" * 20)
                elif reload_best:
                    # something found but requested to load best instead of
                    # last
                    best_step, _, _ = find_best_step_eval(
                        self.exp_name, self.run_name, self.run_number,
                        log_dir=self.log_dir)
                    # self.logger.info(
                    #     "Reloading BEST step {} instead of "
                    #     "LAST step {}".format(
                    #         best_step, reload_step))
                    reload_step = best_step
            if reload_step > -1:
                # reload infos only if models exist
                self.logger.info("{}_{}_{} RELOADING step {}".format(
                    self.exp_name, self.run_name, self.run_number,
                    reload_step))
                self.load_infos(reload_step)
                self.load_agent(reload_step)

    def run_training(self):
        """ Train agent on environment with given settings.

        Returns:
            metrics dictionary, metrics_eval dictionary
        """
        if self.cfg.quantize_flag:
            # epsilon scheduler is only relevant for quantized actions
            eps = self.eps_scheduler.reset(start_step=self.current_step)
        else:
            eps = 0
        start_episode = self.current_episode
        start_step = self.current_step
        # setup timing
        time_start = self.current_time_total
        time_lap = timer()
        # start training
        self.logger.info("Training from {} to {}".format(
            self.current_step, self.cfg.max_steps))
        # loop over episodes
        while self.current_step < self.cfg.max_steps and not self.is_solved:
            # reset observation and score
            observ = self.env.reset()
            # transform observation
            observ = self.observ_transformer.transform_observ(observ)
            score = 0
            n_steps = 0
            for t in range(0, self.cfg.max_steps_per_episode):
                if self.verbose and t % 20 == 0:
                    print("step {} / {}\r".format(t, self.cfg.max_steps_per_episode), end="")
                # get action from environment
                if self.cfg.quantize_flag:
                    # agents in discrete environment need epsilon parameter
                    # to sample actions
                    action = self.agent.act(observ, eps=eps)
                    # step epsilon scheduler for quantized actions
                    eps = self.eps_scheduler.step()
                else:
                    # sample continuous action
                    action = self.agent.act(observ)
                # do one step in the environment
                next_observ, reward, done, _ = self.env.step(action)
                # transform observation
                next_observ = self.observ_transformer.transform_observ(
                    next_observ)
                # do one optimization step of the agent
                self.agent.step(
                    observ, action, reward, next_observ, done,
                    self.current_step)
                # accumulate score, change observation, break if done
                score += reward
                observ = next_observ
                self.current_step += 1
                n_steps += 1

                # ---------- eval, saving etc ----------

                # evaluate: evaluation steps have one higher episode step count
                # (looks nicer to have train (0,1,2,3,4) and eval (5)
                is_best = False
                if (self.current_step % self.cfg.eval_every_n_steps == 0
                        and self.cfg.eval_every_n_steps > -1):
                    is_best_eval = self.eval_step()
                    # only use eval score for is_best if train solve score
                    # is not set
                    if self.cfg.solve_score_train is None:
                        is_best = is_best_eval
                    self.save_infos()
                    self.save_agent()

                # save at every n steps and the best configurations
                if (self.current_step % self.cfg.save_every_n_steps == 0 and
                    self.cfg.save_every_n_steps > -1) or (
                        is_best and self.cfg.save_best):
                    self.logger.info(
                        "***** SAVING: step {} is_best {} "
                        "best eval score {:.3f} "
                        "best train score {:.3f}".format(
                            self.current_step, is_best,
                            self.best_eval_score, self.best_train_avg_score
                        ))
                    # if save every n steps is -1 we want to delete all the old
                    # saves that are not the best anymore
                    if self.cfg.save_every_n_steps == -1 and is_best:
                        old_agents = glob.glob(str(self.exp_dir / "*.pth"))
                        for f in old_agents:
                            agent_num = int(f.split("_")[-1].split(".pth")[0])
                            if agent_num == self.current_step:
                                continue
                            os.remove(f)
                            self.logger.info("Removing {}".format(f))
                    self.save_infos()
                    self.save_agent()

                    # check if problem is solved
                    if (is_best and self.best_eval_score >=
                            self.cfg.solve_score):
                        self.logger.info(
                            "\nProblem solved (eval metric)! "
                            "eval score {:.3f}\n".format(
                                self.best_eval_score))
                        self.is_solved = True

                # stop episode if reward is terminal
                if done:
                    break

            # ---------- episode is done ----------

            # calculate training metrics for this episode
            time_now = timer()
            time_deff = time_now - time_lap
            self.current_time_total += time_deff
            time_lap = time_now
            # print("time diff", time_deff, "total", self.current_time_total)

            self.metrics["episodes"].append(self.current_episode)
            self.metrics["scores"].append(score)
            avg_score = np.mean(self.metrics["scores"][
                                -self.cfg.avg_score_episodes:])
            if avg_score > self.best_train_avg_score:
                # train score is the metric for is_best
                self.best_train_avg_score = avg_score
            if self.cfg.solve_score_train is not None:
                # solve train score is the metric
                if avg_score >= self.cfg.solve_score_train:
                    # done
                    self.logger.info(
                        "\nProblem solved (train metric)! "
                        "train score {:.3f} "
                        "best eval score {:.3f}\n".format(
                            avg_score, self.best_eval_score))
                    self.is_solved = True

            self.metrics["avg_scores"].append(avg_score)
            self.metrics["time_total"].append(self.current_time_total)
            self.metrics["time_eval"].append(self.current_time_eval)
            self.metrics["step"].append(self.current_step)
            # field for additional metrics (tuple)
            self.metrics["other"].append((eps,))

            # output episode info
            if self.current_episode % self.print_every == 0 or \
                    self.current_step >= self.cfg.max_steps - 1:
                # determine some additional log information depending on
                # quantized action environment or not
                if self.cfg.quantize_flag:
                    # add epsilon to log
                    other_info = "eps {:.3f}".format(eps)
                else:
                    # nothing (yet)
                    other_info = ""

                self.logger.info(
                    "{}-{}-{} E{:4} S{:7}/{:7} ({:3}) {} score {:8.3f} "
                    "avg_score {:8.3f}".format(
                        self.exp_name, self.run_name, self.run_number,
                        self.current_episode, self.current_step,
                        self.cfg.max_steps, n_steps,
                        other_info, score, avg_score))

            # increase episode step count
            self.current_episode += 1

            # check if training is done: then do another eval and save
            if self.current_step >= self.cfg.max_steps or self.is_solved:
                is_best = self.eval_step()
                if self.cfg.save_last or self.cfg.save_best and is_best:
                    if self.verbose:
                        self.logger.info(
                            "***** SAVING: step {} is_best {}".format(
                                self.current_step, is_best))
                    # if save every n steps is -1 AND the last agent is the
                    # best, we want to delete all the old
                    # saves that are not the best anymore
                    if self.cfg.save_every_n_steps == -1 and is_best:
                        old_agents = glob.glob(str(self.exp_dir / "*.pth"))
                        for f in old_agents:
                            if f == "agent_{}.pth".format(self.current_step):
                                continue
                            os.remove(f)
                            if self.verbose:
                                self.logger.info("Removing {}".format(f))
                    self.save_infos()
                    self.save_agent()

        self.logger.info("Done training.")

        # output time deltas
        delta_ep = self.current_episode - start_episode
        delta_step = self.current_step - start_step
        delta_time = self.current_time_total - time_start
        if delta_step > 0:
            self.logger.info(
                "{}-{}-{} training {} episodes {} step took {:.3f}s,"
                " {:.3f}ms per step, cuda {}, eval time {:.3f}s".format(
                    self.exp_name, self.run_name, self.run_number, delta_ep,
                    delta_step, delta_time, 1000 * delta_time / delta_step,
                    self.use_cuda, self.current_time_eval))
        return self.metrics, self.metrics_eval

    def eval_step(self):
        """Do one step of evaluation and metrics update

        Returns:
            is_best (bool) whether this is the best agent
        """
        # evaluate
        time_lap = timer()
        eval_scores = self.run_evaluation()
        eval_score = np.mean(eval_scores)
        is_best = False
        if eval_score >= self.best_eval_score:
            is_best = True
            self.best_eval_score = eval_score
        time_delta = timer() - time_lap
        self.current_time_eval += time_delta
        self.metrics_eval["episodes"].append(self.current_episode)
        self.metrics_eval["steps"].append(self.current_step)
        self.metrics_eval["scores_list"].append(eval_scores)
        self.metrics_eval["avg_scores"].append(eval_score)
        return is_best

    def run_evaluation(self, num_ep=-1, num_steps=-1, print_steps=False):
        """Evaluate agent in the eval environment (always sparse reward).

        Args:
            num_ep: How many episodes to evaluate (default -1 use experiment
                settings)
            num_steps: Max steps to terminate episode (default -1
                use experiment settings)

        Returns:
            list of score (total reward) per episode
        """
        if num_ep == -1:
            num_ep = self.cfg.eval_n_episodes
        if num_steps == -1:
            num_steps = self.cfg.max_steps_per_episode_eval
        scores = []
        for ep in range(0, num_ep):
            if self.verbose:
                print("eval {:4}/{:4}".format(ep, num_ep),
                      end="\r")
            observ = self.env_eval.reset()
            # transform observation
            observ = self.observ_transformer.transform_observ(observ)

            score = 0
            for t in range(0, num_steps):
                if print_steps and t % 50 == 0:
                    print("step {}/{}".format(t, num_steps), end="\r")
                if self.cfg.quantize_flag:
                    # set epsilon to 0 for evaluation of quantized action
                    # space methods
                    action = self.agent.act(observ, eps=0)
                else:
                    # sample action in continuous environment
                    action = self.agent.act(observ)
                # step environment
                next_observ, reward, done, _ = self.env_eval.step(action)
                # transform observation
                next_observ = self.observ_transformer.transform_observ(
                    next_observ)

                # do NOT step the agent
                # update scores, change observation and break if done
                score += reward
                observ = next_observ
                if done:
                    break
            # collect scores over episodes
            scores.append(score)
        # print some stats on the evaluation
        s = Stats(scores)
        self.logger.info(
            "{}-{}-{} E{:4} S{:7}/{:7} Eval {:7.3f} +- {:7.3f}".format(
                self.exp_name, self.run_name, self.run_number,
                self.current_episode, self.current_step, self.cfg.max_steps,
                s.mean, s.stddev))

        # return total scores over episodes
        return scores

    def save_agent(self):
        """Save state dict of agent """
        filename_agent = get_filename_agent(
            self.exp_name, self.run_name, self.run_number,
            self.current_step, log_dir=self.log_dir)
        # save agent state dict (model, optimizer etc)
        agent_dict = self.agent.get_state_dict()
        with filename_agent.open("wb") as fh:
            torch.save(agent_dict, fh)
        # if self.verbose:
        #     self.logger.info("Saved agent to {}".format(filename_agent))

    def save_infos(self):
        # save infos to json
        filename_infos = get_filename_infos(
            self.exp_name, self.run_name, self.run_number,
            self.current_step, log_dir=self.log_dir)
        info_dict = {
            "episode": self.current_episode,
            "metrics": self.metrics,
            "metrics_eval": self.metrics_eval,
            "current_time_eval": self.current_time_eval,
            "current_time_total": self.current_time_total,
            "current_step": self.current_step,
            "best_eval_score": float(self.best_eval_score),
            "best_train_avg_score": float(self.best_train_avg_score),
            "is_solved": self.is_solved
        }
        with filename_infos.open("wt", encoding="utf8") as fh:
            # json.dump(info_dict, fh, indent=4, sort_keys=True)
            fh.write(to_json(info_dict))
        # if self.verbose:
        #     self.logger.info("Saved stats to {}".format(filename_infos))

    def load_agent(self, step):
        """Load state dict of agent

        Args:
            step(int, optional): what step to load, default -1: current step
        """
        if step == -1:
            step = self.current_step
        filename_agent = get_filename_agent(
            self.exp_name, self.run_name, self.run_number,
            step, log_dir=self.log_dir)

        # load agent
        if self.verbose:
            self.logger.info("Loading agent from {}".format(filename_agent))
        try:
            with filename_agent.open("rb") as fh:
                agent_dict = torch.load(fh)
                self.agent.set_state_dict(agent_dict)
        except FileNotFoundError:
            self.logger.info(
                "WARNING: {}_{}_{} Could not load agent model.\n{}\n"
                "Weights are not loaded. If you only read the "
                "stats of this experiment, this is fine.".format(
                    self.exp_name, self.run_name, self.run_number,
                    filename_agent))

    def load_infos(self, step):
        """Load info (stats etc.)
        Args:
            step(int, optional): what step to load, default -1: current step
        """
        # load infos
        try:
            filename_infos = get_filename_infos(
                self.exp_name, self.run_name, self.run_number, step,
                log_dir=self.log_dir)
            with filename_infos.open("rt", encoding="utf8") as fh:
                info_dict = json.load(fh)
        except FileNotFoundError:
            # ---------- HACK for accidentally deleting too many scores.json
            last_step = find_last_step(
                self.exp_name, self.run_name, self.run_number, self.log_dir)
            filename_infos = get_filename_infos(
                self.exp_name, self.run_name, self.run_number, last_step,
                log_dir=self.log_dir)
            with filename_infos.open("rt", encoding="utf8") as fh:
                info_dict = json.load(fh)
            if self.verbose:
                 self.logger.warning(
                    "WARNING: {}_{}_{} Reloading agent episode {} but scores episode "
                    "{}, probably some files are missing. do NOT train.".format(
                    self.exp_name, self.run_name, self.run_number,
                    step, last_step))
        if self.verbose:
            self.logger.info("Loading infos from {}".format(filename_infos))
        self.current_episode = info_dict["episode"]
        self.current_time_eval = info_dict["current_time_eval"]
        self.current_time_total = info_dict["current_time_total"]
        self.current_step = step  # info_dict["current_step"]
        self.metrics = info_dict["metrics"]
        self.metrics_eval = info_dict["metrics_eval"]
        self.best_eval_score = info_dict["best_eval_score"]
        self.best_train_avg_score = info_dict["best_train_avg_score"]
        self.is_solved = info_dict["is_solved"]

    def close(self):
        """Close the experiment. It is important to close the logger between
        multiple experiments since otherwise problems may occur:
            - same message printed multiple times
            - folders/files stay locked
        We do not want to close external loggers that have been passed to the
        __init__, since they will be closed by the caller.
        """
        if not self.logger_is_external:
            close_logger(self.logger)
