"""
Multiprocessing convenience classes for torch.
Most importantly makes sure that error stacktraces do not get lost.
"""

import subprocess
import traceback
from timeit import default_timer as timer
from typing import List

import time
import tqdm
from torch import multiprocessing  # import multiprocessing


def systemcall(call):
    pipe = subprocess.PIPE
    process = subprocess.Popen(call, stdout=pipe, stderr=pipe, shell=True)
    out, err = process.communicate()
    retcode = process.poll()
    charset = 'utf-8'
    out = out.decode(charset)
    err = err.decode(charset)
    return out, err, retcode


class Worker(multiprocessing.Process):
    def __init__(self, task_q, result_q, error_q, verbose=False):
        super().__init__()
        self.task_q = task_q
        self.result_q = result_q
        self.error_q = error_q
        self.verbose = verbose

    def run(self):
        proc_name = self.name
        while True:
            next_task = self.task_q.get()
            if next_task is None:
                # poison pill means shutdown
                if self.verbose:
                    print("{:s}: exiting".format(proc_name))
                self.task_q.task_done()
                break
            if self.verbose:
                print(str(next_task))
            try:
                result = next_task()
                pass
            except Exception as e:
                self.error_q.put((e, traceback.format_exc()))
                result = None
            self.task_q.task_done()
            self.result_q.put(result)


class MultiProcessor(object):
    """convenience class for multiprocessing jobs"""

    def __init__(
            self, num_workers=0, verbose=True, progressbar=True,
            sleep=1):
        self._num_workers = num_workers
        self.verbose = verbose
        self.progressbar = progressbar
        if self._num_workers == 0:
            self._num_workers = multiprocessing.cpu_count()
        self._tasks = multiprocessing.JoinableQueue()
        self._results = multiprocessing.Queue()
        self._errors = multiprocessing.Queue()
        self._workers = []  # type: List[Worker]
        self._num_tasks = 0
        self.total_time = 0
        self.sleep = sleep

    def get_num_tasks(self):
        return self._num_tasks

    def add_task(self, task):
        self._tasks.put(task)
        self._num_tasks += 1

    def close(self):
        self._results.close()
        self._errors.close()
        for w in self._workers:
            w.terminate()

    def run(self):
        # start N _workers
        start = timer()
        if self.verbose:
            print('Creating {:d} workers'.format(self._num_workers))
        self._workers = [Worker(self._tasks, self._results, self._errors)
                         for _ in range(self._num_workers)]
        for w in self._workers:
            w.start()

        # add poison pills for _workers
        for i in range(self._num_workers):
            self._tasks.put(None)

        # write start message
        if self.verbose:
            print("Running {:d} enqueued tasks and {:d} stop signals".format(
                self._num_tasks, self._num_workers))

        # check info on the queue, with a nice (somewhat stable) progressbar
        if self.progressbar:
            print("waiting for the task queue to be filled...")
            num_wait = 0
            while self._tasks.empty():
                time.sleep(1)
                num_wait += 1
                if num_wait >= 5:
                    break
            tasks_now = self._num_tasks + self._num_workers
            pbar = tqdm.tqdm(total=tasks_now)
            while not self._tasks.empty():
                time.sleep(self.sleep)
                tasks_before = tasks_now
                tasks_now = self._tasks.qsize()
                resolved = tasks_before - tasks_now
                pbar.set_description(
                    "~{:7d} tasks remaining...".format(tasks_now))
                pbar.update(resolved)
            pbar.close()

        # join _tasks
        if self.verbose:
            print("waiting for all tasks to finish...")
        self._tasks.join()

        # check _errors
        if self.verbose:
            print("reading error queue... ")
        num_err = 0
        while not self._errors.empty():
            e, stacktrace = self._errors.get()
            num_err += 1
            print()
            print(stacktrace)
        if num_err >= 0:
            print("{} errors, check the log.".format(num_err))
        elif self.verbose:
            print("no errors found.")

        # read _results and return them
        if self.verbose:
            print("reading results...")
        results = []
        # # this can lead to some results missing
        # while not self._results.empty():
        while self._num_tasks > 0:
            result = self._results.get()
            results.append(result)
            self._num_tasks -= 1
        stop = timer()
        self.total_time = stop - start
        if self.verbose:
            print("Operation took {:.3f}s".format(self.total_time))
        return results
