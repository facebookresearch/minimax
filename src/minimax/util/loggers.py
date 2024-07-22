"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This file is modified from 
https://github.com/openai/baselines

Licensed under the MIT License;
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://opensource.org/license/mit/
"""

import os
import sys
import json
import csv
import time
import datetime
import copy
import logging
import git
from typing import Dict

import minimax.util.checkpoint as _checkpoint_util


class KVWriter(object):
    def writekvs(self, kvs):
        raise NotImplementedError


class SeqWriter(object):
    def writeseq(self, seq):
        raise NotImplementedError


class HumanOutputFormat(KVWriter, SeqWriter):
    def __init__(self, filename_or_file):
        if isinstance(filename_or_file, str):
            self.file = open(filename_or_file, "wt")
            self.own_file = True
        else:
            assert hasattr(filename_or_file, "read"), (
                "expected file or str, got %s" % filename_or_file
            )
            self.file = filename_or_file
            self.own_file = False

    def writekvs(self, kvs):
        # Create strings for printing
        key2str = {}
        for key, val in sorted(kvs.items()):
            if hasattr(val, "__float__"):
                valstr = "%-8.3g" % val
            else:
                valstr = str(val)
            key2str[self._truncate(key)] = self._truncate(valstr)

        # Find max widths
        if len(key2str) == 0:
            print("WARNING: tried to write empty key-value dict")
            return
        else:
            keywidth = max(map(len, key2str.keys()))
            valwidth = max(map(len, key2str.values()))

        # Write out the data
        dashes = "-" * (keywidth + valwidth + 7)
        lines = [dashes]
        for key, val in sorted(key2str.items(), key=lambda kv: kv[0].lower()):
            lines.append(
                "| %s%s | %s%s |"
                % (
                    key,
                    " " * (keywidth - len(key)),
                    val,
                    " " * (valwidth - len(val)),
                )
            )
        lines.append(dashes)
        self.file.write("\n".join(lines) + "\n")

        # Flush the output to the file
        self.file.flush()

    def _truncate(self, s):
        maxlen = 64
        return s[: maxlen - 3] + "..." if len(s) > maxlen else s

    def writeseq(self, seq):
        seq = list(seq)
        for i, elem in enumerate(seq):
            self.file.write(elem)
            if i < len(seq) - 1:  # add space unless this is the last one
                self.file.write(" ")
        self.file.write("\n")
        self.file.flush()

    def close(self):
        if self.own_file:
            self.file.close()


def gather_metadata() -> Dict:
    date_start = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    # Gathering git metadata.
    try:
        import git

        try:
            repo = git.Repo(search_parent_directories=True)
            git_sha = repo.commit().hexsha
            git_data = dict(
                commit=git_sha,
                branch=None if repo.head.is_detached else repo.active_branch.name,
                is_dirty=repo.is_dirty(),
                path=repo.git_dir,
            )
        except git.InvalidGitRepositoryError:
            git_data = None
    except ImportError:
        git_data = None

    # Gathering slurm metadata.
    if "SLURM_JOB_ID" in os.environ:
        slurm_env_keys = [k for k in os.environ if k.startswith("SLURM")]
        slurm_data = {}
        for k in slurm_env_keys:
            d_key = k.replace("SLURM_", "").replace("SLURMD_", "").lower()
            slurm_data[d_key] = os.environ[k]
    else:
        slurm_data = None
    return dict(
        date_start=date_start,
        date_end=None,
        successful=False,
        git=git_data,
        slurm=slurm_data,
        env=os.environ.copy(),
    )


class Logger:
    def __init__(
        self,
        log_dir="~/logs/minimax",
        xpid=None,
        xp_args=None,
        callback=None,
        from_last_checkpoint=False,
        verbose=False,
    ):
        # Set up checkpoint meta
        self.verbose = verbose
        if self.verbose:
            self._stdout = HumanOutputFormat(sys.stdout)

        self._callback = callback

        formatter = logging.Formatter("%(message)s")
        self._logger = logging.getLogger("logs/out")
        shandle = logging.StreamHandler()
        shandle.setFormatter(formatter)
        self._logger.addHandler(shandle)
        self._logger.setLevel(logging.INFO)

        # Set up main paths for logs and checkpoints
        self.paths = {}
        log_dir_path = os.path.expandvars(os.path.expanduser(log_dir))
        xpid_dir_path = os.path.join(log_dir_path, xpid)
        if not xpid:
            xpid = "{proc}_{unixtime}".format(
                proc=os.getpid(), unixtime=int(time.time())
            )
        if not os.path.exists(xpid_dir_path):
            self._logger.info("Creating log directory: %s", xpid_dir_path)
            os.makedirs(xpid_dir_path, exist_ok=True)
        self.paths["log_dir"] = log_dir_path
        self.paths["xpid_dir"] = xpid_dir_path
        self.paths["checkpoint"] = os.path.join(xpid_dir_path, "checkpoint.pkl")

        # Create logs.csv file
        logs_csv_path = os.path.join(xpid_dir_path, "logs.csv")
        self.paths["log_csv"] = logs_csv_path
        self._last_n_logged_lines = 0
        self._last_logged_tick = self._get_last_logged_tick()

        self.append_to_existing_logs = (
            self._last_logged_tick >= 0
            and from_last_checkpoint
            and os.path.exists(self.paths["checkpoint"])
        )
        log_mode = "a" if self.append_to_existing_logs else "w+"
        self._logfile = open(logs_csv_path, log_mode)
        self._logwriter = None

        # Create meta file
        if xp_args is not None:
            meta_path = os.path.join(xpid_dir_path, "meta.json")

            meta = gather_metadata()
            meta["config"] = dict(xp_args)
            meta["xpid"] = xpid

            self._save_metadata(meta_path, meta)

    def _save_metadata(self, meta_path, meta):
        with open(meta_path, "w") as jsonfile:
            json.dump(meta, jsonfile, indent=4, sort_keys=True)

    def _get_last_logged_tick(self):
        last_tick = -1
        logs_csv_path = self.paths["log_csv"]
        if os.path.exists(logs_csv_path):
            with open(logs_csv_path, "r") as csvfile:
                reader = csv.reader(csvfile)
                lines = list(reader)
                # Need at least two lines in order to read the last tick:
                # the first is the csv header and the second is the first line
                # of data.
                if len(lines) > 1:
                    self._last_n_logged_lines = len(lines)
                    try:
                        last_tick = int(lines[-1][0])
                    except:
                        last_tick = -1

        return last_tick

    def log(self, stats, _tick, ignore_val=None):
        if ignore_val is not None:
            stats = {k: v if v != ignore_val else None for k, v in stats.items()}

        _stats = {"_tick": _tick, "_time": time.time()}
        _stats.update(stats)
        stats = _stats

        if self._logwriter is None:
            fieldnames = list(stats.keys())
            self._logwriter = csv.DictWriter(self._logfile, fieldnames=fieldnames)

        if _tick > self._last_logged_tick or not self.append_to_existing_logs:
            if self._last_n_logged_lines == 0:
                fieldnames = list(stats.keys())
                self._logfile.write("# %s\n" % ",".join(fieldnames))
                self._logfile.flush()
                self._last_n_logged_lines = 1

            self._logwriter.writerow(stats)
            self._logfile.flush()

            if self._callback is not None:
                self._callback(stats)

        if self.verbose:
            self._stdout.writekvs(stats)

    @property
    def checkpoint_path(self):
        return self.paths["checkpoint"]

    def checkpoint(
        self, runner_state, name="checkpoint", index=None, archive_interval=None
    ):
        _checkpoint_util.safe_checkpoint(
            runner_state, self.paths["xpid_dir"], name, index, archive_interval
        )

    def load_last_checkpoint_state(self):
        checkpoint_path = os.path.join(self.paths["xpid_dir"], f"checkpoint.pkl")

        if os.path.exists(checkpoint_path):
            self._logger.info(f"Loading previous checkpoint from {checkpoint_path}...")
            return _checkpoint_util.load_pkl_object(checkpoint_path)
        else:
            return None
