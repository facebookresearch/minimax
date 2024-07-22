"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from collections import defaultdict
import argparse
import sys
import re
import pprint

from minimax.util import DefaultDotDict, DotDict


def append_subparser_prefix(prefix, func):
    def prefixed_add_argument(*args, **kwargs):
        if len(args) > 0:
            name = args[0]
            if name.startswith("--"):
                name = f"--{prefix}_{name[2:]}"
            args = (name,) + args[1:]

        return func(*args, **kwargs)

    return prefixed_add_argument


def ensure_args_suffix(name):
    if not name.endswith("args"):
        name = f"{name}_args"

    return name


def get_all_cmd_arg_names():
    cmd_args = [s.removeprefix("--") for s in sys.argv if s.startswith("--")]
    arg_names = [x.split("=")[0] for x in cmd_args]

    return set(arg_names)


def get_argument_kwargs(subparser):
    kwargs = []
    skip_list = ["-h", "--help"]
    for k, info in subparser.__dict__["_option_string_actions"].items():
        if k in skip_list:
            continue

        info_dict = info.__dict__
        kwargs_dict = dict(
            # option_strings=info_dict['option_strings'],
            name=info_dict["dest"],
            const=info_dict["const"],
            default=info_dict["default"],
            type=info_dict["type"],
            choices=info_dict["choices"],
            required=info_dict["required"],
        )

        nargs = info_dict["nargs"]
        if nargs == "?" or (nargs is not None and int(nargs) > 0):
            kwargs_dict.update(dict(nargs=info_dict["nargs"]))

        kwargs.append(kwargs_dict)

    return kwargs


class Parsnip:
    """
    Wraps a collection of argparse instances
    to enable convenient grouping of arguments and
    access via a DotDict-style interface.
    """

    def __init__(self, description=None):
        self._base_parser = argparse.ArgumentParser(description=description)
        self._subparsers = {}
        self._prefixes = []
        self._dependencies = defaultdict()
        self._dests = defaultdict()
        self._dependent_args = set()

    def add_subparser(
        self,
        name,
        prefix=None,
        dest=None,
        depends_on=None,
        dependency=None,
        is_individual_arg=False,
        description=None,
    ):
        if not is_individual_arg:
            name = ensure_args_suffix(name)

        assert (
            name not in self._subparsers
        ), f"Multiple subparsers named {name} detected."

        if dependency is not None:
            if depends_on is not None:
                depends_on = ensure_args_suffix(depends_on)

                assert (
                    depends_on in self._subparsers
                ), f"Missing subparse {depends_on} must be added before dependent {name}."

            assert isinstance(
                dependency, dict
            ), f"Subparser dependencies must be specified as dicts."

            self._dependencies[name] = (depends_on, dependency)

        if dest is not None:
            dest = ensure_args_suffix(dest)
            assert (
                dest in self._subparsers
            ), f"Missing dest {dest} must be specified before source {name}."

        subparser = argparse.ArgumentParser(description=description, allow_abbrev=False)
        if prefix is not None:
            subparser.add_argument = append_subparser_prefix(
                prefix,
                subparser.add_argument,
            )

        self._subparsers[name] = subparser
        self._prefixes.append(prefix)
        self._dests[name] = dest

        return subparser

    def add_dependent_argument(
        self,
        *args,
        **kwargs,
    ):

        assert "dependency" in kwargs, "Must specify dependency in kwargs."
        dependency = kwargs.pop("dependency")

        name = args[0].removeprefix("--")

        prefix = kwargs.pop("prefix", None)
        dest = kwargs.pop("dest", None)

        subparser = self.add_subparser(
            name,
            prefix=prefix,
            dependency=dependency,
            dest=dest,
            is_individual_arg=True,
            description=kwargs.pop("description", ""),
        )
        subparser.add_argument(*args, **kwargs)

        self._dependent_args.add(name)

    def copy_arguments(self, src, dest, arg_prefix=None):
        all_subparsers = [k for k in self._subparsers.keys()]
        src_name = f"{src}_args"
        src_subparser = self._subparsers[src_name]
        src_idx = all_subparsers.index(src_name)
        src_prefix = self._prefixes[src_idx]

        dest_name = f"{dest}_args"
        dest_subparser = self._subparsers[dest_name]
        dest_idx = all_subparsers.index(dest_name)

        arg_prefix = f"{arg_prefix}_" if arg_prefix else ""

        argument_kwargs = get_argument_kwargs(src_subparser)
        for kwargs in argument_kwargs:
            name = kwargs.pop("name")
            flag = f"--{arg_prefix}{name.removeprefix(f'{src_prefix}_')}"

            dest_subparser.add_argument(flag, **kwargs)

    def parse_args(self, preview=False):
        cmd_arg_names, args, arg_data, _ = self._parse_cmd_line_flags()

        for name in cmd_arg_names:
            if name not in arg_data:
                raise ValueError(f"Unknown argument {name}.")

        if preview:
            print("🥕 Parsnip harvested the following arguments:")
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(args)

        return args

    def parse_cmd_line_flags(self, as_grid_json=False):
        _, _, arg_data, _ = self._parse_cmd_line_flags()

        if as_grid_json:
            arg_data = {k: [v] for k, v in arg_data.items() if v is not None}

        return arg_data

    def _parse_cmd_line_flags(self, keep_structure=False):
        cmd_arg_names = get_all_cmd_arg_names()

        arg_data = {}
        argname2keypath = {}

        args = DefaultDotDict(vars(self._base_parser.parse_known_args()[0]))
        for k in args:
            arg_data[k] = args[k]
            argname2keypath[k] = [k]

        for i, (name, subparser) in enumerate(self._subparsers.items()):
            if name in self._dependencies:
                depends_on, dependencies = self._dependencies[name]
                if depends_on is None:
                    dsubargs = arg_data
                else:
                    dsubargs = arg_data[depends_on]
                skip_subparser = False
                for dk, dv in dependencies.items():
                    if not isinstance(dv, (tuple, list)):
                        dv = [dv]

                    for v in dv:
                        if isinstance(v, str):
                            match_regex = f"^{v.replace('*', '.*')}$"
                            match = re.match(match_regex, dsubargs[dk]) is not None
                        else:
                            match = dsubargs[dk] == v
                        if match:
                            break

                    if not match:
                        skip_subparser = True
                        break

                if skip_subparser:
                    continue

            prefix = self._prefixes[i]
            subargs, _ = subparser.parse_known_args()

            subargs = vars(subargs)
            for k in subargs:
                arg_data[k] = subargs[k]

            subargs = DotDict(
                {k.removeprefix(f"{prefix}_"): v for k, v in subargs.items()}
            )

            # Check for flatten and merge conditions
            dest = self._dests.get(name)
            if name in self._dependent_args:
                if dest is None:
                    args[name] = subargs[name]
                else:
                    args[dest].update({name: subargs[name]})
            else:
                if dest is None:
                    args[name] = subargs
                else:
                    args[dest].update(subargs)

            for k in subargs:
                if prefix is not None:
                    argname = f"{prefix}_{k}"
                else:
                    argname = k
                if dest is None:
                    sp_name = name
                else:
                    sp_name = dest
                if sp_name == k:
                    argname2keypath[argname] = [k]
                else:
                    argname2keypath[argname] = [sp_name, k]

        return cmd_arg_names, args, arg_data, argname2keypath

    @property
    def argname2keypath(self):
        args = DefaultDotDict(vars(self._base_parser.parse_known_args()[0]))
        argname2keypath = {}
        for k in args:
            argname2keypath[k] = args[k]

        for i, (name, subparser) in enumerate(self._subparsers.items()):
            if name in self._dependencies:
                depends_on, dependencies = self._dependencies[name]
                if depends_on is None:
                    dsubargs = arg_data
                else:
                    dsubargs = arg_data[depends_on]
                skip_subparser = False
                for dk, dv in dependencies.items():
                    if not isinstance(dv, (tuple, list)):
                        dv = [dv]

                    for v in dv:
                        if isinstance(v, str):
                            match_regex = f"^{v.replace('*', '.*')}$"
                            match = re.match(match_regex, dsubargs[dk]) is not None
                        else:
                            match = dsubargs[dk] == v
                        if match:
                            break

                    if not match:
                        skip_subparser = True
                        break

                if skip_subparser:
                    continue

            prefix = self._prefixes[i]
            subargs, _ = subparser.parse_known_args()

    def __getattr__(self, attr):
        # Default missing attr to _base_argparse instance
        return getattr(self._base_parser, attr)
