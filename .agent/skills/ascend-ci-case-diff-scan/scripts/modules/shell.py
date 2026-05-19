#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shell-level helpers for command parsing and signature normalization."""

from __future__ import annotations

import re
import shlex

ENV_PREFIX_RE = re.compile(r"^(?:[A-Za-z_][A-Za-z0-9_]*=(?:\"[^\"]*\"|'[^']*'|\S+)\s+)+")
TORCHRUN_RE = re.compile(r"\btorchrun\b")
PATH_VALUE_OPTIONS = {"--ignore", "--ignore-glob"}
PYTEST_OPTIONS_WITH_VALUE = {
    "-k",
    "-m",
    "--maxfail",
    "--rootdir",
    "--log-level",
    "--capture",
    "--asyncio-mode",
    "--durations",
}
IGNORED_CASE_TARGETS = {"tests/"}
PYTEST_DIRECTORY_TARGETS = {"tests", "tests/"}


def normalize_path_text(value: str) -> str:
    """Normalize filesystem-like text so matching stays stable across platforms."""
    return value.replace("\\", "/").strip()


def normalize_signature(command: str, target: str) -> str:
    """Keep only the command prefixes that materially affect parity matching."""
    prefix = command
    idx = command.find(target) if target else -1
    if idx != -1:
        prefix = command[:idx]
    env_match = ENV_PREFIX_RE.match(prefix or "")
    env_prefix = env_match.group(0).strip() if env_match else ""
    parts: list[str] = []
    for key in ("ROLLOUT_NAME", "ENGINE", "STRATEGY", "MODE", "RESUME_MODE", "BACKEND"):
        match = re.search(rf"{key}=([^\s]+)", env_prefix)
        if match:
            parts.append(match.group(0))
    if "--nproc_per_node" in command or "--nproc-per-node" in command:
        parts.append("torchrun-distributed")
    return " ".join(parts).strip()


def split_shell_commands(command: str) -> list[str]:
    """Split a shell line on common shell separators while respecting simple quote scopes."""
    parts: list[str] = []
    current: list[str] = []
    in_single = False
    in_double = False
    idx = 0
    while idx < len(command):
        ch = command[idx]
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        if not in_single and not in_double:
            separator = None
            if ch == ";":
                separator = ";"
            elif command.startswith("&&", idx):
                separator = "&&"
            elif command.startswith("||", idx):
                separator = "||"
            if separator is not None:
                part = "".join(current).strip()
                if part:
                    parts.append(part)
                current = []
                idx += len(separator)
                continue

        current.append(ch)
        idx += 1
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def tokenize_command(command: str) -> list[str]:
    """Tokenize shell text conservatively and fall back when quoting is malformed."""
    try:
        return shlex.split(command, posix=True)
    except ValueError:
        return command.split()


def command_uses_torchrun(command: str, tokens: list[str]) -> bool:
    return "torchrun" in tokens or bool(TORCHRUN_RE.search(command))


def command_uses_pytest(tokens: list[str]) -> bool:
    return "pytest" in tokens
