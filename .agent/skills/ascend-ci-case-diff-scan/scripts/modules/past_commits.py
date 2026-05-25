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

"""Past-N-days commit analysis for workflow and test case changes."""

from __future__ import annotations

import datetime as dt
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .compare import compare_cases_by_pair
from .config import ST_KIND, UT_KIND, WorkflowConfig
from .extractors import extract_test_functions_from_text, normalize_path_text
from .workflows import parse_workflow_content

CI_RELATED_PREFIXES = (".github/workflows/", "tests/", "examples/")
REPORT_STATUSES = {
    "matched": "aligned",
    "cpu_gpu_only": "missing_in_npu_workflows",
    "manual_review": "manual_review_needed",
    "npu_only": "npu_only",
}


@dataclass(frozen=True)
class CommitInfo:
    commit_hash: str
    commit_time: str
    commit_title: str
    changed_files: tuple[str, ...]


def _run_git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _is_ci_related(path_text: str) -> bool:
    normalized = normalize_path_text(path_text)
    return normalized.startswith(CI_RELATED_PREFIXES)


def _is_workflow_path(path_text: str) -> bool:
    normalized = normalize_path_text(path_text)
    return normalized.startswith(".github/workflows/") and normalized.endswith((".yml", ".yaml"))


def _is_test_python_path(path_text: str) -> bool:
    normalized = normalize_path_text(path_text)
    return normalized.startswith("tests/") and normalized.endswith(".py")


def _is_test_script_path(path_text: str) -> bool:
    normalized = normalize_path_text(path_text)
    return normalized.startswith(("tests/", "examples/")) and normalized.endswith(".sh")


def list_recent_commits(repo_root: Path, since_days: int) -> list[CommitInfo]:
    """Return commits in HEAD history during the last N days that touch CI-related paths."""
    cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=since_days)).isoformat()
    output = _run_git(
        repo_root,
        "log",
        "--since",
        cutoff,
        "--pretty=format:%H%x1f%cI%x1f%s",
        "--name-only",
        "--no-renames",
        "HEAD",
    )
    commits: list[CommitInfo] = []
    current_hash = ""
    current_time = ""
    current_title = ""
    current_files: list[str] = []

    def flush() -> None:
        if not current_hash:
            return
        related_files = tuple(path for path in current_files if _is_ci_related(path))
        if related_files:
            commits.append(
                CommitInfo(
                    commit_hash=current_hash,
                    commit_time=current_time,
                    commit_title=current_title,
                    changed_files=related_files,
                )
            )

    for line in output.splitlines():
        if not line.strip():
            continue
        if "\x1f" in line:
            flush()
            current_hash, current_time, current_title = line.split("\x1f", 2)
            current_files = []
            continue
        current_files.append(normalize_path_text(line))
    flush()
    return commits


def build_past_commit_report(repo_root: Path, config: WorkflowConfig, since_days: int, head_cases: list[dict]) -> dict:
    """Build a past-N-days report using the current HEAD scan as the NPU baseline."""
    commits = list_recent_commits(repo_root, since_days)
    status_index = _build_head_status_index(head_cases)
    details: list[dict] = []
    for commit in commits:
        cases = _collect_commit_cases(repo_root, config, commit)
        for case in cases:
            status, npu_refs = _lookup_npu_support(case, status_index)
            details.append(
                {
                    "commit_hash": commit.commit_hash,
                    "commit_time": commit.commit_time,
                    "commit_title": commit.commit_title,
                    "changed_files": tuple(commit.changed_files),
                    "case_kind": case["case_kind"],
                    "case_name": case["display_name"],
                    "command_type": case["command_type"],
                    "workflow_name": case["workflow_name"],
                    "workflow_path": case["workflow_path"],
                    "job_name": case["job_name"],
                    "step_name": case["step_name"],
                    "line_number": case["line_number"],
                    "raw_command": case["raw_command"],
                    "npu_status": status,
                    "npu_refs": npu_refs,
                }
            )

    summary = _summarize_details(details)
    return {
        "repo_root": str(repo_root),
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "since_days": since_days,
        "commit_count": len(commits),
        "summary": summary,
        "details": sorted(
            details,
            key=lambda row: (
                row["commit_time"],
                row["commit_hash"],
                row["case_kind"],
                row["workflow_name"],
                row["case_name"],
            ),
        ),
    }


def _collect_commit_cases(repo_root: Path, config: WorkflowConfig, commit: CommitInfo) -> list[dict]:
    cases: list[dict] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for path_text in commit.changed_files:
        if _is_workflow_path(path_text):
            content = _load_git_file(repo_root, commit.commit_hash, path_text)
            if content is not None:
                _workflow_info, workflow_cases = parse_workflow_content(
                    Path(path_text).name,
                    path_text,
                    content,
                    repo_root,
                    config,
                )
                cases.extend(workflow_cases)
        elif _is_test_python_path(path_text):
            content = _load_git_file(repo_root, commit.commit_hash, path_text)
            cases.extend(_build_test_file_cases(commit, path_text, content))
        elif _is_test_script_path(path_text):
            cases.append(_build_script_case(commit, path_text))

    deduped: list[dict] = []
    for case in cases:
        if case["workflow_kind"] == "npu":
            continue
        key = (
            case["case_kind"],
            case["command_type"],
            case["target"],
            case["workflow_path"],
            case["raw_command"],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(case)
    return deduped


def _load_git_file(repo_root: Path, commit_hash: str, path_text: str) -> str | None:
    try:
        return _run_git(repo_root, "show", f"{commit_hash}:{path_text}")
    except subprocess.CalledProcessError:
        return None


def _build_test_file_cases(commit: CommitInfo, path_text: str, content: str | None) -> list[dict]:
    functions = extract_test_functions_from_text(content) if content is not None else []
    case_names = [f"{path_text}::{name}" for name in functions] if functions else [path_text]
    cases: list[dict] = []
    for case_name in case_names:
        cases.append(
            {
                "workflow_name": commit.commit_hash[:12],
                "workflow_path": path_text,
                "file_name": Path(path_text).name,
                "workflow_kind": "cpu",
                "pair_key": path_text,
                "job_name": "commit",
                "step_name": path_text,
                "line_number": 1,
                "command_type": "pytest",
                "case_kind": UT_KIND,
                "target": case_name,
                "raw_command": f"git show {commit.commit_hash}:{path_text}",
                "signature": "commit-file-change",
                "case_id": f"{commit.commit_hash}|{path_text}|{case_name}",
                "display_name": case_name,
            }
        )
    return cases


def _build_script_case(commit: CommitInfo, path_text: str) -> dict:
    return {
        "workflow_name": commit.commit_hash[:12],
        "workflow_path": path_text,
        "file_name": Path(path_text).name,
        "workflow_kind": "cpu",
        "pair_key": path_text,
        "job_name": "commit",
        "step_name": path_text,
        "line_number": 1,
        "command_type": "bash",
        "case_kind": ST_KIND,
        "target": path_text,
        "raw_command": f"git show {commit.commit_hash}:{path_text}",
        "signature": "commit-file-change",
        "case_id": f"{commit.commit_hash}|{path_text}",
        "display_name": path_text,
    }


def _build_head_status_index(head_cases: list[dict]) -> dict[str, dict[str, tuple[str, list[dict]]]]:
    index = {
        UT_KIND: {},
        ST_KIND: {},
    }
    for case_kind, details in (
        (UT_KIND, compare_cases_by_pair(head_cases, UT_KIND)),
        (ST_KIND, compare_cases_by_pair(head_cases, ST_KIND)),
    ):
        for section_key, status in REPORT_STATUSES.items():
            for item in details[section_key]:
                current = index[case_kind].get(item["name"])
                if current and _status_rank(current[0]) <= _status_rank(status):
                    continue
                index[case_kind][item["name"]] = (status, item["npu_refs"])
    return index


def _lookup_npu_support(
    case: dict, status_index: dict[str, dict[str, tuple[str, list[dict]]]]
) -> tuple[str, list[dict]]:
    return status_index.get(case["case_kind"], {}).get(case["target"], ("missing_in_npu_workflows", []))


def _status_rank(status: str) -> int:
    order = {
        "aligned": 0,
        "manual_review_needed": 1,
        "missing_in_npu_workflows": 2,
        "npu_only": 3,
    }
    return order.get(status, 99)


def _summarize_details(details: list[dict]) -> list[dict]:
    buckets: dict[tuple[str, str, str], dict] = defaultdict(lambda: {"case_count": 0, "commits": set()})
    for row in details:
        key = (row["case_kind"], row["workflow_name"], row["npu_status"])
        buckets[key]["case_count"] += 1
        buckets[key]["commits"].add(row["commit_hash"])
    return [
        {
            "case_kind": case_kind,
            "workflow_name": workflow_name,
            "npu_status": status,
            "case_count": payload["case_count"],
            "commit_count": len(payload["commits"]),
        }
        for (case_kind, workflow_name, status), payload in sorted(buckets.items())
    ]
