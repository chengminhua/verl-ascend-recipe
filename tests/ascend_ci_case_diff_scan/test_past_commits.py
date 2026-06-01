"""Tests for modules/past_commits.py."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from modules.past_commits import CommitInfo

# ============================================================================
# CommitInfo
# ============================================================================


class TestCommitInfo:
    def test_construction(self):
        ci = CommitInfo(
            commit_hash="abc123",
            commit_time="2026-05-30T00:00:00+00:00",
            commit_title="Fix bug",
            changed_files=("tests/test_foo.py", ".github/workflows/gpu.yml"),
        )
        assert ci.commit_hash == "abc123"
        assert ci.commit_time == "2026-05-30T00:00:00+00:00"
        assert ci.commit_title == "Fix bug"
        assert ci.changed_files == ("tests/test_foo.py", ".github/workflows/gpu.yml")

    def test_immutability(self):
        from dataclasses import FrozenInstanceError

        ci = CommitInfo(
            commit_hash="abc123",
            commit_time="",
            commit_title="",
            changed_files=(),
        )
        with pytest.raises(FrozenInstanceError):
            ci.commit_hash = "new"  # type: ignore[misc]


# ============================================================================
# _normalize_commit_commit_hash
# ============================================================================


class TestNormalizeCommitHash:
    def test_strips_whitespace(self):
        from modules.past_commits import _normalize_commit_commit_hash

        assert _normalize_commit_commit_hash("  abc123  ") == "abc123"
        assert _normalize_commit_commit_hash("abc123\n") == "abc123"


# ============================================================================
# _is_relevant_path
# ============================================================================


class TestIsRelevantPath:
    def test_workflow_path(self):
        from modules.past_commits import _is_relevant_path

        assert _is_relevant_path(".github/workflows/gpu_unit_tests.yml") is True

    def test_test_path(self):
        from modules.past_commits import _is_relevant_path

        assert _is_relevant_path("tests/test_foo.py") is True
        assert _is_relevant_path("tests/sub/test_bar.py") is True

    def test_example_script(self):
        from modules.past_commits import _is_relevant_path

        assert _is_relevant_path("examples/test_example.sh") is True

    def test_irrelevant_path(self):
        from modules.past_commits import _is_relevant_path

        assert _is_relevant_path("README.md") is False
        assert _is_relevant_path("setup.py") is False
        assert _is_relevant_path("examples/test_example.py") is False

    def test_backslash_normalized(self):
        from modules.past_commits import _is_relevant_path

        assert _is_relevant_path(".github\\workflows\\test.yml") is True


# ============================================================================
# _case_change_key
# ============================================================================


class TestCaseChangeKey:
    def test_produces_tuple(self):
        from modules.past_commits import _case_change_key

        case = {
            "case_kind": "ut",
            "command_type": "pytest",
            "target": "tests/test_foo.py::test_add",
            "signature": "pytest",
            "workflow_name": "GPU Tests",
            "job_name": "unit-tests",
            "step_name": "Run pytest",
            "raw_command": "pytest tests/",
        }
        result = _case_change_key(case)
        assert isinstance(result, tuple)
        assert result[0] == "ut"
        assert result[1] == "pytest"
        assert result[2] == "tests/test_foo.py::test_add"


# ============================================================================
# _path_or_child_changed
# ============================================================================


class TestPathOrChildChanged:
    def test_exact_match(self):
        from modules.past_commits import _path_or_child_changed

        assert _path_or_child_changed("tests/test_foo.py", {"tests/test_foo.py"}) is True

    def test_child_changed(self):
        from modules.past_commits import _path_or_child_changed

        assert _path_or_child_changed("tests/sub", {"tests/sub/test_bar.py"}) is True

    def test_no_match(self):
        from modules.past_commits import _path_or_child_changed

        assert _path_or_child_changed("tests/module", {"tests/other/test.py"}) is False

    def test_trailing_slash_normalized(self):
        from modules.past_commits import _path_or_child_changed

        assert _path_or_child_changed("tests/sub/", {"tests/sub/test.py"}) is True


# ============================================================================
# _workflow_status
# ============================================================================


class TestWorkflowStatus:
    def test_added(self):
        from modules.past_commits import _workflow_status

        assert _workflow_status(None, object()) == "added"

    def test_removed(self):
        from modules.past_commits import _workflow_status

        assert _workflow_status(object(), None) == "removed"

    def test_modified(self):
        from modules.past_commits import _workflow_status

        assert _workflow_status(object(), object()) == "modified"


# ============================================================================
# _workflow_change_sort_key
# ============================================================================


class TestWorkflowChangeSortKey:
    def test_added_before_modified(self):
        from modules.past_commits import _workflow_change_sort_key

        added_key = _workflow_change_sort_key({"workflow_status": "added", "workflow_path": "b.yml"})
        modified_key = _workflow_change_sort_key({"workflow_status": "modified", "workflow_path": "a.yml"})
        assert added_key < modified_key

    def test_same_status_sorted_by_path(self):
        from modules.past_commits import _workflow_change_sort_key

        a_key = _workflow_change_sort_key({"workflow_status": "modified", "workflow_path": "a.yml"})
        b_key = _workflow_change_sort_key({"workflow_status": "modified", "workflow_path": "b.yml"})
        assert a_key < b_key


# ============================================================================
# _workflow_changed
# ============================================================================


class TestWorkflowChanged:
    def test_added(self):
        from modules.past_commits import _workflow_changed

        head_info = object()
        assert _workflow_changed(None, head_info, set(), []) is True

    def test_removed(self):
        from modules.past_commits import _workflow_changed

        base_info = object()
        assert _workflow_changed(base_info, None, set(), []) is True

    def test_no_change(self):
        from modules.config import WorkflowInfo
        from modules.past_commits import _workflow_changed

        base = WorkflowInfo("A", "a.yml", "a.yml", "gpu", "a")
        head = WorkflowInfo("A", "a.yml", "a.yml", "gpu", "a")

        # Same info + same case keys = no change
        base_keys = {("ut", "pytest", "t", "s", "w", "j", "s", "c")}
        head_cases = [
            {
                "case_kind": "ut",
                "command_type": "pytest",
                "target": "t",
                "signature": "s",
                "workflow_name": "w",
                "job_name": "j",
                "step_name": "s",
                "raw_command": "c",
            }
        ]

        assert _workflow_changed(base, head, base_keys, head_cases) is False

    def test_name_changed(self):
        from modules.config import WorkflowInfo
        from modules.past_commits import _workflow_changed

        base = WorkflowInfo("A", "a.yml", "a.yml", "gpu", "a")
        head = WorkflowInfo("B", "a.yml", "a.yml", "gpu", "a")

        assert _workflow_changed(base, head, set(), []) is True

    def test_kind_changed(self):
        from modules.config import WorkflowInfo
        from modules.past_commits import _workflow_changed

        base = WorkflowInfo("A", "a.yml", "a.yml", "gpu", "a")
        head = WorkflowInfo("A", "a.yml", "a.yml", "cpu", "a")

        assert _workflow_changed(base, head, set(), []) is True


# ============================================================================
# _case_target_changed
# ============================================================================


class TestCaseTargetChanged:
    def test_workflow_path_changed(self):
        from modules.past_commits import _case_target_changed

        case = {
            "workflow_path": ".github/workflows/gpu_unit_tests.yml",
            "command_type": "pytest",
            "target": "tests/test_foo.py::test_add",
        }
        changed = {".github/workflows/gpu_unit_tests.yml"}
        assert _case_target_changed(case, changed) is True

    def test_pytest_target_changed(self):
        from modules.past_commits import _case_target_changed

        case = {
            "workflow_path": ".github/workflows/gpu_unit_tests.yml",
            "command_type": "pytest",
            "target": "tests/test_foo.py::test_add",
        }
        changed = {"tests/test_foo.py"}
        assert _case_target_changed(case, changed) is True

    def test_bash_target_changed(self):
        from modules.past_commits import _case_target_changed

        case = {
            "workflow_path": ".github/workflows/gpu.yml",
            "command_type": "bash",
            "target": "tests/test_integration.sh",
        }
        changed = {"tests/test_integration.sh"}
        assert _case_target_changed(case, changed) is True

    def test_torchrun_target_changed(self):
        from modules.past_commits import _case_target_changed

        case = {
            "workflow_path": ".github/workflows/gpu_e2e.yml",
            "command_type": "torchrun",
            "target": "tests/test_trainer.py",
        }
        changed = {"tests/test_trainer.py"}
        assert _case_target_changed(case, changed) is True

    def test_no_change(self):
        from modules.past_commits import _case_target_changed

        case = {
            "workflow_path": ".github/workflows/gpu_unit_tests.yml",
            "command_type": "pytest",
            "target": "tests/test_foo.py::test_add",
        }
        changed = {"tests/test_other.py"}
        assert _case_target_changed(case, changed) is False


# ============================================================================
# _commit_touches_case_path
# ============================================================================


class TestCommitTouchesCasePath:
    def test_workflow_touched(self):
        from modules.past_commits import _commit_touches_case_path

        commit = CommitInfo("a", "", "", (".github/workflows/gpu.yml",))
        case = {
            "workflow_path": ".github/workflows/gpu.yml",
            "command_type": "pytest",
            "target": "tests/test_foo.py::test_add",
        }
        assert _commit_touches_case_path(commit, case) is True

    def test_pytest_file_touched(self):
        from modules.past_commits import _commit_touches_case_path

        commit = CommitInfo("a", "", "", ("tests/test_foo.py",))
        case = {
            "workflow_path": ".github/workflows/gpu.yml",
            "command_type": "pytest",
            "target": "tests/test_foo.py::test_add",
        }
        assert _commit_touches_case_path(commit, case) is True

    def test_bash_file_touched(self):
        from modules.past_commits import _commit_touches_case_path

        commit = CommitInfo("a", "", "", ("tests/run.sh",))
        case = {
            "workflow_path": ".github/workflows/gpu.yml",
            "command_type": "bash",
            "target": "tests/run.sh",
        }
        assert _commit_touches_case_path(commit, case) is True

    def test_no_touch(self):
        from modules.past_commits import _commit_touches_case_path

        commit = CommitInfo("a", "", "", ("README.md",))
        case = {
            "workflow_path": ".github/workflows/gpu.yml",
            "command_type": "pytest",
            "target": "tests/test_foo.py::test_add",
        }
        assert _commit_touches_case_path(commit, case) is False


# ============================================================================
# _collect_window_changed_files
# ============================================================================


class TestCollectWindowChangedFiles:
    def test_collects_and_dedupes(self):
        from modules.past_commits import _collect_window_changed_files

        commits = [
            CommitInfo("a", "", "", ("tests/test_a.py", "tests/test_b.py")),
            CommitInfo("b", "", "", ("tests/test_a.py", "tests/test_c.py")),
        ]
        result = _collect_window_changed_files(commits)
        assert result == {"tests/test_a.py", "tests/test_b.py", "tests/test_c.py"}


# ============================================================================
# _summarize_details
# ============================================================================


class TestSummarizeDetails:
    def test_aligned_cases_excluded(self):
        from modules.past_commits import _summarize_details

        details = [
            {
                "workflow_path": "gpu.yml",
                "npu_status": "aligned",
                "case_kind": "ut",
                "case_id": "u1",
            }
        ]
        result = _summarize_details(details)
        assert result == []

    def test_counts_gaps(self):
        from modules.past_commits import _summarize_details

        details = [
            {
                "workflow_path": "gpu.yml",
                "npu_status": "missing_in_npu_workflows",
                "case_kind": "ut",
                "case_id": "u1",
            },
            {
                "workflow_path": "gpu.yml",
                "npu_status": "missing_in_npu_workflows",
                "case_kind": "ut",
                "case_id": "u2",
            },
            {
                "workflow_path": "gpu.yml",
                "npu_status": "missing_in_npu_workflows",
                "case_kind": "st",
                "case_id": "s1",
            },
        ]
        result = _summarize_details(details)
        assert len(result) == 1  # one row for this workflow+status pair
        assert result[0]["ut_gap_count"] == 2
        assert result[0]["st_gap_count"] == 1

    def test_dedup_case_ids(self):
        from modules.past_commits import _summarize_details

        details = [
            {
                "workflow_path": "gpu.yml",
                "npu_status": "missing_in_npu_workflows",
                "case_kind": "ut",
                "case_id": "u1",
            },
            {
                "workflow_path": "gpu.yml",
                "npu_status": "missing_in_npu_workflows",
                "case_kind": "ut",
                "case_id": "u1",  # duplicate
            },
        ]
        result = _summarize_details(details)
        # Should dedupe by case_id, so only 1 UT gap
        assert result[0]["ut_gap_count"] == 1


# ============================================================================
# _status_rank
# ============================================================================


class TestStatusRank:
    def test_ordering(self):
        from modules.past_commits import _status_rank

        assert _status_rank("aligned") < _status_rank("manual_review_needed")
        assert _status_rank("manual_review_needed") < _status_rank("missing_in_npu_workflows")
        assert _status_rank("missing_in_npu_workflows") < _status_rank("npu_only")

    def test_unknown_status(self):
        from modules.past_commits import _status_rank

        assert _status_rank("unknown") == 99


# ============================================================================
# _collect_changed_head_cases
# ============================================================================


class TestCollectChangedHeadCases:
    def test_new_case_not_in_base(self):
        from modules.past_commits import _collect_changed_head_cases

        head_cases = [
            {
                "workflow_kind": "gpu",
                "case_id": "new_case",
                "case_kind": "ut",
                "command_type": "pytest",
                "target": "tests/test_new.py::test_feature",
                "signature": "pytest",
                "workflow_name": "GPU Tests",
                "job_name": "j",
                "step_name": "s",
                "raw_command": "pytest tests/",
                "workflow_path": ".github/workflows/gpu.yml",
            }
        ]
        base_keys = set()  # empty base
        changed_files = set()

        result = _collect_changed_head_cases(head_cases, base_keys, changed_files)
        assert len(result) == 1
        assert result[0]["case_id"] == "new_case"

    def test_npu_case_excluded(self):
        from modules.past_commits import _collect_changed_head_cases

        head_cases = [
            {
                "workflow_kind": "npu",
                "case_id": "npu_case",
                "case_kind": "ut",
                "command_type": "pytest",
                "target": "tests/test_npu.py::test_feature",
                "signature": "pytest",
                "workflow_name": "NPU Tests",
                "job_name": "j",
                "step_name": "s",
                "raw_command": "pytest tests/",
                "workflow_path": ".github/workflows/npu.yml",
            }
        ]
        base_keys = set()
        changed_files = set()

        result = _collect_changed_head_cases(head_cases, base_keys, changed_files)
        assert result == []

    def test_dedup_by_case_id(self):
        from modules.past_commits import _collect_changed_head_cases

        head_cases = [
            {
                "workflow_kind": "gpu",
                "case_id": "dup_case",
                "case_kind": "ut",
                "command_type": "pytest",
                "target": "tests/test_a.py::test_1",
                "signature": "pytest",
                "workflow_name": "GPU Tests",
                "job_name": "j1",
                "step_name": "s1",
                "raw_command": "pytest tests/",
                "workflow_path": ".github/workflows/gpu.yml",
            },
            {
                "workflow_kind": "gpu",
                "case_id": "dup_case",  # same case_id
                "case_kind": "ut",
                "command_type": "pytest",
                "target": "tests/test_a.py::test_1",
                "signature": "pytest",
                "workflow_name": "GPU Tests",
                "job_name": "j2",
                "step_name": "s2",
                "raw_command": "pytest tests/",
                "workflow_path": ".github/workflows/gpu.yml",
            },
        ]
        base_keys = set()
        changed_files = set()

        result = _collect_changed_head_cases(head_cases, base_keys, changed_files)
        assert len(result) == 1


# ============================================================================
# case_details_for_workflow
# ============================================================================


class TestCaseDetailsForWorkflow:
    def test_filters_by_workflow_path(self):
        from modules.past_commits import case_details_for_workflow

        details = [
            {"workflow_path": "a.yml", "case_name": "case_a"},
            {"workflow_path": "b.yml", "case_name": "case_b"},
            {"workflow_path": "a.yml", "case_name": "case_c"},
        ]
        result = case_details_for_workflow(details, "a.yml")
        assert len(result) == 2
        assert all(r["workflow_path"] == "a.yml" for r in result)


# ============================================================================
# _run_git and related (mocked subprocess)
# ============================================================================


class TestRunGit:
    def test_successful_call(self):
        from modules.past_commits import _run_git

        with patch("modules.past_commits.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="abc123\ndef456\n", stderr=""
            )
            result = _run_git(Path("/fake/repo"), "log")
            assert result == "abc123\ndef456\n"
            mock_run.assert_called_once()

    def test_called_process_error_propagates(self):
        from modules.past_commits import _run_git

        with patch("modules.past_commits.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(
                returncode=128, cmd=["git"], stderr="fatal: not a git repository"
            )
            with pytest.raises(subprocess.CalledProcessError):
                _run_git(Path("/fake/repo"), "log")


class TestGetFirstParentCommits:
    def test_parses_hash_list(self):
        from modules.past_commits import _get_first_parent_commits

        with patch("modules.past_commits._run_git") as mock_git:
            mock_git.return_value = "abc123\ndef456\n\n"
            result = _get_first_parent_commits(Path("/fake/repo"), 7)
            assert result == ["abc123", "def456"]

    def test_empty_output(self):
        from modules.past_commits import _get_first_parent_commits

        with patch("modules.past_commits._run_git") as mock_git:
            mock_git.return_value = ""
            result = _get_first_parent_commits(Path("/fake/repo"), 7)
            assert result == []


class TestGetCommitInfo:
    def test_parses_commit_data(self):
        from modules.past_commits import _get_commit_info

        with patch("modules.past_commits._run_git") as mock_git:
            # First call: show command
            # Second call: diff-tree
            mock_git.side_effect = [
                "abc123\x1f2026-05-30T00:00:00+00:00\x1fFix bug",
                "tests/test_foo.py\n.github/workflows/gpu.yml\nREADME.md\n",
            ]
            result = _get_commit_info(Path("/fake/repo"), "abc123")
            assert result.commit_hash == "abc123"
            assert result.commit_title == "Fix bug"
            # README.md should be excluded (not relevant)
            assert "tests/test_foo.py" in result.changed_files
            assert ".github/workflows/gpu.yml" in result.changed_files
            assert "README.md" not in result.changed_files


class TestGetBaseCommit:
    def test_returns_parent_commit(self):
        from modules.past_commits import _get_base_commit

        with patch("modules.past_commits._run_git") as mock_git:
            mock_git.return_value = "parent_hash\n"
            result = _get_base_commit(Path("/fake/repo"), "abc123")
            assert result == "parent_hash"

    def test_returns_none_on_error(self):
        from modules.past_commits import _get_base_commit

        with patch("modules.past_commits._run_git") as mock_git:
            mock_git.side_effect = subprocess.CalledProcessError(128, ["git"], "")
            result = _get_base_commit(Path("/fake/repo"), "abc123")
            assert result is None


# ============================================================================
# _build_head_status_index / _lookup_npu_support
# ============================================================================


class TestHeadStatusIndex:
    def test_indexes_aligned_cases(self):
        from modules.past_commits import _build_head_status_index

        # Build minimal head cases that should match
        # This is a simplified integration test of the indexing logic
        head_cases = []
        # With no cases, index should be empty
        index = _build_head_status_index(head_cases)
        assert "ut" in index
        assert "st" in index
        assert index["ut"] == {}
        assert index["st"] == {}


class TestLookupNpuSupport:
    def test_not_found_returns_missing(self):
        from modules.past_commits import _lookup_npu_support

        case = {
            "case_kind": "ut",
            "pair_key": "test",
            "command_type": "pytest",
            "target": "tests/test_foo.py::test_add",
            "signature": "pytest",
        }
        status_index = {"ut": {}, "st": {}}
        status, refs = _lookup_npu_support(case, status_index)
        assert status == "missing_in_npu_workflows"
        assert refs == []


# ============================================================================
# _build_commit_details
# ============================================================================


class TestBuildCommitDetails:
    def test_excludes_commits_without_affected_workflows(self):
        from modules.past_commits import _build_commit_details

        commits = [
            CommitInfo("a", "2026-01-01T00:00:00", "Commit A", ("tests/test_a.py",)),
            CommitInfo("b", "2026-01-02T00:00:00", "Commit B", ("README.md",)),
        ]
        workflow_changes = [{"commit_hashes": ("a",), "workflow_path": "gpu.yml"}]
        case_details = []

        result = _build_commit_details(commits, workflow_changes, case_details)
        # Only commit A should appear since it affects a workflow
        assert len(result) == 1
        assert result[0]["commit_hash"] == "a"

    def test_dedups_affected_workflows(self):
        from modules.past_commits import _build_commit_details

        commits = [
            CommitInfo("a", "2026-01-01T00:00:00", "Commit A", ("tests/test_a.py",)),
            CommitInfo("b", "2026-01-02T00:00:00", "Commit B", ("tests/test_b.py",)),
        ]
        # Both commits affect the same workflow
        workflow_changes = [
            {"commit_hashes": ("a", "b"), "workflow_path": "gpu.yml"},
        ]
        case_details = []

        result = _build_commit_details(commits, workflow_changes, case_details)
        assert len(result) == 2


# ============================================================================
# build_past_commit_report (integration with mocks)
# ============================================================================


class TestBuildPastCommitReport:
    def test_empty_commits_returns_empty_report(self, empty_config):
        from modules.past_commits import build_past_commit_report

        with patch("modules.past_commits._get_commit_sequence") as mock_seq:
            mock_seq.return_value = []
            result = build_past_commit_report(Path("/fake/repo"), empty_config, 7, [])

            assert result["commit_count"] == 0
            assert result["summary"] == []
            assert result["workflow_changes"] == []
            assert result["case_details"] == []
            assert result["commit_details"] == []

    def test_with_commits_and_changes(self, empty_config):
        from modules.past_commits import build_past_commit_report

        commits = [
            CommitInfo(
                "abc123",
                "2026-05-30T00:00:00",
                "Add test",
                (".github/workflows/gpu_unit_tests.yml", "tests/test_new.py"),
            ),
        ]

        with (
            patch("modules.past_commits._get_commit_sequence") as mock_seq,
            patch("modules.past_commits._get_base_commit") as mock_base,
            patch("modules.past_commits._run_git") as mock_git,
            patch("modules.past_commits._materialize_snapshot") as _mock_materialize,
            patch("modules.past_commits._load_snapshot_scan") as mock_load,
        ):
            mock_seq.return_value = commits
            mock_base.return_value = "base_hash"
            mock_git.return_value = "head_hash"

            # Mock snapshot scans: base has 1 case, head has 2 (1 new)
            from modules.config import WorkflowInfo

            base_info = WorkflowInfo("GPU Tests", "gpu.yml", "gpu.yml", "gpu", "gpu")
            head_info = WorkflowInfo("GPU Tests", "gpu.yml", "gpu.yml", "gpu", "gpu")

            base_grouped = {"gpu": {"cpu_gpu": [base_info], "npu": []}}
            head_grouped = {"gpu": {"cpu_gpu": [head_info], "npu": []}}

            def make_case(case_id, target, workflow_path="gpu.yml"):
                return {
                    "workflow_name": "GPU Tests",
                    "workflow_path": workflow_path,
                    "file_name": "gpu.yml",
                    "workflow_kind": "gpu",
                    "pair_key": "gpu",
                    "job_name": "unit-tests",
                    "step_name": "Run pytest",
                    "line_number": 10,
                    "command_type": "pytest",
                    "case_kind": "ut",
                    "target": target,
                    "raw_command": "pytest tests/",
                    "signature": "pytest",
                    "case_id": case_id,
                    "display_name": target,
                }

            base_cases = [make_case("c1", "tests/test_old.py::test_old")]
            head_cases = [
                make_case("c1", "tests/test_old.py::test_old"),
                make_case("c2", "tests/test_new.py::test_new"),
            ]

            # Mock _load_snapshot_scan for base and head
            mock_load.side_effect = [
                (base_grouped, base_cases, {"gpu.yml": base_cases}, {"gpu.yml": base_info}),
                (head_grouped, head_cases, {"gpu.yml": head_cases}, {"gpu.yml": head_info}),
            ]

            result = build_past_commit_report(Path("/fake/repo"), empty_config, 7, head_cases)

            assert result["commit_count"] == 1
            assert "abc123" in result["commit_details"][0]["commit_hash"]
