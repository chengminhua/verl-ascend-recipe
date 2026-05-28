---
name: ascend-ci-case-diff-scan
description: Scan an external verl repository for Ascend/NPU CI coverage gaps by comparing CPU/GPU workflow cases against NPU workflows. Use when Codex needs to audit workflow and case-level parity or generate Markdown and Excel reports about workflow execution differences.
---

# Ascend CI Case Diff Scan

Audit Ascend CI coverage in a target `verl` repository with the scanner shipped in this skill.

## Overview

Use this skill when you need to compare CPU/GPU workflow test coverage with NPU workflow coverage and produce English Markdown and Excel reports. The scanner is static: it reads GitHub Actions workflow `run:` commands, preserves workflow/job/step context, and does not execute tests.

The full reports show ignored workflows, paired CPU/GPU and NPU workflows, UT/ST case counts, matched cases, missing cases, NPU-only cases, and manual-review cases. When `--since-days N` is set, the scanner also emits a past-N-days report that surfaces only CPU/GPU workflows and UT/ST cases with effective changes in the selected window, then checks those changed cases against the current HEAD NPU support evidence.

## Instructions

1. Read [references/repo-signals.md](./references/repo-signals.md) for repo-specific boundaries.
2. Identify the target `verl` repository root, for example `{PATH}/verl`.
3. Run the scanner. The commands below assume the working directory is the root of `verl-ascend-recipe` (where `.agent/` lives). You can also `cd` into `scripts/` and run `python scan_ascend_ci_case_diff.py ...` directly.

```shell
python .agent/skills/ascend-ci-case-diff-scan/scripts/scan_ascend_ci_case_diff.py \
  --repo-root {PATH}/verl \
  --output-dir ./report/ascend-ci-case-diff-scan
```

Optional:

```shell
python .agent/skills/ascend-ci-case-diff-scan/scripts/scan_ascend_ci_case_diff.py \
  --repo-root {PATH}/verl \
  --output-dir ./report/ascend-ci-case-diff-scan \
  --since-days 7
```

`--since-days` must be a positive integer. If it is omitted, only the full report is generated.

## Extraction Rules

Recognize only workflow commands that visibly execute tests (see [references/repo-signals.md](./references/repo-signals.md#extractable-command-forms) for details):

- `pytest ... tests/...`
- `bash tests/.../*.sh` or `bash examples/.../*.sh`
- `torchrun ... tests/...`

For each extracted case, preserve: `workflow_name`, `job_name`, `step_name`, `command_type`, `target`, `raw_command`, `signature`.

When the same command appears in different workflow/job/step contexts, keep cases distinct. For UT, expand to function-level or test-method-level whenever the target file can be parsed. For ST scripts, record only scripts explicitly invoked by the workflow step — do not inspect the script body.

Case matching uses `command_type`, `target`, and `signature` (see [references/repo-signals.md](./references/repo-signals.md#matching-expectations) for detailed matching rules).

## Past-N-Days Analysis

When `--since-days N` is set, the scanner walks the current branch first-parent history for commits merged in the last `N` days, compares the window-start snapshot with current `HEAD`, and reports only effective final-state changes. See [references/repo-signals.md](./references/repo-signals.md#past-n-days-expectations) for the full scoping and column semantics.

## Boundaries

- Ignore commented workflow lines.
- Ignore workflows matched by `config/workflow_scope.json`, including shared baseline and workflows without meaningful CPU/GPU-vs-NPU test coverage.
- Do not execute tests.
- Do not expand GitHub Actions matrices.
- Keep the final report in English only.

## Classification Rules

Output categories: `aligned`, `missing_in_npu_workflows`, `manual_review_needed`, `npu_only`.

Exact target matches are the strongest signal; different signatures for the same target fall back to `manual_review_needed`. See [references/repo-signals.md](./references/repo-signals.md#matching-expectations) for the full matching rules.

## Reporting

The scanner writes `report.md` and `report.xlsx` to the requested output directory.
If `--since-days` is provided, it also writes `report-past-N.md` and `report-past-N.xlsx`.

The reports contain:

- ignored workflows
- scanned workflows with CPU/GPU and NPU case counts
- UT details
- ST details

Within the UT and ST sections, matched, CPU/GPU-only, NPU-only, and manual-review cases are shown in that order, with CPU/GPU and NPU references adjacent for easier comparison. The Excel workbook stores these sections as four sheets: `Ignored Workflows`, `Scanned Workflows`, `UT Cases`, and `ST Cases`.
