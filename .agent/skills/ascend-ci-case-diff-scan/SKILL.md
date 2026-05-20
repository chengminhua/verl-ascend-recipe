---
name: ascend-ci-case-diff-scan
description: Scan an external verl repository for Ascend/NPU CI coverage gaps by comparing CPU/GPU workflow cases against NPU workflows. Use when Codex needs to audit workflow and case-level parity or generate Markdown and Excel reports about workflow execution differences.
---

# Ascend CI Case Diff Scan

Audit Ascend CI coverage in a target `verl` repository with the scanner shipped in this skill.

## Overview

Use this skill when you need to compare CPU/GPU workflow test coverage with NPU workflow coverage and produce English Markdown and Excel reports. The scanner is static: it reads GitHub Actions workflow `run:` commands, preserves workflow/job/step context, and does not execute tests.

The reports show ignored workflows, paired CPU/GPU and NPU workflows, UT/ST case counts, matched cases, missing cases, NPU-only cases, and manual-review cases.

## Instructions

1. Read [references/repo-signals.md](./references/repo-signals.md) for repo-specific boundaries.
2. Identify the target `verl` repository root, for example `{PATH}/verl`.
3. Run the scanner from this repository:

```shell
python .agent/skills/ascend-ci-case-diff-scan/scripts/scan_ascend_ci_case_diff.py \
  --repo-root {PATH}/verl \
  --output-dir ./report/ascend-ci-case-diff-scan
```

## Extraction Rules

Recognize only workflow commands that visibly execute tests:

- `pytest ... tests/...`
- `bash tests/.../*.sh`
- `bash examples/.../*.sh`
- `torchrun ... tests/...`

For each extracted case, preserve:

- `workflow_name`
- `job_name`
- `step_name`
- `command_type`
- `target`
- `raw_command`
- `signature`

When the same command is repeated, keep cases distinct by `workflow_name`, `job_name`, and `step_name`.
For UT expansion, treat Python test functions and `Test*::test_*` methods as the reporting unit whenever the target file can be parsed.
For ST scripts, record only scripts explicitly invoked by the workflow step; do not inspect the script body for nested commands.

## Boundaries

- Ignore commented workflow lines.
- Ignore workflows matched by `config/workflow_scope.json`, including shared baseline and workflows without meaningful CPU/GPU-vs-NPU test coverage.
- Do not execute tests.
- Do not expand GitHub Actions matrices.
- Keep the final report in English only.

## Classification Rules

Use these output categories:

- `aligned`
- `missing_in_npu_workflows`
- `manual_review_needed`
- `npu_only`

Treat matching conservatively:

- Exact target matches are the strongest signal.
- Compatible signatures can be aligned.
- Different signatures for the same target should fall back to manual review.

## Reporting

The scanner writes `report.md` and `report.xlsx` to the requested output directory.

The reports contain:

- ignored workflows
- scanned workflows with CPU/GPU and NPU case counts
- UT details
- ST details

Within the UT and ST sections, matched, CPU/GPU-only, NPU-only, and manual-review cases are shown in that order, with CPU/GPU and NPU references adjacent for easier comparison. The Excel workbook stores these sections as four sheets: `Ignored Workflows`, `Scanned Workflows`, `UT Cases`, and `ST Cases`.
