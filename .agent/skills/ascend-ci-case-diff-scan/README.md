# Ascend CI Case Diff Scan

This skill statically audits a target `verl` repository and reports workflow/case alignment between CPU/GPU and NPU workflows.

## What it does

- Excludes non-test and non-comparable workflows through `config/workflow_scope.json`
- Pairs CPU/GPU workflows with their NPU counterparts
- Counts UT cases at the test-function or test-method level
- Counts ST cases at the command level
- Writes English `report.md` and `report.xlsx` files
- Optionally writes `report-past-N.md` and `report-past-N.xlsx` for recent merged commits when `--since-days N` is set

## Workflow pairing

- `foo.yml` pairs with `foo_ascend.yml`
- `gpu_unit_tests.yml` pairs with `npu_unit_tests.yml`
- Standalone NPU workflows such as `e2e_ascend.yml` and `nightly_ascend.yml` remain NPU-only

## How cases are classified

- `pytest` commands → **UT** cases, expanded to function-level (`test_*` / `Test*::test_*`) whenever the Python file can be parsed
- `torchrun`, `bash tests/*.sh`, and `bash examples/*.sh` → **ST** cases, recorded at the command level
- Repeated commands are kept distinct by `workflow name`, `job name`, and `step name`

## CLI usage

```bash
python scripts/scan_ascend_ci_case_diff.py \
  --repo-root {PATH}/verl \
  --output-dir ./outputs
```

- `--repo-root` — path to the target `verl` repository root
- `--since-days N` — (optional) also generate a past-N-days report; must be a positive integer

## Past-N-days behavior

- Walks first-parent history for merged commits in the last `N` days
- Reports only effective final-state changes (additions later removed or reverted are excluded)
- `Related Commits` lists commits that touched the workflow or the changed case target

## Output

The reports include:

- ignored workflows
- scanned workflows with CPU/GPU and NPU case counts
- UT details
- ST details

Past-N-days reports also include:

- a summary of CPU/GPU workflows that are not fully aligned with NPU
- changed workflow rows with window-start and current-HEAD case counts
- changed case details with signatures and related commits
- commit details for the merged commits inside the selected window

Within UT and ST sections, the report shows matched, CPU/GPU-only, NPU-only, and manual-review cases in that order, with adjacent CPU/GPU and NPU references for easy comparison.

The Excel workbook contains four sheets: `Ignored Workflows`, `Scanned Workflows`, `UT Cases`, and `ST Cases`.

The past-N-days workbook contains summary, changed-workflow, changed-case, and commit-detail sheets.
