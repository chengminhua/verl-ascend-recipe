# Repo Signals

Use these target-repo facts when running `ascend-ci-case-diff-scan` against an external `verl` checkout.

## Primary inputs

- `.github/workflows/*.yml` and `.github/workflows/*.yaml`

## Workflow scope

Exclude workflows that are not part of meaningful CPU/GPU-vs-NPU test coverage. The authoritative ignored set is maintained in `config/workflow_scope.json` (supports exact names and `glob` patterns like `docker-*.yml`).

See that file for the current list.

## Workflow families

- `foo.yml` pairs with `foo_ascend.yml`
- `gpu_unit_tests.yml` pairs with `npu_unit_tests.yml`
- Standalone NPU workflows such as `e2e_ascend.yml` and `nightly_ascend.yml` are treated as NPU-only unless a matching CPU/GPU workflow exists

## Extractable command forms

Recognize only workflow commands that visibly execute tests:

- `pytest ... tests/...`
- `bash tests/.../*.sh`
- `bash examples/.../*.sh`
- `torchrun ... tests/...`

## Matching expectations

- Exact target matches are the strongest signal.
- Same target with materially different env or argument prefixes should usually become `manual_review_needed`.
- Repeated commands should remain distinct when they appear in different workflow, job, or step contexts.
- For UT, prefer function-level or test-method-level comparison over file-level comparison because broad `pytest tests/...` commands and `--ignore-glob` options can hide partial support.

## Past-N-days expectations

- The past-N-days report is scoped to effective final-state changes between the window-start snapshot and current `HEAD`.
- Commits that add or modify a case and are later reverted or removed in the same window should not appear as changed cases.
- The changed-workflow table is CPU/GPU oriented. NPU workflows should appear as support references for changed CPU/GPU cases, not as primary changed workflow rows.
- `UT/ST Not Fully Aligned with NPU Count` covers changed CPU/GPU cases whose current `HEAD` NPU state is either missing or divergent enough to require manual review.
- Direct changes to `tests/**` or `examples/**/*.sh` are relevant only when they are reachable from an extracted workflow case.
