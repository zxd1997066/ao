---
name: weekly-xpu-report
description: 'Generate weekly test report from ao_status.csv and XPU CI logs. Supports fetching latest scheduled run from GitHub Actions xpu_test.yml. Use when user asks weekly report, XPU failure analysis, CUDA/XPU summary, or 周报/失败分析/每周报告.'
argument-hint: 'Optional: week tag like 20260506'
user-invocable: true
---

# Weekly XPU Report

Generate a weekly report that includes:
- Current CSV summary from `reports/ao_status_<week>.csv` (default)
- Failure analysis from XPU logs (local file or latest scheduled GitHub Actions run)
- CUDA UT by device summary (including H100/L4 when available)
- Per-file device matrix with `\\` placeholder for missing device data
- Snapshot artifacts under `reports/`
- Optional one-command refresh that rebuilds `ao_status.csv` before report generation
- Default CUDA log source from latest successful `regression_test.yml` job

## When To Use
- User asks for weekly report / 周报
- User asks to summarize CUDA vs XPU pass/skip
- User asks to analyze failures in XPU logs or latest scheduled workflow run
- User asks to produce reproducible report output files
- User asks to include H100/L4 results in the report or CSV matrix

## Procedure
1. Refresh status is enabled by default.
   - Use `--no-refresh-status` to skip rebuilding `ao_status.csv`.
2. Run [weekly report script](./scripts/generate_weekly_report.py).
3. GitHub scheduled-run mode is enabled by default.
   - Use `--no-fetch-latest-scheduled` to force local xpu-log only.
4. CUDA refresh log fetch is enabled by default.
   - Source: latest successful `regression_test.yml` CUDA job
   - Use `--no-fetch-latest-cuda` to force local cuda-log only.
5. For device-aware report (H100/L4), enable these options:
   - `--include-cuda-ut-results`
   - `--include-per-file-device-matrix`
   - `--cuda-ut-workflows 1xL4_tests.yml,1xH100_tests.yml`
   - Optional local backfill: `--device-log-mapping L4:l4.txt,H100:h100.txt`
6. Confirm generated files under `reports/`:
   - `weekly_report_<week>.md`
   - `ao_status_<week>.csv`
   - `xpu_failures_<week>.csv`
   - `cuda_ut_by_device_<week>.csv` (when CUDA UT by device is enabled)
   - `ao_status_by_device_<week>.csv` (when per-file device matrix is enabled)
7. Share key numbers in chat and point user to output files.

## Script
- [generate_weekly_report.py](./scripts/generate_weekly_report.py)

Cross-platform note:
- Linux/macOS: standard Python + optional `curl` fallback
- Windows: standard Python + PowerShell fallback

## One-Command Example (H100/L4 Included)
`python .github/skills/weekly-xpu-report/scripts/generate_weekly_report.py --week-tag 20260506 --xpu-run-url https://github.com/pytorch/ao/actions/runs/<run_id>/job/<job_id> --include-cuda-ut-results --include-per-file-device-matrix --cuda-ut-workflows 1xL4_tests.yml,1xH100_tests.yml --device-log-mapping L4:l4.txt,H100:h100.txt`

Notes:
- `cuda_0407.txt` is blocked and must not be used.
- If remote per-file fetch is unstable, keep `--device-log-mapping` enabled and provide local logs.

By default, all generated files stay under `reports/` and no root `ao_status.csv` is required.

## Output Format
The markdown report includes this summary format:
- `Total | Pass | Skip | Pass Rate`
- Rows for `CUDA` and `XPU`

It also includes:
- total failed tests from XPU log
- failure distribution by test file
- top failure messages
- complete failure list reference in CSV
- optional `CUDA_UT_BY_DEVICE` section in `ao_status_<week>.csv`
- optional `PER_FILE_DEVICE_MATRIX` section in `ao_status_<week>.csv`
