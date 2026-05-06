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
- Snapshot artifacts under `reports/`
- Optional one-command refresh that rebuilds `ao_status.csv` before report generation
- Default CUDA log source from latest successful `regression_test.yml` job `CUDA 2.10`

## When To Use
- User asks for weekly report / 周报
- User asks to summarize CUDA vs XPU pass/skip
- User asks to analyze failures in XPU logs or latest scheduled workflow run
- User asks to produce reproducible report output files

## Procedure
1. Refresh status is enabled by default.
   - Use `--no-refresh-status` to skip rebuilding `ao_status.csv`.
2. Run [weekly report script](./scripts/generate_weekly_report.py).
3. GitHub scheduled-run mode is enabled by default.
   - Use `--no-fetch-latest-scheduled` to force local xpu-log only.
4. CUDA refresh log fetch is enabled by default.
   - Source: latest successful `regression_test.yml` job matching `CUDA 2.10`
   - Use `--no-fetch-latest-cuda` to force local cuda-log only.
5. Confirm generated files under `reports/`:
   - `weekly_report_<week>.md`
   - `ao_status_<week>.csv`
   - `xpu_failures_<week>.csv`
6. Share key numbers in chat and point user to output files.

## Script
- [generate_weekly_report.py](./scripts/generate_weekly_report.py)

Cross-platform note:
- Linux/macOS: standard Python + optional `curl` fallback
- Windows: standard Python + PowerShell fallback

## One-Command Example
`python .github/skills/weekly-xpu-report/scripts/generate_weekly_report.py --cuda-log cuda_0407.txt --week-tag 20260506`

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
