# Weekly Workflow

Run from repo root:

```bash
python .github/skills/weekly-xpu-report/scripts/generate_weekly_report.py --week-tag 20260506
```

One command to refresh ao_status.csv and generate report:

```bash
python .github/skills/weekly-xpu-report/scripts/generate_weekly_report.py \
  --output-dir reports \
  --week-tag 20260506
```

This command defaults to:
- Fetch XPU log from latest scheduled `xpu_test.yml`
- Fetch CUDA log from latest successful `regression_test.yml` job containing `CUDA 2.10`
- Refresh and write `reports/ao_status_<week>.csv`

Default outputs are self-contained in `reports/`:
- `reports/ao_status_<week>.csv`
- `reports/xpu_failures_<week>.csv`
- `reports/weekly_report_<week>.md`

Disable refresh if you only want to use existing ao_status.csv:

```bash
python .github/skills/weekly-xpu-report/scripts/generate_weekly_report.py \
  --no-refresh-status \
  --week-tag 20260506
```

Force local CUDA log only:

```bash
python .github/skills/weekly-xpu-report/scripts/generate_weekly_report.py \
  --no-fetch-latest-cuda \
  --cuda-log cuda_0407.txt \
  --week-tag 20260506
```

Optional args:

```bash
python .github/skills/weekly-xpu-report/scripts/generate_weekly_report.py \
  --csv-path ao_status.csv \
  --xpu-log xpu_0407.txt \
  --output-dir reports \
  --week-tag 20260506
```

Fetch latest completed scheduled run directly from GitHub Actions:

```bash
python .github/skills/weekly-xpu-report/scripts/generate_weekly_report.py \
  --csv-path ao_status.csv \
  --github-repo pytorch/ao \
  --github-workflow xpu_test.yml \
  --output-dir reports \
  --week-tag 20260506
```

Disable scheduled fetch and use local xpu-log only:

```bash
python .github/skills/weekly-xpu-report/scripts/generate_weekly_report.py \
  --no-fetch-latest-scheduled \
  --xpu-log xpu_0407.txt \
  --week-tag 20260506
```

If rate-limited, set a token first:

```bash
$env:GITHUB_TOKEN = "<token>"
python .github/skills/weekly-xpu-report/scripts/generate_weekly_report.py --fetch-latest-scheduled --week-tag 20260506
```

Linux/macOS:

```bash
export GITHUB_TOKEN="<token>"
python .github/skills/weekly-xpu-report/scripts/generate_weekly_report.py --week-tag 20260506
```

Outputs:
- `reports/weekly_report_<week>.md`
- `reports/ao_status_<week>.csv`
- `reports/xpu_failures_<week>.csv`
