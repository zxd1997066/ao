# Failure Fix Workflow

## Quick Start

Run from repo root:

```bash
# Rule-based fixes only (fast, no API call)
python .github/skills/xpu-failure-fix/scripts/fix_failures.py --week-tag 20260513 --apply-fixes

# LLM auto-fix for failures not covered by known rules (dry-run first)
$env:GITHUB_TOKEN = (Get-Content .\token.txt -Raw).Trim()  # Windows
export GITHUB_TOKEN=$(cat token.txt)                        # Linux/macOS
python .github/skills/xpu-failure-fix/scripts/fix_failures.py --week-tag 20260513 --llm-fix

# LLM + apply
python .github/skills/xpu-failure-fix/scripts/fix_failures.py --week-tag 20260513 --llm-fix --apply-fixes
```

Single entrypoint:
- `fix_failures.py` generates both analysis outputs and fix outputs

This will:
1. Read `reports/xpu_failures_20260513.csv`
2. Categorize failures
3. Generate `reports/failure_analysis_20260513.md`
4. Generate `reports/failure_summary_20260513.json`
5. Apply known safe code fixes
6. Generate `reports/fix_report_20260513.md`

## Full Usage

```bash
python .github/skills/xpu-failure-fix/scripts/fix_failures.py \
  --week-tag 20260513 \
  --output-dir reports \
  --apply-fixes
```

Dry-run only (no source changes):

```bash
python .github/skills/xpu-failure-fix/scripts/fix_failures.py \
  --week-tag 20260513 \
  --output-dir reports
```

## Failure Categories

- **import_error**: Missing dependencies, version mismatches
- **hardware**: GPU/memory/device issues
- **quantization_test**: Quantization module tests
- **float8_test**: Float8 dtype tests
- **sparsity_test**: Sparsity module tests
- **logic_error**: Assertion/logic failures
- **flaky**: Intermittent/timing issues
- **unknown**: Needs investigation

## Next Steps

1. Review `failure_analysis_<week>.md`
2. Review `fix_report_<week>.md`
3. Fix remaining manual items (if any)
4. Re-run targeted tests to verify
5. Create PR with patch summary
