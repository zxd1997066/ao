---
name: xpu-failure-fix
description: 'Analyze and fix XPU test failures from weekly reports. Use when analyzing failure patterns, fixing failing tests, or planning failure mitigation. Use keywords: fix failures, failure analysis, XPU debug, 修复失败, 失败分析.'
---

# XPU Failure Fix

Analyze XPU test failures from weekly reports, then apply known safe fixes automatically.

## When to Use
- User has a weekly report with XPU failures
- Need to categorize and prioritize failures
- Want fixing recommendations or root cause analysis
- Planning failure mitigation strategies

## Procedure
1. **Get the failure list**: Use the latest `xpu_failures_<week>.csv` from `reports/`
2. **Run the single entrypoint**: [fix script](./scripts/fix_failures.py)
3. **Optionally apply known fixes**: Add `--apply-fixes`
4. **Review output**:
   - `failure_analysis_<week>.md` — categorized failures with recommendations
   - `failure_summary_<week>.json` — structured summary for automation
   - `fix_report_<week>.md` — what was patched and what still needs manual fix
5. **Verify and PR**:
   - Run targeted tests for changed files
   - Submit PR with fix report attached

## Script
- [fix_failures.py](./scripts/fix_failures.py) — Apply known safe fixes and generate fix report

## Commands

Dry-run fix planning with auto-detect:
`python .github/skills/xpu-failure-fix/scripts/fix_failures.py --week-tag <YYYYMMDD>`

Apply known rule-based fixes:
`python .github/skills/xpu-failure-fix/scripts/fix_failures.py --week-tag <YYYYMMDD> --apply-fixes`

Let LLM analyze and suggest fixes for unknown failures (dry-run):
`python .github/skills/xpu-failure-fix/scripts/fix_failures.py --week-tag <YYYYMMDD> --llm-fix`

Let LLM analyze and apply fixes automatically:
`python .github/skills/xpu-failure-fix/scripts/fix_failures.py --week-tag <YYYYMMDD> --llm-fix --apply-fixes`

Override failure CSV manually:
`python .github/skills/xpu-failure-fix/scripts/fix_failures.py --failure-csv reports/xpu_failures_<YYYYMMDD>.csv --week-tag <YYYYMMDD> --llm-fix --apply-fixes`

## Failure Categories
- **Import/Environment**: Missing dependencies, version mismatches
- **Hardware**: GPU/memory issues, device-specific failures
- **Logic**: Test logic errors, incorrect assertions
- **Flaky**: Intermittent failures, timing issues
- **Unknown**: Needs manual investigation

## Output
- `reports/failure_analysis_<week>.md` — Human-readable analysis with recommendations
- `reports/failure_summary_<week>.json` — Structured data for tooling
- `reports/fix_report_<week>.md` — Applied fixes and remaining manual actions
- `reports/fix_report_<week>.json` — Structured fix execution summary
