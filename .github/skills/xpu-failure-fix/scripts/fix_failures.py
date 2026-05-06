#!/usr/bin/env python3
"""Analyze XPU failures and optionally apply fixes via GitHub Models LLM."""

import argparse
import csv
import json
import os
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def read_failures(csv_path: Path):
    failures = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            failures.append(row)
    return failures


def resolve_failure_csv(failure_csv_arg: str, week_tag: str, output_dir: Path):
    if failure_csv_arg:
        return Path(failure_csv_arg)
    if week_tag:
        tagged = output_dir / f"xpu_failures_{week_tag}.csv"
        if tagged.exists():
            return tagged
    candidates = sorted(output_dir.glob('xpu_failures_*.csv'), reverse=True)
    if candidates:
        return candidates[0]
    return output_dir / f"xpu_failures_{week_tag or 'latest'}.csv"


def group_failures(failures):
    by_file = defaultdict(list)
    for failure in failures:
        by_file[failure.get("file", "")].append(failure)
    return by_file


# ---------------------------------------------------------------------------
# LLM-based auto-fix helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, max_chars: int = 6000) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + f"\n\n... [{len(text) - max_chars} chars truncated] ...\n\n" + text[-half:]


def guess_source_file(test_file: str, repo_root: Path):
    """Heuristically map a test file path to the corresponding source file."""
    test_path = Path(test_file)
    parts = test_path.parts
    if not parts or parts[0] != 'test':
        return None

    # test/quantization/foo.py  ->  torchao/quantization/foo.py
    candidate = repo_root / Path('torchao') / Path(*parts[1:])
    if candidate.exists():
        return candidate

    # test/quantization/test_foo.py  ->  torchao/quantization/foo.py
    stem = test_path.stem
    if stem.startswith('test_'):
        module = stem[5:]
        candidate2 = repo_root / Path('torchao') / Path(*parts[1:-1]) / f"{module}.py"
        if candidate2.exists():
            return candidate2

    return None


def llm_suggest_fix(failures: list, test_file: str, repo_root: Path,
                    token: str, model: str = "gpt-4o-mini") -> dict:
    """Call GitHub Models API to suggest a fix for a failure group."""
    error_messages = list(dict.fromkeys(
        f.get('message', '') for f in failures if f.get('message')
    ))[:5]
    nodeids = [f.get('nodeid', '') for f in failures][:5]

    test_path = repo_root / test_file
    test_content = _truncate(test_path.read_text(encoding='utf-8')) if test_path.exists() else '(file not found)'

    src_file = guess_source_file(test_file, repo_root)
    src_content = _truncate(src_file.read_text(encoding='utf-8')) if src_file else '(not found)'
    src_rel = str(src_file.relative_to(repo_root)).replace('\\', '/') if src_file else 'unknown'

    prompt = (
        "You are analyzing XPU (Intel GPU) test failures in the TorchAO PyTorch library.\n\n"
        "Failed tests:\n" + "\n".join(f"  {n}" for n in nodeids) + "\n\n"
        "Error messages:\n" + "\n".join(f"  {m}" for m in error_messages) + "\n\n"
        f"Test file: {test_file}\n```python\n{test_content}\n```\n\n"
        f"Source file: {src_rel}\n```python\n{src_content}\n```\n\n"
        "Identify the root cause and provide the minimal code fix.\n"
        "Return ONLY a JSON object with these keys:\n"
        '  "root_cause": one sentence describing the root cause\n'
        '  "fix_description": what to change and why\n'
        '  "patches": list of {"file": "repo-relative/path.py", "search": "exact text to find", "replace": "replacement text"}\n'
        '  "confidence": "high", "medium", or "low"\n\n'
        "If the issue is environment/hardware (not fixable via code), return empty patches list."
    )

    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "max_tokens": 2000,
        "temperature": 0,
    }).encode()

    url = "https://models.inference.ai.azure.com/chat/completions"
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST",
    )
    raw = None
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            raw = resp.read().decode()
    except urllib.error.HTTPError as exc:
        return {"error": f"HTTP {exc.code}: {exc.read().decode()[:300]}"}
    except Exception:
        # urllib failed (common on Windows) – fall back to PowerShell / curl
        body_str = payload.decode()
        ps_body = body_str.replace("'", "''")
        if sys.platform == "win32":
            import subprocess
            ps_cmd = (
                f"$r=Invoke-WebRequest -Uri '{url}' -Method POST"
                f" -Headers @{{'Content-Type'='application/json';'Authorization'='Bearer {token}'}}"
                f" -Body '{ps_body}' -UseBasicParsing; $r.Content"
            )
            try:
                result = subprocess.run(
                    ["powershell", "-Command", ps_cmd],
                    capture_output=True, text=True, timeout=120,
                )
                if result.returncode == 0:
                    raw = result.stdout.strip()
                else:
                    return {"error": f"PowerShell fallback failed: {result.stderr[:300]}"}
            except Exception as exc2:
                return {"error": f"PowerShell fallback exception: {exc2}"}
        else:
            import subprocess, tempfile
            tmp = tempfile.NamedTemporaryFile(mode='wb', suffix='.json', delete=False)
            tmp.write(payload)
            tmp.close()
            try:
                result = subprocess.run(
                    ["curl", "-s", "-X", "POST", url,
                     "-H", "Content-Type: application/json",
                     "-H", f"Authorization: Bearer {token}",
                     "--data-binary", f"@{tmp.name}"],
                    capture_output=True, text=True, timeout=120,
                )
                if result.returncode == 0:
                    raw = result.stdout.strip()
                else:
                    return {"error": f"curl fallback failed: {result.stderr[:300]}"}
            except Exception as exc2:
                return {"error": f"curl fallback exception: {exc2}"}
            finally:
                Path(tmp.name).unlink(missing_ok=True)

    if raw is None:
        return {"error": "no response received"}
    try:
        body = json.loads(raw)
        return json.loads(body["choices"][0]["message"]["content"])
    except Exception as exc:
        return {"error": f"parse error: {exc}; raw[:200]={raw[:200]}"}


def apply_llm_patches(patches: list, repo_root: Path) -> list:
    results = []
    for patch in patches:
        fpath = repo_root / patch.get("file", "")
        if not fpath.exists():
            results.append({"file": patch.get("file"), "changed": False, "detail": "file not found"})
            continue
        search = patch.get("search", "")
        replace = patch.get("replace", "")
        if not search:
            results.append({"file": patch.get("file"), "changed": False, "detail": "empty search string"})
            continue
        text = fpath.read_text(encoding="utf-8")
        if search not in text:
            results.append({"file": patch.get("file"), "changed": False, "detail": "search string not found in file"})
            continue
        if text.count(search) > 1:
            results.append({"file": patch.get("file"), "changed": False,
                             "detail": f"ambiguous: {text.count(search)} matches"})
            continue
        fpath.write_text(text.replace(search, replace, 1), encoding="utf-8")
        results.append({"file": patch.get("file"), "changed": True, "detail": "patched by LLM suggestion"})
    return results


# ---------------------------------------------------------------------------
# Failure analysis (categorize + report)
# ---------------------------------------------------------------------------

def categorize_failure(nodeid, message):
    message_lower = (message or "").lower()
    nodeid_lower = (nodeid or "").lower()

    if any(x in message_lower for x in ['importerror', 'modulenotfounderror', 'no module', 'cannot find', 'missing']):
        return 'import_error'
    if any(x in message_lower for x in ['cuda', 'gpu', 'device', 'out of memory', 'oom', 'not available']):
        return 'hardware'
    if any(x in message_lower for x in ['assertionerror', 'assertion failed', 'assert', 'valueerror', 'typeerror', 'runtimeerror']):
        return 'logic_error'
    if any(x in message_lower for x in ['timeout', 'hang', 'deadlock', 'flaky', 'intermittent']):
        return 'flaky'
    if 'float8' in nodeid_lower:
        return 'float8_test'
    if 'quantiz' in nodeid_lower or 'quant' in nodeid_lower:
        return 'quantization_test'
    if 'sparse' in nodeid_lower:
        return 'sparsity_test'
    return 'unknown'


def analyze_failure_groups(failures):
    analysis = {
        'total': len(failures),
        'by_category': defaultdict(list),
        'by_file': defaultdict(list),
        'by_message': defaultdict(int),
    }
    for failure in failures:
        category = categorize_failure(failure.get('nodeid', ''), failure.get('message', ''))
        analysis['by_category'][category].append(failure)
        analysis['by_file'][failure.get('file', '')].append(failure)
        if failure.get('message'):
            analysis['by_message'][failure['message']] += 1
    analysis['by_category'] = dict(analysis['by_category'])
    analysis['by_file'] = dict(analysis['by_file'])
    return analysis


def write_analysis_report(output_dir: Path, week_tag: str, analysis):
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / f"failure_analysis_{week_tag}.md"
    json_path = output_dir / f"failure_summary_{week_tag}.json"

    priorities = {
        'import_error': 'HIGH', 'hardware': 'HIGH',
        'quantization_test': 'MEDIUM', 'float8_test': 'MEDIUM',
        'sparsity_test': 'LOW', 'logic_error': 'MEDIUM',
        'flaky': 'LOW', 'unknown': 'MEDIUM',
    }

    lines = [
        f"# XPU Failure Analysis ({week_tag})",
        "", f"**Total failures: {analysis['total']}**", "",
        "## Summary by Category",
        "| Category | Count | Priority |", "|---|---:|---|",
    ]
    for category in sorted(analysis['by_category'].keys()):
        count = len(analysis['by_category'][category])
        lines.append(f"| {category} | {count} | {priorities.get(category, 'MEDIUM')} |")

    lines.extend(["", "## Failures by File (Top 20)", "| File | Count |", "|---|---:|"])
    files_sorted = sorted(analysis['by_file'].items(), key=lambda x: (-len(x[1]), x[0]))
    for file_path, file_failures in files_sorted[:20]:
        lines.append(f"| {file_path} | {len(file_failures)} |")

    lines.extend(["", "## Top Failure Messages (Top 15)", "| Message | Count |", "|---|---:|"])
    msg_sorted = sorted(analysis['by_message'].items(), key=lambda x: (-x[1], x[0]))
    for msg, count in msg_sorted[:15]:
        safe_msg = msg.replace('|', '\\|')[:100]
        lines.append(f"| {safe_msg}... | {count} |")

    lines.extend([
        "", "## Recommendations", "",
        "### High Priority",
        "- **Import/Environment errors**: Check dependencies, Python version compatibility",
        "- **Hardware errors**: Verify XPU environment setup, device availability",
        "", "### Medium Priority",
        "- **Quantization/Float8/Logic errors**: Review test assertions and implementation",
        "- **Unknown**: Requires manual investigation",
        "", "### Low Priority",
        "- **Flaky tests**: Add retry logic or stabilize timing",
        "- **Sparsity tests**: Verify format and data alignment",
    ])

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {
        'week': week_tag,
        'total_failures': analysis['total'],
        'by_category': {cat: len(items) for cat, items in analysis['by_category'].items()},
        'top_files': [{'file': name, 'count': len(items)} for name, items in files_sorted[:10]],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return md_path, json_path


# ---------------------------------------------------------------------------
# Fix report
# ---------------------------------------------------------------------------

def write_fix_report(output_dir: Path, week_tag: str, llm_results: list, dry_run: bool):
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / f"fix_report_{week_tag}.md"
    json_path = output_dir / f"fix_report_{week_tag}.json"

    total_patched = sum(
        1 for item in llm_results
        for pr in item.get('patch_results', []) if pr.get('changed')
    )

    lines = [
        f"# XPU Failure Fix Report ({week_tag})",
        "",
        f"Files analyzed by LLM: {len(llm_results)}",
        f"Patches applied: {total_patched}",
        f"Dry run: {dry_run}",
    ]

    if not llm_results:
        lines.extend(["", "No LLM analysis performed (pass --llm-fix to enable)."])
    else:
        lines.extend(["", "## LLM Fix Results"])
        for item in llm_results:
            file_ = item.get('file', '?')
            suggestion = item.get('suggestion', {})
            error = suggestion.get('error', '')
            if error:
                lines.append(f"- **{file_}**: LLM error — {error}")
                continue
            confidence = suggestion.get('confidence', '?')
            root_cause = suggestion.get('root_cause', '?')
            desc = suggestion.get('fix_description', '')
            patch_results = item.get('patch_results', [])
            lines.append(f"- **{file_}** (confidence: {confidence})")
            lines.append(f"  - Root cause: {root_cause}")
            if desc:
                lines.append(f"  - Fix: {desc}")
            for pr in patch_results:
                lines.append(f"  - patch `{pr['file']}`: changed={pr['changed']}, {pr['detail']}")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps({"week": week_tag, "dry_run": dry_run,
                                     "llm_results": llm_results}, indent=2), encoding="utf-8")
    return md_path, json_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze and fix XPU test failures via LLM")
    parser.add_argument("--failure-csv", default="",
                        help="Path to xpu_failures CSV. If omitted, auto-detect latest under --output-dir")
    parser.add_argument("--week-tag", default="", help="Week tag for report naming (default: today YYYYMMDD)")
    parser.add_argument("--output-dir", default="reports", help="Output directory (default: reports)")
    parser.add_argument("--llm-fix", action="store_true",
                        help="Call GitHub Models API to analyze and suggest fixes for each failing test file")
    parser.add_argument("--apply-fixes", action="store_true",
                        help="Apply LLM-suggested patches to source files (requires --llm-fix)")
    parser.add_argument("--llm-model", default="gpt-4o-mini",
                        help="GitHub Models model name (default: gpt-4o-mini)")
    parser.add_argument("--github-token", default="",
                        help="GitHub token for Models API (falls back to GITHUB_TOKEN env var)")
    parser.add_argument("--repo-root", default=".",
                        help="Repository root where source files are located")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    week_tag = args.week_tag or datetime.now().strftime('%Y%m%d')
    failure_csv = resolve_failure_csv(args.failure_csv, week_tag, output_dir)
    if not failure_csv.exists():
        print(f"Error: failure csv not found: {failure_csv}", file=sys.stderr)
        return 1

    week_tag = args.week_tag or failure_csv.stem.replace("xpu_failures_", "") or week_tag

    failures = read_failures(failure_csv)
    analysis = analyze_failure_groups(failures)
    analysis_md, analysis_json = write_analysis_report(output_dir, week_tag, analysis)
    by_file = group_failures(failures)

    llm_results = []
    if args.llm_fix:
        token = args.github_token or os.environ.get('GITHUB_TOKEN', '')
        if not token:
            print("Warning: --llm-fix requires a GitHub token (set GITHUB_TOKEN env var or --github-token)",
                  file=sys.stderr)
        else:
            for file, file_failures in by_file.items():
                if not file:
                    continue
                print(f"  LLM analyzing: {file} ({len(file_failures)} failures)...")
                suggestion = llm_suggest_fix(
                    file_failures, file, Path(args.repo_root), token, args.llm_model
                )
                patch_results = []
                if not suggestion.get('error') and suggestion.get('patches'):
                    if args.apply_fixes:
                        patch_results = apply_llm_patches(suggestion['patches'], Path(args.repo_root))
                    else:
                        patch_results = [{"file": p.get('file'), "changed": False,
                                          "detail": "dry-run: use --apply-fixes to apply"}
                                         for p in suggestion['patches']]
                llm_results.append({"file": file, "suggestion": suggestion, "patch_results": patch_results})

    md_path, json_path = write_fix_report(output_dir, week_tag, llm_results, dry_run=not args.apply_fixes)

    print(f"failure_csv={failure_csv}")
    print(f"analysis_md={analysis_md}")
    print(f"analysis_json={analysis_json}")
    print(f"fix_report_md={md_path}")
    print(f"fix_report_json={json_path}")
    for item in llm_results:
        changed_any = any(p.get('changed') for p in item.get('patch_results', []))
        conf = item.get('suggestion', {}).get('confidence', '?')
        err = item.get('suggestion', {}).get('error', '')
        status = f"error:{err}" if err else f"confidence:{conf},patched:{changed_any}"
        print(f"llm_{Path(item['file']).stem}={status}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
