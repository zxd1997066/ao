#!/usr/bin/env python3
"""Analyze XPU failures and optionally apply fixes via GitHub Models LLM."""

import argparse
import csv
import json
import os
import re
import subprocess
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


def _likely_source_file_from_test_file(test_file: str) -> str:
    p = Path(test_file)
    parts = p.parts
    if not parts:
        return "unknown"
    if parts[0] != "test":
        return "unknown"

    if p.stem.startswith("test_"):
        mod = p.stem[5:]
        return str(Path("torchao") / Path(*parts[1:-1]) / f"{mod}.py").replace("\\", "/")
    return str(Path("torchao") / Path(*parts[1:])).replace("\\", "/")


def _summarize_message(msg: str, limit: int = 220) -> str:
    text = (msg or "").replace("\n", " ").strip()
    return text if len(text) <= limit else (text[:limit] + " ...")


def _root_cause_hypothesis(file_failures: list, category: str) -> dict:
    messages = [f.get("message", "") for f in file_failures if f.get("message")]
    top_msg = messages[0] if messages else ""
    msg_lower = top_msg.lower()

    if "does not require grad" in msg_lower:
        return {
            "problem": "Backward path receives a tensor that is detached or never marked for autograd.",
            "chain": [
                "A test executes backward() with expected trainable activations.",
                "An operand in backward matmul/add path has requires_grad=False.",
                "Autograd rejects the graph with 'does not require grad'.",
            ],
            "checks": [
                "Inspect backward implementation for .detach() / .data usage.",
                "Print requires_grad for forward output and backward operands.",
                "Validate custom tensor .to()/dequantization path preserves intended grad behavior.",
            ],
        }

    if "not implemented" in msg_lower and "xpu" in msg_lower:
        return {
            "problem": "Operator is missing XPU kernel support on current torch/oneAPI stack.",
            "chain": [
                "Test invokes an op on XPU device.",
                "Dispatcher cannot find XPU implementation.",
                "Runtime raises not implemented for XPU.",
            ],
            "checks": [
                "Confirm op availability on target torch/xpu version.",
                "Add fallback path or skip guard for unsupported op.",
                "Reduce operator surface in test to minimal supported variant.",
            ],
        }

    if category == "import_error":
        return {
            "problem": "Dependency or module resolution issue in test/runtime environment.",
            "chain": [
                "Test imports feature-specific module.",
                "Module/package is missing or version-mismatched.",
                "Import fails before execution.",
            ],
            "checks": [
                "Verify package installation and version pins.",
                "Gate optional dependency tests with robust skip conditions.",
                "Check PYTHONPATH / editable install state.",
            ],
        }

    if category == "hardware":
        return {
            "problem": "Device/runtime capability mismatch (availability, memory, or driver/runtime setup).",
            "chain": [
                "Test assumes accelerator capability.",
                "Runtime/device state cannot satisfy assumption.",
                "Execution fails during device op dispatch.",
            ],
            "checks": [
                "Validate torch.accelerator.is_available() and device identity.",
                "Check driver/runtime compatibility and memory headroom.",
                "Add stricter precondition checks or skip decorators.",
            ],
        }

    return {
        "problem": "Logic/assertion failure likely caused by behavior drift between backend paths.",
        "chain": [
            "Test expectation diverges from observed backend behavior.",
            "One or more intermediate values differ from reference path.",
            "Assertion/runtime guard fails.",
        ],
        "checks": [
            "Compare intermediate tensors between reference and target backend.",
            "Narrow repro to smallest failing nodeid.",
            "Audit recent changes in related module and test assumptions.",
        ],
    }


def _extract_code_snippet(path: Path, needles: list, context: int = 4):
    if not path.exists() or not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []

    out = []
    for needle in needles:
        line_no = -1
        for idx, line in enumerate(lines, start=1):
            if needle in line:
                line_no = idx
                break
        if line_no == -1:
            continue
        start = max(1, line_no - context)
        end = min(len(lines), line_no + context)
        snippet = "\n".join(lines[start - 1:end])
        out.append({
            "needle": needle,
            "line": line_no,
            "snippet": snippet,
        })
    return out


def _run_git(repo_root: Path, args: list):
    try:
        res = subprocess.run(
            ["git", "-C", str(repo_root)] + args,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return ""
    if res.returncode != 0:
        return ""
    return (res.stdout or "").strip()


def _version_context(repo_root: Path):
    ctx = {}
    version_file = repo_root / "version.txt"
    if version_file.exists():
        try:
            ctx["repo_version"] = version_file.read_text(encoding="utf-8").strip().splitlines()[0]
        except Exception:
            pass

    torch_constraints = []
    req_line = re.compile(r"^(torch|torchvision|torchaudio|pytorch)([<>=!~].*)?$", re.IGNORECASE)
    for rel in ["dev-requirements.txt", "docs/requirements.txt", "requirements.txt"]:
        p = repo_root / rel
        if not p.exists():
            continue
        try:
            for raw in p.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                norm = line.split("#", 1)[0].strip()
                if not req_line.match(norm):
                    continue
                torch_constraints.append(f"{rel}: {norm}")
        except Exception:
            continue

    if torch_constraints:
        ctx["torch_constraints"] = list(dict.fromkeys(torch_constraints))[:8]
    return ctx


def _why_now_hypothesis(repo_root: Path, test_file: str, source_file: str, top_message: str):
    test_log = _run_git(repo_root, ["log", "--oneline", "-n", "8", "--", test_file])
    src_log = _run_git(repo_root, ["log", "--oneline", "-n", "8", "--", source_file]) if source_file else ""
    head_log = _run_git(repo_root, ["log", "--oneline", "-n", "15"])
    versions = _version_context(repo_root)

    reasons = []
    confidence = "low"

    msg = (top_message or "").lower()
    if "does not require grad" in msg:
        reasons.append("Autograd contract around backward operands likely tightened, exposing latent requires_grad mismatch.")
        confidence = "medium"
    if src_log:
        reasons.append("Recent commits touched likely source module, suggesting behavior drift on current branch.")
        confidence = "medium"
    if test_log:
        reasons.append("Recent test updates may have increased coverage and now execute previously untested gradient paths.")
        confidence = "medium"
    if not reasons:
        reasons.append("No strong change signal found; likely environment/runtime variance (backend, toolchain, or dependency versions).")

    evidence = {
        "source_recent_commits": src_log.splitlines()[:5] if src_log else [],
        "test_recent_commits": test_log.splitlines()[:5] if test_log else [],
        "head_recent_commits": head_log.splitlines()[:8] if head_log else [],
        "version_context": versions,
    }
    return {
        "confidence": confidence,
        "reasons": reasons,
        "evidence": evidence,
    }


def _deep_root_cause(file_path: str, file_failures: list, category: str, repo_root: Path):
    messages = [f.get("message", "") for f in file_failures if f.get("message")]
    nodeids = [f.get("nodeid", "") for f in file_failures if f.get("nodeid")]
    top_msg = messages[0] if messages else ""
    msg_lower = top_msg.lower()
    likely_source = _likely_source_file_from_test_file(file_path)
    likely_source_path = repo_root / likely_source

    result = {
        "likely_source": likely_source,
        "problem_statement": "",
        "mechanism": [],
        "code_locations": [],
        "fix_options": [],
        "impact_scope": [],
        "validation_steps": [],
        "why_now": {
            "confidence": "low",
            "reasons": [],
            "evidence": {},
        },
    }

    if "does not require grad" in msg_lower:
        result["problem_statement"] = (
            "Backward path consumes a tensor that is not tracked by autograd, causing grad graph break on runtime checks."
        )
        result["mechanism"] = [
            "Failing tests call backward() and expect input activation gradients.",
            "NF4 conversion/dequantization path may return a plain tensor with requires_grad=False.",
            "Backward matmul uses that tensor, autograd detects non-differentiable operand path and throws runtime error.",
        ]
        result["code_locations"] = _extract_code_snippet(
            likely_source_path,
            [
                "class LinearNF4",
                "def backward(ctx, grad_output)",
                "def to_dtype(func, *args, **kwargs)",
                "def get_original_weight(self)",
            ],
        )
        result["fix_options"] = [
            {
                "title": "Option A: Make backward explicit about non-trainable NF4 weight",
                "change": "Use an explicit non-grad path for weight operand in backward and keep gradient only for activation.",
                "pros": "Minimal and aligned with frozen-weight design.",
                "cons": "Needs careful check to avoid masking real grad-flow bugs.",
            },
            {
                "title": "Option B: Adjust dtype conversion path grad semantics",
                "change": "Ensure conversion/dequantization path preserves intended requires_grad behavior for backward operands.",
                "pros": "Fixes root behavior in a central place.",
                "cons": "Wider blast radius across NF4 ops and potentially other codepaths.",
            },
            {
                "title": "Option C: Save reconstructed tensor explicitly for backward",
                "change": "Store and reuse reconstructed weight tensor with clear grad semantics in autograd function.",
                "pros": "Makes backward contract explicit and easier to reason about.",
                "cons": "Potential memory and perf overhead.",
            },
        ]
        result["impact_scope"] = [
            "NF4 linear backward paths in quantization workflow.",
            "Tests covering backward dtype behavior and quantize API compile/non-compile paths.",
            "Potentially any training/inference path mixing NF4 wrappers and autograd-sensitive ops.",
        ]
        first_nodeid = nodeids[0] if nodeids else "test/quantization/quantize_/workflows/nf4/test_nf4_tensor.py::TestNF4Linear::test_backward_dtype_match_bfloat16"
        result["validation_steps"] = [
            f"pytest {first_nodeid} -xvs",
            "pytest test/quantization/quantize_/workflows/nf4/test_nf4_tensor.py::TestNF4Linear::test_quantize_api -xvs",
            "Run same subset on XPU and compare with CUDA/CPU behavior for regressions.",
        ]
        result["why_now"] = _why_now_hypothesis(repo_root, file_path, likely_source, top_msg)
        return result

    # Generic deep-dive fallback for non-NF4/non-grad signatures
    result["problem_statement"] = "Primary failure appears to be backend-specific behavior mismatch against test expectations."
    result["mechanism"] = [
        "Test expectation and runtime backend behavior diverged.",
        "One or more intermediate values/dispatch paths differ from reference path.",
        "Assertion/runtime guard fails and surfaces as grouped failure signature.",
    ]
    result["code_locations"] = _extract_code_snippet(
        likely_source_path,
        ["def forward", "def backward", "@implements", "torch.accelerator"],
    )
    result["fix_options"] = [
        {
            "title": "Option A: Narrow failing repro and patch nearest backend-specific branch",
            "change": "Use smallest failing nodeid and instrument intermediate states before changing logic.",
            "pros": "Low regression risk.",
            "cons": "May take longer to converge.",
        },
        {
            "title": "Option B: Guard unsupported backend path",
            "change": "Add robust skip/fallback checks where backend capability is missing.",
            "pros": "Stabilizes CI quickly.",
            "cons": "Can hide missing feature implementation.",
        },
    ]
    result["impact_scope"] = [
        "Current failing test module and nearby backend-conditional tests.",
        "Operator coverage for target backend runtime.",
    ]
    if nodeids:
        result["validation_steps"].append(f"pytest {nodeids[0]} -xvs")
    result["validation_steps"].append("Re-run related test file on target backend after patch.")
    result["why_now"] = _why_now_hypothesis(repo_root, file_path, likely_source, top_msg)
    return result


def write_analysis_report(output_dir: Path, week_tag: str, analysis, repo_root: Path):
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

    lines.extend(["", "## Detailed Root Cause Breakdown (Top 10 Files)"])
    for file_path, file_failures in files_sorted[:10]:
        if not file_path:
            continue
        sample_failure = file_failures[0]
        category = categorize_failure(sample_failure.get('nodeid', ''), sample_failure.get('message', ''))
        likely_source = _likely_source_file_from_test_file(file_path)
        hypothesis = _root_cause_hypothesis(file_failures, category)
        deep = _deep_root_cause(file_path, file_failures, category, repo_root)

        nodeids = [f.get('nodeid', '') for f in file_failures if f.get('nodeid')]
        unique_nodeids = list(dict.fromkeys(nodeids))

        top_local_messages = defaultdict(int)
        for ff in file_failures:
            if ff.get('message'):
                top_local_messages[ff['message']] += 1
        local_msg_sorted = sorted(top_local_messages.items(), key=lambda x: (-x[1], x[0]))

        lines.extend([
            "",
            f"### {file_path}",
            f"- Failure count: {len(file_failures)}",
            f"- Primary category: {category}",
            f"- Likely source module: `{likely_source}`",
            f"- Representative error: `{_summarize_message(sample_failure.get('message', ''))}`",
            "- Failing tests (up to 12):",
        ])
        for nodeid in unique_nodeids[:12]:
            lines.append(f"  - `{nodeid}`")

        lines.append("- Top error signatures:")
        for msg, count in local_msg_sorted[:5]:
            lines.append(f"  - ({count}) `{_summarize_message(msg)}`")

        lines.append(f"- Root cause hypothesis: {hypothesis['problem']}")
        lines.append("- Causal chain:")
        for step in hypothesis['chain']:
            lines.append(f"  - {step}")

        lines.append("- Verification checklist:")
        for item in hypothesis['checks']:
            lines.append(f"  - {item}")

        if unique_nodeids:
            lines.append("- Suggested targeted repro:")
            lines.append(f"  - `pytest {unique_nodeids[0]} -xvs`")

        lines.extend([
            "",
            "#### Deep Dive",
            f"- Problem statement: {deep['problem_statement']}",
            "- Mechanism:",
        ])
        for item in deep["mechanism"]:
            lines.append(f"  - {item}")

        lines.append("- Key code locations:")
        if deep["code_locations"]:
            for loc in deep["code_locations"]:
                lines.append(f"  - `{deep['likely_source']}:{loc['line']}` (matched `{loc['needle']}`)")
                lines.append("```python")
                lines.append(loc["snippet"])
                lines.append("```")
        else:
            lines.append("  - No source snippet found automatically; review likely source manually.")

        lines.append("- Fix options:")
        for opt in deep["fix_options"]:
            lines.append(f"  - {opt['title']}")
            lines.append(f"    - Change: {opt['change']}")
            lines.append(f"    - Pros: {opt['pros']}")
            lines.append(f"    - Cons: {opt['cons']}")

        lines.append("- Impact scope:")
        for scope in deep["impact_scope"]:
            lines.append(f"  - {scope}")

        lines.append("- Validation plan:")
        for step in deep["validation_steps"]:
            lines.append(f"  - {step}")

        lines.append("- Why it fails now (heuristic):")
        lines.append(f"  - Confidence: {deep['why_now'].get('confidence', 'low')}")
        for reason in deep["why_now"].get("reasons", []):
            lines.append(f"  - {reason}")

        evidence = deep["why_now"].get("evidence", {})
        src_commits = evidence.get("source_recent_commits", [])
        test_commits = evidence.get("test_recent_commits", [])
        ver_ctx = evidence.get("version_context", {})
        if src_commits or test_commits or ver_ctx:
            lines.append("- Why-now evidence:")
            for c in src_commits[:3]:
                lines.append(f"  - source commit: `{c}`")
            for c in test_commits[:3]:
                lines.append(f"  - test commit: `{c}`")
            if ver_ctx.get("repo_version"):
                lines.append(f"  - repo version: `{ver_ctx['repo_version']}`")
            for constraint in ver_ctx.get("torch_constraints", [])[:3]:
                lines.append(f"  - torch constraint: `{constraint}`")

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
        'top_files': [
            {
                'file': name,
                'count': len(items),
                'likely_source_module': _likely_source_file_from_test_file(name),
                'top_nodeids': list(dict.fromkeys([f.get('nodeid', '') for f in items if f.get('nodeid')]))[:10],
                'deep_root_cause': _deep_root_cause(
                    name,
                    items,
                    categorize_failure(items[0].get('nodeid', ''), items[0].get('message', '')) if items else 'unknown',
                    repo_root,
                ),
                'top_messages': [
                    {'message': _summarize_message(m), 'count': c}
                    for m, c in sorted(
                        defaultdict(int, {
                            msg: sum(1 for ff in items if ff.get('message') == msg)
                            for msg in [ff.get('message', '') for ff in items if ff.get('message')]
                        }).items(),
                        key=lambda x: (-x[1], x[0])
                    )[:5]
                ],
            }
            for name, items in files_sorted[:10]
        ],
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


def write_pr_template(output_dir: Path, week_tag: str, analysis: dict,
                      llm_results: list, dry_run: bool, analysis_only: bool):
    output_dir.mkdir(parents=True, exist_ok=True)
    pr_path = output_dir / f"pr_template_fix_{week_tag}.md"

    changed_files = []
    for item in llm_results:
        for pr in item.get("patch_results", []):
            if pr.get("changed") and pr.get("file"):
                changed_files.append(pr["file"])
    changed_files = list(dict.fromkeys(changed_files))

    category_counts = {cat: len(items) for cat, items in analysis.get("by_category", {}).items()}
    top_files = sorted(
        analysis.get("by_file", {}).items(), key=lambda x: (-len(x[1]), x[0])
    )[:10]

    lines = [
        f"# [XPU Failure Fix] {week_tag}",
        "",
        "## Summary",
        f"- Week tag: {week_tag}",
        f"- Total failures analyzed: {analysis.get('total', 0)}",
        f"- Mode: {'analysis-only' if analysis_only else ('dry-run' if dry_run else 'apply-fixes')}",
    ]

    if category_counts:
        lines.append("- Failure categories:")
        for cat, cnt in sorted(category_counts.items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"  - {cat}: {cnt}")

    lines.extend([
        "",
        "## Root Cause Highlights",
    ])
    if top_files:
        for f, items in top_files[:5]:
            sample_msg = ""
            if items and items[0].get("message"):
                sample_msg = _summarize_message(items[0].get("message", ""))
            lines.append(f"- {f}")
            lines.append(f"  - Failures: {len(items)}")
            if sample_msg:
                lines.append(f"  - Representative error: {sample_msg}")
    else:
        lines.append("- No failing files found in current analysis payload.")

    lines.extend([
        "",
        "## Fixes Included",
    ])
    if analysis_only:
        lines.append("- Analysis only: no code changes were applied in this run.")
    elif dry_run:
        lines.append("- Dry-run only: proposed fixes were analyzed but not applied.")
    elif changed_files:
        for f in changed_files:
            lines.append(f"- {f}")
    else:
        lines.append("- No patches were applied by automation in this run.")

    lines.extend([
        "",
        "## Validation",
        "- [ ] Re-run targeted failing tests on XPU",
        "- [ ] Re-run relevant tests on CUDA/CPU to check regressions",
        "- [ ] Verify updated failure report and fix report artifacts",
        "",
        "## Artifacts",
        f"- reports/failure_analysis_{week_tag}.md",
        f"- reports/failure_summary_{week_tag}.json",
        f"- reports/fix_report_{week_tag}.md",
        f"- reports/fix_report_{week_tag}.json",
        f"- reports/pr_template_fix_{week_tag}.md",
        "",
        "## Checklist",
        "- [ ] Confirm root cause and fix scope are accurate",
        "- [ ] Confirm no unintended file changes",
        "- [ ] Attach any additional benchmark/regression evidence if needed",
    ])

    pr_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return pr_path


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
    parser.add_argument("--analysis-only", action="store_true",
                        help="Generate failure analysis reports only (skip LLM fix and fix report)")
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
    analysis_md, analysis_json = write_analysis_report(output_dir, week_tag, analysis, Path(args.repo_root))

    if args.analysis_only:
        pr_template = write_pr_template(output_dir, week_tag, analysis, [], dry_run=True, analysis_only=True)
        print(f"failure_csv={failure_csv}")
        print(f"analysis_md={analysis_md}")
        print(f"analysis_json={analysis_json}")
        print(f"pr_template={pr_template}")
        return 0

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
    pr_template = write_pr_template(
        output_dir, week_tag, analysis, llm_results,
        dry_run=not args.apply_fixes,
        analysis_only=False,
    )

    print(f"failure_csv={failure_csv}")
    print(f"analysis_md={analysis_md}")
    print(f"analysis_json={analysis_json}")
    print(f"fix_report_md={md_path}")
    print(f"fix_report_json={json_path}")
    print(f"pr_template={pr_template}")
    for item in llm_results:
        changed_any = any(p.get('changed') for p in item.get('patch_results', []))
        conf = item.get('suggestion', {}).get('confidence', '?')
        err = item.get('suggestion', {}).get('error', '')
        status = f"error:{err}" if err else f"confidence:{conf},patched:{changed_any}"
        print(f"llm_{Path(item['file']).stem}={status}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
