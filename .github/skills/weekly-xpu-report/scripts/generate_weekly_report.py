#!/usr/bin/env python3
"""
Generate weekly CUDA/XPU test report from logs and CSV.

Usage:
    python generate_weekly_report.py [--csv-path ao_status.csv] [--xpu-log xpu_0407.txt] 
                                     [--xpu-run-url https://github.com/<owner>/<repo>/actions/runs/<id>]
                                     [--output-dir reports] [--week-tag 20260506]
"""

import argparse
import csv
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def parse_csv(csv_path):
    """Parse ao_status.csv and return per-file counts."""
    data = {}
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, fieldnames=['file', 'cp', 'cs', 'xp', 'xs'])
        for i, row in enumerate(reader):
            if i < 2:  # Skip header rows
                continue
            if not row['file'] or not row['file'].startswith('test/'):
                continue
            try:
                data[row['file']] = {
                    'cuda_pass': int(row['cp']),
                    'cuda_skip': int(row['cs']),
                    'xpu_pass': int(row['xp']),
                    'xpu_skip': int(row['xs']),
                }
            except (ValueError, KeyError):
                pass
    return data


def parse_xpu_log(log_path):
    """Extract FAILED tests from XPU log."""
    failures = []
    
    # ANSI escape sequence pattern
    ansi_escape = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
    
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # Remove ANSI codes
                clean_line = ansi_escape.sub('', line)
                
                # Match FAILED lines with nodeid
                if 'FAILED' in clean_line:
                    # Pattern 1: "path/file.py::test_name FAILED - message"
                    match = re.search(
                        r'((?:torchao/)?test/\S+?)(?:\s+-\s+(.*))?$',
                        clean_line
                    )
                    if match:
                        nodeid = match.group(1)
                        message = match.group(2) or ""
                        
                        if nodeid.startswith('torchao/'):
                            nodeid = nodeid[8:]
                        
                        # Extract file path
                        file_match = re.match(r'(test/[^:]+\.py)::', nodeid)
                        file_path = file_match.group(1) if file_match else nodeid
                        
                        failures.append({
                            'file': file_path,
                            'nodeid': nodeid,
                            'message': message.strip()
                        })
    except FileNotFoundError:
        print(f"Warning: XPU log not found: {log_path}", file=sys.stderr)
    
    return failures


def _strip_ansi(text):
    return re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', text)


def parse_stateful_log_text(log_text):
    """Parse pytest-like logs to per-file PASSED/SKIPPED with collection skips."""
    lines = log_text.splitlines()
    cur = None
    collection_skips = 0
    counts = {}

    for seg in lines:
        if not seg or not seg.strip():
            continue
        line = _strip_ansi(seg)

        m_collect = re.search(r'collecting\s+\d+\s+items\s*/\s*(\d+)\s+skipped', line)
        if m_collect:
            collection_skips = max(collection_skips, int(m_collect.group(1)))
            continue

        m_explicit = re.search(r'((?:torchao/)?test/\S+?)::.*?\b(PASSED|SKIPPED)\b', line)
        if m_explicit:
            f = m_explicit.group(1)
            st = m_explicit.group(2)
            if f.startswith('torchao/'):
                f = f[8:]
            counts.setdefault(f, {'PASSED': 0, 'SKIPPED': 0})
            counts[f][st] += 1
            cur = f
            continue

        m_ctx = re.search(r'((?:torchao/)?test/\S+?)::', line)
        if m_ctx:
            f = m_ctx.group(1)
            if f.startswith('torchao/'):
                f = f[8:]
            cur = f
            continue

        m_bare = re.search(r'\b(PASSED|SKIPPED)\s*$', line)
        if m_bare and 'test/' not in line and cur:
            st = m_bare.group(1)
            counts.setdefault(cur, {'PASSED': 0, 'SKIPPED': 0})
            counts[cur][st] += 1

    return counts, collection_skips


def parse_stateful_log_file(log_path):
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    return parse_stateful_log_text(text)


def _distribute_collection_skips(rows, skip_field, collection_skips, excluded_file):
    if collection_skips <= 0:
        return

    eligible = [r for r in rows if r['file'] != excluded_file]
    if not eligible:
        return

    skip_sum = sum(int(r[skip_field]) for r in eligible)
    if skip_sum <= 0:
        eligible[0][skip_field] += int(collection_skips)
        return

    alloc = []
    assigned = 0
    for r in eligible:
        exact = collection_skips * (float(r[skip_field]) / float(skip_sum))
        base = int(exact)
        frac = exact - base
        alloc.append((r, base, frac))
        assigned += base

    left = collection_skips - assigned
    alloc.sort(key=lambda x: x[2], reverse=True)
    for i in range(left):
        r, base, frac = alloc[i]
        alloc[i] = (r, base + 1, frac)

    for r, base, _ in alloc:
        r[skip_field] += int(base)


def refresh_ao_status(csv_path, cuda_log_path, xpu_log_path):
    """Rebuild ao_status.csv in Python, applying normalization and summary rules."""
    cuda_map, cuda_col_skip = parse_stateful_log_file(cuda_log_path)
    xpu_map, xpu_col_skip = parse_stateful_log_file(xpu_log_path)

    prev_rows = {}
    file_order = []
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            for i, raw in enumerate(f):
                if i < 2:
                    continue
                row = raw.rstrip('\n').split(',')
                if len(row) < 5:
                    continue
                name = row[0]
                if not name.startswith('test/'):
                    continue
                try:
                    prev_rows[name] = {
                        'cp': int(row[1]),
                        'cs': int(row[2]),
                        'xp': int(row[3]),
                        'xs': int(row[4]),
                    }
                    file_order.append(name)
                except ValueError:
                    continue

    all_files = list(file_order)
    for f in sorted(set(list(cuda_map.keys()) + list(xpu_map.keys()))):
        if f not in all_files:
            all_files.append(f)

    rows = []
    for f in all_files:
        cp = int(cuda_map.get(f, {}).get('PASSED', 0))
        cs = int(cuda_map.get(f, {}).get('SKIPPED', 0))
        xp = int(xpu_map.get(f, {}).get('PASSED', 0))
        xs = int(xpu_map.get(f, {}).get('SKIPPED', 0))
        rows.append({'file': f, 'cp': cp, 'cs': cs, 'xp': xp, 'xs': xs})

    excluded_file = 'test/quantization/pt2e/test_x86inductor_fusion.py'
    for r in rows:
        if r['file'] == excluded_file:
            r['cp'] = r['cs'] = r['xp'] = r['xs'] = 0

    _distribute_collection_skips(rows, 'cs', cuda_col_skip, excluded_file)
    _distribute_collection_skips(rows, 'xs', xpu_col_skip, excluded_file)

    for r in rows:
        if r['file'] == excluded_file:
            continue
        cuda_cases = int(r['cp']) + int(r['cs'])
        xpu_cases = int(r['xp']) + int(r['xs'])
        if cuda_cases > xpu_cases:
            r['xs'] += cuda_cases - xpu_cases
        elif xpu_cases > cuda_cases:
            r['cs'] += xpu_cases - cuda_cases

    included = [r for r in rows if r['file'] != excluded_file]
    sum_cuda_pass = sum(r['cp'] for r in included)
    sum_cuda_skip = sum(r['cs'] for r in included)
    sum_xpu_pass = sum(r['xp'] for r in included)
    sum_xpu_skip = sum(r['xs'] for r in included)
    sum_cuda_total = sum_cuda_pass + sum_cuda_skip
    sum_xpu_total = sum_xpu_pass + sum_xpu_skip

    cuda_rate = (sum_cuda_pass / sum_cuda_total) if sum_cuda_total else 0.0
    xpu_rate = (sum_xpu_pass / sum_xpu_total) if sum_xpu_total else 0.0

    lines = [
        'Test file,CUDA,,XPU,',
        ',PASSED,SKIPPED,PASSED,SKIPPED',
    ]
    for r in rows:
        lines.append(f"{r['file']},{r['cp']},{r['cs']},{r['xp']},{r['xs']}")

    lines.extend([
        '',
        'Summary,Total,Pass,Skip,Pass Rate',
        f'CUDA,{sum_cuda_total},{sum_cuda_pass},{sum_cuda_skip},{cuda_rate:.4f}',
        f'XPU,{sum_xpu_total},{sum_xpu_pass},{sum_xpu_skip},{xpu_rate:.4f}',
    ])

    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        f.write('\n'.join(lines) + '\n')

    return (
        f'Parsing {cuda_log_path} ...\n'
        f'Parsing {xpu_log_path} ...\n\n'
        f'Totals from logs:\n'
        f'  CUDA: {sum_cuda_pass} passed  {sum_cuda_skip} skipped\n'
        f'  XPU : {sum_xpu_pass} passed  {sum_xpu_skip} skipped\n\n'
        f'Written: {csv_path}  ({len(rows)} test files)'
    )


def parse_xpu_log_text(log_text):
    """Extract FAILED tests from a raw log string."""
    failures = []
    ansi_escape = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')

    for line in log_text.splitlines():
        clean_line = ansi_escape.sub('', line)
        if 'FAILED' not in clean_line:
            continue

        match = re.search(r'((?:torchao/)?test/\S+?)(?:\s+-\s+(.*))?$', clean_line)
        if not match:
            continue

        nodeid = match.group(1)
        message = match.group(2) or ""
        if nodeid.startswith('torchao/'):
            nodeid = nodeid[8:]

        file_match = re.match(r'(test/[^:]+\.py)::', nodeid)
        file_path = file_match.group(1) if file_match else nodeid

        failures.append({
            'file': file_path,
            'nodeid': nodeid,
            'message': message.strip(),
        })

    return failures


def _github_request_json(url, token=None):
    headers = {
        'Accept': 'application/vnd.github+json',
        'User-Agent': 'torchao-weekly-xpu-report',
        'X-GitHub-Api-Version': '2022-11-28',
    }
    if token:
        headers['Authorization'] = f'Bearer {token}'

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode('utf-8')
        return json.loads(body)
    except (urllib.error.URLError, TimeoutError):
        body = _fallback_download(url, token=token).decode('utf-8', errors='ignore')
        return json.loads(body)


def _github_request_bytes(url, token=None):
    headers = {
        'Accept': 'application/vnd.github+json',
        'User-Agent': 'torchao-weekly-xpu-report',
        'X-GitHub-Api-Version': '2022-11-28',
    }
    if token:
        headers['Authorization'] = f'Bearer {token}'

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read()
    except (urllib.error.URLError, TimeoutError):
        return _fallback_download(url, token=token)


def _ps_quote(value):
    return value.replace("'", "''")


def _fallback_download(url, token=None):
    if os.name == 'nt':
        return _powershell_download(url, token=token)
    return _curl_download(url, token=token)


def _powershell_download(url, token=None):
    """Download URL content via PowerShell as fallback for urllib connectivity issues."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as tmp:
        out_path = tmp.name

    header_lines = [
        "'Accept'='application/vnd.github+json'",
        "'User-Agent'='torchao-weekly-xpu-report'",
        "'X-GitHub-Api-Version'='2022-11-28'",
    ]
    if token:
        header_lines.append(f"'Authorization'='Bearer {_ps_quote(token)}'")
    header_ps = '; '.join(header_lines)

    cmd = (
        "$ProgressPreference='SilentlyContinue'; "
        f"$h=@{{{header_ps}}}; "
        f"Invoke-WebRequest -UseBasicParsing -Headers $h -Uri '{_ps_quote(url)}' -OutFile '{_ps_quote(out_path)}'"
    )

    try:
        proc = subprocess.run(
            ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', cmd],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or '').strip()
            raise RuntimeError(f'PowerShell download failed: {err}')

        with open(out_path, 'rb') as f:
            return f.read()
    finally:
        try:
            os.remove(out_path)
        except OSError:
            pass


def _curl_download(url, token=None):
    """Download URL content via curl as fallback for urllib issues on Linux/macOS."""
    cmd = [
        'curl',
        '--silent',
        '--show-error',
        '--location',
        '--fail',
        '--connect-timeout',
        '30',
        '--max-time',
        '120',
        '-H',
        'Accept: application/vnd.github+json',
        '-H',
        'User-Agent: torchao-weekly-xpu-report',
        '-H',
        'X-GitHub-Api-Version: 2022-11-28',
    ]
    if token:
        cmd.extend(['-H', f'Authorization: Bearer {token}'])
    cmd.append(url)

    try:
        proc = subprocess.run(cmd, capture_output=True, check=False)
    except FileNotFoundError as e:
        raise RuntimeError('curl is not installed and urllib fallback failed') from e

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout).decode('utf-8', errors='ignore').strip()
        raise RuntimeError(f'curl download failed: {err}')

    return proc.stdout


def fetch_latest_scheduled_xpu_log(repo, workflow, token=None):
    """Fetch logs of the latest completed scheduled workflow run from GitHub Actions."""
    runs_url = (
        f'https://api.github.com/repos/{repo}/actions/workflows/{workflow}/runs'
        '?event=schedule&status=completed&per_page=1'
    )
    runs_data = _github_request_json(runs_url, token=token)
    workflow_runs = runs_data.get('workflow_runs', [])
    if not workflow_runs:
        raise RuntimeError(
            f'No completed scheduled runs found for {repo} workflow {workflow}'
        )

    run = workflow_runs[0]
    run_id = run['id']
    logs_url = f'https://api.github.com/repos/{repo}/actions/runs/{run_id}/logs'
    logs_zip = _github_request_bytes(logs_url, token=token)

    all_logs = []
    with zipfile.ZipFile(io.BytesIO(logs_zip)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name_lower = info.filename.lower()
            if not (name_lower.endswith('.txt') or name_lower.endswith('.log')):
                continue
            text = zf.read(info).decode('utf-8', errors='ignore')
            all_logs.append(f'\n===== {info.filename} =====\n')
            all_logs.append(text)

    combined_log = ''.join(all_logs)
    return {
        'run_id': run_id,
        'run_html_url': run.get('html_url', ''),
        'run_created_at': run.get('created_at', ''),
        'log_text': combined_log,
    }


def fetch_xpu_log_by_run_id(repo, run_id, token=None):
    """Fetch logs for a specific GitHub Actions run id."""
    run_url = f'https://api.github.com/repos/{repo}/actions/runs/{run_id}'
    run = _github_request_json(run_url, token=token)

    logs_url = f'https://api.github.com/repos/{repo}/actions/runs/{run_id}/logs'
    logs_zip = _github_request_bytes(logs_url, token=token)

    all_logs = []
    with zipfile.ZipFile(io.BytesIO(logs_zip)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name_lower = info.filename.lower()
            if not (name_lower.endswith('.txt') or name_lower.endswith('.log')):
                continue
            text = zf.read(info).decode('utf-8', errors='ignore')
            all_logs.append(f'\n===== {info.filename} =====\n')
            all_logs.append(text)

    combined_log = ''.join(all_logs)
    return {
        'run_id': int(run_id),
        'run_html_url': run.get('html_url', ''),
        'run_created_at': run.get('created_at', ''),
        'log_text': combined_log,
    }


def _extract_run_id_from_url(run_url):
    """Extract numeric run id from a GitHub Actions run URL."""
    if not run_url:
        return None
    match = re.search(r'/actions/runs/(\d+)', run_url)
    if not match:
        return None
    return match.group(1)


def _sanitize_filename(value):
    return re.sub(r'[^A-Za-z0-9._-]+', '_', value)


def fetch_latest_successful_cuda_job_log(repo, workflow, job_keyword, token=None):
    """Fetch log text from latest successful run's successful CUDA job (e.g. CUDA 2.10)."""
    runs_url = (
        f'https://api.github.com/repos/{repo}/actions/workflows/{workflow}/runs'
        '?status=completed&per_page=100'
    )
    runs_data = _github_request_json(runs_url, token=token)
    workflow_runs = runs_data.get('workflow_runs', [])

    fallback_keywords = [
        'CUDA 2.12',
        'CUDA 2.11',
        'CUDA 2.10',
        'CUDA Nightly',
        'CUDA',
    ]
    fallback_keywords = [
        k for k in fallback_keywords if k.lower() != job_keyword.lower()
    ]
    observed_cuda_names = []

    def _search_runs(runs):
        for run in runs:
            run_id = run['id']
            jobs_url = f'https://api.github.com/repos/{repo}/actions/runs/{run_id}/jobs?per_page=100'
            jobs_data = _github_request_json(jobs_url, token=token)
            jobs = jobs_data.get('jobs', [])

            # Pass 1: exact user keyword
            for job in jobs:
                name = job.get('name', '')
                if 'cuda' in name.lower() and name not in observed_cuda_names:
                    observed_cuda_names.append(name)
                if job_keyword.lower() not in name.lower():
                    continue
                if job.get('conclusion') != 'success':
                    continue

                job_id = job['id']
                logs_url = f'https://api.github.com/repos/{repo}/actions/jobs/{job_id}/logs'
                log_bytes = _github_request_bytes(logs_url, token=token)
                log_text = log_bytes.decode('utf-8', errors='ignore')
                return {
                    'run_id': run_id,
                    'run_html_url': run.get('html_url', ''),
                    'job_id': job_id,
                    'job_name': name,
                    'matched_keyword': job_keyword,
                    'log_text': log_text,
                }

            # Pass 2: fallback keywords when exact keyword is not available
            for fallback in fallback_keywords:
                for job in jobs:
                    name = job.get('name', '')
                    if fallback.lower() not in name.lower():
                        continue
                    if job.get('conclusion') != 'success':
                        continue

                    job_id = job['id']
                    logs_url = f'https://api.github.com/repos/{repo}/actions/jobs/{job_id}/logs'
                    log_bytes = _github_request_bytes(logs_url, token=token)
                    log_text = log_bytes.decode('utf-8', errors='ignore')
                    return {
                        'run_id': run_id,
                        'run_html_url': run.get('html_url', ''),
                        'job_id': job_id,
                        'job_name': name,
                        'matched_keyword': fallback,
                        'log_text': log_text,
                    }
        return None

    # First prefer fully successful workflow runs.
    successful_runs = [r for r in workflow_runs if r.get('conclusion') == 'success']
    result = _search_runs(successful_runs)
    if result is not None:
        return result

    # Fallback: scan all completed runs in case workflow-level conclusion is not success
    # but a CUDA job inside still succeeded and is usable for snapshot refresh.
    result = _search_runs(workflow_runs)
    if result is not None:
        return result

    observed_summary = ', '.join(observed_cuda_names[:6]) if observed_cuda_names else 'none'
    raise RuntimeError(
        f'No successful job matching "{job_keyword}" found in recent successful runs '
        f'for {repo} workflow {workflow}. '
        f'Observed CUDA jobs: {observed_summary}'
    )


def main():
    parser = argparse.ArgumentParser(
        description='Generate weekly CUDA/XPU test report'
    )
    parser.add_argument(
        '--csv-path', default='',
        help='Path to ao_status.csv. If omitted, use <output-dir>/ao_status_<week>.csv'
    )
    parser.add_argument(
        '--xpu-log', default='xpu_0407.txt',
        help='Path to XPU CI log'
    )
    parser.add_argument(
        '--cuda-log', default='cuda_0407.txt',
        help='Path to CUDA CI log (used when --refresh-status is enabled)'
    )
    parser.add_argument(
        '--fetch-latest-cuda',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Fetch latest successful CUDA job log for refresh-status (default: enabled)'
    )
    parser.add_argument(
        '--cuda-github-repo', default='pytorch/ao',
        help='GitHub repo for CUDA workflow log fetch'
    )
    parser.add_argument(
        '--cuda-github-workflow', default='regression_test.yml',
        help='Workflow file for CUDA log fetch'
    )
    parser.add_argument(
        '--cuda-job-keyword', default='CUDA 2.11',
        help='Job name keyword to pick CUDA job in regression workflow'
    )
    parser.add_argument(
        '--refresh-status',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Rebuild ao_status.csv from CUDA/XPU logs before report generation (default: enabled)'
    )
    parser.add_argument(
        '--fetch-latest-scheduled',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Fetch latest completed scheduled run logs from GitHub Actions (default: enabled)'
    )
    parser.add_argument(
        '--github-repo', default='pytorch/ao',
        help='GitHub repo in owner/repo format'
    )
    parser.add_argument(
        '--github-workflow', default='xpu_test.yml',
        help='Workflow file name or workflow id for Actions API'
    )
    parser.add_argument(
        '--xpu-run-url', default='',
        help='GitHub Actions run URL for XPU log (takes precedence over --fetch-latest-scheduled)'
    )
    parser.add_argument(
        '--xpu-run-id', default='',
        help='GitHub Actions run id for XPU log (takes precedence over --fetch-latest-scheduled)'
    )
    parser.add_argument(
        '--github-token', default=os.environ.get('GITHUB_TOKEN', ''),
        help='GitHub token (optional). Defaults to env GITHUB_TOKEN'
    )
    parser.add_argument(
        '--output-dir', default='reports',
        help='Output directory for reports'
    )
    parser.add_argument(
        '--week-tag', default=None,
        help='Week tag (e.g., 20260506). Defaults to today\'s date.'
    )
    
    args = parser.parse_args()
    
    if args.week_tag is None:
        args.week_tag = datetime.now().strftime('%Y%m%d')
    
    # Create output directory early for downloaded logs.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    working_csv_path = args.csv_path.strip() if args.csv_path else ''
    if not working_csv_path:
        working_csv_path = str(output_dir / f"ao_status_{args.week_tag}.csv")

    xpu_input_desc = args.xpu_log
    refresh_xpu_log = args.xpu_log
    xpu_failures = []
    explicit_xpu_run_id = (args.xpu_run_id or '').strip()
    if not explicit_xpu_run_id:
        explicit_xpu_run_id = _extract_run_id_from_url((args.xpu_run_url or '').strip()) or ''

    if explicit_xpu_run_id:
        try:
            fetched = fetch_xpu_log_by_run_id(
                args.github_repo,
                explicit_xpu_run_id,
                token=args.github_token or None,
            )
            fetched_log_path = output_dir / f"xpu_run_{fetched['run_id']}.log"
            with open(fetched_log_path, 'w', encoding='utf-8') as f:
                f.write(fetched['log_text'])

            refresh_xpu_log = str(fetched_log_path)
            xpu_input_desc = (
                f"github://{args.github_repo}/actions/runs/{fetched['run_id']}"
            )
            xpu_failures = parse_xpu_log_text(fetched['log_text'])
            print(f"fetched_run_id={fetched['run_id']}")
            print(f"fetched_run_url={fetched['run_html_url']}")
            print(f"fetched_run_log={fetched_log_path}")
        except (RuntimeError, urllib.error.URLError, urllib.error.HTTPError) as e:
            print(f"Error: failed to fetch XPU run by id/url: {e}", file=sys.stderr)
            if os.path.exists(args.xpu_log):
                print(
                    f"Fallback: using local xpu log {args.xpu_log}",
                    file=sys.stderr,
                )
                refresh_xpu_log = args.xpu_log
                xpu_input_desc = args.xpu_log
            else:
                sys.exit(1)
    elif args.fetch_latest_scheduled:
        try:
            fetched = fetch_latest_scheduled_xpu_log(
                args.github_repo,
                args.github_workflow,
                token=args.github_token or None,
            )
            fetched_log_path = output_dir / f"xpu_scheduled_run_{fetched['run_id']}.log"
            with open(fetched_log_path, 'w', encoding='utf-8') as f:
                f.write(fetched['log_text'])

            refresh_xpu_log = str(fetched_log_path)
            xpu_input_desc = (
                f"github://{args.github_repo}/{args.github_workflow}"
                f" run_id={fetched['run_id']}"
            )
            xpu_failures = parse_xpu_log_text(fetched['log_text'])
            print(f"fetched_run_id={fetched['run_id']}")
            print(f"fetched_run_url={fetched['run_html_url']}")
            print(f"fetched_run_log={fetched_log_path}")
        except (RuntimeError, urllib.error.URLError, urllib.error.HTTPError) as e:
            print(f"Error: failed to fetch latest scheduled run: {e}", file=sys.stderr)
            if os.path.exists(args.xpu_log):
                print(
                    f"Fallback: using local xpu log {args.xpu_log}",
                    file=sys.stderr,
                )
                refresh_xpu_log = args.xpu_log
                xpu_input_desc = args.xpu_log
            else:
                sys.exit(1)
    else:
        if not os.path.exists(args.xpu_log):
            print(f"Warning: XPU log not found: {args.xpu_log}", file=sys.stderr)
        refresh_xpu_log = args.xpu_log

    cuda_input_desc = args.cuda_log
    refresh_cuda_log = args.cuda_log
    if args.refresh_status and args.fetch_latest_cuda:
        try:
            fetched_cuda = fetch_latest_successful_cuda_job_log(
                args.cuda_github_repo,
                args.cuda_github_workflow,
                args.cuda_job_keyword,
                token=args.github_token or None,
            )
            safe_job = _sanitize_filename(fetched_cuda['job_name'])
            cuda_log_path = output_dir / (
                f"cuda_job_{fetched_cuda['run_id']}_{fetched_cuda['job_id']}_{safe_job}.log"
            )
            with open(cuda_log_path, 'w', encoding='utf-8') as f:
                f.write(fetched_cuda['log_text'])
            refresh_cuda_log = str(cuda_log_path)
            cuda_input_desc = (
                f"github://{args.cuda_github_repo}/{args.cuda_github_workflow} "
                f"run_id={fetched_cuda['run_id']} job={fetched_cuda['job_name']}"
            )
            matched_keyword = fetched_cuda.get('matched_keyword', args.cuda_job_keyword)
            if matched_keyword.lower() != args.cuda_job_keyword.lower():
                print(
                    f"cuda_keyword_fallback={args.cuda_job_keyword}->{matched_keyword}"
                )
            print(f"fetched_cuda_run_id={fetched_cuda['run_id']}")
            print(f"fetched_cuda_job_id={fetched_cuda['job_id']}")
            print(f"fetched_cuda_log={cuda_log_path}")
        except (RuntimeError, urllib.error.URLError, urllib.error.HTTPError) as e:
            print(f"Error: failed to fetch latest CUDA log: {e}", file=sys.stderr)
            if os.path.exists(args.cuda_log):
                print(
                    f"Fallback: using local cuda log {args.cuda_log}",
                    file=sys.stderr,
                )
                refresh_cuda_log = args.cuda_log
                cuda_input_desc = args.cuda_log
            else:
                sys.exit(1)

    if args.refresh_status:
        try:
            refresh_output = refresh_ao_status(working_csv_path, refresh_cuda_log, refresh_xpu_log)
            print('refresh_status=ok')
            if refresh_output:
                print('refresh_status_detail_begin')
                print(refresh_output)
                print('refresh_status_detail_end')
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    # Validate inputs
    if not os.path.exists(working_csv_path):
        print(f"Error: CSV not found: {working_csv_path}", file=sys.stderr)
        sys.exit(1)
    
    # Parse data
    csv_data = parse_csv(working_csv_path)

    if not xpu_failures:
        xpu_failures = parse_xpu_log(refresh_xpu_log)
    
    # Calculate totals
    cuda_pass = sum(row['cuda_pass'] for row in csv_data.values())
    cuda_skip = sum(row['cuda_skip'] for row in csv_data.values())
    xpu_pass = sum(row['xpu_pass'] for row in csv_data.values())
    xpu_skip = sum(row['xpu_skip'] for row in csv_data.values())
    
    cuda_total = cuda_pass + cuda_skip
    xpu_total = xpu_pass + xpu_skip
    
    cuda_rate = cuda_pass / cuda_total if cuda_total > 0 else 0
    xpu_rate = xpu_pass / xpu_total if xpu_total > 0 else 0
    
    # Analyze failures
    unique_failures = {f['nodeid']: f for f in xpu_failures}.values()
    
    failures_by_file = defaultdict(int)
    for f in unique_failures:
        failures_by_file[f['file']] += 1
    
    failures_by_file = sorted(
        failures_by_file.items(),
        key=lambda x: (-x[1], x[0])
    )
    
    # Top failure messages
    msg_counts = defaultdict(int)
    for f in unique_failures:
        if f['message']:
            msg_counts[f['message']] += 1
    
    top_messages = sorted(
        msg_counts.items(),
        key=lambda x: (-x[1], x[0])
    )[:20]
    
    # Generate report paths
    report_path = output_dir / f"weekly_report_{args.week_tag}.md"
    csv_snapshot_path = output_dir / f"ao_status_{args.week_tag}.csv"
    failure_csv_path = output_dir / f"xpu_failures_{args.week_tag}.csv"
    
    # Write CSV snapshot
    if os.path.abspath(working_csv_path) != os.path.abspath(str(csv_snapshot_path)):
        with open(working_csv_path, 'r', encoding='utf-8') as src:
            with open(csv_snapshot_path, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
    
    # Write failure CSV
    with open(failure_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['file', 'nodeid', 'message'])
        writer.writeheader()
        for failure in unique_failures:
            writer.writerow(failure)
    
    # Generate markdown report
    report_lines = [
        f"# Weekly CUDA/XPU Report ({args.week_tag})",
        "",
        "## Inputs",
        f"- CSV: {working_csv_path}",
        f"- CUDA log source: {cuda_input_desc}",
        f"- XPU log source: {xpu_input_desc}",
        "",
        "## Summary",
        "| Platform | Total | Pass | Skip | Pass Rate |",
        "|---|---:|---:|---:|---:|",
        f"| CUDA | {cuda_total} | {cuda_pass} | {cuda_skip} | {cuda_rate:.4f} |",
        f"| XPU | {xpu_total} | {xpu_pass} | {xpu_skip} | {xpu_rate:.4f} |",
        "",
        "## XPU Failure Analysis",
        f"- Total unique failed tests: {len(unique_failures)}",
        f"- Files with failures: {len(failures_by_file)}",
        "",
        "### Failures By File (Top 30)",
        "| File | Failed Tests |",
        "|---|---:|",
    ]
    
    for file_path, count in failures_by_file[:30]:
        report_lines.append(f"| {file_path} | {count} |")
    
    report_lines.extend([
        "",
        "### Top Failure Messages (Top 20)",
        "| Message | Count |",
        "|---|---:|",
    ])
    
    for msg, count in top_messages:
        # Escape pipe characters in message
        safe_msg = msg.replace('|', '\\|')
        report_lines.append(f"| {safe_msg} | {count} |")
    
    report_lines.extend([
        "",
        "### Artifacts",
        f"- CSV snapshot: {csv_snapshot_path}",
        f"- Failure list CSV: {failure_csv_path}",
    ])
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    # Print summary to stdout
    print(f"report={report_path}")
    print(f"csv_snapshot={csv_snapshot_path}")
    print(f"failure_csv={failure_csv_path}")
    print(f"summary_cuda={cuda_total},{cuda_pass},{cuda_skip},{cuda_rate:.4f}")
    print(f"summary_xpu={xpu_total},{xpu_pass},{xpu_skip},{xpu_rate:.4f}")
    print(f"xpu_failed_unique_tests={len(unique_failures)}")


if __name__ == '__main__':
    main()
