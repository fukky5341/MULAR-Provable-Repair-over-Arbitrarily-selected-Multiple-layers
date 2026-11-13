

import os
import re


RESULT_COMPLETION_MARKER = "=== Summary of Repair Results ==="


def completed_result_exists(logfile):
    """Return True only for a result log that reached its final summary."""
    if not os.path.isfile(logfile):
        return False

    with open(logfile, "r", errors="replace") as f:
        return RESULT_COMPLETION_MARKER in f.read()


def completed_external_runs_exist(
        logfile, expected_runs, run_heading_pattern, terminal_pattern):
    """Return whether an external-tool log contains every completed run.

    APRNN and ProRepair store all runs in one file and do not emit the standard
    summary marker. A file is reusable only when its declared run count matches
    the request, its headings are exactly 1..N, and every run block has a
    terminal success or failure result.
    """

    if not os.path.isfile(logfile):
        return False
    with open(logfile, "r", errors="replace") as f:
        text = f.read()

    declared = re.search(r"^Repair num runs\s+(\d+)\s*$", text, re.MULTILINE)
    if declared is None or int(declared.group(1)) != expected_runs:
        return False

    headings = list(re.finditer(run_heading_pattern, text, re.MULTILINE))
    if [int(match.group(1)) for match in headings] != list(
            range(1, expected_runs + 1)):
        return False

    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        if re.search(terminal_pattern, text[heading.end():end], re.MULTILINE) is None:
            return False
    return True


def logging(message, option="a", line_break=1, border_type='-', border=None, logfile=None):
    print(message, flush=True)

    if logfile is None:
        return

    line_break_str = "\n" * line_break
    
    border_top = None
    border_bottom = None
    if border == "top":
        str_len = len(message)
        border_top = border_type * str_len
    if border == "bottom":
        str_len = len(message)
        border_bottom = border_type * str_len
    elif border == "both":
        str_len = len(message)
        border_top = border_type * str_len
        border_bottom = border_type * str_len

    with open(logfile, option) as f:
        f.write(line_break_str)
        if border_top is not None:
            f.write(border_top + "\n")
        f.write(message + "\n")
        if border_bottom is not None:
            f.write(border_bottom + "\n")
