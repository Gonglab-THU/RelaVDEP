import os
import timeit

import numpy as np


def format_duration(seconds):
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


def print_section(title):
    print(f"\n{'=' * 60}")
    print(title)
    print(f"{'=' * 60}")


def print_section_header(title):
    print(f"\n{'=' * 60}")
    print(f"   {title.upper()}")
    print(f"{'=' * 60}")


def print_key_values(title, items):
    print(f"\n--- {title} ---")
    width = max(len(str(key)) for key, _ in items) if items else 0
    for key, value in items:
        print(f"{key:<{width}} : {value}")
    print("-" * 30)


def print_stage_start(stage_num, total_stages, title):
    print(f"\n{'#' * 60}")
    print(f"Stage {stage_num}/{total_stages}: {title}")
    print(f"{'#' * 60}")
    return timeit.default_timer()


def print_stage_end(stage_num, start_time, status="completed"):
    elapsed = timeit.default_timer() - start_time
    print(f"Stage {stage_num} {status}. Duration: {format_duration(elapsed)}")
    return elapsed


def print_step_start(step_num, total_steps, description):
    print(f"\n[Step {step_num}/{total_steps}] {description}")
    return timeit.default_timer()


def print_step_end(start_time, message="Done"):
    print(f"  -> {message}. Duration: {format_duration(timeit.default_timer() - start_time)}")


def absolute_path(path):
    return os.path.abspath(path)


def format_config_value(value, max_items=20):
    if isinstance(value, np.ndarray):
        return f"ndarray(shape={value.shape}, dtype={value.dtype})"
    if isinstance(value, (list, tuple)):
        if len(value) > max_items:
            head = list(value[:max_items // 2])
            tail = list(value[-max_items // 2:])
            return f"{type(value).__name__}(len={len(value)}, values={head} ... {tail})"
        return repr(value)
    if isinstance(value, set):
        values = sorted(value)
        if len(values) > max_items:
            head = values[:max_items // 2]
            tail = values[-max_items // 2:]
            return f"set(len={len(values)}, values={head} ... {tail})"
        return repr(value)
    if isinstance(value, dict):
        keys = list(value.keys())
        if len(keys) > max_items:
            shown_keys = keys[:max_items]
            return f"dict(len={len(value)}, keys={shown_keys} ...)"
        return repr(value)
    return repr(value)


def print_config(config_obj):
    print_section("Resolved Config")
    for key, value in sorted(vars(config_obj).items()):
        print(f"{key}: {format_config_value(value)}")
