import time

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours"

def print_stage_start(stage_num, total_stages, description):
    print(f"\n{'#'*60}")
    print(f"### [START] STAGE {stage_num}/{total_stages}: {description}")
    print(f"{'#'*60}")
    return time.time()

def print_stage_end(stage_num, start_time, result_msg="Completed"):
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\n{'#'*60}")
    print(f"### [END] STAGE {stage_num} finished. {result_msg}")
    print(f"### DURATION: {format_time(elapsed)}")
    print(f"{'#'*60}\n")
    return elapsed

def print_section_header(title):
    print(f"\n{'=' * 60}")
    print(f"   {title.upper()}")
    print(f"{'=' * 60}")