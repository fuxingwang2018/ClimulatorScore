from datetime import datetime, timedelta

def parse_datetime_safe(date_input, date_format="%Y-%m-%d %H:%M"):
    """
    Safely parse a datetime string or object, supporting '24:00' as next-day 00:00.
    """
    if isinstance(date_input, datetime):
        return date_input
    if "24:00" in date_input:
        temp = datetime.strptime(date_input.replace(" 24:00", " 00:00"), date_format)
        return temp + timedelta(days=1)
    return datetime.strptime(date_input, date_format)


def generate_time_series(start_date, end_date, step_hours=6, date_format="%Y-%m-%d %H:%M"):
    """
    Generate a list of datetime objects from start_date to end_date (inclusive)
    with a given step in hours.
    """
    start_date = parse_datetime_safe(start_date, date_format)
    end_date = parse_datetime_safe(end_date, date_format)

    delta = timedelta(hours=step_hours)
    times = []
    t = start_date
    while t <= end_date:
        times.append(t)
        t += delta
    return times


def get_time_indices(times, start_date, end_date, date_format="%Y-%m-%d %H:%M"):
    """
    Get start and end indices from a list of datetime objects (inclusive).
    """
    start_dt = parse_datetime_safe(start_date, date_format)
    end_dt = parse_datetime_safe(end_date, date_format)

    # --- Find start index ---
    try:
        start_idx = next(i for i, t in enumerate(times) if t >= start_dt)
    except StopIteration:
        raise ValueError("Start date is beyond the available time range.")

    # --- Find end index safely ---
    try:
        end_idx = next(i for i, t in enumerate(times) if t > end_dt) - 1
    except StopIteration:
        # end_dt is after or equal to the last timestamp
        end_idx = len(times) - 1

    return start_idx, end_idx


"""
if __name__ == "__main__":
    # Example usage
    all_times = generate_time_series("2000-01-01 00:00", "2009-12-31 24:00")
    start_date = "2005-07-01 00:00"
    end_date = "2005-08-31 24:00"

    start_idx, end_idx = get_time_indices(all_times, start_date, end_date)
    print(f"Start index: {start_idx}, End index: {end_idx}")
    print(f"Start time: {all_times[start_idx]}, End time: {all_times[end_idx]}")
"""
