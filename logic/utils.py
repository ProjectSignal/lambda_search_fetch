from datetime import datetime, timedelta
from pytz import utc

# --- Constants ---
VALID_STAGES = ["LOW", "MEDIUM", "STRONG", "VERY_STRONG"]
VALID_FREQUENCIES = ["WEEKLY", "FORTNIGHT", "MONTHLY", "CUSTOM"]
VALID_GROUP_STATUSES = ["ACTIVE", "COMPLETED", "ARCHIVED"]
VALID_CADENCE_STATUSES = ["UPCOMING", "SKIPPED", "DONE"]
VALID_NOTE_TYPES = ["MEETING", "COFFEE_CHAT", "EVENT", "FOLLOW_UP", "INTERVIEW", "OTHER"]
VALID_NOTE_CHANNELS = ["IN_PERSON", "PHONE_CALL", "VIDEO_CALL", "EMAIL", "OTHER"]
VALID_NOTE_OUTCOMES = ["VERY_POSITIVE", "POSITIVE", "NEUTRAL", "NEGATIVE"]

NETWORK_STAGE_WEIGHTS = {
    "LOW": 1,
    "MEDIUM": 2,
    "STRONG": 3,
    "VERY_STRONG": 4
}

# --- Helpers ---
def get_utc_now():
    """Returns the current UTC datetime with tzinfo."""
    return datetime.utcnow().replace(tzinfo=utc)

def ensure_utc_datetime(dt):
    """Ensures that a datetime (or ISO‐formatted string) is in UTC."""
    if dt is None:
        return None
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=utc)
    return dt

def format_datetime(dt):
    """Converts a datetime to an ISO 8601 string."""
    if dt is None:
        return None
    if isinstance(dt, str):
        dt = ensure_utc_datetime(dt)
    return dt.isoformat()

def validate_stage(stage):
    if stage not in VALID_STAGES:
        raise ValueError(f"Invalid stage. Must be one of: {', '.join(VALID_STAGES)}")
    return stage

def validate_frequency(frequency):
    if frequency not in VALID_FREQUENCIES:
        raise ValueError(f"Invalid frequency. Must be one of: {', '.join(VALID_FREQUENCIES)}")
    return frequency

def validate_group_status(status):
    if status not in VALID_GROUP_STATUSES:
        raise ValueError(f"Invalid group status. Must be one of: {', '.join(VALID_GROUP_STATUSES)}")
    return status

def validate_cadence_status(status):
    if status not in VALID_CADENCE_STATUSES:
        raise ValueError(f"Invalid cadence status. Must be one of: {', '.join(VALID_CADENCE_STATUSES)}")
    return status

def validate_note_type(type_):
    if type_ is None:
        return None
    if type_ not in VALID_NOTE_TYPES:
        raise ValueError(f"Invalid type. Must be one of: {', '.join(VALID_NOTE_TYPES)}")
    return type_

def validate_channel(channel):
    if channel is None:
        return None
    if channel not in VALID_NOTE_CHANNELS:
        raise ValueError(f"Invalid channel. Must be one of: {', '.join(VALID_NOTE_CHANNELS)}")
    return channel

def validate_outcome(outcome):
    if outcome is None:
        return None
    if outcome not in VALID_NOTE_OUTCOMES:
        raise ValueError(f"Invalid outcome. Must be one of: {', '.join(VALID_NOTE_OUTCOMES)}")
    return outcome

def validate_duration(duration):
    if duration is None:
        return None
    VALID_DURATIONS = ["15", "30", "45", "60", "60+"]
    if not isinstance(duration, str):
        raise ValueError("Duration must be one of: 15, 30, 45, 60, 60+")
    if duration not in VALID_DURATIONS:
        raise ValueError("Duration must be one of: 15, 30, 45, 60, 60+")
    return duration

def calculate_next_reachout_date(frequency, custom_days=None, base_date=None):
    """Calculate the next reachout date based on frequency and optional custom days."""
    if base_date is None:
        base_date = datetime.utcnow()
    elif isinstance(base_date, str):
        try:
            base_date = datetime.fromisoformat(base_date.replace('Z', '+00:00'))
        except ValueError:
            base_date = datetime.strptime(base_date, "%Y-%m-%d")
            base_date = datetime.combine(base_date, datetime.min.time())
    frequency_map = {
        "WEEKLY": 7,
        "FORTNIGHT": 14,
        "MONTHLY": 30,
    }
    if frequency in frequency_map:
        return base_date + timedelta(days=frequency_map[frequency])
    elif frequency == "CUSTOM" and custom_days:
        return base_date + timedelta(days=custom_days)
    return base_date + timedelta(days=14)

def get_start_of_current_week_utc():
    now = datetime.utcnow().replace(tzinfo=utc)
    start = now - timedelta(days=now.weekday())
    return start.replace(hour=0, minute=0, second=0, microsecond=0)