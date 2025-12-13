from datetime import datetime, timedelta, timezone
from typing import Dict

from langchain.tools import tool


_CITY_OFFSETS: Dict[str, int] = {
    "helsinki": 2,       
    "moscow": 3,         
    "ottawa": -5,        
    "washington d.c.": -5,
    "washington dc": -5,
    "washington": -5,
    "beijing": 8,        
    "brasilia": -3,      
    "canberra": 11,      
}


def _normalize_city_name(city: str) -> str:
    return city.strip().lower()


def get_current_time(city: str) -> str:
    """Return the current local time."""
    norm = _normalize_city_name(city or "")

    if norm not in _CITY_OFFSETS:
        norm = "helsinki"

    offset_hours = _CITY_OFFSETS[norm]

    now_utc = datetime.now(timezone.utc)
    local_dt = now_utc + timedelta(hours=offset_hours)

    display_city = city.strip() or "Helsinki"
    if _normalize_city_name(display_city) not in _CITY_OFFSETS:
        display_city = "Helsinki"

    return local_dt.strftime("%Y-%m-%d %H:%M:%S"), display_city


@tool
def time(city: str) -> str:
    """Get the current local time."""
    current_time, display_city = get_current_time(city)
    return f"{display_city}: {current_time}"
