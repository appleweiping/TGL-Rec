"""Shared dataframe column names for processed recommendation data."""

from __future__ import annotations

USER_ID = "user_id"
ITEM_ID = "item_id"
RAW_USER_ID = "raw_user_id"
RAW_ITEM_ID = "raw_item_id"
TIMESTAMP = "timestamp"
RATING = "rating"
EVENT_ID = "event_id"
SPLIT_LOO = "split_loo"
SPLIT_GLOBAL = "split_global"

INTERACTION_COLUMNS = [
    EVENT_ID,
    USER_ID,
    ITEM_ID,
    RAW_USER_ID,
    RAW_ITEM_ID,
    TIMESTAMP,
    RATING,
    SPLIT_LOO,
    SPLIT_GLOBAL,
]

