# Copyright Amity
"""
Retail Environment Data Loader

Loads data from db.json in the same directory.
"""

import json
import os
from typing import Any, Dict

FOLDER_PATH = os.path.dirname(__file__)


def load_data() -> Dict[str, Any]:
    """
    Load data from db.json in the retail environment folder.
    
    Returns a dictionary with 'orders', 'products', and 'users' keys.
    """
    db_path = os.path.join(FOLDER_PATH, "db.json")
    with open(db_path, "r") as f:
        data = json.load(f)
    return {
        "orders": data.get("orders", {}),
        "products": data.get("products", {}),
        "users": data.get("users", {}),
    }
