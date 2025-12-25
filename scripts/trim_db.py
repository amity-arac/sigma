#!/usr/bin/env python3
"""
Script to trim db.json files to only keep sample data for seeding scenario generation.
"""

import json
from pathlib import Path

# Configuration: how many items to keep from each section
AIRLINE_CONFIG = {
    "flights": 5,  # Keep 5 sample flights
    "users": 5,    # Keep 5 sample users
    "reservations": 5,  # Keep 5 sample reservations
}

RETAIL_CONFIG = {
    "products": 5,  # Keep 5 sample products (with all their variants)
    "users": 5,     # Keep 5 sample users
    "orders": 5,    # Keep 5 sample orders
}

def trim_dict(data: dict, max_items: int) -> dict:
    """Keep only first max_items from a dictionary."""
    if not isinstance(data, dict):
        return data
    return dict(list(data.items())[:max_items])

def trim_airline_db(db_path: Path) -> None:
    """Trim the airline db.json file."""
    print(f"Processing airline db: {db_path}")
    
    with open(db_path, 'r') as f:
        data = json.load(f)
    
    original_stats = {
        "flights": len(data.get("flights", {})),
        "users": len(data.get("users", {})),
        "reservations": len(data.get("reservations", {})),
    }
    
    # Trim each section
    trimmed_data = {}
    
    if "flights" in data:
        trimmed_data["flights"] = trim_dict(data["flights"], AIRLINE_CONFIG["flights"])
    
    if "users" in data:
        # Keep users that have reservations if possible
        users = data["users"]
        users_with_reservations = {k: v for k, v in users.items() 
                                    if v.get("reservations") and len(v["reservations"]) > 0}
        
        if len(users_with_reservations) >= AIRLINE_CONFIG["users"]:
            trimmed_data["users"] = trim_dict(users_with_reservations, AIRLINE_CONFIG["users"])
        else:
            # Mix users with reservations and some without
            trimmed_users = dict(list(users_with_reservations.items()))
            remaining = AIRLINE_CONFIG["users"] - len(trimmed_users)
            users_without = {k: v for k, v in users.items() if k not in users_with_reservations}
            trimmed_users.update(trim_dict(users_without, remaining))
            trimmed_data["users"] = trimmed_users
    
    if "reservations" in data:
        trimmed_data["reservations"] = trim_dict(data["reservations"], AIRLINE_CONFIG["reservations"])
    
    new_stats = {
        "flights": len(trimmed_data.get("flights", {})),
        "users": len(trimmed_data.get("users", {})),
        "reservations": len(trimmed_data.get("reservations", {})),
    }
    
    print(f"  Original: {original_stats}")
    print(f"  Trimmed:  {new_stats}")
    
    # Write back
    with open(db_path, 'w') as f:
        json.dump(trimmed_data, f, indent=2)
    
    print(f"  Saved trimmed file.")

def trim_retail_db(db_path: Path) -> None:
    """Trim the retail db.json file."""
    print(f"Processing retail db: {db_path}")
    
    with open(db_path, 'r') as f:
        data = json.load(f)
    
    original_stats = {
        "products": len(data.get("products", {})),
        "users": len(data.get("users", {})),
        "orders": len(data.get("orders", {})),
    }
    
    # Trim each section
    trimmed_data = {}
    
    if "products" in data:
        trimmed_data["products"] = trim_dict(data["products"], RETAIL_CONFIG["products"])
    
    if "users" in data:
        # Keep users that have orders if possible
        users = data["users"]
        users_with_orders = {k: v for k, v in users.items() 
                             if v.get("orders") and len(v["orders"]) > 0}
        
        if len(users_with_orders) >= RETAIL_CONFIG["users"]:
            trimmed_data["users"] = trim_dict(users_with_orders, RETAIL_CONFIG["users"])
        else:
            trimmed_users = dict(list(users_with_orders.items()))
            remaining = RETAIL_CONFIG["users"] - len(trimmed_users)
            users_without = {k: v for k, v in users.items() if k not in users_with_orders}
            trimmed_users.update(trim_dict(users_without, remaining))
            trimmed_data["users"] = trimmed_users
    
    if "orders" in data:
        trimmed_data["orders"] = trim_dict(data["orders"], RETAIL_CONFIG["orders"])
    
    new_stats = {
        "products": len(trimmed_data.get("products", {})),
        "users": len(trimmed_data.get("users", {})),
        "orders": len(trimmed_data.get("orders", {})),
    }
    
    print(f"  Original: {original_stats}")
    print(f"  Trimmed:  {new_stats}")
    
    # Write back
    with open(db_path, 'w') as f:
        json.dump(trimmed_data, f, indent=2)
    
    print(f"  Saved trimmed file.")

def main():
    base_path = Path(__file__).parent / "data" / "envs"
    
    airline_db = base_path / "airline" / "db.json"
    retail_db = base_path / "retail" / "db.json"
    
    if airline_db.exists():
        trim_airline_db(airline_db)
    else:
        print(f"Airline db not found: {airline_db}")
    
    print()
    
    if retail_db.exists():
        trim_retail_db(retail_db)
    else:
        print(f"Retail db not found: {retail_db}")
    
    print("\nDone! Both db.json files have been trimmed to sample data only.")

if __name__ == "__main__":
    main()
