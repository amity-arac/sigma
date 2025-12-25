# Copyright Amity
"""
Airline Environment Tools

All tools for the airline customer service environment.
"""

import json
from copy import deepcopy
from typing import Any, Dict, List

from sigma.envs.tool import Tool


# Helper functions
def _get_user(data: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    """Get user from database."""
    if user_id not in data["users"]:
        raise ValueError(f"User {user_id} not found")
    return data["users"][user_id]


def _get_reservation(data: Dict[str, Any], reservation_id: str) -> Dict[str, Any]:
    """Get reservation from database."""
    if reservation_id not in data["reservations"]:
        raise ValueError(f"Reservation {reservation_id} not found")
    return data["reservations"][reservation_id]


def _get_flight(data: Dict[str, Any], flight_number: str) -> Dict[str, Any]:
    """Get flight from database."""
    if flight_number not in data["flights"]:
        raise ValueError(f"Flight {flight_number} not found")
    return data["flights"][flight_number]


def _get_flight_instance(data: Dict[str, Any], flight_number: str, date: str) -> Dict[str, Any]:
    """Get flight instance from database."""
    flight = _get_flight(data, flight_number)
    if date not in flight["dates"]:
        raise ValueError(f"Flight {flight_number} not found on date {date}")
    return flight["dates"][date]


def _get_new_reservation_id(data: Dict[str, Any]) -> str:
    """Get a new reservation id."""
    for reservation_id in ["HATHAT", "HATHAU", "HATHAV"]:
        if reservation_id not in data["reservations"]:
            return reservation_id
    raise ValueError("Too many reservations")


def _get_new_payment_id() -> List[int]:
    """Get a new payment id."""
    return [3221322, 3221323, 3221324]


def _get_datetime() -> str:
    """Get the current datetime."""
    return "2024-05-15T15:00:00"


def _search_direct_flight(
    data: Dict[str, Any],
    date: str,
    origin: str = None,
    destination: str = None,
    leave_after: str = None,
) -> List[Dict[str, Any]]:
    """Search for direct flights."""
    results = []
    for flight in data["flights"].values():
        check = (
            (origin is None or flight["origin"] == origin)
            and (destination is None or flight["destination"] == destination)
            and (date in flight["dates"])
            and (flight["dates"][date].get("status", "available") == "available")
            and (
                leave_after is None
                or flight["scheduled_departure_time_est"] >= leave_after
            )
        )
        if check:
            flight_date = flight["dates"][date]
            direct_flight = {
                "flight_number": flight["flight_number"],
                "origin": flight["origin"],
                "destination": flight["destination"],
                "status": "available",
                "scheduled_departure_time_est": flight["scheduled_departure_time_est"],
                "scheduled_arrival_time_est": flight["scheduled_arrival_time_est"],
                "available_seats": flight_date.get("available_seats", {}),
                "prices": flight_date.get("prices", {}),
            }
            results.append(direct_flight)
    return results


class BookReservation(Tool):
    @staticmethod
    def invoke(
        data: Dict[str, Any],
        user_id: str,
        origin: str,
        destination: str,
        flight_type: str,
        cabin: str,
        flights: List[Dict[str, Any]],
        passengers: List[Dict[str, Any]],
        payment_methods: List[Dict[str, Any]],
        total_baggages: int,
        nonfree_baggages: int,
        insurance: str,
    ) -> str:
        user = _get_user(data, user_id)
        reservation_id = _get_new_reservation_id(data)

        reservation = {
            "reservation_id": reservation_id,
            "user_id": user_id,
            "origin": origin,
            "destination": destination,
            "flight_type": flight_type,
            "cabin": cabin,
            "flights": [],
            "passengers": deepcopy(passengers),
            "payment_history": deepcopy(payment_methods),
            "created_at": _get_datetime(),
            "total_baggages": total_baggages,
            "nonfree_baggages": nonfree_baggages,
            "insurance": insurance,
            "status": "active",
        }

        # Update flights and calculate price
        total_price = 0
        all_flights_date_data = []

        for flight_info in flights:
            flight_number = flight_info["flight_number"]
            flight = _get_flight(data, flight_number)
            flight_date_data = _get_flight_instance(data, flight_number, flight_info["date"])

            # Checking flight availability
            if flight_date_data.get("status") == "cancelled":
                return f"Error: Flight {flight_number} not available on date {flight_info['date']}"

            # Checking seat availability
            available_seats = flight_date_data.get("available_seats", {})
            if available_seats.get(cabin, 0) < len(passengers):
                return f"Error: Not enough seats on flight {flight_number}"

            # Calculate price
            prices = flight_date_data.get("prices", {})
            price = prices.get(cabin, 0)

            # Update reservation
            reservation["flights"].append({
                "origin": flight["origin"],
                "destination": flight["destination"],
                "flight_number": flight_number,
                "date": flight_info["date"],
                "price": price,
            })
            all_flights_date_data.append(flight_date_data)
            total_price += price * len(passengers)

        # Add insurance fee
        if insurance == "yes":
            total_price += 30 * len(passengers)

        # Add baggage fee
        total_price += 50 * nonfree_baggages

        for payment_method in payment_methods:
            payment_id = payment_method["payment_id"]
            amount = payment_method["amount"]
            if payment_id not in user.get("payment_methods", {}):
                return f"Error: Payment method {payment_id} not found"

            user_payment_method = user["payment_methods"][payment_id]
            if user_payment_method.get("source") in {"gift_card", "certificate"}:
                if user_payment_method.get("amount", 0) < amount:
                    return f"Error: Not enough balance in payment method {payment_id}"

        total_payment = sum(pm["amount"] for pm in payment_methods)
        if total_payment != total_price:
            return f"Error: Payment amount does not add up, total price is {total_price}, but paid {total_payment}"

        # if checks pass, deduct payment
        for payment_method in payment_methods:
            payment_id = payment_method["payment_id"]
            amount = payment_method["amount"]
            user_payment_method = user["payment_methods"][payment_id]
            if user_payment_method.get("source") == "gift_card":
                user_payment_method["amount"] -= amount
            elif user_payment_method.get("source") == "certificate":
                del user["payment_methods"][payment_id]

        # Update DB
        for flight_date_data in all_flights_date_data:
            if "available_seats" in flight_date_data:
                flight_date_data["available_seats"][cabin] -= len(passengers)
        data["reservations"][reservation_id] = reservation
        data["users"][user_id]["reservations"].append(reservation_id)

        return json.dumps(reservation)

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "book_reservation",
                "description": "Book a reservation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {
                            "type": "string",
                            "description": "The ID of the user to book the reservation such as 'sara_doe_496'.",
                        },
                        "origin": {
                            "type": "string",
                            "description": "The IATA code for the origin city such as 'SFO'.",
                        },
                        "destination": {
                            "type": "string",
                            "description": "The IATA code for the destination city such as 'JFK'.",
                        },
                        "flight_type": {
                            "type": "string",
                            "enum": ["one_way", "round_trip"],
                            "description": "The type of flight such as 'one_way' or 'round_trip'.",
                        },
                        "cabin": {
                            "type": "string",
                            "enum": ["basic_economy", "economy", "business"],
                            "description": "The cabin class such as 'basic_economy', 'economy', or 'business'.",
                        },
                        "flights": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "flight_number": {"type": "string"},
                                    "date": {"type": "string"},
                                },
                            },
                            "description": "An array of objects containing details about each piece of flight.",
                        },
                        "passengers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "first_name": {"type": "string"},
                                    "last_name": {"type": "string"},
                                    "dob": {"type": "string"},
                                },
                            },
                            "description": "An array of objects containing details about each passenger.",
                        },
                        "payment_methods": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "payment_id": {"type": "string"},
                                    "amount": {"type": "number"},
                                },
                            },
                            "description": "An array of objects containing details about each payment method.",
                        },
                        "total_baggages": {
                            "type": "integer",
                            "description": "The total number of baggage items to book the reservation.",
                        },
                        "nonfree_baggages": {
                            "type": "integer",
                            "description": "The number of non-free baggage items to book the reservation.",
                        },
                        "insurance": {
                            "type": "string",
                            "enum": ["yes", "no"],
                            "description": "Whether the reservation has insurance.",
                        },
                    },
                    "required": [
                        "user_id",
                        "origin",
                        "destination",
                        "flight_type",
                        "cabin",
                        "flights",
                        "passengers",
                        "payment_methods",
                        "total_baggages",
                        "nonfree_baggages",
                        "insurance",
                    ],
                },
            },
        }


class Calculate(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any], expression: str) -> str:
        if not all(char in "0123456789+-*/(). " for char in expression):
            return "Error: invalid characters in expression"
        try:
            return str(round(float(eval(expression, {"__builtins__": None}, {})), 2))
        except Exception as e:
            return f"Error: {e}"

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Calculate the result of a mathematical expression.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The mathematical expression to calculate, such as '2 + 2'. The expression can contain numbers, operators (+, -, *, /), parentheses, and spaces.",
                        },
                    },
                    "required": ["expression"],
                },
            },
        }


class CancelReservation(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any], reservation_id: str) -> str:
        if reservation_id not in data["reservations"]:
            return "Error: reservation not found"

        reservation = data["reservations"][reservation_id]

        # reverse the payment
        refunds = []
        for payment in reservation["payment_history"]:
            refunds.append({
                "payment_id": payment["payment_id"],
                "amount": -payment["amount"],
            })
        reservation["payment_history"].extend(refunds)
        reservation["status"] = "cancelled"

        return json.dumps(reservation)

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "cancel_reservation",
                "description": (
                    "Cancel the whole reservation. The agent needs to explain the cancellation detail "
                    "and ask for explicit user confirmation (yes/no) to proceed."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reservation_id": {
                            "type": "string",
                            "description": "The reservation ID, such as 'ZFA04Y'.",
                        },
                    },
                    "required": ["reservation_id"],
                },
            },
        }


class GetReservationDetails(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any], reservation_id: str) -> str:
        if reservation_id not in data["reservations"]:
            return "Error: reservation not found"
        return json.dumps(data["reservations"][reservation_id])

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "get_reservation_details",
                "description": "Get the details of a reservation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reservation_id": {
                            "type": "string",
                            "description": "The reservation ID, such as '8JX2WO'.",
                        },
                    },
                    "required": ["reservation_id"],
                },
            },
        }


class GetUserDetails(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any], user_id: str) -> str:
        if user_id not in data["users"]:
            return "Error: user not found"
        return json.dumps(data["users"][user_id])

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "get_user_details",
                "description": "Get the details of a user, including their reservations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {
                            "type": "string",
                            "description": "The user ID, such as 'sara_doe_496'.",
                        },
                    },
                    "required": ["user_id"],
                },
            },
        }


class ListAllAirports(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any]) -> str:
        airports = [
            {"iata": "SFO", "city": "San Francisco"},
            {"iata": "JFK", "city": "New York"},
            {"iata": "LAX", "city": "Los Angeles"},
            {"iata": "ORD", "city": "Chicago"},
            {"iata": "DFW", "city": "Dallas"},
            {"iata": "DEN", "city": "Denver"},
            {"iata": "SEA", "city": "Seattle"},
            {"iata": "ATL", "city": "Atlanta"},
            {"iata": "MIA", "city": "Miami"},
            {"iata": "BOS", "city": "Boston"},
            {"iata": "PHX", "city": "Phoenix"},
            {"iata": "IAH", "city": "Houston"},
            {"iata": "LAS", "city": "Las Vegas"},
            {"iata": "MCO", "city": "Orlando"},
            {"iata": "EWR", "city": "Newark"},
            {"iata": "CLT", "city": "Charlotte"},
            {"iata": "MSP", "city": "Minneapolis"},
            {"iata": "DTW", "city": "Detroit"},
            {"iata": "PHL", "city": "Philadelphia"},
            {"iata": "LGA", "city": "LaGuardia"},
        ]
        return json.dumps(airports)

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "list_all_airports",
                "description": "Returns a list of all available airports.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }


class SearchDirectFlight(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any], origin: str, destination: str, date: str) -> str:
        results = _search_direct_flight(data, date=date, origin=origin, destination=destination)
        return json.dumps(results)

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "search_direct_flight",
                "description": "Search for direct flights between two cities on a specific date.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "origin": {
                            "type": "string",
                            "description": "The origin city airport in three letters, such as 'JFK'.",
                        },
                        "destination": {
                            "type": "string",
                            "description": "The destination city airport in three letters, such as 'LAX'.",
                        },
                        "date": {
                            "type": "string",
                            "description": "The date of the flight in the format 'YYYY-MM-DD', such as '2024-01-01'.",
                        },
                    },
                    "required": ["origin", "destination", "date"],
                },
            },
        }


class SearchOnestopFlight(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any], origin: str, destination: str, date: str) -> str:
        results = []
        for result1 in _search_direct_flight(data, date=date, origin=origin, destination=None):
            result1["date"] = date
            date2 = (
                f"2024-05-{int(date[-2:]) + 1}"
                if "+1" in result1["scheduled_arrival_time_est"]
                else date
            )
            for result2 in _search_direct_flight(
                data,
                date=date2,
                origin=result1["destination"],
                destination=destination,
                leave_after=result1["scheduled_arrival_time_est"],
            ):
                result2["date"] = date2
                results.append([result1, result2])
        return json.dumps(results)

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "search_onestop_flight",
                "description": "Search for one-stop flights between two cities on a specific date.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "origin": {
                            "type": "string",
                            "description": "The origin city airport in three letters, such as 'JFK'.",
                        },
                        "destination": {
                            "type": "string",
                            "description": "The destination city airport in three letters, such as 'LAX'.",
                        },
                        "date": {
                            "type": "string",
                            "description": "The date of the flight in the format 'YYYY-MM-DD', such as '2024-05-01'.",
                        },
                    },
                    "required": ["origin", "destination", "date"],
                },
            },
        }


class SendCertificate(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any], user_id: str, amount: int) -> str:
        if user_id not in data["users"]:
            return "Error: user not found"

        user = data["users"][user_id]

        # add a certificate, assume at most 3 cases per task
        for payment_id in [f"certificate_{id}" for id in _get_new_payment_id()]:
            if payment_id not in user.get("payment_methods", {}):
                if "payment_methods" not in user:
                    user["payment_methods"] = {}
                user["payment_methods"][payment_id] = {
                    "id": payment_id,
                    "amount": amount,
                    "source": "certificate",
                }
                return f"Certificate {payment_id} added to user {user_id} with amount {amount}."

        return "Error: Too many certificates"

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "send_certificate",
                "description": "Send a certificate to a user. Be careful!",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {
                            "type": "string",
                            "description": "The ID of the user to send the certificate to, such as 'sara_doe_496'.",
                        },
                        "amount": {
                            "type": "integer",
                            "description": "The amount of the certificate to send.",
                        },
                    },
                    "required": ["user_id", "amount"],
                },
            },
        }


class Think(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any], thought: str) -> str:
        return ""

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "think",
                "description": (
                    "Use the tool to think about something. It will not obtain new information or change the database, "
                    "but just append the thought to the log. Use it when complex reasoning or some cache memory is needed."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "thought": {
                            "type": "string",
                            "description": "A thought to think about.",
                        },
                    },
                    "required": ["thought"],
                },
            },
        }


class TransferToHumanAgents(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any], summary: str) -> str:
        return "Transfer successful"

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "transfer_to_human_agents",
                "description": (
                    "Transfer the user to a human agent, with a summary of the user's issue. "
                    "Only transfer if the user explicitly asks for a human agent, or if the user's issue cannot be resolved by the agent with the available tools."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "A summary of the user's issue.",
                        },
                    },
                    "required": ["summary"],
                },
            },
        }


class UpdateReservationBaggages(Tool):
    @staticmethod
    def invoke(
        data: Dict[str, Any],
        reservation_id: str,
        total_baggages: int,
        nonfree_baggages: int,
        payment_id: str,
    ) -> str:
        if reservation_id not in data["reservations"]:
            return "Error: reservation not found"

        reservation = data["reservations"][reservation_id]
        user_id = reservation["user_id"]

        if user_id not in data["users"]:
            return "Error: user not found"

        user = data["users"][user_id]

        # Calculate price
        total_price = 50 * max(0, nonfree_baggages - reservation.get("nonfree_baggages", 0))

        # Check payment
        if payment_id not in user.get("payment_methods", {}):
            return "Error: payment method not found"

        payment_method = user["payment_methods"][payment_id]
        if payment_method.get("source") == "certificate":
            return "Error: certificate cannot be used to update reservation"
        elif payment_method.get("source") == "gift_card" and payment_method.get("amount", 0) < total_price:
            return "Error: gift card balance is not enough"

        # Deduct payment
        if payment_method.get("source") == "gift_card":
            payment_method["amount"] -= total_price

        # Create payment if total price is not 0
        if total_price != 0:
            reservation["payment_history"].append({
                "payment_id": payment_id,
                "amount": total_price,
            })

        # Update reservation
        reservation["total_baggages"] = total_baggages
        reservation["nonfree_baggages"] = nonfree_baggages

        return json.dumps(reservation)

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "update_reservation_baggages",
                "description": "Update the baggage information of a reservation. The agent needs to explain the modification detail and ask for explicit user confirmation (yes/no) to proceed.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reservation_id": {
                            "type": "string",
                            "description": "The reservation ID, such as 'ZFA04Y'.",
                        },
                        "total_baggages": {
                            "type": "integer",
                            "description": "The updated total number of baggage items included in the reservation.",
                        },
                        "nonfree_baggages": {
                            "type": "integer",
                            "description": "The updated number of non-free baggage items included in the reservation.",
                        },
                        "payment_id": {
                            "type": "string",
                            "description": "The payment id stored in user profile, such as 'credit_card_7815826', 'gift_card_7815826', 'certificate_7815826'.",
                        },
                    },
                    "required": ["reservation_id", "total_baggages", "nonfree_baggages", "payment_id"],
                },
            },
        }


class UpdateReservationFlights(Tool):
    @staticmethod
    def invoke(
        data: Dict[str, Any],
        reservation_id: str,
        cabin: str,
        flights: List[Dict[str, Any]],
        payment_id: str,
    ) -> str:
        if reservation_id not in data["reservations"]:
            return "Error: reservation not found"

        reservation = data["reservations"][reservation_id]
        user_id = reservation["user_id"]

        if user_id not in data["users"]:
            return "Error: user not found"

        user = data["users"][user_id]

        # Check payment method
        if payment_id not in user.get("payment_methods", {}):
            return "Error: payment method not found"

        payment_method = user["payment_methods"][payment_id]
        if payment_method.get("source") == "certificate":
            return "Error: certificate cannot be used to update reservation"

        # Update flights and calculate price
        total_price = 0
        reservation_flights = []

        for flight_info in flights:
            # if existing flight, keep it
            matching_reservation_flight = next(
                (
                    rf
                    for rf in reservation.get("flights", [])
                    if rf["flight_number"] == flight_info["flight_number"]
                    and rf["date"] == flight_info["date"]
                    and cabin == reservation.get("cabin")
                ),
                None,
            )
            if matching_reservation_flight:
                total_price += matching_reservation_flight["price"] * len(reservation.get("passengers", []))
                reservation_flights.append(matching_reservation_flight)
                continue

            # If new flight:
            if flight_info["flight_number"] not in data["flights"]:
                return f"Error: Flight {flight_info['flight_number']} not found"

            flight = data["flights"][flight_info["flight_number"]]
            flight_date = flight_info["date"]

            if flight_date not in flight["dates"]:
                return f"Error: Flight {flight_info['flight_number']} not available on date {flight_date}"

            flight_date_data = flight["dates"][flight_date]

            # Check flight availability
            if flight_date_data.get("status") == "cancelled":
                return f"Error: Flight {flight_info['flight_number']} not available on date {flight_date}"

            # Check seat availability
            available_seats = flight_date_data.get("available_seats", {})
            if available_seats.get(cabin, 0) < len(reservation.get("passengers", [])):
                return f"Error: Not enough seats on flight {flight_info['flight_number']}"

            # Calculate price and add to reservation
            prices = flight_date_data.get("prices", {})
            price = prices.get(cabin, 0)

            reservation_flight = {
                "flight_number": flight_info["flight_number"],
                "date": flight_date,
                "price": price,
                "origin": flight["origin"],
                "destination": flight["destination"],
            }
            total_price += price * len(reservation.get("passengers", []))
            reservation_flights.append(reservation_flight)

        # Deduct amount already paid for reservation
        total_price -= sum(f["price"] for f in reservation.get("flights", [])) * len(
            reservation.get("passengers", [])
        )

        # Check gift card balance
        if payment_method.get("source") == "gift_card" and payment_method.get("amount", 0) < total_price:
            return "Error: gift card balance is not enough"

        # Deduct payment
        if payment_method.get("source") == "gift_card":
            payment_method["amount"] -= total_price

        # Create payment if total price is not 0
        if total_price != 0:
            reservation["payment_history"].append({
                "payment_id": payment_id,
                "amount": total_price,
            })

        # Update reservation
        reservation["flights"] = reservation_flights
        reservation["cabin"] = cabin

        return json.dumps(reservation)

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "update_reservation_flights",
                "description": "Update the flight information of a reservation. The agent needs to explain the modification detail and ask for explicit user confirmation (yes/no) to proceed.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reservation_id": {
                            "type": "string",
                            "description": "The reservation ID, such as 'ZFA04Y'.",
                        },
                        "cabin": {
                            "type": "string",
                            "enum": ["basic_economy", "economy", "business"],
                            "description": "The cabin class of the reservation.",
                        },
                        "flights": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "flight_number": {"type": "string"},
                                    "date": {"type": "string"},
                                },
                            },
                            "description": "An array of objects containing details about each piece of flight in the ENTIRE new reservation. Even if a flight segment is not changed, it should still be included in the array.",
                        },
                        "payment_id": {
                            "type": "string",
                            "description": "The payment id stored in user profile, such as 'credit_card_7815826', 'gift_card_7815826', 'certificate_7815826'.",
                        },
                    },
                    "required": ["reservation_id", "cabin", "flights", "payment_id"],
                },
            },
        }


class UpdateReservationPassengers(Tool):
    @staticmethod
    def invoke(
        data: Dict[str, Any],
        reservation_id: str,
        passengers: List[Dict[str, Any]],
    ) -> str:
        if reservation_id not in data["reservations"]:
            return "Error: reservation not found"

        reservation = data["reservations"][reservation_id]

        if len(passengers) != len(reservation.get("passengers", [])):
            return "Error: number of passengers does not match"

        reservation["passengers"] = deepcopy(passengers)
        return json.dumps(reservation)

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "update_reservation_passengers",
                "description": "Update the passenger information of a reservation. The agent needs to explain the modification detail and ask for explicit user confirmation (yes/no) to proceed.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reservation_id": {
                            "type": "string",
                            "description": "The reservation ID, such as 'ZFA04Y'.",
                        },
                        "passengers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "first_name": {"type": "string"},
                                    "last_name": {"type": "string"},
                                    "dob": {"type": "string"},
                                },
                            },
                            "description": "An array of objects containing details about each passenger.",
                        },
                    },
                    "required": ["reservation_id", "passengers"],
                },
            },
        }


class GetFlightDetails(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any], flight_number: str, date: str) -> str:
        if flight_number not in data["flights"]:
            return "Error: flight not found"

        flight = data["flights"][flight_number]
        if date not in flight["dates"]:
            return "Error: flight not found on this date"

        flight_date_data = flight["dates"][date]
        status = flight_date_data.get("status", "available")
        
        result = {
            "flight_number": flight["flight_number"],
            "origin": flight["origin"],
            "destination": flight["destination"],
            "scheduled_departure_time_est": flight["scheduled_departure_time_est"],
            "scheduled_arrival_time_est": flight["scheduled_arrival_time_est"],
            "date": date,
            "status": status,
        }
        
        # Include seat availability and prices for available flights
        if status == "available":
            result["available_seats"] = flight_date_data.get("available_seats", {})
            result["prices"] = flight_date_data.get("prices", {})
        
        # Include actual times for landed/delayed flights
        if status in ["landed", "delayed", "on time", "flying"]:
            if "actual_departure_time_est" in flight_date_data:
                result["actual_departure_time_est"] = flight_date_data["actual_departure_time_est"]
            if "actual_arrival_time_est" in flight_date_data:
                result["actual_arrival_time_est"] = flight_date_data["actual_arrival_time_est"]
        
        return json.dumps(result)

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "get_flight_details",
                "description": "Get detailed information about a specific flight on a specific date, including status, schedule, and for available flights: seat availability and prices by cabin class.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "flight_number": {
                            "type": "string",
                            "description": "The flight number, such as 'HAT001'.",
                        },
                        "date": {
                            "type": "string",
                            "description": "The date of the flight in the format 'YYYY-MM-DD', such as '2024-05-16'.",
                        },
                    },
                    "required": ["flight_number", "date"],
                },
            },
        }


class GetFlightStatus(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any], flight_number: str, date: str) -> str:
        if flight_number not in data["flights"]:
            return "Error: flight not found"

        flight = data["flights"][flight_number]
        if date not in flight["dates"]:
            return "Error: flight not found on this date"

        flight_date_data = flight["dates"][date]
        status = flight_date_data.get("status", "available")
        return status

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "get_flight_status",
                "description": "Get the status of a flight (available, cancelled, landed, delayed, on time, flying).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "flight_number": {
                            "type": "string",
                            "description": "The flight number.",
                        },
                        "date": {
                            "type": "string",
                            "description": "The date of the flight in the format 'YYYY-MM-DD'.",
                        },
                    },
                    "required": ["flight_number", "date"],
                },
            },
        }


ALL_TOOLS = [
    BookReservation,
    Calculate,
    CancelReservation,
    GetFlightDetails,
    GetFlightStatus,
    GetReservationDetails,
    GetUserDetails,
    ListAllAirports,
    SearchDirectFlight,
    SearchOnestopFlight,
    SendCertificate,
    Think,
    TransferToHumanAgents,
    UpdateReservationBaggages,
    UpdateReservationFlights,
    UpdateReservationPassengers,
]
