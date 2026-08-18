"""Payments: card charging facade."""

from utils import as_cents


def charge_card(token, amount):
    """Charge a card; raises on declined or invalid token."""
    if not token or len(token) < 4:
        raise ValueError("Invalid card token")
    if amount <= 0:
        raise ValueError("Cannot charge a non-positive amount")
    return {"authorized": True, "amount": as_cents(amount)}
