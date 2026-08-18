"""Pricing: discount codes and price computation."""

from utils import as_cents


DISCOUNTS = {
    "SAVE10": 0.10,
    "SAVE20": 0.20,
}


def apply_discount(subtotal, code):
    """Return discounted total for a valid discount code."""
    rate = DISCOUNTS.get(code, 0.0)
    discount = as_cents(subtotal * rate)
    return subtotal - discount


def price_item(unit_price, quantity):
    """Compute total price for a quantity of one item."""
    return as_cents(unit_price * quantity)
