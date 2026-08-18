"""Order orchestration: turns a cart into an order."""

from cart import Cart, CartItem
from inventory import check_stock, reserve_stock
from payments import charge_card
from pricing import apply_discount
from tax import compute_tax


def build_order(cart, discount_code=None):
    """Build an order from a cart, applying discount and tax."""
    subtotal = cart.subtotal()
    discounted = apply_discount(subtotal, discount_code or "")
    tax = compute_tax(discounted)
    total = discounted + tax
    return {
        "subtotal": subtotal,
        "discounted": discounted,
        "tax": tax,
        "total": total,
    }


def place_order(cart, discount_code, card_token):
    """Reserve stock, charge the card, and finalize the order."""
    for item in cart.items:
        if not check_stock(item.sku, item.quantity):
            raise ValueError(f"Insufficient stock for {item.sku}")
        reserve_stock(item.sku, item.quantity)

    totals = build_order(cart, discount_code)
    charge_card(card_token, totals["total"])
    return totals
