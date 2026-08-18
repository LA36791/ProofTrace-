"""Shipping: estimate shipping cost from weight."""

BASE_RATE = 5.0
PER_KG = 1.5


def estimate_shipping(total_weight_kg):
    return BASE_RATE + total_weight_kg * PER_KG
