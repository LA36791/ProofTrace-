"""Utils: shared helpers."""

def as_cents(amount):
    """Round a monetary amount to integer cents."""
    return round(amount * 100) / 100
