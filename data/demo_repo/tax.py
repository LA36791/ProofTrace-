"""Tax: compute sales tax on a price."""

TAX_RATE = 0.08


def compute_tax(amount):
    """Compute sales tax on the given pre-tax amount."""
    return amount * TAX_RATE
