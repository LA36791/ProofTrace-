"""Inventory: stock availability and reservation."""

STOCK = {
    "SKU-A": 10,
    "SKU-B": 5,
    "SKU-C": 0,
}


def check_stock(sku, quantity):
    return STOCK.get(sku, 0) >= quantity


def reserve_stock(sku, quantity):
    if not check_stock(sku, quantity):
        raise ValueError(f"Cannot reserve {quantity} of {sku}")
    STOCK[sku] = STOCK[sku] - quantity
