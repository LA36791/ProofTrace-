"""Shopping cart: line items, subtotal, and quantity handling."""

class CartItem:
    def __init__(self, sku, name, unit_price, quantity=1):
        self.sku = sku
        self.name = name
        self.unit_price = unit_price
        self.quantity = quantity

    def line_total(self):
        return self.unit_price * self.quantity


class Cart:
    def __init__(self):
        self.items = []

    def add(self, item):
        self.items.append(item)

    def subtotal(self):
        return sum(i.line_total() for i in self.items)

    def item_count(self):
        return sum(i.quantity for i in self.items)
