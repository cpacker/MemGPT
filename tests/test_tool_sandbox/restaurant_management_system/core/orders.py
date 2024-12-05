from typing import Dict


class Order:
    def __init__(self, customer_name: str, items: Dict[str, int]):
        self.customer_name = customer_name
        self.items = items  # Dictionary of item names to quantities

    def calculate_total(self, menu):
        total = 0
        for item_name, quantity in self.items.items():
            menu_item = next((item for item in menu.items if item.name == item_name), None)
            if menu_item is None:
                raise ValueError(f"Menu item '{item_name}' not found.")
            total += menu_item.price * quantity
        return total
