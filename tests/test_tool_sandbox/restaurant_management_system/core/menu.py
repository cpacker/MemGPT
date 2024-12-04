from typing import List


class MenuItem:
    def __init__(self, name: str, price: float, category: str):
        self.name = name
        self.price = price
        self.category = category

    def __repr__(self):
        return f"{self.name} (${self.price:.2f}) - {self.category}"


class Menu:
    def __init__(self):
        self.items: List[MenuItem] = []

    def add_item(self, item: MenuItem):
        self.items.append(item)

    def update_price(self, name: str, new_price: float):
        for item in self.items:
            if item.name == name:
                item.price = new_price
                return
        raise ValueError(f"Menu item '{name}' not found.")
