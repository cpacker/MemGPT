class Customer:
    def __init__(self, name: str, loyalty_points: int = 0):
        self.name = name
        self.loyalty_points = loyalty_points

    def add_loyalty_points(self, points: int):
        self.loyalty_points += points
