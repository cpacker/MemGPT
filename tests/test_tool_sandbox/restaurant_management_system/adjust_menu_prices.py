def adjust_menu_prices(percentage: float) -> str:
    """
    Tool: Adjust Menu Prices
    Description: Adjusts the prices of all menu items by a given percentage.
    Args:
        percentage (float): The percentage by which to adjust prices. Positive for an increase, negative for a decrease.
    Returns:
        str: A formatted string summarizing the price adjustments.
    """
    import cowsay
    from core.menu import Menu, MenuItem  # Import a class from the codebase
    from core.utils import format_currency  # Use a utility function to test imports

    if not isinstance(percentage, (int, float)):
        raise TypeError("percentage must be a number")

    # Generate dummy menu object
    menu = Menu()
    menu.add_item(MenuItem("Burger", 8.99, "Main"))
    menu.add_item(MenuItem("Fries", 2.99, "Side"))
    menu.add_item(MenuItem("Soda", 1.99, "Drink"))

    # Make adjustments and record
    adjustments = []
    for item in menu.items:
        old_price = item.price
        item.price += item.price * (percentage / 100)
        adjustments.append(f"{item.name}: {format_currency(old_price)} -> {format_currency(item.price)}")

    # Cowsay the adjustments because why not
    cowsay.cow("Hello World")

    return "Price Adjustments:\n" + "\n".join(adjustments)
