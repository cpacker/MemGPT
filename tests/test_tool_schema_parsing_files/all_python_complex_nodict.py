from typing import List, Optional


def check_order_status(
    order_number: int,
    customer_name: str,
    related_tickets: List[str],
    severity: float,
    metadata: Optional[str],
):
    """
    Check the status for an order number (integer value).

    Args:
        order_number (int): The order number to check on.
        customer_name (str): The name of the customer who placed the order.
        related_tickets (List[str]): A list of ticket numbers related to the order.
        severity (float): The severity of the request (between 0 and 1).
        metadata (Optional[str]): Additional metadata about the order.

    Returns:
        str: The status of the order (e.g. cancelled, refunded, processed, processing, shipping).
    """
    # TODO replace this with a real query to a database
    dummy_message = f"Order {order_number} is currently processing."
    return dummy_message
