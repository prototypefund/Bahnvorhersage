def remove_item_from_list(lst: list, item: any) -> list:
    """
    Removes an item from a list and returns the resulting list.
    Args:
        lst: list to remove item from
        item: item to remove

    Returns: resulting list
    """
    if item in lst:
        lst.remove(item)
    return lst