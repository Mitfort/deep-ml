def label_encode_ordinal(values: list, order: list) -> list:
    """
    Encode ordinal categorical values to integers based on specified order.
    
    Args:
        values: List of categorical values to encode
        order: List specifying the order of categories from lowest (0) to highest
    
    Returns:
        List of integers representing the encoded values
    """
    n:int = len(values)
    res:list[int] = [0] * n 

    order_dict = {val:encoded for encoded,val in enumerate(order)}

    for i,val in enumerate(values):
        res[i] = order_dict.get(val,-1)

    return res
    