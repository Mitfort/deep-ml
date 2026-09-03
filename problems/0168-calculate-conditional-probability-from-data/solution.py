def conditional_probability(data, x, y):
    """
    Returns the probability P(Y=y|X=x) from list of (X, Y) pairs.
    Args:
      data: List of (X, Y) tuples
      x: value of X to condition on
      y: value of Y to check
    Returns:
      float: conditional probability, rounded to 4 decimal places
    """
    
    # P(A | B) = P(AnB) / P(B)

    # P(B) - is it a cat
    # P(AnB) - is it a blue cat

    counter_X = {}
    counter_Y = {}
    counter_XY = {}

    for x_val,y_val in data:
      counter_X[x_val] = counter_X.get(x_val,0) + 1
      counter_Y[y_val] = counter_Y.get(y_val,0) + 1
      counter_XY[(x_val,y_val)] = counter_XY.get((x_val,y_val),0) + 1

    if counter_X.get(x,0) == 0:
      return 0.0

    P_x = counter_X[x]
    P_y = counter_Y[y]
    P_xy = counter_XY.get((x,y), 0)

    return round(P_xy / P_x,4)



    