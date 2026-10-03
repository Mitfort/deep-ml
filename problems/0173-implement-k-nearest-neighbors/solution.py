import numpy as np

def k_nearest_neighbors(points, query_point, k):
    """
    Find k nearest neighbors to a query point
    
    Args:
        points: List of tuples representing points [(x1, y1), (x2, y2), ...]
        query_point: Tuple representing query point (x, y)
        k: Number of nearest neighbors to return
    
    Returns:
        List of k nearest neighbor points as tuples
        When distances are tied, points appearing earlier in the input list come first.
    """
    points = np.array(points)
    
    dist = [
        (np.sqrt((x - query_point[0])**2 + (y - query_point[1])**2),(x,y)) 
        for x,y in points
    ]

    dist.sort(key=lambda item: item[0])


    return [point for distance,point in dist[:k]]
