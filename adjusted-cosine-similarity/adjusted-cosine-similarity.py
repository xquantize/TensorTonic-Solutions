import math

def adjusted_cosine_similarity(ratings_matrix: list, item_i: int, item_j: int) -> float:
    """
    Returns the adjusted cosine similarity between the requested items.
    """
    numerator = 0.0
    sum_sq_i = 0.0
    sum_sq_j = 0.0
    
    for row in ratings_matrix:
        if row[item_i] != 0 and row[item_j] != 0:
            nonzero_ratings = [r for r in row if r != 0]
            user_mean = sum(nonzero_ratings) / len(nonzero_ratings)
            
            centered_i = row[item_i] - user_mean
            centered_j = row[item_j] - user_mean
            
            numerator += centered_i * centered_j
            sum_sq_i += centered_i ** 2
            sum_sq_j += centered_j ** 2
            
    denominator = math.sqrt(sum_sq_i) * math.sqrt(sum_sq_j)
    
    if denominator == 0.0:
        return 0.0
        
    return float(numerator / denominator)
