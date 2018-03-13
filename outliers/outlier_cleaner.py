  
import math


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """

    data = [(age, net_worth, abs(pred - net_worth)) for pred, age, net_worth in zip(predictions, ages, net_worths)]
    data.sort(key=lambda tup: tup[2])
    return data[:math.floor(0.9 * len(data))]

print(outlierCleaner([50, 1000, 10000000, 20,
                      43, 43, 54, 75, 43232, 54353, 432435 ,43, 4343], [40,50,60, 43, 43, 43, 43, 43, 43,
                                                                        43, 43, 34, 34], [500, 900, 1000000,33333, 44444, 444444,
                                                                              43333, 44444, 44444, 33333, 3333333 , 3333, 3333]))