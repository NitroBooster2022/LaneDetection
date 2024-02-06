import numpy as np

def find_center_indices(hist, threshold):
    # Find indices where histogram values are above the threshold
    above_threshold = np.where(hist > threshold)[0]

    # Find consecutive groups of five or more indices
    consecutive_groups = np.split(above_threshold, np.where(np.diff(above_threshold) != 1)[0] + 1)

    # Filter groups with five or more consecutive indices
    valid_groups = [group for group in consecutive_groups if len(group) >= 5]

    # Find the center index for each valid group
    center_indices = [(group[0] + group[-1]) // 2 for group in valid_groups]

    return center_indices


def find_center_conv(hist):
    # Convert the array into a boolean array based on your condition
    bool_array = hist> 1500

    # Use convolution with a window of ones
    kernel = np.ones(5, dtype=int)
    conv_result = np.convolve(bool_array, kernel, mode='valid')

    # Find the indices where the convolution result equals 5
    indices = np.where(conv_result == 5)[0]
    return indices

def find_sequences(arr, length, condition):
    indices = []
    for i in range(len(arr) - length + 1):
        if condition(arr[i:i+length]):
            indices.append(i)
    return indices

# Condition: all elements in the sequence should be greater than 1500
condition = lambda x: np.all(x > 1500)


# Example usage:
histogram = np.random.randint(0, 3001, size=640)
threshold_value = 1500
result = find_center_indices(histogram, threshold_value)
# Find indices
result2 = find_sequences(histogram, 5, condition)
print("Center indices for consecutive groups above threshold:", result)
print("Center indices for convolution:", result2)