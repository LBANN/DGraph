def largest_split(size, num_splits):
    split_size = size // num_splits
    split_size += 1 if size % num_splits > 0 else 0
    return split_size
