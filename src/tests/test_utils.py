


from src.generative.utils import (
    sequential_train_val_split,
    sequential_undersample_3d_arr,
)

"""
sequential_train_val_split
test cases:
    arr = np.array([[[1, 2, 3]], [[-1, -2, -3]], [[4, 5, 6]], [[4, 5, 6]]])

    arr[0:15, :, :], arr[0:0, :, :], arr[0:2, :, :]

sequential_undersample_3d_arr
1 test: assert train_data.shape[0] == len(train_idx) + len(val_idx)

"""
