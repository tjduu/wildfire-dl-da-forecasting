"""testing module for the generative utils.py."""

import numpy as np

from src.generative.utils import (
    sequential_train_val_split,
    sequential_undersample_3d_arr,
)


def test_sequential_undersample_3d_arr_no_jump():
    """Test sequential_undersample_3d_arr() when jump is < len arr."""
    test_arr = np.array([[[1, 2, 3]], [[-1, -2, -3]], [[4, 5, 6]], [[4, 5, 6]]])
    actual = sequential_undersample_3d_arr(arr=test_arr, sequence_jump=1, _print=False)
    desired = np.array(
        [[[1, 2, 3]], [[-1, -2, -3]], [[4, 5, 6]], [[4, 5, 6]]]
    )  # same as original.

    assert np.allclose(actual, desired)


def test_sequential_undersample_3d_arr_jump_size_within_len_arr():
    """Test sequential_undersample_3d_arr() when jump is < len arr."""
    test_arr = np.array([[[1, 2, 3]], [[-1, -2, -3]], [[4, 5, 6]], [[4, 5, 6]]])
    actual = sequential_undersample_3d_arr(arr=test_arr, sequence_jump=2, _print=False)
    desired = np.array([[[1, 2, 3]], [[4, 5, 6]]])

    assert np.allclose(actual, desired)


def test_sequential_undersample_3d_arr_jump_size_outside_len_arr():
    """Test sequential_undersample_3d_arr() when jump is > len arr."""
    test_arr = np.array([[[1, 2, 3]], [[-1, -2, -3]], [[4, 5, 6]], [[4, 5, 6]]])
    actual = sequential_undersample_3d_arr(
        arr=test_arr, sequence_jump=100, _print=False
    )
    desired = np.array([[[1, 2, 3]]])  # only 1st row by default.

    assert np.allclose(actual, desired)

def test_test_sequential_train_val_split():
    

    test_data_path = "src/tests/data/test_forecasting_dataset.npy"
    test_data = np.load(test_data_path)



# # TODO: do this.
# def test_sequential_train_val_split():
#     test_arr = np.array([[[1, 2, 3]], [[-1, -2, -3]], [[4, 5, 6]], [[4, 5, 6]]])


"""
sequential_train_val_split
test cases:
    arr = np.array([[[1, 2, 3]], [[-1, -2, -3]], [[4, 5, 6]], [[4, 5, 6]]])

    arr[0:15, :, :], arr[0:0, :, :], arr[0:2, :, :]

sequential_undersample_3d_arr
1 test: assert train_data.shape[0] == len(train_idx) + len(val_idx)

"""