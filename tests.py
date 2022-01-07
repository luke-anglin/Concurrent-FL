from helpers import * 
import pytest
import numpy as np

def test_split_data():
    old_data = np.array([5,1,2,3,4,5])
    get = split_data(old_data, 3)[0].shape
    expect = (2,)
    assert(get == expect)
    # assert all shapes are equal 

