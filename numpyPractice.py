import marimo

__generated_with = "0.15.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    return mo, np


@app.cell
def _(mo):
    mo.md("""## Datatypes in Python and Numpy""")
    return


@app.cell
def _(np):
    # integer array
    np.array([1, 2, 4, 5, 3])
    return


@app.cell
def _(np):
    np.array([3.14, 4, 2, 3])
    return


@app.cell
def _(np):
    np.array([1, 2, 4, 3], dtype="float32")
    return


@app.cell
def _(np):
    # nested lists result in multi-dimensional arrays
    np.array([range(i, i + 3) for i in [2, 4, 6]])
    return


@app.cell
def _(np):
    # Create a length-10 integer array filled with zeros
    np.zeros(10, dtype="int")
    return


@app.cell
def _(np):
    # Create a 3x5 floating-point array filled with ones
    np.ones((3, 5), dtype=float)
    return


@app.cell
def _(np):
    # Create a 3x5 array filled with 3.14
    np.full((3, 5), 3.14)
    return


@app.cell
def _(np):
    # Create an array filled with a linear sequence
    # Starting at 0, ending at 20, stepping by 2
    # (this is similar to the built-in range() function)
    np.arange(0, 20, 2)
    return


@app.cell
def _(np):
    # Create an array of five values evenly spaced between 0 and 1
    np.linspace(0, 1, 5)
    return


@app.cell
def _(np):
    # Create a 3x3 array of uniformly distributed
    # random values between 0 and 1
    np.random.random((3, 3))
    return


@app.cell
def _(np):
    # Create a 3x3 array of normally distributed random values
    # with mean 0 and standard deviation 1
    np.random.normal(0, 1, (3, 3))
    return


@app.cell
def _(np):
    # Create a 3x3 array of random integers in the interval [0, 10)
    np.random.randint(0, 10, (3, 3))
    return


@app.cell
def _(np):
    # Create a 3x3 identity matrix
    np.eye(3)
    return


@app.cell
def _(np):
    # Create an uninitialized array of three integers
    # The values will be whatever happens to already exist at that memory location
    np.empty(3)
    return


@app.cell
def _(mo):
    mo.md("""## Numpy Arrays""")
    return


@app.cell
def _(np):
    np.random.seed(0)  # seed for reproducibility
    x1 = np.random.randint(10, size=6)  # One-dimensional array
    x2 = np.random.randint(10, size=(3, 4))  # Two-dimensional array
    x3 = np.random.randint(10, size=(3, 4, 5))  # Three-dimensional array
    return x1, x2, x3


@app.cell
def _(x3):
    # ndim - number of dimensions
    # shape - size of ech dimension
    # size - total size of array
    print("x3 ndim: ", x3.ndim)
    print("x3 shape:", x3.shape)
    print("x3 size: ", x3.size)
    return


@app.cell
def _(x3):
    print("dtype:", x3.dtype)
    return


@app.cell
def _(x3):
    # itemsize - the size in bytes of each array element
    # nbytes - total size of the array in bytes
    print("itemsize:", x3.itemsize, "bytes")
    print("nbytes:", x3.nbytes, "bytes")
    return


@app.cell
def _(x1):
    x1
    return


@app.cell
def _(x1):
    x1[0]
    return


@app.cell
def _(x1):
    x1[4]
    return


@app.cell
def _(x1):
    x1[-1]
    return


@app.cell
def _(x1):
    x1[-2]
    return


@app.cell
def _(x2):
    x2
    return


@app.cell
def _(x2):
    x2[0, 0]
    return


@app.cell
def _(x2):
    x2[2, 0]
    return


@app.cell
def _(x2):
    x2[2, -1]
    return


@app.cell
def _(x2):
    x2[0, 0] = 12
    x2
    return


@app.cell
def _(x1):
    x1[0] = 3.14159  # this will be truncated!
    x1
    return


@app.cell
def _(np):
    x = np.arange(10)
    x
    return (x,)


@app.cell
def _(x):
    x[:5]  # first five elements
    return


@app.cell
def _(x):
    x[5:]  # elements after index 5
    return


@app.cell
def _(x):
    x[4:7]  # middle sub-array
    return


@app.cell
def _(x):
    x[::2]  # every other element
    return


@app.cell
def _(x):
    x[1::2]  # every other element, starting at index 1
    return


@app.cell
def _(x):
    x[::-1]  # all elements, reversed
    return


@app.cell
def _(x):
    x[5::-2]  # reversed every other from index 5
    return


@app.cell
def _(x2):
    x2
    return


@app.cell
def _(x2):
    x2[:2, :3]  # two rows, three columns
    return


@app.cell
def _(x2):
    x2[:3, ::2]  # all rows, every other column
    return


@app.cell
def _(x2):
    x2[::-1, ::-1]
    return


@app.cell
def _(x2):
    print(x2[:, 0])  # first column of x2
    return


@app.cell
def _(x2):
    print(x2[0, :])  # first row of x2
    return


@app.cell
def _(x2):
    print(x2[0])  # equivalent to x2[0, :]
    return


@app.cell
def _(x2):
    print(x2)
    return


@app.cell
def _(x2):
    x2_sub = x2[:2, :2]
    print(x2_sub)
    return (x2_sub,)


@app.cell
def _(x2_sub):
    x2_sub[0, 0] = 99
    print(x2_sub)
    return


@app.cell
def _(x2):
    print(x2)
    return


@app.cell
def _(x2):
    x2_sub_copy = x2[:2, :2].copy()
    print(x2_sub_copy)
    return (x2_sub_copy,)


@app.cell
def _(x2_sub_copy):
    x2_sub_copy[0, 0] = 42
    print(x2_sub_copy)
    return


@app.cell
def _(x2):
    print(x2)
    return


@app.cell
def _(np):
    grid = np.arange(1, 10).reshape((3, 3))
    print(grid)
    return


@app.cell
def _(np):
    _x = np.array([1, 2, 3])

    # row vector via reshape
    print(_x.reshape((1, 3)))
    print()

    # row vector via newaxis
    print(_x[np.newaxis, :])
    print()

    # column vector via reshape
    print(_x.reshape((3, 1)))
    print()

    # column vector via newaxis
    print(_x[:, np.newaxis])
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
