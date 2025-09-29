import marimo

__generated_with = "0.16.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo 
    return (mo,)


@app.cell
def _(mo):
    mo.md("""#Data Types in Python""")
    return


@app.cell
def _(mo):
    mo.md("""##Python Lists""")
    return


@app.cell
def _():
    L = list(range(10))
    L
    return (L,)


@app.cell
def _(L):
    type(L[0])
    return


@app.cell
def _(L):
    L2 = [str(c) for c in L]
    L2
    return (L2,)


@app.cell
def _(L2):
    type(L2[0])
    return


@app.cell
def _():
    L3 = [True, "2", 3.0, 4]
    [type(item) for item in L3]
    return


@app.cell
def _(mo):
    mo.md("""##Fixed-Type Arrays in Python""")
    return


@app.cell
def _(L):
    import array 
    A = array.array('i', L)
    A
    return


@app.cell
def _():
    import numpy as np 
    return (np,)


@app.cell
def _(mo):
    mo.md("""## Creating Arrays from Python Lists""")
    return


@app.cell
def _(np):
    np.array([1, 4, 2, 5, 3])
    return


@app.cell
def _(np):
    np.array([3.14, 4, 2, 3])
    return


@app.cell
def _(np):
    np.array([1, 2, 3, 4], dtype='float32')
    return


@app.cell
def _(np):
    # nested lists result in multi-dimensional arrays
    np.array([range(i, i + 3) for i in [2, 4, 6]])
    return


@app.cell
def _(mo):
    mo.md("""## Creating Arrays from Scratch""")
    return


@app.cell
def _(np):
    # Create a length-10 integer array filled with zeros
    np.zeros(10, dtype=int)
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
    mo.md("""## NumPy Standard Data Types""")
    return


@app.cell
def _(np):
    np.zeros(10, dtype='int16')
    return


@app.cell
def _(np):
    np.zeros(10, dtype=np.int16)
    return


@app.cell
def _(mo):
    mo.md("""#Basics of NumPy Arrays""")
    return


@app.cell
def _(mo):
    mo.md("""## Numpy Array Attributes""")
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
    print("x3 ndim: ", x3.ndim) # number of dimensions 
    print("x3 shape:", x3.shape) # size of each dimension 
    print("x3 size: ", x3.size) # the total size of the array 
    return


@app.cell
def _(x3):
    print("dtype:", x3.dtype) # dtype = array data type 
    return


@app.cell
def _(x3):
    print("itemsize:", x3.itemsize, "bytes") # size in bytes of each element 
    print("nbytes:", x3.nbytes, "bytes") # total size in bytes of array 
    return


@app.cell
def _(mo):
    mo.md("""## Array Indexing: Accessing Single Elements""")
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
def _(mo):
    mo.md("""## Array Slicing: Accessing Subarrays""")
    return


@app.cell
def _(mo):
    mo.md("""### One-dimensional subarrays""")
    return


@app.cell
def _(np):
    x4 = np.arange(10)
    x4
    return (x4,)


@app.cell
def _(x4):
    x4[:5]  # first five elements
    return


@app.cell
def _(x4):
    x4[5:]  # elements after index 5
    return


@app.cell
def _(x4):
    x4[4:7]  # middle sub-array
    return


@app.cell
def _(x4):
    x4[::2]  # every other element
    return


@app.cell
def _(x4):
    x4[1::2]  # every other element, starting at index 1
    return


@app.cell
def _(x4):
    x4[::-1]  # all elements, reversed
    return


@app.cell
def _(x4):
    x4[5::-2]  # reversed every other from index 5
    return


@app.cell
def _(mo):
    mo.md("""### Multi-dimensional subarrays""")
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
    x2[::-1, ::-1] # subarray dimensions reversed
    return


@app.cell
def _(mo):
    mo.md("""### Accessing array rows and columns""")
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
def _(mo):
    mo.md("""##Subarrays as no-copy views""")
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
def _(mo):
    mo.md("""## Create Copies of arrays""")
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
def _(mo):
    mo.md("""## Reshaping Arrays""")
    return


@app.cell
def _(np):
    grid = np.arange(1, 10).reshape((3, 3))
    print(grid)
    return


@app.cell
def _(np):
    x5 = np.array([1, 2, 3])
    return (x5,)


@app.cell
def _(x5):
    x5.reshape((1, 3)) # row vector via reshape
    return


@app.cell
def _(np, x5):
    x5[np.newaxis, :]# row vector via new axis 

    return


@app.cell
def _(x5):
    x5.reshape((3, 1)) # column vector via reshape
    return


@app.cell
def _(np, x5):
    x5[:, np.newaxis] # column vector via newaxis
    return


@app.cell
def _(mo):
    mo.md("""## Array Concatentation and Splitting""")
    return


@app.cell
def _(np):
    x6 = np.array([1, 2, 3])
    y1 = np.array([3, 2, 1])
    np.concatenate([x6, y1])
    return x6, y1


@app.cell
def _(np, x6, y1):
    z = [99, 99, 99]
    print(np.concatenate([x6, y1, z]))
    return


@app.cell
def _(np):
    grid2 = np.array([[1, 2, 3],
                     [4, 5, 6]])
    return (grid2,)


@app.cell
def _(grid2, np):
    np.concatenate([grid2, grid2])
    return


@app.cell
def _(np):
    x7 = np.array([1, 2, 3])
    grid3 = np.array([[9, 8, 7],
                     [6, 5, 4]])
    return grid3, x7


@app.cell
def _(grid3, np, x7):
    np.vstack([x7, grid3])
    return


@app.cell
def _(grid3, np):
    y2 = np.array([[99],
                  [99]])
    np.hstack([grid3, y2])
    return


@app.cell
def _(mo):
    mo.md("""## Splitting Arrays""")
    return


@app.cell
def _(np):
    x8 = [1, 2, 3, 99, 99, 3, 2, 1]
    x9, x10, x11 = np.split(x8, [3, 5])
    print(x9, x10, x11)
    return


@app.cell
def _(np):
    grid4 = np.arange(16).reshape((4, 4))
    grid4
    return (grid4,)


@app.cell
def _(grid4, np):
    upper, lower = np.vsplit(grid4, [2])
    print(upper)
    print(lower)
    return


@app.cell
def _(grid4, np):
    left, right = np.hsplit(grid4, [2])
    print(left)
    print(right)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
