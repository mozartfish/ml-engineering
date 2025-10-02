import marimo

__generated_with = "0.16.3"
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

    A = array.array("i", L)
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
    np.array([1, 2, 3, 4], dtype="float32")
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
    np.zeros(10, dtype="int16")
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
    print("x3 ndim: ", x3.ndim)  # number of dimensions
    print("x3 shape:", x3.shape)  # size of each dimension
    print("x3 size: ", x3.size)  # the total size of the array
    return


@app.cell
def _(x3):
    print("dtype:", x3.dtype)  # dtype = array data type
    return


@app.cell
def _(x3):
    print("itemsize:", x3.itemsize, "bytes")  # size in bytes of each element
    print("nbytes:", x3.nbytes, "bytes")  # total size in bytes of array
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
    x2[::-1, ::-1]  # subarray dimensions reversed
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
    x5.reshape((1, 3))  # row vector via reshape
    return


@app.cell
def _(np, x5):
    x5[np.newaxis, :]  # row vector via new axis
    return


@app.cell
def _(x5):
    x5.reshape((3, 1))  # column vector via reshape
    return


@app.cell
def _(np, x5):
    x5[:, np.newaxis]  # column vector via newaxis
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
    grid2 = np.array([[1, 2, 3], [4, 5, 6]])
    return (grid2,)


@app.cell
def _(grid2, np):
    np.concatenate([grid2, grid2])
    return


@app.cell
def _(np):
    x7 = np.array([1, 2, 3])
    grid3 = np.array([[9, 8, 7], [6, 5, 4]])
    return grid3, x7


@app.cell
def _(grid3, np, x7):
    np.vstack([x7, grid3])
    return


@app.cell
def _(grid3, np):
    y2 = np.array([[99], [99]])
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
def _(mo):
    mo.md("""#Computation Numpy Arrays - Universal Functions""")
    return


@app.cell
def _(mo):
    mo.md("""## For Loops and Vectorization""")
    return


@app.cell
def _(np):
    np.random.seed(0)


    def compute_reciprocals(values):
        output = np.empty(len(values))
        for i in range(len(values)):
            output[i] = 1.0 / values[i]
        return output


    values = np.random.randint(1, 10, size=5)
    compute_reciprocals(values)
    return compute_reciprocals, values


@app.cell
def _(compute_reciprocals, np):
    big_array = np.random.randint(1, 100, size=1000000)

    import time

    start_time = time.time()
    compute_reciprocals(big_array)
    end_time = time.time()

    print(f"Elapsed time: {end_time - start_time} seconds")
    return big_array, time


@app.cell
def _(compute_reciprocals, values):
    print(compute_reciprocals(values))
    print(1.0 / values)
    return


@app.cell
def _(big_array, time):
    _start_time = time.time()
    1.0 / big_array
    _end_time = time.time()
    print(f"Elapsed time: {_end_time - _start_time} seconds")
    return


@app.cell
def _(mo):
    mo.md("""## Ufuncs""")
    return


@app.cell
def _(np):
    np.arange(5) / np.arange(1, 6)
    return


@app.cell
def _(np):
    x = np.arange(9).reshape((3, 3))
    2**x
    return (x,)


@app.cell
def _(mo):
    mo.md("""### Array Arithmetic""")
    return


@app.cell
def _(np):
    x12 = np.arange(4)
    print("x     =", x12)
    print("\nx + 5 =", x12 + 5)
    print("\nx - 5 =", x12 - 5)
    print("\nx * 2 =", x12 * 2)
    print("\nx / 2 =", x12 / 2)
    print("\nx // 2 =", x12 // 2)
    return (x12,)


@app.cell
def _(x12):
    print("-x     = ", -x12)
    print("\nx ** 2 = ", x12**2)
    print("\nx % 2  = ", x12 % 2)
    return


@app.cell
def _(x12):
    -((0.5 * x12 + 1) ** 2)
    return


@app.cell
def _(np, x12):
    np.add(x12, 2)
    return


@app.cell
def _(mo):
    mo.md("""### Absolute Value""")
    return


@app.cell
def _(np):
    x13 = np.array([-2, -1, 0, 1, 2])
    abs(x13)
    return (x13,)


@app.cell
def _(np, x13):
    np.absolute(x13)
    return


@app.cell
def _(np, x13):
    np.abs(x13)
    return


@app.cell
def _(np):
    x14 = np.array([3 - 4j, 4 - 3j, 2 + 0j, 0 + 1j])
    np.abs(x14)
    return


@app.cell
def _(mo):
    mo.md("""### Trignometric Functions""")
    return


@app.cell
def _(np):
    theta = np.linspace(0, np.pi, 3)

    print("theta      = ", theta)
    print("\nsin(theta) = ", np.sin(theta))
    print("\ncos(theta) = ", np.cos(theta))
    print("\ntan(theta) = ", np.tan(theta))
    return


@app.cell
def _(np):
    x15 = [-1, 0, 1]
    print("x         = ", x15)
    print("\narcsin(x) = ", np.arcsin(x15))
    print("\narccos(x) = ", np.arccos(x15))
    print("\narctan(x) = ", np.arctan(x15))
    return


@app.cell
def _(mo):
    mo.md("""## Exponents and Logarithms""")
    return


@app.cell
def _(np):
    x16 = [1, 2, 3]
    print("x     =", x16)
    print("\ne^x   =", np.exp(x16))
    print("\n2^x   =", np.exp2(x16))
    print("\n3^x   =", np.power(3, x16))
    return


@app.cell
def _(np):
    x17 = [1, 2, 4, 10]
    print("x        =", x17)
    print("\nln(x)    =", np.log(x17))
    print("\nlog2(x)  =", np.log2(x17))
    print("\nlog10(x) =", np.log10(x17))
    return


@app.cell
def _(np):
    x18 = [0, 0.001, 0.01, 0.1]
    print("exp(x) - 1 =", np.expm1(x18))
    print("\nlog(1 + x) =", np.log1p(x18))
    return


@app.cell
def _(mo):
    mo.md("""## Special Ufuncs""")
    return


@app.cell
def _():
    from scipy import special
    return (special,)


@app.cell
def _(special):
    # Gamma functions (generalized factorials) and related functions
    x19 = [1, 5, 10]
    print("gamma(x)     =", special.gamma(x19))
    print("\nln|gamma(x)| =", special.gammaln(x19))
    print("\nbeta(x, 2)   =", special.beta(x19, 2))
    return


@app.cell
def _(np, special):
    # Error function (integral of Gaussian)
    # its complement, and its inverse
    x20 = np.array([0, 0.3, 0.7, 1.0])
    print("\nerf(x)  =", special.erf(x20))
    print("\nerfc(x) =", special.erfc(x20))
    print("\nerfinv(x) =", special.erfinv(x20))
    return


@app.cell
def _(mo):
    mo.md("""## Advanced Ufuncs""")
    return


@app.cell
def _(np):
    x21 = np.arange(5)
    y3 = np.empty(5)
    np.multiply(x21, 10, out=y3)
    print(y3)
    return (x21,)


@app.cell
def _(np, x21):
    y4 = np.zeros(10)
    np.power(2, x21, out=y4[::2])
    print(y4)
    return


@app.cell
def _(mo):
    mo.md("""## Aggregates""")
    return


@app.cell
def _(np):
    x22 = np.arange(1, 6)
    np.add.reduce(x22)
    return (x22,)


@app.cell
def _(np, x22):
    np.multiply.reduce(x22)
    return


@app.cell
def _(np, x22):
    np.add.accumulate(x22)
    return


@app.cell
def _(np, x22):
    np.multiply.accumulate(x22)
    return


@app.cell
def _(mo):
    mo.md("""## Outer Product""")
    return


@app.cell
def _(np):
    x23 = np.arange(1, 6)
    np.multiply.outer(x23, x23)
    return


@app.cell
def _(mo):
    mo.md("""# Aggregations - Min, Max, Everything in Between""")
    return


@app.cell
def _(mo):
    mo.md("""## Summing Values in Array""")
    return


@app.cell
def _(np):
    L4 = np.random.random(100)
    sum(L4)
    return (L4,)


@app.cell
def _(L4, np):
    np.sum(L4)
    return


@app.cell
def _(mo):
    mo.md("""Min and Max""")
    return


@app.cell
def _(np):
    big_array2= np.random.rand(1000000)
    min(big_array2), max(big_array2)
    return (big_array2,)


@app.cell
def _(big_array2):
    print(big_array2.min(), big_array2.max(), big_array2.sum())
    return


@app.cell
def _(mo):
    mo.md("""## Multi-dimensional Aggregates""")
    return


@app.cell
def _(np):
    M = np.random.random((3, 4))
    print(M)
    return (M,)


@app.cell
def _(M):
    M.sum()
    return


@app.cell
def _(M):
    M.min(axis=0)
    return


@app.cell
def _(M):
    M.max(axis=1)
    return


@app.cell
def _(mo):
    mo.md("""## Aggregation Practice""")
    return


@app.cell
def _(np):
    import pandas as pd 
    president_data = pd.read_csv("data/president_heights.csv")
    heights = np.array(president_data['height(cm)'])
    print(heights)
    return heights, pd


@app.cell
def _(heights):
    print("Mean height:       ", heights.mean())
    print("Standard deviation:", heights.std())
    print("Minimum height:    ", heights.min())
    print("Maximum height:    ", heights.max())
    return


@app.cell
def _(heights, np):
    print("25th percentile:   ", np.percentile(heights, 25))
    print("Median:            ", np.median(heights))
    print("75th percentile:   ", np.percentile(heights, 75))
    return


@app.cell
def _(heights):
    import matplotlib.pyplot as plt 
    import seaborn 
    seaborn.set()
    plt.hist(heights)
    plt.title('Height Distribution of US Presidents')
    plt.xlabel('height (cm)')
    plt.ylabel('number')
    return (plt,)


@app.cell
def _(mo):
    mo.md("""# Computation on Arrays - Broadcasting""")
    return


@app.cell
def _(np):
    a1 = np.arange(0, 3)
    b1 = np.array([5, 5, 5])
    a1 + b1
    return (a1,)


@app.cell
def _(a1):
    a1 + 5
    return


@app.cell
def _(np):
    M2 = np.ones((3, 3))
    M2
    return (M2,)


@app.cell
def _(M2, a1):
    M2 + a1
    return


@app.cell
def _(np):
    a2 = np.arange(3)
    b2 = np.arange(3)[:, np.newaxis]
    print(a2)
    print(b2)
    return a2, b2


@app.cell
def _(a2, b2):
    a2 + b2
    return


@app.cell
def _(mo):
    mo.md("""## Rules of Broadcasting""")
    return


@app.cell
def _(mo):
    mo.md("""
    1. If the two arrays differ in their number of dimensions, the shape of the one with fewer dimensions is padded with ones on its leading (left) side.

    2. If the shape of the two arrays does not match in any dimension, the array with shape equal to 1 in that dimension is stretched to match the other shape.

    3. If in any dimension the sizes disagree and neither is equal to 1, an error is raised.
    """)
    return


@app.cell
def _(mo):
    mo.md("""### Broadcasting - Two-dimensional-One Dimensional Case""")
    return


@app.cell
def _(a1, np):
    M3 = np.ones((2, 3))
    a3 = np.arange(3)

    print(M3)
    print(a1)
    return (M3,)


@app.cell
def _(M3, a2):
    M3 + a2
    return


@app.cell
def _(mo):
    mo.md("""### Broadcasting - Different Dimensions""")
    return


@app.cell
def _(np):
    a4 = np.arange(3).reshape((3, 1))
    b4 = np.arange(3)

    print(a4)
    print(b4)
    return a4, b4


@app.cell
def _(a4, b4):
    a4 + b4
    return


@app.cell
def _(mo):
    mo.md("""### Broadcasting 3 - Non-Compatible Arrays""")
    return


@app.cell
def _(np):
    M4 = np.ones((3, 2))
    a5 = np.arange(3)
    return M4, a5


@app.cell
def _(M4, a5, np):
    # M4 + a5
    a5[:, np.newaxis].shape
    M4 + a5[:, np.newaxis]
    return


@app.cell
def _(mo):
    mo.md("""### Other Functions""")
    return


@app.cell
def _(M4, a5, np):
    np.logaddexp(M4, a5[:, np.newaxis])
    return


@app.cell
def _(mo):
    mo.md("""## Applying Broadcasting""")
    return


@app.cell
def _(mo):
    mo.md("""### Centering Array""")
    return


@app.cell
def _(np, x):
    X = np.random.random((10, 3))
    x
    return (X,)


@app.cell
def _(X):
    Xmean = X.mean(0)
    Xmean
    return (Xmean,)


@app.cell
def _(X, Xmean):
    X_centered = X - Xmean
    X_centered
    return


@app.cell
def _(mo):
    mo.md("""### Plotting 2-D Functions""")
    return


@app.cell
def _(np, plt):
    x24 = np.linspace(0, 5, 50)
    y5 = np.linspace(0, 5, 50)[:, np.newaxis]
    z2 = np.sin(x24) ** 10 + np.cos(10 + y5 * x24) * np.cos(x24)

    plt.imshow(z2, origin='lower', extent=[0, 5, 0, 5],
               cmap='viridis')
    plt.colorbar();
    plt.show()
    return


@app.cell
def _(mo):
    mo.md("""# Comparisons, Masks and Booleans""")
    return


@app.cell
def _(pd):
    rainfall = pd.read_csv('data/Seattle2014.csv')['PRCP'].values 
    inches = rainfall / 254.0 
    inches.shape
    return (inches,)


@app.cell
def _(inches, plt):
    plt.hist(inches, 40)
    return


@app.cell
def _(mo):
    mo.md("""## Comparison UFuncs""")
    return


@app.cell
def _(np):
    x25 = np.arange(1, 6)
    return (x25,)


@app.cell
def _(x25):
    x25 < 3
    return


@app.cell
def _(x25):
    x25 > 3
    return


@app.cell
def _(x25):
    x25 <= 3
    return


@app.cell
def _(x25):
    x25 >= 3
    return


@app.cell
def _(x25):
    x25 != 3
    return


@app.cell
def _(x25):
    x25 == 3
    return


@app.cell
def _(x25):
    (2 * x25) == (x25 ** 2)
    return


@app.cell
def _(np):
    rng = np.random.RandomState(0)
    x26 = rng.randint(10, size=(3, 4))
    x26
    return (x26,)


@app.cell
def _(x26):
    x26 < 6
    return


@app.cell
def _(mo):
    mo.md("""## Boolean Arrays""")
    return


@app.cell
def _(x26):
    print(x26)
    return


@app.cell
def _(mo):
    mo.md("""### Counting""")
    return


@app.cell
def _(np, x26):
    # how many values less than 6?
    np.count_nonzero(x26 < 6)
    return


@app.cell
def _(np, x26):
    np.sum(x26 < 6)
    return


@app.cell
def _(np, x26):
    # how many values less than 6 in each row?
    np.sum(x26 < 6, axis=1)
    return


@app.cell
def _(np, x26):
    # are there any values greater than 8?
    np.any(x26 > 8)
    return


@app.cell
def _(np, x26):
    # are there any values less than zero?
    np.any(x26 < 0)
    return


@app.cell
def _(np, x26):
    # are all values less than 10?
    np.all(x26 < 10)
    return


@app.cell
def _(np, x26):
    # are all values equal to 6?
    np.all(x26 == 6)
    return


@app.cell
def _(np, x26):
    # are all values in each row less than 8?
    np.all(x26 < 8, axis=1)
    return


@app.cell
def _(mo):
    mo.md("""### Boolean Operators""")
    return


@app.cell
def _(inches, np):
    np.sum((inches > 0.5) & (inches < 1))
    return


@app.cell
def _(inches, np):
    np.sum(~( (inches <= 0.5) | (inches >= 1) ))
    return


@app.cell
def _(inches, np):
    print("Number days without rain:      ", np.sum(inches == 0))
    print("Number days with rain:         ", np.sum(inches != 0))
    print("Days with more than 0.5 inches:", np.sum(inches > 0.5))
    print("Rainy days with < 0.2 inches  :", np.sum((inches > 0) &
                                                    (inches < 0.2)))
    return


@app.cell
def _(mo):
    mo.md("""## Boolean Arrays as Masks""")
    return


@app.cell
def _(x26):
    x26
    return


@app.cell
def _(x26):
    x26 < 5
    return


@app.cell
def _(x26):
    x26[x26 < 5]
    return


@app.cell
def _(inches, np):
    # construct a mask of all rainy days
    rainy = (inches > 0)

    # construct a mask of all summer days (June 21st is the 172nd day)
    days = np.arange(365)
    summer = (days > 172) & (days < 262)

    print("Median precip on rainy days in 2014 (inches):   ",
          np.median(inches[rainy]))
    print("Median precip on summer days in 2014 (inches):  ",
          np.median(inches[summer]))
    print("Maximum precip on summer days in 2014 (inches): ",
          np.max(inches[summer]))
    print("Median precip on non-summer rainy days (inches):",
          np.median(inches[rainy & ~summer]))
    return


@app.cell
def _(mo):
    mo.md("""## Aside - Keywords""")
    return


@app.cell
def _():
    bool(42), bool(0)
    return


@app.cell
def _():
    bool(42 and 0)
    return


@app.cell
def _():
    bool(42 or 0)
    return


@app.cell
def _():
    bin(42)
    return


@app.cell
def _():
    bin(59)
    return


@app.cell
def _():
    bin(42 & 59)
    return


@app.cell
def _():
    bin(42 | 59)
    return


@app.cell
def _(np):
    A = np.array([1, 0, 1, 0, 1, 0], dtype=bool)
    B = np.array([1, 1, 1, 0, 1, 1], dtype=bool)
    A | B
    return


@app.cell
def _(np):
    # A or B
    x27 = np.arange(10)
    (x27 > 4) & (x27 < 8) # always use this for numpy arrays 
    # (x > 4) and (x < 8)
    return


@app.cell
def _(mo):
    mo.md("""# Fancy Indexing""")
    return


@app.cell
def _(mo):
    mo.md("""# Structured Data - Numpy Structured Arrays""")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
