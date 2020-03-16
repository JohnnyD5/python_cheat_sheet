# Author:
Zhizhong John DIng
# Preface:
This is a cheatsheet and tutorial for the Numpy library.
# Table of contents
- [1. Advantage of Numpy over python list](#1-advantage-of-numpy-over-python-list)
- [2. Basics](#2-basics)
  * [Data type conversion](#data-type-conversion)
- [3. Array creation](#3-array-creation)
  * [3.1. From known value](#31-from-known-value)
    + [Create from Python list](#create-from-python-list)
    + [Create using function](#create-using-function)
  * [3.2 Initializing with placeholder](#32-initializing-with-placeholder)
  * [3.3 Import](#33-import)
    + [3.3.1 Import from csv](#331-import-from-csv)
      - [Method 1. use csv reader](#method-1-use-csv-reader)
      - [Method 2. load csv with Numpy function](#method-2-load-csv-with-numpy-function)
      - [Method 3. load csv with Pandas](#method-3-load-csv-with-pandas)
- [4. Basic operations](#4-basic-operations)

# 1. Advantage of Numpy over python list
NumPy is more compact, convenient than Python Lists. The vector and matrix operations in NumPy is well and efficiently implemented. NumPy is much faster and it also have more functionality such as FFTs, convolutions, fast searching, basic statistics, linear algebra, histograms, etc.
# 2. Basics
* In Numpy, dimension are called axes
* Numpy's class type is called ndarray
```Python
import numpy as py
a = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
```
| Operator             |                Discription               | Example       | Return                                                       |
|----------------------|:----------------------------------------:|---------------|--------------------------------------------------------------|
| ndarray.ndim         | Return number of axis  (dimension)                  | a.ndim        | 2                                                            |
| ndarray.shape        | Return shape (m,n)                   | a.shape       | (4,3)                                                        |
| ndarray.size         | Return total number of elements          | a.size        | 12                                                           |
| ndarray.dtype        | Return object describing the type        | a.dtype       | int 32                                                       |
| ndarray.itemsize     | Return the size in terms of each element | a.itemsize    | 4                                                            |
| ndarray.data         | Return the buffer (memory address)       | a.data        | <memory at 0x000000016D29B40>                                |
| ndarray.astype(type) | Convert to another data type             | a.astype(str) | [['1' '2' '3'],['4' '5' '6'],['7' '8' '9'],['10' '11' '12']] |
| len(ndarray)         | Return number of rows                    | len(a)        | 4                                                            |

## Data type conversion
`ndarray.astype(type)` is a great tool to convert data to int, float or str.
# 3. Array creation
## 3.1. From known value
Create from python list, python tuple or python list of tuples, or python list of list.
Results are same no matter from list of list or from tuple of list. Recommend using List to avoid confusion
### Create from Python list
| Operator                          | ndarray.shape | Return              |
|-----------------------------------|---------------|---------------------|
| a = np.array([1,2,3,4])           | 4             | [1 2 3 4]           |
| a = np.array([[1,2],[3,4],[5,6]]) | (3,2)         | [[1 2],[3 4],[5 6]] |

**Note:** You can also use tuple to create numpy Array
```Python
a = np.array((1,2,3,4))
a = np.array([(1,2),(3,4),(5,6)])
a = np.array(((1,2),(3,4),(5,6)))
```
But personally I don't recommend using tuple. Because using list is consistent with python list for either array and matrix, easy to remember.
### Create using function
| Operator               | Description          | Return             |
|------------------------|----------------------|--------------------|
| a = np.arange(10,30,5) | same as python range | [10 15 20 25]      |
| a = np.linspace(0,2,4) | start, stop, number  | [0. 0.667 1.333 2] |
| a = np.fromfunction(f,(5,4)) | Construct an array by executing over each coord  |  |

```Python
def f(x,y):
    return 10*x+y    
a = np.fromfunction(f,(5,4))
print(a)
```
> [[ 0.  1.  2.  3.]  
 [10. 11. 12. 13.]  
 [20. 21. 22. 23.]  
 [30. 31. 32. 33.]  
 [40. 41. 42. 43.]]  

## 3.2 Initializing with placeholder
| Operator                    | Description                | Return                             |
|-----------------------------|----------------------------|------------------------------------|
| a = np.empty((3,2))         | Create empty matrix              |     
| a = np.empty(3)         | Create empty matrix              |     
| a = np.ones(4)              | Create list (vector)       | [1. 1. 1. 1.]                      |
| a = np.ones((3,2))          | Create matrix              | [[1. 1.] [1. 1.] [1. 1.]]          |
| a = np.zeros((3,2))         | Create matrix              | [[0. 0.] [0. 0.] [0. 0.]]          |
| a = np.eye(3)               | Create Identity matrix (n) | [1., 0., 0.],  [0., 1., 0.], [0., 0., 1.]                                   |
| a = np.random.rand(3,2)     | Create random matrix       | uniform distribution between [0,1) |
| a = np.random.random((3,2)) | same as last one           | uniform distribution between [0,1) |
| a = np.random.randn(3,2)    | Create random matrix       | normal distribution, mean 0        |

## 3.3 Import
### 3.3.1 Import from csv  
#### Method 1. use csv reader
Original data: test_import.csv
> Latitude,Longitude,Elevation  
48.89016000,2.689270000,71.0  
48.89000000,2.689730000,72.0   
48.88987000,2.689810000,72.0  
48.88924000,2.689570000,67.0  
48.88934000,2.690050000,67.0  
48.88949000,2.691400000,65.0

```Python
import csv
import numpy as np
data_path = 'test_import.csv'
with open(data_path, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    # get header from first row
    headers = next(reader)
    # get all the rows as a list
    data = list(reader)
    # transform data into numpy array
    data = np.array(data).astype(float)
```
output looks like:
```Python
print(headers)
print(data.shape)
print(data[:3])
```
> ['Latitude', 'Longitude', 'Elevation']  
(6, 3)  
[[48.89016  2.68927 71.     ]  
 [48.89     2.68973 72.     ]  
 [48.88987  2.68981 72.     ]]

**Comment:**
1. This method could be slow, if the data file is huge.  
2. next() returns the next item from the iterator

#### Method 2. load csv with Numpy function
Two functions:
1. `numpy.loadtxt`
2. `numpy.genfromtxt`

`numpy.genfromtxt` is recommended over the other because `np.genfromtxt` can read CSV files with missing data and gives you options like the parameters `missing_values` and `filling_values` that help with missing values in the CSV.

Original data: test_import.csv
> X,Y,Name,small,large,circular,mini_hoop,total_rack  
982903.56993819773,205129.99858243763,1 7 AV S,5,0,0,0,5  
987330.41607135534,191302.73030526936,1 BOERUM PL,1,0,0,0,1  
983210.95318169892,199016.51343409717,1 CENTRE ST,10,0,0,0,10  
985897.83954019845,207157.88527469337,1 E 13 ST,1,0,0,0,1  
1010993.9694659412,252137.33960694075,1 E 183 ST,0,0,2,0,2  
987774.37089210749,210586.44665901363,1 E 28 ST,1,0,0,0,1  

```Python
import numpy as np
data_path = "test_import2.csv"
types = ['f8', 'f8', 'U50', 'i4', 'i4', 'i4', 'i4', 'i4']
data = np.genfromtxt(data_path, dtype=types, delimiter=',',names=True)
a = data['X']
```
output looks like:
```Python
print(data)
print(data['X'])
```
> [( 982903.5699382 , 205129.99858244, '1 7 AV S',  5, 0, 0, 0,  5)  
 ( 987330.41607136, 191302.73030527, '1 BOERUM PL',  1, 0, 0, 0,  1)  
 ( 983210.9531817 , 199016.5134341 , '1 CENTRE ST', 10, 0, 0, 0, 10)  
 ( 985897.8395402 , 207157.88527469, '1 E 13 ST',  1, 0, 0, 0,  1)  
 (1010993.96946594, 252137.33960694, '1 E 183 ST',  0, 0, 2, 0,  2)  
 ( 987774.37089211, 210586.44665901, '1 E 28 ST',  1, 0, 0, 0,  1)]

> [ 982903.5699382   987330.41607136  983210.9531817   985897.8395402
 1010993.96946594  987774.37089211]

 **Comment:**
 1. This method returns a tuple list
 2. `names = True` to access the header and use it as column name to return a specific column

#### Method 3. load csv with Pandas
**Note**: if the data is generated in excel file, should export the data as csv file. 
1. get array from data
```Python
import pandas as pd
path = "test_import2.csv"
df=pd.read_csv(path, delimiter = ',', header = 0)
a = df['X'].values
```
output looks like:
```
print(a)
print(type(a))
```
> [ 982903.5699382   987330.41607136  983210.9531817   985897.8395402
 1010993.96946594  987774.37089211]  
<class 'numpy.ndarray'>

2. get matrix from data
```Python
import pandas as pd
path = "test_import2.csv"
df=pd.read_csv(path, delimiter = ',', header = 0)
a = df.iloc[:,:2].values
```
output looks like:
```
print(a)
print(type(a))
```
> [[ 982903.5699382   205129.99858244]  
 [ 987330.41607136  191302.73030527]  
 [ 983210.9531817   199016.5134341 ]  
 [ 985897.8395402   207157.88527469]  
 [1010993.96946594  252137.33960694]  
 [ 987774.37089211  210586.44665901]]  
<class 'numpy.ndarray'>

# 4. Math
## 4.1 Basic math
### 4.1.1 Vector
**Rule:** all operations for array (vector) are element wise.
```Python
a = np.array([20,30,40,50])
b = np.arange(4) # b = np.array([0,1,2,3])
```
| Operator                   | Description             | Return                       |
|----------------------------|-------------------------|------------------------------|
| a+b /  np.add(a,b)          | Addition                | [20 31 42 53]                |
| a-b /  np.substract(a,b)    | Subtraction             | [20 29 38 47]                |
| a/b /  np.divide(a,b)       | Division                | [inf 30. 20. 16.67]          |
| a*b  /  np.multiply(a,b)    | Multiplication          | [ 0 30 80 150]               |
| b**2                       | Square                  | [0 1 4 9]                    |
| np.sqrt(b)                 | Square Root             | [0. 1. 1.4141.732]           |
| np.sin(a)                  | Treat element as Radian | [ 0.912 -0.988 0.745 -0.262] |
| np.cos(a)                  | Treat element as Radian |                              |
| np.log(a)                  | Base 2                  |                              |
| np.dot(a,b)  /  np.sum(a*b) | Dot product             | 260                          |
| a<35                       | comparison              | [True True False False]      |

### 4.1.2 Matrix
**Rule:** except matrix product, all operations are element wise.
```Python
a = np.array([[1,1],[0,1]])
[[1 1]
[0 1]]
b = np.array([[2,0],[3,4]])
[[2 0]
[3 4]]
```
| Operator                 | Description             | Return                              |
|--------------------------|-------------------------|-------------------------------------|
| a+b /  np.add(a,b)       | Addition                | [[3 1] [3 5]]                       |
| b-a /  np.substract(b,a) | Subtraction             | [[ 1 -1] [ 3 3]]                    |
| b/a /  np.divide(b,a)    | Division                | [[ 2. 0.] [inf 4.]]                 |
| a*b /  np.multiply(a,b)  | Multiplication          | [[2 0] [0 4]]                       |
| b**2                     | Square                  | [[ 4 0] [ 9 16]]                    |
| np.sqrt(b)               | Square Root             | [[1.41421356 0. ] [1.73205081 2. ]] |
| np.sin(a)                | Treat element as Radian |                                     |
| np.cos(a)                | Treat element as Radian |                                     |
| np.log(a)                | Base 2                  |                                     |
|  a@b /a.dot(b)        | Matrix dot product      | [[5 4] [3 4]]                       |
| a<35                     | comparison              | [[ True True] [ True True]]         |

<span style="color:red">cumsum min exp etc needs update</span>
## 4.2 Linear algebra
<span style="color:red">under construction</span>

# 5. Shape manipulation
## 5.1 Change shape
| Operator        | Description                    |
|-----------------|--------------------------------|
| a.reshape(2,6)    | Won't change array itself      |
| a.resize(2,6) | Will change array itself       |
| b = np.ravel(a) | Return 1D flattened array, itself doesn't change      |
| a.flatten()       | Return 1D flattened array, itself doesn't change |

**Comment:**
1. A key difference between `flatten()` and `ravel()` is that `flatten()` is a method of an `ndarray` object and hence can only be called for true numpy arrays. In contrast `ravel()` is a library-level function and hence can be called on any object that can successfully be parsed.
2. Turn 1D array into 2D array:
```Python
import numpy as np
A = np.arange(8)
A = A.reshape(1,8)
#or
A = A.reshape(8,1)
```

```Python
def f(x,y):
    return 10*x+y
a = np.fromfunction(f,(3,4))
```
```Python
print(a)
```
> [[ 0.  1.  2.  3.]  
 [10. 11. 12. 13.]  
 [20. 21. 22. 23.]]  

```Python
b = a.reshape(2,6)
print(b)
print(a)
```
> [[ 0.  1.  2.  3. 10. 11.]  
 [12. 13. 20. 21. 22. 23.]]  

> [[ 0.  1.  2.  3.]  
 [10. 11. 12. 13.]  
 [20. 21. 22. 23.]]

```Python
 b = a.resize(2,6)
 print(b)
 print(a)
 ```
 > None

 **Comment:** resize works on the array itself, it returns None

 > [[ 0.  1.  2.  3. 10. 11.]  
  [12. 13. 20. 21. 22. 23.]]  

```Python
b = np.ravel(a)
 print(b)
 print(a)
 ```
 > [ 0.  1.  2.  3. 10. 11. 12. 13. 20. 21. 22. 23.]

 > [[ 0.  1.  2.  3.]  
 [10. 11. 12. 13.]  
 [20. 21. 22. 23.]]

```Python
b = a.flatten()
print(b)
print(a)
```
> [ 0.  1.  2.  3. 10. 11. 12. 13. 20. 21. 22. 23.]

> [[ 0.  1.  2.  3.]  
[10. 11. 12. 13.]  
[20. 21. 22. 23.]]

## 5.2 Stack and append
| Operator               | Description                                  | Graph   |
|------------------------|----------------------------------------------|---------|
| np.hstack((A,B))       | Stack horizontally                           | AB      |
| np.vstack((A,B))       | Stack vertically                             | A <br>  B   |
| np.column_stack((A,B)) | Stack 1-D arrays as columns into a 2-D array | A.T B.T |
| np.append(A,B) | form 1D array of AB | AB |
| np.append(A,B, axis=0) | stack vertically | A <br>  B  |
| np.append(A,B, axis=1) | stack horizontally | AB |

**Comment:**
1. I don't recommend using concatenate. It's basically same as operators mentioned above by changing axis. And it could be confusing.
2. For `np.append`, when axis is specified, values must have the correct dimension(`2D and 2D` or `1D and 1D`)

```Python
import numpy as np
A = np.arange(2,6)
B = np.arange(1,5)*2
print(A,'\n',B)
```
> [2 3 4 5]   
 [2 4 6 8]

```Python
print(np.hstack((A,B)))
```
> [2 3 4 5 2 4 6 8]

```Python
print(np.column_stack((A,B)))
```
> [[2 2]  
 [3 4]  
 [4 6]  
 [5 8]]   

```Python
print(np.vstack((A,B)))
```
 > [[2 3 4 5]  
 [2 4 6 8]]

```Python
np.append([[1, 2, 3], [4, 5, 6]], [[7, 8, 9]], axis=0)
```
> [[1, 2, 3]  
       [4, 5, 6]  
       [7, 8, 9]]

```Python
np.append([1, 2, 3], [[4, 5, 6], [7, 8, 9]])
```
> [1, 2, 3, ..., 7, 8, 9]

# 6. Indexing and slicing
## 6.1 One-D
Same as python list.  
```Python
import numpy as np
a = np.arange(10)**2
print(a)
```
> [ 0  1  4  9 16 25 36 49 64 81]

```Python
print(a[2])
```
> 4

```Python
print(a[2:5])
```
> [ 4  9 16]

```Python
print(a[::-1])
```
> [81 64 49 36 25 16  9  4  1  0]
## 6.2 multiple-D
Same as python 2D list
```Python
def f(x,y):
    return 10*x+y
a = np.fromfunction(f,(5,4))
```
```Python
print(a[2,3])
```
> 23

```Python
print(a[1:4,2])
```
> [12. 22. 32.]

```Python
print(a[1:3,:])
```
> [[10. 11. 12. 13.]  
 [20. 21. 22. 23.]]

```Python
print(a[-1])
```
> [40. 41. 42. 43.]
## 6.3 Advanced indexing
### 6.3.1 ndarray.flat[]
`a.flat` is a 1D iterator over the array
```Python
def f(x,y):
    return 10*x+y
a = np.fromfunction(f,(3,4))
print(a)
print(a.flat[6])
```
> [[ 0.  1.  2.  3.]  
[10. 11. 12. 13.]  
[20. 21. 22. 23.]]   
12.0
