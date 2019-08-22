# Author:
Zhizhong John DIng
# Preface:
This is a cheatsheet and tutorial for the Numpy library.
# Table of contents
1. [Advantage of Numpy over python list](#1. advantage of Numpy over python list)
2. [Basics](# 2. Basics)
3. [Array creation](# 3. Array creation)
4.

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
| ndarray.ndim         | Return number of axis                    | a.ndim        | 2                                                            |
| ndarray.shape        | Return dimension (m,n)                   | a.shape       | (4,3)                                                        |
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

# 4. Basic operations




<span style="color:red">needs further contribution</span>
