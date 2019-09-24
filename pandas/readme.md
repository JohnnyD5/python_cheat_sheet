# A question stack overflow deleted but extremely useful in fluid dynamics
## plot probe data
I am trying to turn a data file into a pandas dataframe. The first few lines of the file look like this:
```
# Forces     
# CofR       : (0.000000e+00 0.000000e+00 0.000000e+00)  
# Time       forces(pressure viscous porous) moment(pressure viscous porous)  
1.000000e-03    ((0.000000e+00 -1.602836e-08 0.000000e+00) (-1.922779e+00 1.540206e-10 0.000000e+00) (0.000000e+00 0.000000e+00 0.000000e+00)) ((0.000000e+00 0.000000e+00 1.185805e-06) (0.000000e+00 0.000000e+00 1.922779e+00) (0.000000e+00 0.000000e+00 0.000000e+00))  
2.000000e-03    ((0.000000e+00 -3.904013e-09 0.000000e+00) (-1.893677e+00 2.888988e-10 0.000000e+00) (0.000000e+00 0.000000e+00 0.000000e+00)) ((0.000000e+00 0.000000e+00 3.537284e-06) (0.000000e+00 0.000000e+00 1.893677e+00) (0.000000e+00 0.000000e+00 0.000000e+00))  
3.000000e-03    ((0.000000e+00 9.851879e-08 0.000000e+00) (-1.453065e+00 4.435274e-11 0.000000e+00) (0.000000e+00 0.000000e+00 0.000000e+00)) ((0.000000e+00 0.000000e+00 3.650183e-06) (0.000000e+00 0.000000e+00 1.453065e+00) (0.000000e+00 0.000000e+00 0.000000e+00))  
```

The data follows a strict format:
```
'%f ((%f %f %f) (%f %f %f) (%f %f %f)) ((%f %f %f) (%f %f %f) (%f %f %f))'
```

I only want to keep the numbers, here below is the code I used to turn csv into pandas dataframe, basically I used multiple sep to parse out '(' and ')'

```python
df = pd.read_csv('/test.txt',skiprows = 3, 
   sep='\)\s+\(|\s+\(\(|\s|\)\)\s\(\(',engine ='python', 
   names = ["t", "pfx", "pfy", "pfz",
   "vfx", "vfy", "vfz", "pofx", "pofy", "pofz", "pmx", "pmy", 
   "pmz","vmx", "vmy", "vmz", "pmfx", "pmfy", "pmfz"])   
```

And here is the result showed in Spider:

```
       t  pfx           pfy  pfz       vfx           vfy  vfz  pofx  pofy  \
0  0.001  0.0 -1.602836e-08  0.0 -1.922779  1.540206e-10  0.0   0.0   0.0   
1  0.002  0.0 -3.904013e-09  0.0 -1.893677  2.888988e-10  0.0   0.0   0.0   
2  0.003  0.0  9.851879e-08  0.0 -1.453065  4.435274e-11  0.0   0.0   0.0   

   pofz  pmx  pmy       pmz  vmx  vmy       vmz  pmfx  pmfy            pmfz  
0   0.0  0.0  0.0  0.000001  0.0  0.0  1.922779   0.0   0.0  0.000000e+00))  
1   0.0  0.0  0.0  0.000004  0.0  0.0  1.893677   0.0   0.0  0.000000e+00))  
2   0.0  0.0  0.0  0.000004  0.0  0.0  1.453065   0.0   0.0  0.000000e+00))
```

But the problem is: see the last value, it still has '))' followed with. How should I modify my code to get a clean pure numbers result?

## Answer:
df['pmfz'] = df['pmfz'].str.replace('\)\)', '').astype(float)
