import numpy as np
import pandas as pd

print("This is a Test")

x = [1, 2, 3, 4]
y = np.square(x)

print(list(y))


df = pd.DataFrame(data = {
    "col1" : [1, 2], 
    "col2": [3, 4]
})

print(df)
print(df.to_string())


