import pandas as pd
import numpy as np

data_1 = pd.DataFrame({'One': [1, 1, 1, 1, 2, 4, 5, 6, 6, 6, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 10]},
                      columns=['One'])
print(data_1.describe())
