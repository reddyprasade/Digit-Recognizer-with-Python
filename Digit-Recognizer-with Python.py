# geport numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_digits

# Data Set Loading Form Scikit-Learn
Data = load_digits()
print("Description of the Data Optical recognition of handwritten digits dataset",Data.DESCR)
print("Orginal Data",Data.data)
print("Data Target_Names",Data.target_names)
print("Data Target",Data.target)
print("Images Data Loading",Data.images)
print("Finding the Shape of Data",Data.data.shape)

### Data Set Loading to Pandas
##Digits_DataSets = pd.DataFrame(Data,columns=Data.target_names)
##print(Digits_DataSets.head())



# Data Visulization
import matplotlib.pyplot as plt
#plt.gray() 
plt.imshow(Data.images[1]) 
plt.show()
