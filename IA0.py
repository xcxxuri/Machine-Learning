import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# plt.switch_backend('agg')

# loading csv file
location = 'pa0(train-only).csv'
data = pd.read_csv(location)

# a. remove id from data
data = data.drop(columns=['id'])

# b. split the date feature into month, day, year
monthDayYear = data['date'].str.split('/',expand=True) # Convert series type to DataFrame type
monthDayYear.columns = ['month', 'day', 'year']
data = pd.concat([data, monthDayYear],axis = 1) # connect monthDayYear's column into data
data = data.drop(columns=['date']) # delete date column

# c. generate boxplots
bedrooms = pd.Series(data['bedrooms']) # boxplot for bedrooms:
bedrooms = pd.unique(bedrooms)
data.boxplot(column=['price'],by='bedrooms')
plt.style.use('ggplot')
plt.savefig('bedrooms')
plt.show()

bathrooms = pd.Series(data['bathrooms']) # boxplot for bathrooms:
bathrooms = pd.unique(bathrooms)
data.boxplot(column=['price'],by='bathrooms')
plt.style.use('ggplot')
plt.savefig('bathrooms')
plt.show()

floors = pd.Series(data['floors']) # boxplot for floors:
floors = pd.unique(floors)
data.boxplot(column=['price'],by='floors')
plt.style.use('ggplot')
plt.savefig('floors')
plt.show()

# d.  co-variance matrix
sqft_living = pd.Series(data['sqft_living'])
sqft_living15 = pd.Series(data['sqft_living15'])
sqft_lot = pd.Series(data['sqft_lot'])
sqft_lot15 = pd.Series(data['sqft_lot15'])

# generate the co-variance matrix of these four features.
covariance_matrix = np.cov(np.array([sqft_living,sqft_lot,sqft_living15,sqft_lot15]))
print('The co-variance matrix of these four features are: ',covariance_matrix)

# scatter plot using sqrt_living with sqrt_living15
plt.scatter(sqft_living,sqft_living15)
plt.savefig('sqft_living & sqft_living15')
plt.show()

# scatter plot using sqrt_lot with sqrt_lot15
plt.scatter(sqft_lot,sqft_lot15)
plt.savefig('sqft_lot & sqft_lot15')
plt.show()













