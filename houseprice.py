import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.stats import skew
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


sns.set(style= 'ticks')

tran_data =pd.read_csv('....house-prices-advanced-regression-techniques/train.csv')
# figure = plt.figure()
sns.pairplot(x_vars=['OverallQual','GrLivArea','YearBuilt','TotalBsmtSF'],
             y_vars=['SalePrice'],data=tran_data,dropna=True)
data = pd.DataFrame(tran_data)
feature = data[["OverallQual","GrLivArea","YearBuilt","TotalBsmtSF"]]

# sns.pairplot(data=feature)
# plt.show()
# print(tran_data.describe())
# print(tran_data.head())
tran_data.info()

plt.show()
tran_data.drop(tran_data[(tran_data['OverallQual']<5) & (tran_data['SalePrice']>200000)].index,inplace=True)
tran_data.drop(tran_data[(tran_data['GrLivArea']>4000) & (tran_data['SalePrice']<200000)].index,inplace=True)
tran_data.drop(tran_data[(tran_data['YearBuilt']<1900) & (tran_data['SalePrice']>200000)].index,inplace=True)
tran_data.drop(tran_data[(tran_data['TotalBsmtSF']>6000) & (tran_data['SalePrice']<200000)].index,inplace=True)
tran_data.reset_index(drop=True,inplace=True)

test_data = pd.read_csv('/Users/luowenyan/Downloads/house-prices-advanced-regression-techniques/test.csv')
my_data = pd.concat([tran_data,test_data],axis=0)
my_data.reset_index(drop=True,inplace=True)
test_index = list(set(my_data.index).difference(set(tran_data.index)))

all_data=pd.concat([tran_data,test_data])
count=all_data.isnull().sum().sort_values(ascending=False)
ratio=count/len(all_data)
nulldata=pd.concat([count,ratio],axis=1,keys=['count','ratio'])
print(nulldata)

def fill_missings (res):
    res['Alley'] = res['Alley'].fillna('missing')
    res['PoolQC'] = res['PoolQC'].fillna(res['PoolQC'].mode()[0])
    res['MasVnrType'] = res['MasVnrType'].fillna('None')
    res['BsmtQual'] = res['BsmtQual'].fillna(res['BsmtQual'].mode()[0])
    res['BsmtCond'] = res['BsmtCond'].fillna(res['BsmtCond'].mode()[0])
    res['FireplaceQu'] = res['FireplaceQu'].fillna(res['FireplaceQu'].mode()[0])
    res['GarageType'] = res['GarageType'].fillna('missing')
    res['GarageFinish'] = res['GarageFinish'].fillna(res['GarageFinish'].mode()[0])
    res['GarageQual'] = res['GarageQual'].fillna(res['GarageQual'].mode()[0])
    res['GarageCond'] = res['GarageCond'].fillna('missing')
    res['Fence'] = res['Fence'].fillna('missing')
    res['Street'] = res['Street'].fillna('missing')
    res['LotShape'] = res['LotShape'].fillna('missing')
    res['LandContour'] = res['LandContour'].fillna('missing')
    res['BsmtExposure'] = res['BsmtExposure'].fillna(res['BsmtExposure'].mode()[0])
    res['BsmtFinType1'] = res['BsmtFinType1'].fillna('missing')
    res['BsmtFinType2'] = res['BsmtFinType2'].fillna('missing')
    res['CentralAir'] = res['CentralAir'].fillna('missing')
    res['Electrical'] = res['Electrical'].fillna(res['Electrical'].mode()[0])
    res['MiscFeature'] = res['MiscFeature'].fillna('missing')
    res['MSZoning'] = res['MSZoning'].fillna(res['MSZoning'].mode()[0])
    res['Utilities'] = res['Utilities'].fillna('missing')
    res['Exterior1st'] = res['Exterior1st'].fillna(res['Exterior1st'].mode()[0])
    res['Exterior2nd'] = res['Exterior2nd'].fillna(res['Exterior2nd'].mode()[0])
    res['KitchenQual'] = res['KitchenQual'].fillna(res['KitchenQual'].mode()[0])
    res["Functional"] = res["Functional"].fillna("Typ")
    res['SaleType'] = res['SaleType'].fillna(res['SaleType'].mode()[0])
    flist = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
             'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
             'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
             'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
             'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']
    for fl in flist:
        res[fl] = res[fl].fillna(0)
    res['TotalBsmtSF'] = res['TotalBsmtSF'].apply(lambda x: np.exp(6) if x <= 0.0 else x)
    res['2ndFlrSF'] = res['2ndFlrSF'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)
    res['GarageArea'] = res['GarageArea'].apply(lambda x: np.exp(6) if x <= 0.0 else x)
    res['GarageCars'] = res['GarageCars'].apply(lambda x: 0 if x <= 0.0 else x)
    res['LotFrontage'] = res['LotFrontage'].apply(lambda x: np.exp(4.2) if x <= 0.0 else x)
    res['MasVnrArea'] = res['MasVnrArea'].apply(lambda x: np.exp(4) if x <= 0.0 else x)
    res['BsmtFinSF1'] = res['BsmtFinSF1'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)
    return res
mydata = fill_missings(my_data)

all_data=pd.concat([tran_data,test_data])
count=all_data.isnull().sum().sort_values(ascending=False)
ratio=count/len(all_data)
nulldata=pd.concat([count,ratio],axis=1,keys=['count','ratio'])
print(nulldata)

mydata['MSSubClass']=mydata['MSSubClass'].apply(str)
mydata['YrSold']= mydata['YrSold'].astype(str)
mydata['MoSold'] = mydata['MoSold'].astype(str)
all_data['OverallCond'] = all_data['OverallCond'].astype(str)






