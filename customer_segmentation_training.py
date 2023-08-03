#%%
#1. Import Libraries and Packages
import os
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, OneHotEncoder

#%%
PATH = os.getcwd()
TRAIN_DATASET = os.path.join(PATH, 'data', 'customer_segmentation.csv')
LOG_PATH = os.path.join(PATH,'log')

#%%
#Load the dataset
df = pd.read_csv(TRAIN_DATASET)
# %%
#3. Data inspection
df.info()
#%%
#df['ID'] = df['ID'].astype('string')
# %%
df.isna().sum()
# %%
df.describe().T
# %%
df.nunique()
# %%
#Check distribution of 'Work_Experience' column
df["Work_Experience"].hist(figsize=(5,5))
plt.show()
#Check distribution of 'Family_Size' column
df["Family_Size"].hist(figsize=(5,5))
plt.show()
# %%
#Check fot outliers
numeric_columns = ['Age', 'Work_Experience', 'Family_Size']
df[numeric_columns].boxplot(figsize=(5,5))
# %%
#Data Cleaning
# Fill in the missing values with Forward Fill
df['Ever_Married'].fillna(method='ffill',inplace=True)
df['Graduated'].fillna(method='ffill', inplace=True)
df['Profession'].fillna(method='ffill', inplace=True)
df['Var_1'].fillna(method='ffill',inplace=True)

#%%
# Fill in the missing values with Median
df['Family_Size'] = df['Family_Size'].replace(np.NaN, df['Family_Size'].median())
df['Work_Experience'] = df['Work_Experience'].replace(np.NaN, df['Work_Experience'].median())

#%%
df.info()
# %%
#check for duplicated data
df.duplicated().sum()
# %%
df.isna().sum()
# %%
#Convert labels into numerical form (label encoding)
for col in df.columns:
    if df[col].dtype == 'object':
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])

# %%
df.head()
# %%
X = df.drop(labels=['Segmentation'], axis=1)
y = df['Segmentation']
#%%
#y = keras.utils.to_categorical(y)

#%%
#Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=42)
#%%
#Data Preprocessing
y_train = to_categorical(y_train, num_classes=4)
y_test = to_categorical(y_test, num_classes=4)
#%%
mms = MinMaxScaler()
X_train_scaled = mms.fit_transform(X_train)
X_test_scaled = mms.transform(X_test)

#%%
#ss = StandardScaler()
#X_train_scaled = ss.fit_transform(X_train)
#X_test_scaled = ss.transform(X_test)
# %%
#Model Development
model = keras.Sequential()
model.add(keras.layers.Dense(128, activation='relu', input_shape=[X_train.shape[1],]))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(4, activation='softmax'))

#%%
opt =  keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics='accuracy')
model.summary()
#%%
keras.utils.plot_model(model)

# %%
# Tensorboard call backs
log_dir = os.path.join(LOG_PATH, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tb = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
n_epoch= 100
batch_size = 250
early_stopping = keras.callbacks.EarlyStopping(patience=3)
#%%
#Model Training
hist = model.fit(X_train_scaled, y_train,validation_data=(X_test_scaled,y_test), 
                 epochs=n_epoch, batch_size=batch_size,
                 callbacks=[early_stopping,tb])
# %%
model.evaluate(X_test_scaled, y_test)
# %%
model_save_path = os.path.join(PATH,"model.h5")
keras.models.save_model(model,model_save_path)
# %%
