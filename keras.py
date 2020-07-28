import pymysql.cursors
import pandas as pd
import numpy as np
from tensorflow import keras

args = {'host':'localhost', 'user':'root', 'password':'aybs1196', 'db':'py_vis'}
conn = pymysql.connect(**args)

try:
	with conn.cursor() as cursor:
		query = "SELECT * FROM py_vis.adult"
		cursor.execute(query)
		result = cursor.fetchall()
finally:
	conn.close()

# print(cursor.description[0][0])
data = pd.DataFrame(result)
data.columns = [cursor.description[i][0] for i in range(len(cursor.description))]

# for i in data.index:
# 	if ' ?' in data.iloc[i,:]:
# 		print(i)

catg = ['workclass', 'education', 'married', 'occupation', 'relationship', 'race', 'sex', 'nativecountry', 'income']
for i in catg:
	data[i] = data[i].replace(data[i].unique(), range(len(data[i].unique())))

data_label = np.array(data.iloc[:,-1])
data_input = np.array(data.iloc[:,0:(len(data.columns)-1)])

#Keras parameter options
act_fn = ['elu', 'exponential', 'hard_sigmoid', 'linear', 'relu', 'selu', 'sigmoid', 'softmax']

# Keras Model
inputs = keras.Input(shape = (len(data.columns)-1) )					#Defining input format

layer1 = keras.layers.Dense(units = 4, activation = 'hard_sigmoid')(inputs)	#Defining Intermediate Layers
# layer2 = keras.layers.Dense(units = 4, activation = 'sigmoid')(layer1)
# layer3 = keras.layers.Dense(units = 4, activation = 'sigmoid')(layer2)

outputs = keras.layers.Dense(units = 2, activation = 'softmax')(layer1)	#Defining Output layer

#Initializing Model
model = keras.Model(inputs = inputs, outputs = outputs)
# print(model.summary())

model_result = model(data_input)
pred_class = list([])

for i in range(len(data_label)):
	if model_result[i,0] >= model_result[i,1]:
		pred_class.append(0)
	else:
		pred_class.append(1)

acc = sum(pred_class == data_label)/len(data_label)*100
acc = round(acc,4)
print(pd.DataFrame(list([data_label, pred_class])))
print('acc = {}'.format(acc))
print(pd.Series(pred_class).value_counts())