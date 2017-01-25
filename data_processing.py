import pickle
import numpy as np
import pandas  as pd
from datetime import date
import random
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def get_rawdata():

	pickle_in = open('txfuture_updated.pickle', 'rb')
	df = pickle.load(pickle_in)

	labels = []
	for i in range(len(df.index)):
		if df['settlement'][i] > df['open'][i]:
			labels.append([1,0])
		else:
			labels.append([0,1])

	df['label'] = pd.Series(labels, index= df.index)
	df['close_adjusted'] = df['close']
	df['close_adjusted'].fillna(df['settlement'], inplace=True)

	return (df)

def sample_suffling(sample_x, sample_y, length = 20):

	featureset = []
	sample_length = len(sample_x)-length

	for i in range(sample_length):
		sample_x_chunk = np.array(sample_x[i:i+length])
		min_max_scaler = preprocessing.MinMaxScaler()
		sample_x_chunk = min_max_scaler.fit_transform(sample_x_chunk)
		featureset.append([sample_x_chunk,sample_y[i+length]])

	random.shuffle(featureset)

	return featureset

def sample_visualizing(sample_x):
	picture_array = []
	
	for i in range(len(sample_x)):
		picture = []
		sample_x[i] = np.transpose(sample_x[i])
		for j in range(len(sample_x[i])):
			pictureline = []
			for k in range(len(sample_x[i][j])):
				append_pt = [sample_x[i][j][k],sample_x[i][j][k],sample_x[i][j][k]]
				pictureline.append(append_pt)
			picture.append(pictureline)
		plt.imshow(picture)
		#plt.show()
		picture_array.append(picture)
	return (picture_array)


def data_processing(test_size=0.05, start_year=2000):

	df = get_rawdata()
	df = df.loc[date(start_year,1,1):date(2015,12,31)]
	df['close_mvag5'] = df['close_adjusted'].rolling(window=5).mean()
	df['close_mvag20'] = df['close_adjusted'].rolling(window=20).mean()
	df['close_mvag60'] = df['close_adjusted'].rolling(window=60).mean()
	df.dropna(inplace=True)
	df_X = df[['open','high','low','close_adjusted','close_mvag5','close_mvag20','close_mvag60','volume']]
	df_y = df['label']

	datasets = sample_suffling(df_X,df_y)
	datasets = np.array(datasets)
	testing_size = int(test_size * len(datasets))

	train_x = list(datasets[:,0][:-testing_size])
	train_x = sample_visualizing(train_x)
	train_y = list(datasets[:,1][:-testing_size])

	test_x = list(datasets[:,0][-testing_size:])
	test_x = sample_visualizing(test_x)
	test_y = list(datasets[:,1][-testing_size:])

	return train_x, train_y, test_x, test_y

#data_processing()