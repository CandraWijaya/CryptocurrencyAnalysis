import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

# date, open, hi, low, close, adj, vol
def read_data(str):
	opens = []
	highs = []
	lows = []
	closes = []
	adjusted = []
	volume = []
	with open(str, 'r') as data:
		data.readline()
		for line in data:
			line = line.strip()
			line = line.split(',')
			opens.append(float(line[1]))
			highs.append(float(line[2]))
			lows.append(float(line[3]))
			closes.append(float(line[4]))
			adjusted.append(float(line[5]))
			volume.append(int(line[6]))
	opens = np.array(opens)
	highs = np.array(highs)
	lows = np.array(lows)
	closes = np.array(closes)
	adjusted = np.array(adjusted)
	volume = np.array(volume)
	data = np.zeros((len(opens), 6))
	for i in range(0, len(opens)):
		data[i] = np.array([opens[i], highs[i], lows[i], closes[i], adjusted[i], volume[i]])
	data = data[:len(data)-len(data)%7]
	data = np.reshape(data, (int(len(data)/7), 7, 6))
	return data

eth_data = read_data('ETH-USD.csv') # etherium
train_data = eth_data[:-1,:,:]
vol_labels = np.divide(eth_data[1:, 6, 5], eth_data[:-1, 6, 5]) # change since Xyesterday'sX - last week's volume (divide) (mean_squared_error)
adj_labels = np.divide(eth_data[1:, 6, 4], eth_data[:-1, 6, 4]) # difference since last week's adj (subtract) (poisson)
close_labels = np.divide(eth_data[1:, 6, 3], eth_data[:-1, 6, 3])
low_labels = np.divide(eth_data[1:, 6, 2], eth_data[:-1, 6, 2])
hi_labels = np.divide(eth_data[1:, 6, 1], eth_data[:-1, 6, 1])
op_labels = np.divide(eth_data[1:, 6, 0], eth_data[:-1, 6, 0])

xrp_data = read_data('XRP-USD.csv') # ripple
train_data = np.vstack((train_data, xrp_data[:-1,:,:]))
vol_labels = np.hstack((vol_labels, np.divide(xrp_data[1:, 6, 5], xrp_data[:-1, 6, 5])))
adj_labels = np.hstack((adj_labels, np.divide(xrp_data[1:, 6, 4], xrp_data[:-1, 6, 4])))
close_labels = np.hstack((close_labels, np.divide(xrp_data[1:, 6, 3], xrp_data[:-1, 6, 3])))
low_labels = np.hstack((low_labels, np.divide(xrp_data[1:, 6, 2], xrp_data[:-1, 6, 2])))
hi_labels = np.hstack((hi_labels, np.divide(xrp_data[1:, 6, 1], xrp_data[:-1, 6, 1])))
op_labels = np.hstack((op_labels, np.divide(xrp_data[1:, 6, 0], xrp_data[:-1, 6, 0])))

eos_data = read_data('EOS-USD.csv') # eos
train_data = np.vstack((train_data, eos_data[:-1,:,:]))
vol_labels = np.hstack((vol_labels, np.divide(eos_data[1:, 6, 5], eos_data[:-1, 6, 5])))
adj_labels = np.hstack((adj_labels, np.divide(eos_data[1:, 6, 4], eos_data[:-1, 6, 4])))
close_labels = np.hstack((close_labels, np.divide(eos_data[1:, 6, 3], eos_data[:-1, 6, 3])))
low_labels = np.hstack((low_labels, np.divide(eos_data[1:, 6, 2], eos_data[:-1, 6, 2])))
hi_labels = np.hstack((hi_labels, np.divide(eos_data[1:, 6, 1], eos_data[:-1, 6, 1])))
op_labels = np.hstack((op_labels, np.divide(eos_data[1:, 6, 0], eos_data[:-1, 6, 0])))

trx_data = read_data('TRX-USD.csv') # tron
train_data = np.vstack((train_data, trx_data[:-1,:,:]))
vol_labels = np.hstack((vol_labels, np.divide(trx_data[1:, 6, 5], trx_data[:-1, 6, 5])))
adj_labels = np.hstack((adj_labels, np.divide(trx_data[1:, 6, 4], trx_data[:-1, 6, 4])))
close_labels = np.hstack((close_labels, np.divide(trx_data[1:, 6, 3], trx_data[:-1, 6, 3])))
low_labels = np.hstack((low_labels, np.divide(trx_data[1:, 6, 2], trx_data[:-1, 6, 2])))
hi_labels = np.hstack((hi_labels, np.divide(trx_data[1:, 6, 1], trx_data[:-1, 6, 1])))
op_labels = np.hstack((op_labels, np.divide(trx_data[1:, 6, 0], trx_data[:-1, 6, 0])))

ltc_data = read_data('LTC-USD.csv') # lite
train_data = np.vstack((train_data, ltc_data[:-1,:,:]))
vol_labels = np.hstack((vol_labels, np.divide(ltc_data[1:, 6, 5], ltc_data[:-1, 6, 5])))
adj_labels = np.hstack((adj_labels, np.divide(ltc_data[1:, 6, 4], ltc_data[:-1, 6, 4])))
close_labels = np.hstack((close_labels, np.divide(ltc_data[1:, 6, 3], ltc_data[:-1, 6, 3])))
low_labels = np.hstack((low_labels, np.divide(ltc_data[1:, 6, 2], ltc_data[:-1, 6, 2])))
hi_labels = np.hstack((hi_labels, np.divide(ltc_data[1:, 6, 1], ltc_data[:-1, 6, 1])))
op_labels = np.hstack((op_labels, np.divide(ltc_data[1:, 6, 0], ltc_data[:-1, 6, 0])))

ltc_data = trx_data = eos_data = xrp_data = eth_data = 0


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from tensorflow.keras import regularizers
# MLP input [open, hi, low, close, adj, vol, ... repeat 7] -> next day change

print('Training volume predictor model:')
vol_model = Sequential()
vol_model.add(Conv1D(7, 6, strides=6, input_shape=(7, 6), activation='sigmoid', kernel_regularizer=regularizers.l2(0.1), bias_regularizer=regularizers.l2(0.1)))
vol_model.add(Flatten())
vol_model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.1), bias_regularizer=regularizers.l2(0.1)))
vol_model.add(Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.1), bias_regularizer=regularizers.l2(0.1)))
vol_model.add(Dense(1, activation='relu', kernel_regularizer=regularizers.l2(0.1), bias_regularizer=regularizers.l2(0.1)))
vol_model.compile(loss='mean_squared_error', optimizer='nadam')
vol_model.fit(train_data, vol_labels, batch_size=32, epochs=64)

vol_json = vol_model.to_json()
with open('vol.json', 'w') as json_file:
	json_file.write(vol_json)
vol_model.save_weights('vol.h5')
from tensorflow.keras.models import model_from_json

print('Training adjusted predictor model:')
adj_model = Sequential()
adj_model.add(Conv1D(7, 6, strides=6, input_shape=(7, 6), activation='sigmoid', kernel_regularizer=regularizers.l2(0.1), bias_regularizer=regularizers.l2(0.1)))
adj_model.add(Flatten())
adj_model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.1), bias_regularizer=regularizers.l2(0.1)))
adj_model.add(Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.1), bias_regularizer=regularizers.l2(0.1)))
adj_model.add(Dense(1, activation='relu', kernel_regularizer=regularizers.l2(0.1), bias_regularizer=regularizers.l2(0.1)))
adj_model.compile(loss='mean_squared_error', optimizer='nadam')
adj = adj_model.fit(train_data, adj_labels, batch_size=32, epochs=64)

adj_json = adj_model.to_json()
with open('adj.json', 'w') as json_file:
	json_file.write(adj_json)
adj_model.save_weights('adj.h5')

print('Training close predictor model:')
close_model = Sequential()
close_model.add(Conv1D(7, 6, strides=6, input_shape=(7, 6), activation='sigmoid', kernel_regularizer=regularizers.l2(0.1), bias_regularizer=regularizers.l2(0.1)))
close_model.add(Flatten())
close_model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.1), bias_regularizer=regularizers.l2(0.1)))
close_model.add(Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.1), bias_regularizer=regularizers.l2(0.1)))
close_model.add(Dense(1, activation='relu', kernel_regularizer=regularizers.l2(0.1), bias_regularizer=regularizers.l2(0.1)))
close_model.compile(loss='mean_squared_error', optimizer='nadam')
close_model.fit(train_data, close_labels, batch_size=32, epochs=64)

close_json = close_model.to_json()
with open('close.json', 'w') as json_file:
	json_file.write(close_json)
close_model.save_weights('close.h5')

print('Training lows predictor model:')
low_model = Sequential()
low_model.add(Conv1D(7, 6, strides=6, input_shape=(7, 6), activation='sigmoid', kernel_regularizer=regularizers.l2(0.1), bias_regularizer=regularizers.l2(0.1)))
low_model.add(Flatten())
low_model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.1), bias_regularizer=regularizers.l2(0.1)))
low_model.add(Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.1), bias_regularizer=regularizers.l2(0.1)))
low_model.add(Dense(1, activation='relu', kernel_regularizer=regularizers.l2(0.1), bias_regularizer=regularizers.l2(0.1)))
low_model.compile(loss='mean_squared_error', optimizer='nadam')
low_model.fit(train_data, low_labels, batch_size=32, epochs=64)

low_json = low_model.to_json()
with open('low.json', 'w') as json_file:
	json_file.write(low_json)
low_model.save_weights('low.h5')

print('Training highs predictor model:')
hi_model = Sequential()
hi_model.add(Conv1D(7, 6, strides=6, input_shape=(7, 6), activation='sigmoid', kernel_regularizer=regularizers.l2(0.1), bias_regularizer=regularizers.l2(0.1)))
hi_model.add(Flatten())
hi_model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.1), bias_regularizer=regularizers.l2(0.1)))
hi_model.add(Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.1), bias_regularizer=regularizers.l2(0.1)))
hi_model.add(Dense(1, activation='relu', kernel_regularizer=regularizers.l2(0.1), bias_regularizer=regularizers.l2(0.1)))
hi_model.compile(loss='mean_squared_error', optimizer='nadam')
hi_model.fit(train_data, hi_labels, batch_size=32, epochs=64)

hi_json = hi_model.to_json()
with open('hi.json', 'w') as json_file:
	json_file.write(hi_json)
hi = hi_model.save_weights('hi.h5')

print('Training opening predictor model:')
op_model = Sequential()
op_model.add(Conv1D(7, 6, strides=6, input_shape=(7, 6), activation='sigmoid', kernel_regularizer=regularizers.l2(0.1), bias_regularizer=regularizers.l2(0.1)))
op_model.add(Flatten())
op_model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.1), bias_regularizer=regularizers.l2(0.1)))
op_model.add(Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.1), bias_regularizer=regularizers.l2(0.1)))
op_model.add(Dense(1, activation='relu', kernel_regularizer=regularizers.l2(0.1), bias_regularizer=regularizers.l2(0.1)))
op_model.compile(loss='mean_squared_error', optimizer='nadam')
op_model.fit(train_data, op_labels, batch_size=32, epochs=64)

op_json = op_model.to_json()
with open('op.json', 'w') as json_file:
	json_file.write(op_json)
op_model.save_weights('op.h5')

# # load previously trained models
from tensorflow.keras.models import model_from_json

with open('vol.json', 'r') as json_file:
	vol_json = json_file.read()
vol_model = model_from_json(vol_json)
vol_model.load_weights('vol.h5')
vol_model.compile(optimizer='nadam', loss='mean_squared_error')

with open('adj.json', 'r') as json_file:
	adj_json = json_file.read()
adj_model = model_from_json(adj_json)
adj_model.load_weights('adj.h5')
adj_model.compile(optimizer='nadam', loss='mean_squared_error')

with open('close.json', 'r') as json_file:
	close_json = json_file.read()
close_model = model_from_json(close_json)
close_model.load_weights('close.h5')
close_model.compile(optimizer='nadam', loss='mean_squared_error')

with open('low.json', 'r') as json_file:
	low_json = json_file.read()
low_model = model_from_json(low_json)
low_model.load_weights('low.h5')
low_model.compile(optimizer='nadam', loss='mean_squared_error')

with open('hi.json', 'r') as json_file:
	hi_json = json_file.read()
hi_model = model_from_json(hi_json)
hi_model.load_weights('hi.h5')
hi_model.compile(optimizer='nadam', loss='mean_squared_error')

with open('op.json', 'r') as json_file:
	op_json = json_file.read()
op_model = model_from_json(op_json)
op_model.load_weights('op.h5')
op_model.compile(optimizer='nadam', loss='mean_squared_error')

# stacking model input = predicted [op, hi, lo, close, adj, vol]
# output = next week adj
stacked = Sequential()
stacked.add(Dense(12, kernel_regularizer=regularizers.l2(0.1), bias_regularizer=regularizers.l2(0.1), input_dim=6))
stacked.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.1), bias_regularizer=regularizers.l2(0.1)))
stacked.add(Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.1), bias_regularizer=regularizers.l2(0.1)))
stacked.add(Dense(1, activation='relu', kernel_regularizer=regularizers.l2(0.1), bias_regularizer=regularizers.l2(0.1)))
stacked.compile(loss='mean_squared_error', optimizer='nadam')

meta_dat = np.hstack((op_model.predict(train_data), hi_model.predict(train_data), low_model.predict(train_data), close_model.predict(train_data), adj_model.predict(train_data), vol_model.predict(train_data)))

tested = stacked.fit(meta_dat, adj_labels, batch_size=32, epochs=64)

stacked_json = stacked.to_json()
with open('stacked.json', 'w') as json_file:
	json_file.write(stacked_json)
stacked.save_weights('stacked.h5')

with open('stacked.json', 'r') as json_file:
	stacked_json = json_file.read()
stacked = model_from_json(stacked_json)
stacked.load_weights('stacked.h5')
stacked.compile(optimizer='nadam', loss='mean_squared_error')

# read and predict on test dataset
dates = []
opens = []
highs = []
lows = []
closes = []
adjusted = []
volume = []
with open('BTC-USD.csv', 'r') as data:
	data.readline()
	for line in data:
		line = line.strip()
		line = line.split(',')
		dates.append(line[0])
		opens.append(float(line[1]))
		highs.append(float(line[2]))
		lows.append(float(line[3]))
		closes.append(float(line[4]))
		adjusted.append(float(line[5]))
		volume.append(float(line[6]))
dates = np.array(dates, dtype='datetime64') 
opens = np.array(opens)
highs = np.array(highs)
lows = np.array(lows)
closes = np.array(closes)
adjusted = np.array(adjusted)
volume = np.array(volume)
data = np.zeros((len(opens), 6))
for i in range(0, len(opens)):
	data[i] = np.array([opens[i], highs[i], lows[i], closes[i], adjusted[i], volume[i]])
data = data[:len(data)-len(data)%7]
data = np.reshape(data, (int(len(data)/7), 7, 6))
# predicted = close_model.predict(data[:-1, :, :])
meta_test = np.hstack((op_model.predict(data[:-1, :, :]), hi_model.predict(data[:-1, :, :]), low_model.predict(data[:-1, :, :]), close_model.predict(data[:-1, :, :]), adj_model.predict(data[:-1, :, :]), vol_model.predict(data[:-1, :, :])))
predicted = stacked.predict(meta_test)

# plot setup
years = mdates.YearLocator()
months = mdates.MonthLocator()
years_format = mdates.DateFormatter('%Y')

fig, ax = plt.subplots()
ax.plot(dates, adjusted) # plot truth
# ax.plot(dates[1:], np.divide(adjusted[1:], adjusted[:-1])) # plot truth

# ticks
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_format)
ax.xaxis.set_minor_locator(months)

# bound years
date_min = np.datetime64(dates[0], 'Y')
date_max = np.datetime64(dates[-1], 'Y') + np.timedelta64(1, 'Y')
ax.set_xlim(date_min, date_max)

def price(x):
	return ('$%1.2f' % x)

def quantity(x):
	return ('%d' % x)

ax.format__xdata = mdates.DateFormatter('%Y-%m-%d')
ax.format_ydata = price
ax.grid(True)

fig.autofmt_xdate()

# plot predicted volume
predict_dates = []
for i, d in enumerate(dates):
	if (i%7 == 0):
		predict_dates.append(d)
ax.plot(predict_dates[1:-1], predicted * np.reshape(data[:-1, 6, 4], predicted.shape))
# ax.plot(predict_dates[1:-1], predicted)
plt.show()

#plot loss function
line1, = plt.plot(adj.history['loss'], label='training set')
line2, = plt.plot(tested.history['loss'], label='testing set')
plt.legend([line1,line2],['training set','testing set'])
plt.legend(bbox_to_anchor=(1, 1),
           bbox_transform=plt.gcf().transFigure)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.show()
