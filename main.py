import calendar

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import torch  # pytorch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable


class MyLSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(MyLSTM, self).__init__()
        self.num_classes = num_classes  # number of classe1
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length

        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=0.93)  # lstm drop 0.9 is good
        self.lstm2 = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=0.93)  # lstm drop 0.9 is good
        self.fc_1 = nn.Linear(hidden_size, 128)  # fully connected 1
        self.fc = nn.Linear(128, num_classes)  # fully connected last layer

        self.relu = nn.ReLU()

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm1(x, (h_0, c_0))  # lstm with input, hidden, and internal state
        output, (hn, cn) = self.lstm2(x, (hn, cn))  # lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size)  # reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out)  # first Dense
        out = self.relu(out)  # relu
        out = self.fc(out)  # Final Output
        return out


class Single_Variable_LSTM:
    def __init__(self, train_y: np.ndarray, prev_days_for_train, device: str = None):
        """

        :param train_y: one-dim array
        :param prev_days_for_train: how many days to look back
        :param device: cpu or cuda
        """

        self.num_epochs = 1000  # 1000 epochs  7000 is good
        self.learning_rate = 0.0008  # 0.001 lr

        self.input_size = prev_days_for_train  # number of features
        self.hidden_size = max(1, int(0.6 * prev_days_for_train))  # number of features in hidden state 0.3 is good
        self.num_layers = 1  # number of stacked lstm layers

        self.num_classes = 1  # number of output classes

        # Use MinMax to rescale the training data
        self.scaler = MinMaxScaler()
        self.train_y = train_y
        if type(train_y) == pd.DataFrame:
            self.scaler.fit(train_y.values.reshape(-1, 1))
        else:
            self.scaler.fit(train_y.reshape(-1, 1))
        self.train_y = self.transform_data(train_y).astype(np.float32)

        # Use the data from the previous prev_days_for_train days as input, and remember to shift the data in self.train_y backward by prev_days_for_train positions.
        self.prev_days_for_train = prev_days_for_train
        self.train_x = self.create_dataset(self.train_y, self.prev_days_for_train)
        self.train_y = self.train_y[self.prev_days_for_train:]
        self.init_pred = self.train_y[-self.prev_days_for_train:].copy()

        self.train_x_tensors = Variable(torch.Tensor(self.train_x))
        self.train_x_tensors = torch.reshape(self.train_x_tensors,
                                             (self.train_x_tensors.shape[0], 1, self.train_x_tensors.shape[1]))
        self.train_y_tensors = Variable(torch.Tensor(self.train_y))

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.lstm = MyLSTM(self.num_classes, self.input_size, self.hidden_size, self.num_layers,
                           self.train_x_tensors.shape[1])  # our lstm class

        self.criterion = torch.nn.MSELoss()  # mean-squared error for regression
        self.optimizer = torch.optim.Adam(self.lstm.parameters(), lr=self.learning_rate)  # adam optimizer

    def train(self):
        """
        Train the model
        :return: predicted result
        """
        outputs = None
        for epoch in range(self.num_epochs):
            outputs = self.lstm.forward(self.train_x_tensors)  # forward pass
            self.optimizer.zero_grad()  # caluclate the gradient, manually setting to 0

            # obtain the loss function
            loss = self.criterion(outputs, self.train_y_tensors)

            loss.backward()  # calculates the loss of the loss function

            self.optimizer.step()  # improve from loss, i.e backprop
            if epoch % 100 == 0:
                print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
        return self.inv_transform_data(torch.squeeze(outputs, 1).detach().cpu().numpy().reshape(-1, 1))

    def test(self, test_y):
        """

        :param test_y: one_dim array
        :return:
        """
        y = np.concatenate((self.init_pred, self.transform_data(test_y)), axis=0)
        x_test = self.create_dataset(y.copy(), self.prev_days_for_train)  # old transformers

        x_test = Variable(torch.Tensor(x_test))  # converting to Tensors

        # reshaping the dataset
        x_test = torch.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

        train_predict = self.lstm(x_test)  # forward pass
        train_predict = train_predict.data.numpy()  # numpy conversion
        return self.scaler.inverse_transform(train_predict)

    def predict(self, predict_days, update=False):
        """
        :param predict_days: days to predict
        :param update: choose to update the model while predicting
        :return:
        """
        def single_predict(X):
            new_x = torch.Tensor(X)
            new_x_final = torch.reshape(new_x, (new_x.shape[0], 1, new_x.shape[1]))
            return self.lstm(new_x_final)

        def update_model(X, y):
            new_x = Variable(torch.Tensor(X))
            new_y = Variable(torch.Tensor(y))

            new_x_final = torch.reshape(new_x, (new_x.shape[0], 1, new_x.shape[1]))

            for epoch in range(self.num_epochs):
                outputs = self.lstm.forward(new_x_final)

                self.optimizer.zero_grad()
                loss = self.criterion(outputs, new_y)
                loss.backward()
                self.optimizer.step()

        pred = []
        predict_x = torch.from_numpy(self.init_pred.copy().reshape(1, -1))
        # predict_x = torch.unsqueeze(torch.from_numpy(predict_x), 0)
        for _ in range(predict_days):
            predict_y = single_predict(predict_x)
            pred.append(predict_y[0][0].item())
            if update is True:
                update_model(predict_x, predict_y)
            predict_x = torch.cat((predict_x, predict_y), 1)
            predict_x = predict_x[:, 1:]
            if _ % 10 == 0:
                print(f'predict {_} days')
        start_index = self.prev_days_for_train + self.train_x.shape[0]
        return start_index, start_index + predict_days, self.inv_transform_data(np.array(pred).reshape(-1, 1))

    def create_dataset(self, df, train_days):
        result = []
        for i in range(len(df) - train_days):
            result.extend(df[i:i + train_days])
        result = np.array(result, dtype=np.float32)
        result = result.reshape(-1, train_days)
        return result

    def get_train_data(self):
        return self.prev_days_for_train, self.prev_days_for_train + self.train_x.shape[0] - 1, self.inv_transform_data(
            self.train_y)

    def transform_data(self, data):
        if type(data) == pd.DataFrame:
            return self.scaler.transform(data.values.reshape(-1, 1))
        elif type(data) == np.ndarray:
            return self.scaler.transform(data.reshape(-1, 1))
        else:
            raise TypeError("data type must be pd.DataFrame or np.ndarray")

    def inv_transform_data(self, data):
        return self.scaler.inverse_transform(data)
def main(update_option=False, prev_days_for_training=180, call_by_app=False):
    prev_days_for_train = prev_days_for_training  # pred 150 is good
    df = pd.read_csv('data_daily.csv')
    df = df[['Receipt_Count']].values
    data = np.reshape(df, (len(df)))
    model = Single_Variable_LSTM(data, prev_days_for_train)
    y = data[prev_days_for_train:]

    if not update_option:
        model.lstm.load_state_dict(torch.load('model'))
        a = np.reshape(y, (len(y), 1))
    else:
        a = model.train()

    b = model.predict(365, update=update_option)[2]
    result = np.concatenate((a, b), axis=0)

    result = result.reshape(1, -1)
    result = np.squeeze(result)

    prev = data[:prev_days_for_train]
    y = np.concatenate([prev, y])
    zeroes = np.empty(len(prev), dtype=object)
    zeroes[:] = None
    result = np.concatenate([zeroes, result])
    plt.plot(y, label='real')
    plt.plot(result, label='predict')
    plt.axvline(x=len(y), color='r', linestyle='--')
    plt.legend()
    plt.title('LSTM Predictions')
    plt.savefig('static/predictions.png')

    if call_by_app:
        pass
    else:
        plt.show()

    months = []
    for i in range(12):
        months.append(calendar.monthrange(2022, i + 1)[1])

    wast_predict = len(df) - prev_days_for_train
    predict = result[365:]
    monthly_predict = dict()
    for i in range(12):
        monthly_predict[i + 1] = int(np.sum(predict[sum(months[:i]):sum(months[:i + 1])]))
    print(monthly_predict)
    torch.save(model.lstm.state_dict(), 'model')

    # plt.show()

    return monthly_predict


if __name__ == '__main__':
    update_option = input("Update Option? Y/N      ")
    while True:
        if update_option.lower() == 'y':
            update_option = True
            break
        elif update_option.lower() == 'n':
            update_option = False
            break
        else:
            print('Input Y or N')
    prev_days = input("Input a number for Previous days to look ahead(Less than 180)        ")
    while True:
        if prev_days.isnumeric():
            prev_days = int(prev_days)
            break
        else:
            print('Please input a number')
    monthly = main(update_option, prev_days)
    for i in range(12):
        print('Month' + str(i + 1) + " Total is : " + str(monthly[i + 1]))
