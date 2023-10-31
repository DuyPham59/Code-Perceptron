import numpy as np
import pandas as pd

# trả về 1 nếu x>0 và ngược lại
def unit_step_func(x):
    return np.where(x > 0, 1, 0)

class Perceptron:

    #hàm khởi tạo
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = unit_step_func
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        #gán giá trị số hàng cho n_samples và số cột cho n_features
        n_samples, n_features = X.shape

        # Khởi tạo tham số
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.where(y > 0, 1, 0)

        # Học trọng số
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                # cập nhật perceptron
                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

# Testing
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    # Load data from file
    data = pd.read_csv('heart.csv')
    X = data.iloc[:, 0:-1].values
    y = data.iloc[:, -1].values

    # chia dữ liệu thành các bộ training và test 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # Đào tạo và đánh giá mô hình Perceptron
    p = Perceptron(learning_rate=0.01, n_iters=1000)
    p.fit(X_train, y_train)
    predictions = p.predict(X_test)

    # In các thông số mô hình đã đào tạo
    print("Trained model:")
    print("Weights:", p.weights)
    print("Bias:", p.bias)

    # Thực hiện dự đoán
    print("độ chính xác : ", accuracy(y_test, predictions))

    # Yêu cầu người dùng nhập các giá trị của mảng
    input_str = input("Nhập các giá trị của mảng, cách nhau bằng dấu phẩy: ")

    # Chuyển đổi chuỗi nhập thành mảng numpy
    input_arr = np.fromstring(input_str, dtype=float, sep=',')

    # Sử dụng mô hình để dự đoán trên mảng nhập
    predictions = p.predict(input_arr.reshape(1, -1))

    # In kết quả dự đoán
    print("Kết quả dự đoán:", predictions)

