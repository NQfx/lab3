import numpy as np
import pandas as pd
import time

def mse(actual, pred):
    count = len(actual)
    res = 0.0
    for i in range(count):
        res += (actual[i]-pred[i])**2
    return res/count

source_df = pd.read_csv('Student_Performance.csv')
source_df['Extracurricular Activities'] = source_df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})

X = source_df.loc[:, source_df.columns != 'Performance Index'].to_numpy(dtype=np.float64)
Y = source_df['Performance Index'].to_numpy(dtype=np.float64)

cur_err = np.float64('inf')
w = np.zeros_like(X[0])
b = 0.0

err_history = []
prev_err = np.float64('inf')

start = time.perf_counter()
while True:
    for i in range(len(Y)):
        pred = X[i] @ w + b

        grad_w = 2 * (pred - Y[i]) * X[i].T
        grad_b = 2 * (pred - Y[i])

        w -= 0.00001 * grad_w
        b -= 0.00001 * grad_b

    preds = []
    for x in X:
        preds.append(x @ w + b)

    if len(err_history) > 0:
        prev_err = err_history[-1]

    cur_err = mse(preds, Y)
    err_history.append(cur_err)

    if abs(prev_err - cur_err) / prev_err < 1e-6:
        break

elapsed = time.perf_counter() - start
print(err_history[-1])
print((w, b))
print(f"Время выполнения: {elapsed:.6f} секунд")

# best result:
# MSE = 4.271789653322252
# w = [2.85364572, 1.02314147, 0.6141226 , 0.47832207, 0.19030606]
# bias = -34.038612279508285
# Время выполнения: 143.721061 секунд