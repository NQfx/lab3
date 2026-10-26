import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
    ClassificationMetrics,
)

def mse(actual, predicated):
    count = len(actual)
    res = 0.0
    for i in range(count):
        res += (actual[i] - predicated[i]) ** 2
    return res/count

def get_batches(features, targets, batch_size):
    indices = np.arange(len(features))
    np.random.shuffle(indices)
    for i in range(0, len(features), batch_size):
        batch_idx = indices[i:i + batch_size]
        yield features[batch_idx], targets[batch_idx]

def train_linear_regression(data: RegressionInput) -> RegressionResult:
    features = data.train_features
    targets = data.train_target

    loss_history = []
    weights = np.zeros_like(features[0])
    bias = 0.0

    for i in range (data.epochs):
        for x_batch, y_batch in get_batches(features, targets, int(len(features)*0.1)):
            predictions = x_batch @ weights + bias

            current_error = predictions - y_batch
            grad_w = (2/len(y_batch)) * (x_batch.T@current_error)
            grad_b = (2/len(y_batch)) * np.sum(current_error)

            weights -= data.learning_rate * grad_w
            bias -= data.learning_rate * grad_b

        loss_history.append(mse(features @ weights + bias, targets))

    predictions = data.test_features@weights.T + bias
    return RegressionResult(
        weights=weights,
        bias=bias,
        loss_history=loss_history,
        predictions=predictions
    )

def sigmoid(x):
    return 1/(1+np.exp(-x))


def log_loss(actual, predicted):
    actual = np.array(actual, dtype=np.float64)
    predicted = np.array(predicted, dtype=np.float64)
    eps = np.finfo(np.float64).eps

    predicted = np.clip(predicted, eps, 1 - eps)
    loss = -np.mean(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted))
    return float(loss)


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    features = data.train_features
    targets = data.train_target

    loss_history = []
    weights = np.zeros_like(features[0])
    bias = 0.0

    for i in range(data.epochs):
        for x_batch, y_batch in get_batches(features, targets, int(len(features) * 0.1)):
            probabilities = sigmoid(x_batch @ weights + bias)

            current_error = probabilities - y_batch
            grad_w = (1 / len(y_batch)) * (x_batch.T @ current_error)
            grad_b = (1 / len(y_batch)) * np.sum(current_error)

            weights -= data.learning_rate * grad_w
            bias -= data.learning_rate * grad_b

        loss_history.append(log_loss(sigmoid(features @ weights + bias), targets))

    probabilities = sigmoid(data.test_features @ weights.T + bias)
    return ClassificationResult(
        weights=weights,
        bias=bias,
        loss_history=loss_history,
        predictions=np.round(probabilities),
        probabilities=probabilities
    )

def get_metrics(model, features, target) -> ClassificationMetrics:
    preds = model.predict(features)
    probs = model.predict_proba(features)[:, 1]
    return ClassificationMetrics(
        roc_auc=roc_auc_score(target, probs),
        accuracy=accuracy_score(target, preds),
        precision=precision_score(target, preds),
        recall=recall_score(target, preds),
        f1=f1_score(target, preds)
    )


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    log_res = LogisticRegression().fit(data.train_features, data.train_target)
    dt_res = DecisionTreeClassifier().fit(data.train_features, data.train_target)
    knn_res = KNeighborsClassifier().fit(data.train_features, data.train_target)

    return BaselineComparisonResult(
        logistic_regression=get_metrics(log_res, data.test_features, data.test_target),
        decision_tree=get_metrics(dt_res, data.test_features, data.test_target),
        knn=get_metrics(knn_res, data.test_features, data.test_target)
    )