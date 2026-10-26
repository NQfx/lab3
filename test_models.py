import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from linear_models import (
    train_logistic_classifier,
    train_linear_regression,
    compare_classical_models,
)
from linear_models import log_loss, mse

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)

def standardize_data(df,targets_column):
    scaler = StandardScaler()
    features = scaler.fit_transform(df.loc[:, df.columns != targets_column].to_numpy(dtype=np.float64))
    targets = df[targets_column].to_numpy(dtype=np.float64)
    return features, targets

def get_classifier_metrics(preds, probs, targets):
    metrics = {'accuracy': accuracy_score(targets, preds), 'precision': precision_score(targets, preds),
               'recall': recall_score(targets, preds), 'f1': f1_score(targets, preds),
               'roc_auc': roc_auc_score(targets, probs)}
    return metrics

def prepare_dataset(source_df, quantile_proc, train_set_proc, random_seed=42):
    clean_df = source_df.dropna().copy()

    low_boundary = clean_df.quantile(quantile_proc)
    high_boundary = clean_df.quantile(1-quantile_proc)
    clean_df = clean_df[((clean_df >= low_boundary) & (clean_df <= high_boundary)).all(axis=1)]

    random_df = clean_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    train_count = int(random_df.shape[0] * train_set_proc)
    train_df = random_df[:train_count]
    test_df = random_df[train_count:random_df.shape[0]]
    
    return train_df, test_df

def draw_loss_graphic(loss_history):
    plt.plot(loss_history)
    plt.title("Ход loss функции")
    plt.xlabel("Индекс")
    plt.ylabel("Ошибка")
    plt.show()

def test_linear_model():
    original_df = pd.read_csv('Student_Performance.csv')
    original_df['Extracurricular Activities'] = original_df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})
    train_df, test_df = prepare_dataset(original_df, 0.02, 0.8)
    train_features, train_targets = standardize_data(train_df, 'Extracurricular Activities')
    test_features, test_targets = standardize_data(test_df, 'Extracurricular Activities')

    train_result = train_linear_regression(RegressionInput(
        train_features=train_features,
        train_target=train_targets,
        test_features=test_features,
        test_target=test_targets,
        learning_rate=0.009,
        epochs=100
    ))

    print("== Линейная модель ==")
    print(f"Веса: {train_result.weights, train_result.bias}")
    print(f"Loss: {mse(test_targets, train_result.predictions)}")
    print()

    draw_loss_graphic(train_result.loss_history)

def test_logistic_classifier():
    original_df = pd.read_csv('german.csv', sep=';')
    train_df, test_df = prepare_dataset(original_df, 0.02, 0.8)
    train_features, train_targets = standardize_data(train_df, 'Creditability')
    test_features, test_targets = standardize_data(test_df, 'Creditability')

    train_result = train_logistic_classifier(ClassificationInput(
        train_features=train_features,
        train_target=train_targets,
        test_features=test_features,
        test_target=test_targets,
        learning_rate=0.009,
        epochs=50
    ))

    classifier_metrics = get_classifier_metrics(train_result.predictions,
                                                train_result.probabilities,
                                                test_targets)

    print("== Логистическая модель ==")
    print(f"Веса: {train_result.weights, train_result.bias}")
    print(f"Loss: {log_loss(test_targets, train_result.predictions)}")

    for k, v in classifier_metrics.items():
        print(f"{k}: {v:.4f}")

    print()

    draw_loss_graphic(train_result.loss_history)

def print_comparison(result: BaselineComparisonResult):
    models = {
        "Logistic": result.logistic_regression,
        "Tree": result.decision_tree,
        "KNN": result.knn
    }

    print("== Сравнение Logistic, KNN, Decision Tree ==")
    print(f"{'Model':<12} | {'ROC-AUC'}\t | {'Accuracy'}\t | {'F1'}\t | {'Recall'}\t | {'Precision'}\t")

    for name, m in models.items():
        print(f"{name:<12} | {m.roc_auc:.4f}\t  | {m.accuracy:.4f}\t   | {m.f1:.4f}\t | {m.recall:.4f}\t | {m.precision:.4f}")
    print()

def test_compare_classical_models():
    original_df = pd.read_csv('german.csv', sep=';')
    train_df, test_df = prepare_dataset(original_df, 0.02, 0.8)
    train_features, train_targets = standardize_data(train_df, 'Creditability')
    test_features, test_targets = standardize_data(test_df, 'Creditability')

    comparison = compare_classical_models(ClassificationInput(
        train_features=train_features,
        train_target=train_targets,
        test_features=test_features,
        test_target=test_targets,
        learning_rate=0.08,
        epochs=100
    ))

    print_comparison(comparison)


def print_learn_rate_stat_linear(step, lr_range, epochs):
    original_df = pd.read_csv('Student_Performance.csv')
    original_df['Extracurricular Activities'] = original_df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})
    train_df, test_df = prepare_dataset(original_df, 0.02, 0.8)
    train_features, train_targets = standardize_data(train_df, 'Extracurricular Activities')
    test_features, test_targets = standardize_data(test_df, 'Extracurricular Activities')

    n = int(round((lr_range[1] - lr_range[0]) / step)) + 1
    learning_rates = lr_range[0] + step * np.arange(n)
    plt.figure(figsize=(10, 6))
    for lr in learning_rates:
        data = RegressionInput(
            train_features=train_features,
            train_target=train_targets,
            test_features=test_features,
            test_target=test_targets,
            learning_rate=float(lr),
            epochs=epochs,
        )
        result = train_linear_regression(data)
        plt.plot(result.loss_history, label=f"lr = {lr:.3f}")
        print(f"learning_rate = {lr:.3f} -> финальная ошибка (MSE) = {result.loss_history[-1]:.6f}")

    plt.xlabel("Эпоха")
    plt.ylabel("MSE")
    plt.title("История ошибок при разных learning rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def print_learn_rate_stat_logistic(step, lr_range, epochs):
    original_df = pd.read_csv('german.csv', sep=';')
    train_df, test_df = prepare_dataset(original_df, 0.02, 0.8)
    train_features, train_targets = standardize_data(train_df, 'Creditability')
    test_features, test_targets = standardize_data(test_df, 'Creditability')

    n = int(round((lr_range[1] - lr_range[0]) / step)) + 1
    learning_rates = lr_range[0] + step * np.arange(n)

    plt.figure(figsize=(10, 6))

    for lr in learning_rates:
        data = ClassificationInput(
            train_features=train_features,
            train_target=train_targets,
            test_features=test_features,
            test_target=test_targets,
            learning_rate=float(lr),
            epochs=epochs,
        )
        result = train_logistic_classifier(data)
        plt.plot(result.loss_history, label=f"lr = {lr:.3f}")
        print(f"learning_rate = {lr:.3f} -> финальная ошибка (log_loss) = {result.loss_history[-1]:.6f}")

    plt.xlabel("Эпоха")
    plt.ylabel("log_loss")
    plt.title("История log_loss при разных learning rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

print_learn_rate_stat_linear(0.001, (0.001, 0.01), 100)
test_linear_model()
print_learn_rate_stat_logistic(0.01, (0.01, 0.1), 100)
test_logistic_classifier()
test_compare_classical_models()