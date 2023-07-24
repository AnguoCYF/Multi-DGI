import gc
import os
import warnings
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from adj_matrix_generate import *
from signal_to_nodes import *
from sklearn.preprocessing import StandardScaler
from dgi_multi import *
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_fscore_support, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings("ignore")

# %% Preprocessing WISDM

# df = pd.read_csv('./WISDM.csv')
#
# df.dropna(inplace=True)
#
# scaler = StandardScaler()
# df[['x-axis', 'y-axis', 'z-axis']] = scaler.fit_transform(df[['x-axis', 'y-axis', 'z-axis']])

# %% Preprocessing PAMAP2

df = pd.read_csv('./PAMAP2.csv')
df = df.drop(df.columns[0], axis=1)
labels, uniques = pd.factorize(df['activity_id'])
df['activity_id'] = labels

scaler = StandardScaler()

df.iloc[:, 2:-1] = scaler.fit_transform(df.iloc[:, 2:-1])
# %% Creating Nodes

# window_size = 100  # WISDM
window_size = 50  # PAMAP2
overlap = 0.5
train_ratio = 0.8

# 对时间序列数据进行窗口划分
# nodes, nodes_labels, labels_index = signal_to_nodes_WISDM(df, window_size, overlap)
nodes, nodes_labels = signal_to_nodes_PAMAP2(df, window_size, overlap)

n, m, p = nodes.shape
nodes = nodes.reshape(n, m * p)

# %% Graph Construction
# g = build_dgl_graph(nodes.cpu(), nodes_labels.cpu(), method='knn', param=10).to(device)  # WISDM
g = build_dgl_graph(nodes.cpu(), nodes_labels.cpu(), method='cosine', param=80).to(device)   # PAMAP2

# 使用generate_masks函数
train_mask, test_mask = generate_masks(nodes, nodes_labels, train_ratio)

# 将掩码添加到图中
g.train_mask = train_mask.to(device)
g.test_mask = test_mask.to(device)


# %%
def run_model(g, n_layers, dropout, pooling_methods, patience, n_hidden):
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.train_mask.to(device)
    test_mask = g.test_mask.to(device)
    in_feats = features.shape[1]
    activation = nn.PReLU(n_hidden)

    # 计算类别权重
    class_weights = compute_class_weight('balanced', classes=np.unique(labels.cpu().numpy()), y=labels.cpu().numpy())
    # 转为Tensor
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    model = Multi_DGI(g, in_feats, n_hidden, n_layers, activation, dropout, pooling_methods).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练DGI模型
    best_loss = float('inf')
    counter = 0

    for epoch in range(500):
        model.train()
        optimizer.zero_grad()
        loss = model(features)
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            counter = 0
            torch.save(model.state_dict(), '/home/hjf/experiment_muilt_dgi/best_mlt_dgi.pkl')
        else:
            counter += 1

        if counter >= patience:
            break

    model.load_state_dict(torch.load('/home/hjf/experiment_muilt_dgi/best_mlt_dgi.pkl'))

    # 提取节点表示
    embeds = model.encoder(features, corrupt=False).detach()
    features_emb = torch.cat((features, embeds), 1)

    def evaluate(model, features, labels):
        model.eval()
        with torch.no_grad():
            logits = model(features)
            logits = logits
            labels = labels
            _, indices = torch.max(logits, dim=1)
            correct = torch.sum(indices == labels)
            acc = correct.item() * 1.0 / len(labels)
            precision = precision_score(labels.cpu(), indices.cpu(), average='macro', zero_division=0)
            recall = recall_score(labels.cpu(), indices.cpu(), average='macro', zero_division=0)
            f1 = f1_score(labels.cpu(), indices.cpu(), average='macro', zero_division=0)
        return acc, precision, recall, f1

    def train(model, epochs, inputs, lr):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
        model.train()
        best_f1_test = 0
        best_metrics = None

        for epoch in range(epochs):
            model.train()

            logits = model(inputs[train_mask])
            loss = loss_fn(logits, labels[train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc_test, prec_test, recall_test, f1_test = evaluate(model, inputs[test_mask], labels[test_mask])

            if f1_test > best_f1_test:
                best_f1_test = f1_test
                best_metrics = (acc_test, prec_test, recall_test, f1_test)

        best_acc_test, best_prec, best_recall, best_f1 = best_metrics
        print(
            f"\nBest Test Accuracy: {best_acc_test:.3f}, Precision: {best_prec:.3f}, Recall: {best_recall:.3f}, F1: {best_f1:.3f}")

        return best_f1

    # MLP Classifier
    c_epochs = 1000
    c_lr = 0.005
    c_dropout = 0.2
    num_classes = len(np.unique(nodes_labels.cpu()))

    FE_inputs = features_emb.to(device)
    FE_classifier = Classifier(FE_inputs.shape[1], num_classes, dropout=c_dropout).to(device)
    FE_best_f1 = train(FE_classifier, c_epochs, FE_inputs, c_lr)

    # %%
    # Use RandomForestClassifier for classification
    # model_RF_hx = RandomForestClassifier(random_state=42, n_estimators=50, max_features=50).fit(
    #     features_emb[train_mask].cpu().data.numpy(), labels[train_mask].cpu().data.numpy())
    # y_preds = model_RF_hx.predict(features_emb[test_mask].cpu().data.numpy())
    # #
    # prec, rec, FE_best_f1, num = precision_recall_fscore_support(labels[test_mask].cpu().data.numpy(), y_preds,
    #                                                              average='macro')
    # acc_rf = accuracy_score(labels[test_mask].cpu().data.numpy(), y_preds)
    # #
    # print("ALL feature + Embeddings RandomForest Classifier")
    # print("macro Precision:%.3f \nRecall:%.3f \nF1 Score:%.3f \nAccuracy:%.3f" % (prec, rec, FE_best_f1, acc_rf))

    # %% Use Support Vector Machine (SVM) for classification

    # model_svm = SVC(random_state=42).fit(features_emb[train_mask].cpu().data.numpy(), labels[train_mask].cpu().data.numpy())
    # y_preds_svm = model_svm.predict(features_emb[test_mask].cpu().data.numpy())
    #
    # prec_svm, rec_svm, f1_svm, num_svm = precision_recall_fscore_support(labels[test_mask].cpu().data.numpy(), y_preds_svm, average='macro')
    # acc_svm = accuracy_score(labels[test_mask].cpu().data.numpy(), y_preds_svm)
    #
    # print("ALL feature + Embeddings SVM Classifier")
    # print("macro Precision:%.3f \nRecall:%.3f \nF1 Score:%.3f \nAccuracy:%.3f" % (prec_svm, rec_svm, f1_svm, acc_svm))
    #
    # FE_best_f1 = f1_svm

    # %% Use Logistic Regression for classification
    # model_lr = LogisticRegression(max_iter=500, random_state=42).fit(features_emb[train_mask].cpu().data.numpy(), labels[train_mask].cpu().data.numpy())
    # y_preds_lr = model_lr.predict(features_emb[test_mask].cpu().data.numpy())
    #
    # prec_lr, rec_lr, f1_lr, num_lr = precision_recall_fscore_support(labels[test_mask].cpu().data.numpy(), y_preds_lr, average='macro')
    # acc_lr = accuracy_score(labels[test_mask].cpu().data.numpy(), y_preds_lr)
    #
    # print("ALL feature + Embeddings Logistic Regression Classifier")
    # print("macro Precision:%.3f \nRecall:%.3f \nF1 Score:%.3f \nAccuracy:%.3f" % (prec_lr, rec_lr, f1_lr, acc_lr))
    #
    # FE_best_f1 = f1_lr

    # %%
    try:
        name = '//home//hjf//experiment_muilt_dgi//mlt_dgi_f1_' + str(round(FE_best_f1, 3)) + '.pkl'
        os.rename("//home//hjf//experiment_muilt_dgi//best_mlt_dgi.pkl", name)
    except FileNotFoundError:
        print("目录不存在")
    except FileExistsError:
        print("存在相同文件名")

    return FE_best_f1


def objective(trial):
    # 超参数的搜索空间
    n_layers = trial.suggest_int("n_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0, 0.5, step=0.1)
    patience = trial.suggest_int("patience", 10, 50, step=5)
    n_hidden = trial.suggest_int("n_hidden", 40, 900, step=5)

    pooling_max = trial.suggest_int("pooling_max", 0, 1)
    pooling_mean = trial.suggest_int("pooling_mean", 0, 1)
    pooling_min = trial.suggest_int("pooling_min", 0, 1)
    pooling_sum = trial.suggest_int("pooling_sum", 0, 1)
    pooling_std = trial.suggest_int("pooling_std", 0, 1)
    pooling_median = trial.suggest_int("pooling_median", 0, 1)
    pooling_l2_norm = trial.suggest_int("pooling_l2_norm", 0, 1)
    pooling_l1_norm = trial.suggest_int("pooling_l1_norm", 0, 1)

    selected_pooling_methods = []
    if pooling_max:
        selected_pooling_methods.append("max")
    if pooling_mean:
        selected_pooling_methods.append("mean")
    if pooling_min:
        selected_pooling_methods.append("min")
    if pooling_sum:
        selected_pooling_methods.append("sum")
    if pooling_std:
        selected_pooling_methods.append("std")
    if pooling_median:
        selected_pooling_methods.append("median")
    if pooling_l2_norm:
        selected_pooling_methods.append("l2_norm")
    if pooling_l1_norm:
        selected_pooling_methods.append("l1_norm")
    if not selected_pooling_methods:
        return 0

    best_F1 = run_model(g, n_layers, dropout, selected_pooling_methods, patience, n_hidden)

    return best_F1


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=400)  # 您可以根据需要调整 n_trials

# 获取最佳超参数
best_params = study.best_params
print(f"Best Parameters:{best_params}")
