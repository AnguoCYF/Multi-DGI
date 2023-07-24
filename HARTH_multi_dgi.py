import gc
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from adj_matrix_generate import *
from signal_to_nodes import signal_to_nodes_HARTH
from sklearn.preprocessing import StandardScaler
from dgi_multi import *
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings("ignore")
# %% Preprocessing

df = pd.read_csv('./HARTH.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['timestamp'] = (df['timestamp'] - df['timestamp'].min()) / np.timedelta64(1, 's')
labels, uniques = pd.factorize(df['label'])
df['label'] = labels

scaler = StandardScaler()

df.iloc[:, 2:-1] = scaler.fit_transform(df.iloc[:, 2:-1])
# %% Creating Nodes

window_size = 300
overlap = 0.2
train_ratio = 0.8

# 对时间序列数据进行窗口划分
nodes, nodes_labels = signal_to_nodes_HARTH(df, window_size, overlap)

n, m, p = nodes.shape
nodes = nodes.reshape(n, m * p)

# %% Graph Construction
g = build_dgl_graph(nodes.cpu(), nodes_labels.cpu(), method='cosine', param=80).to(device)

# 使用generate_masks函数
train_mask, test_mask = generate_masks(nodes, nodes_labels, train_ratio)

# 将掩码添加到图的ndata中
g.train_mask = train_mask.to(device)
g.test_mask = test_mask.to(device)

features = nodes
labels = nodes_labels

# 计算类别权重
class_weights = compute_class_weight('balanced', classes=np.unique(labels.cpu().numpy()), y=labels.cpu().numpy())
# 转为Tensor
class_weights = torch.tensor(class_weights, dtype=torch.float32)
# %% 设置超参数
# Multi_DGI
n_layers = 2
dropout = 0.2
lr = 0.001
epochs = 1000
# pooling_methods = ['mean']
pooling_methods = ['mean', 'max', 'min', 'sum', 'std', 'median', 'l2_norm', 'l1_norm']
patience = 50
n_hidden = 1500
activation = nn.PReLU(n_hidden)

# %% 初始化DGI模型
in_feats = features.shape[1]

model = Multi_DGI(g, in_feats, n_hidden, n_layers, activation, dropout, pooling_methods).to(device)
# model = DGI(g, in_feats, n_hidden, n_layers, activation, dropout).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# %% 训练Muilt_DGI模型
best_loss = float('inf')
best_epoch = 0
counter = 0

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    loss = model(features)
    loss.backward()
    optimizer.step()

    if loss.item() < best_loss:
        best_loss = loss.item()
        best_epoch = epoch
        counter = 0
        torch.save(model.state_dict(), '/home/hjf/experiment_muilt_dgi/best_adjacency_matrix.pkl')
    else:
        counter += 1

    if counter >= patience:
        print(f"Early stopping! Epoch: {epoch}")
        break

    if epoch % 5 == 0:
        print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item()}")

# %% 提取节点表示

print(f"Loading the best model")
model.load_state_dict(torch.load('/home/hjf/experiment_muilt_dgi/best_adjacency_matrix.pkl'))

embeds = model.encoder(features, corrupt=False).detach()

# Concatenate h and X to obtain combined feature matrix X'
features_emb = torch.cat((features, embeds), 1)

# %%

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

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    best_f1_test = 0
    best_epoch = 0
    best_metrics = None

    for epoch in range(epochs):
        model.train()

        logits = model(inputs[train_mask])
        loss = loss_fn(logits, labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc_train, prec_train, recall_train, f1_train = evaluate(model, inputs[train_mask], labels[train_mask])
        acc_test, prec_test, recall_test, f1_test = evaluate(model, inputs[test_mask], labels[test_mask])

        if f1_test > best_f1_test:
            best_f1_test = f1_test
            best_epoch = epoch
            best_metrics = (acc_test, prec_test, recall_test, f1_test)

        if (epoch + 1) % 50 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.3f},Train Accuracy: {acc_train:.3f}, Train F1: {f1_train:.3f}, Test Accuracy: {acc_test:.3f},"
                f"Test Precision: {prec_test:.3f},Test Recall: {recall_test:.3f},Test F1: {f1_test:.3f}")

    best_acc_test, best_prec, best_recall, best_f1 = best_metrics
    print(
        f"\nBest Test Accuracy at Epoch {best_epoch + 1}: Accuracy: {best_acc_test:.3f}, Precision: {best_prec:.3f}, Recall: {best_recall:.3f}, F1: {best_f1:.3f}")

    return best_acc_test, best_f1

# %% 设置超参数
gc.collect()

# Classifier
c_epochs = 1000
c_lr = 0.005
c_dropout = 0.5
num_classes = len(np.unique(nodes_labels.cpu()))
AF_inputs = features.to(device)

AF_classifier = Classifier(AF_inputs.shape[1], num_classes, dropout=c_dropout).to(device)
AF_acc_test, AF_best_f1 = train(AF_classifier, c_epochs, AF_inputs, c_lr)

FE_inputs = features_emb.to(device)
FE_classifier = Classifier(FE_inputs.shape[1], num_classes, dropout=c_dropout).to(device)
FE_acc_test, FE_best_f1 = train(FE_classifier, c_epochs, FE_inputs, c_lr)

print(f"\nDifference Accuracy: {FE_acc_test - AF_acc_test:.3f}, F1: {FE_best_f1 - AF_best_f1:.3f}")


# %% train classifier RandomForest
from sklearn.ensemble import RandomForestClassifier
# # ALL feature
# model_RF = RandomForestClassifier(n_estimators=50, max_features=50).fit(features[train_mask].cpu().data.numpy(), labels[train_mask].cpu().data.numpy())
# y_preds = model_RF.predict(features[test_mask].cpu().data.numpy())
#
# prec, rec, f1, num = precision_recall_fscore_support(labels[test_mask].cpu().data.numpy(), y_preds, average='macro')
#
# print("AF RandomForest Classifier")
# print("macro Precision:%.3f \nIllicit  Recall:%.3f \nIllicit F1 Score:%.3f" % (prec, rec, f1))
#
# ALL feature + embeddings
# model_RF_hx = RandomForestClassifier(n_estimators=50, max_features=50).fit(features_emb[train_mask].cpu().data.numpy(), labels[train_mask].cpu().data.numpy())
# y_preds = model_RF_hx.predict(features_emb[test_mask].cpu().data.numpy())
#
# prec, rec, f1, num = precision_recall_fscore_support(labels[test_mask].cpu().data.numpy(), y_preds, average='macro')
#
# print("ALL feature + Embeddings RandomForest Classifier")
# print("macro Precision:%.3f \nIllicit  Recall:%.3f \nIllicit F1 Score:%.3f" % (prec, rec, f1))
# %%
# def plot_activity(activity, df):
#     data = df[df['activity'] == activity][['x-axis', 'y-axis', 'z-axis']][:200]
#     axis = data["x-axis"].plot(subplots=True,
#                      title=activity,color="b")
#     axis = data["y-axis"].plot(subplots=True,
#                  title=activity,color="r")
#     axis = data["z-axis"].plot(subplots=True,
#              title=activity,color="g")
#     for ax in axis:
#         ax.legend(loc='lower left', bbox_to_anchor=(1.0, 0.5))
#
# plot_activity("Sitting", df)
#
# df['activity'].value_counts().plot(kind='bar')
