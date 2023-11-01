import warnings
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from adj_matrix_generate import *
from dgi import DGI
from dgi_multi import *
from signal_to_nodes import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings("ignore")

# %% Preprocessing WISDM

df = pd.read_csv('./WISDM.csv')

df.dropna(inplace=True)

scaler = StandardScaler()
df[['x-axis', 'y-axis', 'z-axis']] = scaler.fit_transform(df[['x-axis', 'y-axis', 'z-axis']])

# %% Creating Nodes

window_size = 100  # WISDM
overlap = 0.5
train_ratio = 0.8

# 对时间序列数据进行窗口划分
nodes, nodes_labels, labels_index = signal_to_nodes_WISDM(df, window_size, overlap)

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

# %% 设置超参数
# Multi_DGI
n_layers = 3
dropout = 0
lr = 0.001
epochs = 1000
pooling_methods = ['mean', 'max', 'min', 'std', 'median', 'l1_norm']
patience = 40
n_hidden = 885
activation = nn.PReLU(n_hidden)

# %% 初始化DGI模型
in_feats = features.shape[1]

# model = Multi_DGI(g, in_feats, n_hidden, n_layers, activation, dropout, pooling_methods).to(device)
model = DGI(g, in_feats, n_hidden, n_layers, activation, dropout).to(device)
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
        torch.save(model.state_dict(), '/home/hjf/experiment_muilt_dgi/best_mlt_dgi.pkl')
    else:
        counter += 1

    if counter >= patience:
        print(f"Early stopping! Epoch: {epoch}")
        break

    if epoch % 10 == 0:
        print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item()}")

    # %% 提取节点表示

print(f"Loading the best model")
model.load_state_dict(torch.load('/home/hjf/experiment_muilt_dgi/best_mlt_dgi.pkl'))

embeds = model.encoder(features, corrupt=False).detach()

# Concatenate feature and embeddings to obtain combined feature matrix
features_emb = torch.cat((features, embeds), 1)

# %% Random Forest Classifier
model_rf = RandomForestClassifier(random_state=42, n_estimators=50, max_features=50)

# Train the downstream classifier
model_rf.fit(features_emb[train_mask].cpu().data.numpy(), labels[train_mask].cpu().data.numpy())
y_preds = model_rf.predict(features_emb[test_mask].cpu().data.numpy())

# Evaluate the Classifier
prec, rec, f1, num = precision_recall_fscore_support(labels[test_mask].cpu().data.numpy(), y_preds,
                                                     average='macro')
acc = accuracy_score(labels[test_mask].cpu().data.numpy(), y_preds)

print("ALL feature + Embeddings RandomForest Classifier")
print("macro Precision:%.3f \nRecall:%.3f \nF1 Score:%.3f \nAccuracy:%.3f" % (prec, rec, f1, acc))

# %%
model_mlp = MLPClassifier(hidden_layer_sizes=(in_feats // 4, in_feats // 8), activation='relu', max_iter=1000,
                          random_state=42)

# Train the downstream classifier
model_mlp.fit(features_emb[train_mask].cpu().data.numpy(), labels[train_mask].cpu().data.numpy())
y_preds = model_mlp.predict(features_emb[test_mask].cpu().data.numpy())

# Evaluate the Classifier
prec, rec, f1, num = precision_recall_fscore_support(labels[test_mask].cpu().data.numpy(), y_preds,
                                                     average='macro')
acc = accuracy_score(labels[test_mask].cpu().data.numpy(), y_preds)

print("ALL feature + DGI Embeddings MLP Classifier")
print("macro Precision:%.3f \nRecall:%.3f \nF1 Score:%.3f \nAccuracy:%.3f" % (prec, rec, f1, acc))

# %% Use Support Vector Machine (SVM) for classification

model_svm = SVC(random_state=42)

model_svm.fit(features_emb[train_mask].cpu().data.numpy(), labels[train_mask].cpu().data.numpy())
y_preds_svm = model_svm.predict(features_emb[test_mask].cpu().data.numpy())

prec_svm, rec_svm, f1_svm, num_svm = precision_recall_fscore_support(labels[test_mask].cpu().data.numpy(), y_preds_svm,
                                                                     average='macro')
acc_svm = accuracy_score(labels[test_mask].cpu().data.numpy(), y_preds_svm)

print("ALL feature + DGI Embeddings SVM Classifier")
print("macro Precision:%.3f \nRecall:%.3f \nF1 Score:%.3f \nAccuracy:%.3f" % (prec_svm, rec_svm, f1_svm, acc_svm))

# %% Use Logistic Regression for classification
model_lr = LogisticRegression(max_iter=500, random_state=42)

model_lr.fit(features_emb[train_mask].cpu().data.numpy(), labels[train_mask].cpu().data.numpy())
y_preds_lr = model_lr.predict(features_emb[test_mask].cpu().data.numpy())

prec_lr, rec_lr, f1_lr, num_lr = precision_recall_fscore_support(labels[test_mask].cpu().data.numpy(), y_preds_lr,
                                                                 average='macro')
acc_lr = accuracy_score(labels[test_mask].cpu().data.numpy(), y_preds_lr)

print("ALL feature + DGI Embeddings Logistic Regression Classifier")
print("macro Precision:%.3f \nRecall:%.3f \nF1 Score:%.3f \nAccuracy:%.3f" % (prec_lr, rec_lr, f1_lr, acc_lr))


