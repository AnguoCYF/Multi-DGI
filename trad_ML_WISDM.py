import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from adj_matrix_generate import *
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

# 使用generate_masks函数划分数据集
train_mask, test_mask = generate_masks(nodes, nodes_labels, train_ratio)

features = nodes
labels = nodes_labels


# %% Random Forest Classifier
model_rf_hx = RandomForestClassifier(random_state=42, n_estimators=50, max_features=50)

# Train the downstream classifier
model_rf_hx.fit(features[train_mask].cpu().data.numpy(), labels[train_mask].cpu().data.numpy())
y_preds = model_rf_hx.predict(features[test_mask].cpu().data.numpy())

# Evaluate the Classifier
prec, rec, f1, num = precision_recall_fscore_support(labels[test_mask].cpu().data.numpy(), y_preds,
                                                     average='macro')
acc = accuracy_score(labels[test_mask].cpu().data.numpy(), y_preds)

print("ALL feature RandomForest Classifier")
print("macro Precision:%.3f \nRecall:%.3f \nF1 Score:%.3f \nAccuracy:%.3f" % (prec, rec, f1, acc))

# %%Use MLP for classification
in_feats = features.shape[1]

model_mlp = MLPClassifier(hidden_layer_sizes=(in_feats // 4, in_feats // 8), activation='relu', max_iter=1000,
                          random_state=42)

# Train the downstream classifier
model_mlp.fit(features[train_mask].cpu().data.numpy(), labels[train_mask].cpu().data.numpy())
y_preds = model_mlp.predict(features[test_mask].cpu().data.numpy())

# Evaluate the Classifier
prec, rec, f1, num = precision_recall_fscore_support(labels[test_mask].cpu().data.numpy(), y_preds,
                                                     average='macro')
acc = accuracy_score(labels[test_mask].cpu().data.numpy(), y_preds)

print("ALL feature MLP Classifier")
print("macro Precision:%.3f \nRecall:%.3f \nF1 Score:%.3f \nAccuracy:%.3f" % (prec, rec, f1, acc))

# %% Use Support Vector Machine (SVM) for classification

model_svm = SVC(random_state=42)

model_svm.fit(features[train_mask].cpu().data.numpy(), labels[train_mask].cpu().data.numpy())
y_preds_svm = model_svm.predict(features[test_mask].cpu().data.numpy())

prec_svm, rec_svm, f1_svm, num_svm = precision_recall_fscore_support(labels[test_mask].cpu().data.numpy(), y_preds_svm,
                                                                     average='macro')
acc_svm = accuracy_score(labels[test_mask].cpu().data.numpy(), y_preds_svm)

print("ALL feature SVM Classifier")
print("macro Precision:%.3f \nRecall:%.3f \nF1 Score:%.3f \nAccuracy:%.3f" % (prec_svm, rec_svm, f1_svm, acc_svm))

# %% Use Logistic Regression for classification
model_lr = LogisticRegression(max_iter=500, random_state=42)
model_lr.fit(features[train_mask].cpu().data.numpy(), labels[train_mask].cpu().data.numpy())

y_preds_lr = model_lr.predict(features[test_mask].cpu().data.numpy())

prec_lr, rec_lr, f1_lr, num_lr = precision_recall_fscore_support(labels[test_mask].cpu().data.numpy(), y_preds_lr,
                                                                 average='macro')
acc_lr = accuracy_score(labels[test_mask].cpu().data.numpy(), y_preds_lr)

print("ALL feature + DGI Embeddings Logistic Regression Classifier")
print("macro Precision:%.3f \nRecall:%.3f \nF1 Score:%.3f \nAccuracy:%.3f" % (prec_lr, rec_lr, f1_lr, acc_lr))
