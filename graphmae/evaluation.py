import copy
from tqdm import tqdm
import torch
import torch.nn as nn

from graphmae.utils import create_optimizer, accuracy
from sklearn.model_selection import  GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn import metrics
from munkres import Munkres
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.preprocessing import normalize, OneHotEncoder
# from sklearn.linear_model import LogisticRegression

# def test_nc(X_train, y_train, X_test, y_test):
#     logreg = LogisticRegression(solver='liblinear')
#     c = 2.0 ** np.arange(-10, 10)
#
#     clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
#                        param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
#                        verbose=0)
#     clf.fit(X_train, y_train)
#
#     y_pred = clf.predict_proba(X_test)
#     y_pred = prob_to_one_hot(y_pred)
#     acc = accuracy_score(y_test, y_pred)
#     micro = f1_score(y_test, y_pred, average="micro")
#     macro = f1_score(y_test, y_pred, average="macro")
#
#     return acc, micro, macro
#
# def label_classification(embeddings, train_mask, val_mask, test_mask, label):
#     X = embeddings.detach().cpu().numpy()
#     Y = label.detach().cpu().numpy()
#     Y = Y.reshape(-1, 1)
#     onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
#     Y = onehot_encoder.transform(Y).toarray().astype(bool)
#
#     if np.isinf(X).any() == True or np.isnan(X).any() == True:
#         return {
#             'F1Mi': 0,
#             'F1Ma': 0,
#             'Acc': 0
#         }
#     X = normalize(X, norm='l2')
#     # if type == 1:
#         # mask = train_test_split(
#         #     label.detach().cpu().numpy(), seed=np.random.randint(0, 35456, size=1), train_examples_per_class=20,
#         #     val_size=500, test_size=None)
#         #
#         # X_train = X[mask['train'].astype(bool)]
#         # X_val = X[mask['val'].astype(bool)]
#         # X_test = X[mask['test'].astype(bool)]
#         # y_train = Y[mask['train'].astype(bool)]
#         # y_val = Y[mask['val'].astype(bool)]
#         # y_test = Y[mask['test'].astype(bool)]
#     #     X_train, X_test, y_train, y_test = train_test_split(X, Y,
#     #                                                         test_size=1 - 0.1)
#     # else:
#     X_train = X[train_mask.cpu().numpy()]
#     X_val = X[val_mask.cpu().numpy()]
#     X_test = X[test_mask.cpu().numpy()]
#     y_train = Y[train_mask.cpu().numpy()]
#     y_val = Y[val_mask.cpu().numpy()]
#     y_test = Y[test_mask.cpu().numpy()]
#
#     acc, micro, macro = test_nc(X_train, y_train, X_test, y_test)
#
#     # return {
#     #     'F1Mi': micro,
#     #     'F1Ma': macro,
#     #     'Acc': acc
#     # }
#     return acc
def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)

    l1 = list(set(y_true))
    numclass1 = len(l1)

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    if numclass1 != numclass2:
        print('Error! Class number not equal...')
        return

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    precision_macro = metrics.precision_score(y_true, new_predict, average='macro')
    recall_macro = metrics.recall_score(y_true, new_predict, average='macro')
    f1_micro = metrics.f1_score(y_true, new_predict, average='micro')
    precision_micro = metrics.precision_score(y_true, new_predict, average='micro')
    recall_micro = metrics.recall_score(y_true, new_predict, average='micro')
    return acc, f1_macro
def eva(y_true, y_pred, state, show):
    # clustering
    acc, f1 = cluster_acc(y_true, y_pred)
    # nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    nmi = nmi_score(y_true, y_pred)
    ari = ari_score(y_true, y_pred)

    if show:
        print(state, 'clustering acc: {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi),
              ', ari {:.4f}'.format(ari), ', f1 {:.4f}'.format(f1))
    return acc, nmi, ari, f1
def node_clustering(model, graph, x, num_classes, epoch, device):
    model.eval()

    with torch.no_grad():
        emb = model.embed(graph.to(device), x.to(device))

    kmeans = KMeans(n_clusters=num_classes, n_init=20)
    labels = graph.ndata["label"].cpu().numpy()
    y_pred = kmeans.fit_predict(emb.cpu().numpy())
    clu_acc = eva(labels, y_pred, epoch, show=True)
    return clu_acc

def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, bool)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret

# def accuracy_score(preds, labels):
#     # correct = (preds == labels).astype(float)
#     correct = (preds == labels).float()
#
#     correct = correct.sum()
#     return correct / len(labels)

def test_classify(feature, labels):
    f1_mac = []
    f1_mic = []
    accs = []
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    for train_index, test_index in kf.split(feature):
        train_X, train_y = feature[train_index], labels[train_index]
        test_X, test_y = feature[test_index], labels[test_index]
        clf = svm.SVC(kernel='rbf', decision_function_shape='ovo')
        clf.fit(train_X, train_y)
        preds = clf.predict(test_X)

        micro = f1_score(test_y, preds, average='micro')
        macro = f1_score(test_y, preds, average='macro')
        acc = accuracy(preds, test_y)
        accs.append(acc)
        f1_mac.append(macro)
        f1_mic.append(micro)
    f1_mic = np.array(f1_mic)
    f1_mac = np.array(f1_mac)
    accs = np.array(accs)
    f1_mic = np.mean(f1_mic)
    f1_mac = np.mean(f1_mac)
    accs = np.mean(accs)
    print('Testing based on svm: ',
          'f1_micro=%.4f' % f1_mic,
          'f1_macro=%.4f' % f1_mac,
          'acc=%.4f' % accs)
    return f1_mic, f1_mac, accs

def node_classification_evaluation(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device,
                                   linear_prob=True, mute=False, logger=None):
    model.eval()
    if linear_prob:
        with torch.no_grad():
            x = model.embed(graph.to(device), x.to(device))
            in_feat = x.shape[1]
        encoder = LogisticRegression(in_feat, num_classes)
    else:
        encoder = model.encoder
        encoder.reset_classifier(num_classes)

    num_finetune_params = [p.numel() for p in encoder.parameters() if p.requires_grad]
    if not mute:
        print(f"num parameters for finetuning: {sum(num_finetune_params)}")

    encoder.to(device)
    optimizer_f = create_optimizer("adam", encoder, lr_f, weight_decay_f)
    final_acc, estp_acc = linear_probing_for_transductive_node_classiifcation(encoder, graph, x, optimizer_f,
                                                                              max_epoch_f, device, mute, logger=logger)
    return final_acc, estp_acc


def linear_probing_for_transductive_node_classiifcation(model, graph, feat, optimizer, max_epoch, device, mute=False,
                                                        logger=None):
    criterion = torch.nn.CrossEntropyLoss()

    graph = graph.to(device)
    x = feat.to(device)

    train_mask = graph.ndata["train_mask"]
    val_mask = graph.ndata["val_mask"]
    test_mask = graph.ndata["test_mask"]
    labels = graph.ndata["label"]

    best_val_acc = 0
    best_val_epoch = 0
    best_model = None

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)

    for epoch in epoch_iter:
        model.train()
        out = model(graph, x)
        loss = criterion(out[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
        optimizer.step()

        with torch.no_grad():
            model.eval()
            pred = model(graph, x)
            val_acc = accuracy(pred[val_mask], labels[val_mask])
            val_loss = criterion(pred[val_mask], labels[val_mask])
            test_acc = accuracy(pred[test_mask], labels[test_mask])
            test_loss = criterion(pred[test_mask], labels[test_mask])

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_model = copy.deepcopy(model)

        if not mute:
            epoch_iter.set_description(
                f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_loss:{val_loss.item(): .4f}, val_acc:{val_acc}, test_loss:{test_loss.item(): .4f}, test_acc:{test_acc: .4f}")

    best_model.eval()
    with torch.no_grad():
        pred = best_model(graph, x)
        estp_test_acc = accuracy(pred[test_mask], labels[test_mask])
    if mute:
        print(
            f"# IGNORE: --- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")
        logger.info(
            f"# IGNORE: --- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")

    else:
        print(
            f"--- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")

    # (final_acc, es_acc, best_acc)
    return test_acc, estp_test_acc


def linear_probing_for_inductive_node_classiifcation(model, x, labels, mask, optimizer, max_epoch, device, mute=False):
    if len(labels.shape) > 1:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    train_mask, val_mask, test_mask = mask

    best_val_acc = 0
    best_val_epoch = 0
    best_model = None

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)

        best_val_acc = 0

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)

    for epoch in epoch_iter:
        model.train()
        out = model(None, x)
        loss = criterion(out[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
        optimizer.step()

        with torch.no_grad():
            model.eval()
            pred = model(None, x)
            val_acc = accuracy(pred[val_mask], labels[val_mask])
            val_loss = criterion(pred[val_mask], labels[val_mask])
            test_acc = accuracy(pred[test_mask], labels[test_mask])
            test_loss = criterion(pred[test_mask], labels[test_mask])

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_model = copy.deepcopy(model)

        if not mute:
            epoch_iter.set_description(
                f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_loss:{val_loss.item(): .4f}, val_acc:{val_acc}, test_loss:{test_loss.item(): .4f}, test_acc:{test_acc: .4f}")

    best_model.eval()
    with torch.no_grad():
        pred = best_model(None, x)
        estp_test_acc = accuracy(pred[test_mask], labels[test_mask])
    if mute:
        print(
            f"# IGNORE: --- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} ")
    else:
        print(
            f"--- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch}")

    return test_acc, estp_test_acc


class LogisticRegression(nn.Module):
    def __init__(self, num_dim, num_class):
        super().__init__()
        self.linear = nn.Linear(num_dim, num_class)

    def forward(self, g, x, *args):
        logits = self.linear(x)
        return logits
