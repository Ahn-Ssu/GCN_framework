import numpy
import random
import pandas
import torch
import xgboost as xgb
import utils.mol_dml
import utils.chem as chem
from torch_geometric.data import DataLoader
from sklearn.metrics import r2_score
from utils.models import GCN


# Experiment settings
dataset_name = 'esol'
batch_size = 32
init_lr = 1e-4
l2_coeff = 1e-7
n_epochs = 100
dim_emb = 128


# Load dataset
print('Load molecular structures...')
data = chem.load_dataset('res/data/' + dataset_name + '.xlsx')
random.shuffle(data)
smiles = [x[0] for x in data]
mols = [x[1] for x in data]


# Generate training and test datasets
n_train = int(0.8 * len(data))
train_data = mols[:n_train]
test_data = mols[n_train:]
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
emb_loader = DataLoader(train_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)
train_smiles = numpy.array(smiles[:n_train]).reshape(-1, 1)
test_smiles = numpy.array(smiles[n_train:]).reshape(-1, 1)
train_targets = numpy.array([x.y.item() for x in train_data]).reshape(-1, 1)
test_targets = numpy.array([x.y.item() for x in test_data]).reshape(-1, 1)


# Model configuration
emb_net = GCN(chem.n_atom_feats, dim_emb)
optimizer = torch.optim.Adam(emb_net.parameters(), lr=init_lr, weight_decay=l2_coeff)


# Train GNN-based embedding network
print('Train the GNN-based embedding network...')
for i in range(0, n_epochs):
    train_loss = utils.mol_dml.train(emb_net, optimizer, train_loader)
    print('Epoch [{}/{}]\tTrain loss: {:.4f}'.format(i + 1, n_epochs, train_loss))


# Generate embeddings of the molecules
train_embs = utils.mol_dml.test(emb_net, emb_loader)
test_embs = utils.mol_dml.test(emb_net, test_loader)
train_emb_results = numpy.concatenate([train_embs, train_smiles, train_targets], axis=1).tolist()
test_emb_results = numpy.concatenate([test_embs, test_smiles, test_targets], axis=1).tolist()
df = pandas.DataFrame(train_emb_results)
df.to_excel('res/embs/embs_' + dataset_name + '_train.xlsx', header=None, index=None)
df = pandas.DataFrame(test_emb_results)
df.to_excel('res/embs/embs_' + dataset_name + '_test.xlsx', header=None, index=None)


# Train XGBoost using the molecular embeddings
print('Train the XGBoost regressor...')
model = xgb.XGBRegressor(max_depth=8, n_estimators=300, subsample=0.8)
model.fit(train_embs, train_targets, eval_metric='mae')
preds = model.predict(test_embs).reshape(-1, 1)
test_mae = numpy.mean(numpy.abs(test_targets - preds))
r2 = r2_score(test_targets, preds)
print('Test MAE: {:.4f}\tTest R2 score: {:.4f}'.format(test_mae, r2))


# Save prediction results
pred_results = list()
for i in range(0, preds.shape[0]):
    pred_results.append([test_smiles[i], test_targets[i].item(), preds[i].item()])
df = pandas.DataFrame(pred_results)
df.columns = ['smiles', 'true_y', 'pred_y']
df.to_excel('res/preds/preds_' + dataset_name + '_dml.xlsx', index=False)
