import numpy
import random
import pandas
import torch
import utils.ml
import utils.chem as chem
from torch_geometric.data import DataLoader
from sklearn.metrics import r2_score
from utils.models import GCN
from utils.models import GAT
from utils.models import GIN


# Experiment settings
gnn = 'gcn'
dataset_name = 'esol'
batch_size = 32
init_lr = 1e-4
l2_coeff = 1e-7
n_epochs = 300


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
test_loader = DataLoader(test_data, batch_size=batch_size)
test_smiles = smiles[n_train:]
test_targets = numpy.array([x.y.item() for x in test_data]).reshape(-1, 1)


# Model configuration
if gnn == 'gcn':
    model = GCN(chem.n_atom_feats, 1)
elif gnn == 'gat':
    model = GAT(chem.n_atom_feats, 1)
elif gnn == 'gin':
    model = GIN(chem.n_atom_feats, 1)

optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=l2_coeff)
criterion = torch.nn.L1Loss()


# Train graph neural network (GNN)
print('Train the GNN-based predictor...')
for i in range(0, n_epochs):
    train_loss = utils.ml.train(model, optimizer, train_loader, criterion)
    print('Epoch [{}/{}]\tTrain loss: {:.4f}'.format(i + 1, n_epochs, train_loss))


# Test the trained GNN
preds = utils.ml.test(model, test_loader)
test_mae = numpy.mean(numpy.abs(test_targets - preds))
r2 = r2_score(test_targets, preds)
print('Test MAE: {:.4f}\tTest R2 score: {:.4f}'.format(test_mae, r2))


# Save prediction results
pred_results = list()
for i in range(0, preds.shape[0]):
    pred_results.append([test_smiles[i], test_targets[i].item(), preds[i].item()])
df = pandas.DataFrame(pred_results)
df.columns = ['smiles', 'true_y', 'pred_y']
df.to_excel('res/preds/preds_' + dataset_name + '_' + gnn + '.xlsx', index=False)
