import torch

def GPU_check():
    MODEL_NAME = 'GNN' 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("MODEL_NAME = {}, DEVICE = {}".format(MODEL_NAME, DEVICE))
    
    return DEVICE


def train(model, optimizer, data_loader, criterion):
    model.train()
    train_loss = 0

    for i, (batch) in enumerate(data_loader):
        pred = model(batch)
        loss = criterion(batch.y, pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.detach().item()

    return train_loss / len(data_loader)


def test(model, data_loader):
    model.eval()
    list_preds = list()

    with torch.no_grad():
        for batch in data_loader:
            preds = model(batch)
            list_preds.append(preds)

    return torch.cat(list_preds, dim=0).cpu().numpy()
