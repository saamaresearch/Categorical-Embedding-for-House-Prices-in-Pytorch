from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class TabularDataset(Dataset):
  def __init__(self, data, cat_cols=None, output_col=None):

    self.n = data.shape[0]

    if output_col:
      self.y = data[output_col].astype(np.float32).values.reshape(-1, 1)
    else:
      self.y =  np.zeros((self.n, 1))

    self.cat_cols = cat_cols if cat_cols else []
    self.cont_cols = [col for col in data.columns
                      if col not in self.cat_cols + [output_col]]

    if self.cont_cols:
      self.cont_X = data[self.cont_cols].astype(np.float32).values
    else:
      self.cont_X = np.zeros((self.n, 1))

    if self.cat_cols:
      self.cat_X = data[cat_cols].astype(np.int64).values
    else:
      self.cat_X =  np.zeros((self.n, 1))

  def __len__(self):

    return self.n

  def __getitem__(self, idx):

    return [self.y[idx], self.cont_X[idx], self.cat_X[idx]]


class FeedForwardNN(nn.Module):
    def __init__(self, emb_dims, no_of_cont, lin_layer_sizes,
                 output_size, emb_dropout, lin_layer_dropouts):
        super().__init__()

        self.emb_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims])

        no_of_embs = sum([y for x,y in emb_dims])
        self.no_of_embs = no_of_embs
        self.no_of_cont = no_of_cont

        first_lin_layer = nn.Linear(self.no_of_embs + self.no_of_cont,
                                    lin_layer_sizes[0])

        self.lin_layers = nn.ModuleList([first_lin_layer] +
                                        [nn.Linear(lin_layer_sizes[i], lin_layer_sizes[i + 1])
                                         for i in range(len(lin_layer_sizes) - 1)])

        for lin_layer in self.lin_layers:
            nn.init.kaiming_normal_(lin_layer.weight.data)

        self.output_layer = nn.Linear(lin_layer_sizes[-1], output_size)
        nn.init.kaiming_normal_(self.output_layer.weight.data)

        self.first_bn_layer = nn.BatchNorm1d(self.no_of_cont)
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(size) for size in lin_layer_sizes])

        self.emb_dropout_layer = nn.Dropout(emb_dropout)
        self.dropout_layers = nn.ModuleList([nn.Dropout(size)
                                             for size in lin_layer_dropouts])

    def forward(self, cont_data, cat_data):
        if self.no_of_embs != 0:
            x = [emb_layer(cat_data[:, i])
                 for i, emb_layer in enumerate(self.emb_layers)]
            x = torch.cat(x, 1)
            x = self.emb_dropout_layer(x)

        if self.no_of_cont != 0:
            normalized_cont_data = self.first_bn_layer(cont_data)

            if self.no_of_embs != 0:
                x = torch.cat([x, normalized_cont_data], 1)
            else:
                x = normalized_cont_data

        for lin_layer, dropout_layer, bn_layer in zip(self.lin_layers, self.dropout_layers, self.bn_layers):
            x = F.relu(lin_layer(x))
            x = bn_layer(x)
            x = dropout_layer(x)

        x = self.output_layer(x)

        return x

data = pd.read_csv("house prices/train.csv", usecols=["SalePrice", "MSSubClass", "MSZoning", "LotFrontage", "LotArea",
                                         "Street", "YearBuilt", "LotShape", "1stFlrSF", "2ndFlrSF"]).dropna()

categorical_features = ["MSSubClass", "MSZoning", "Street", "LotShape", "YearBuilt"]
output_feature = "SalePrice"

label_encoders = {}

for cat_col in categorical_features:
    label_encoders[cat_col] = LabelEncoder()
    data[cat_col] = label_encoders[cat_col].fit_transform(data[cat_col])

dataset = TabularDataset(data=data, cat_cols=categorical_features,
                             output_col=output_feature)
batchsize = 64
dataloader = DataLoader(dataset, batchsize, shuffle=True, num_workers=1)
cat_dims = [int(data[col].nunique()) for col in categorical_features]
emb_dims = [(x, min(50, (x + 1) // 2)) for x in cat_dims]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FeedForwardNN(emb_dims, no_of_cont=4, lin_layer_sizes=[50, 100],
                          output_size=1, emb_dropout=0.04,
                          lin_layer_dropouts=[0.001,0.01]).to(device)

no_of_epochs = 5
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for epoch in range(no_of_epochs):
      for y, cont_x, cat_x in dataloader:
        cat_x = cat_x.to(device)
        cont_x = cont_x.to(device)
        y  = y.to(device)

        # Forward Pass
        preds = model(cont_x, cat_x)
        loss = criterion(preds, y)

        # Backward Pass and Optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

      print(loss)
