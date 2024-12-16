import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GRU, Dropout, Linear, ReLU, Sequential
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.utils import dropout_adj


class GilmerNet(torch.nn.Module):
    def __init__(
        self,
        n_node_features,
        n_edge_features,
        dim=64,
        set2set_steps=3,
        n_atom_jumps=3,
        aggr_function="mean",
    ):
        super().__init__()

        self.n_atom_jumps = n_atom_jumps

        self.lin0 = torch.nn.Linear(n_node_features, dim)

        nn = Sequential(Linear(n_edge_features, 128), ReLU(), Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr=aggr_function)
        self.gru = GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=set2set_steps)
        self.lin1 = torch.nn.Linear(2 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, 1)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(self.n_atom_jumps):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out.view(-1)


class GilmerNetGraphLevelFeatures(nn.Module):
    def __init__(
        self,
        n_node_features,
        n_edge_features,
        n_graph_features,
        dim=64,
        set2set_steps=3,
        n_atom_jumps=3,
        aggr_function="mean",
    ):
        super().__init__()

        self.n_atom_jumps = n_atom_jumps

        self.lin0 = torch.nn.Linear(n_node_features, dim)

        nn = Sequential(Linear(n_edge_features, 128), ReLU(), Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr=aggr_function)
        self.gru = GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=set2set_steps)
        self.lin1 = torch.nn.Linear(2 * dim + n_graph_features, dim)
        self.lin2 = torch.nn.Linear(dim, 1)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(self.n_atom_jumps):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)

        # concatenate graph features to embedding vector
        batch_size = len(np.unique(data.batch.cpu().detach().numpy()))
        graph_attr = data.graph_attr.reshape((batch_size, -1))
        out = torch.cat((out, graph_attr), dim=1)

        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out.view(-1)


class GilmerNetGraphLevelFeaturesLayerNorm(torch.nn.Module):
    def __init__(
        self,
        n_node_features,
        n_edge_features,
        n_graph_features,
        dim=64,
        set2set_steps=3,
        n_atom_jumps=3,
        aggr_function="mean",
    ):
        super().__init__()

        self.n_atom_jumps = n_atom_jumps

        self.lin0 = torch.nn.Linear(n_node_features, dim)

        nn = Sequential(Linear(n_edge_features, 128), ReLU(), Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr=aggr_function)
        self.gru = GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=set2set_steps)
        self.lin1 = torch.nn.Linear(2 * dim + n_graph_features, dim)
        self.lin2 = torch.nn.Linear(dim, 1)

        self.layer_norm = torch.nn.LayerNorm(dim)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(self.n_atom_jumps):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
            out = self.layer_norm(out)

        out = self.set2set(out, data.batch)

        # concatenate graph features to embedding vector
        batch_size = len(np.unique(data.batch.cpu().detach().numpy()))
        graph_attr = data.graph_attr.reshape((batch_size, -1))
        out = torch.cat((out, graph_attr), dim=1)

        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out.view(-1)


class GilmerNetGraphLevelFeaturesEdgeDropout(torch.nn.Module):
    def __init__(
        self,
        n_node_features,
        n_edge_features,
        n_graph_features,
        dim=64,
        set2set_steps=3,
        n_atom_jumps=3,
        aggr_function="mean",
        dropout_rate=0,
        force_undirected=True,
    ):
        super().__init__()

        self.n_atom_jumps = n_atom_jumps

        self.lin0 = torch.nn.Linear(n_node_features, dim)

        nn = Sequential(Linear(n_edge_features, 128), ReLU(), Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr=aggr_function)
        self.gru = GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=set2set_steps)
        self.lin1 = torch.nn.Linear(2 * dim + n_graph_features, dim)
        self.lin2 = torch.nn.Linear(dim, 1)

        self.dropout_rate = dropout_rate
        self.dropout_force_undirected = force_undirected

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(self.n_atom_jumps):
            data.edge_index, data.edge_attr = dropout_adj(
                data.edge_index,
                data.edge_attr,
                p=self.dropout_rate,
                force_undirected=self.dropout_force_undirected,
                training=self.training,
            )
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)

        # concatenate graph features to embedding vector
        batch_size = len(np.unique(data.batch.cpu().detach().numpy()))
        graph_attr = data.graph_attr.reshape((batch_size, -1))
        out = torch.cat((out, graph_attr), dim=1)

        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out.view(-1)


class GilmerNetGraphLevelFeaturesDropout(torch.nn.Module):
    def __init__(
        self,
        n_node_features,
        n_edge_features,
        n_graph_features,
        dim=64,
        set2set_steps=3,
        n_atom_jumps=3,
        aggr_function="mean",
        dropout_rate=0,
    ):
        super().__init__()

        self.n_atom_jumps = n_atom_jumps

        self.lin0 = torch.nn.Linear(n_node_features, dim)

        nn = Sequential(Linear(n_edge_features, 128), ReLU(), Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr=aggr_function)
        self.gru = GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=set2set_steps)
        self.lin1 = torch.nn.Linear(2 * dim + n_graph_features, dim)
        self.lin2 = torch.nn.Linear(dim, 1)

        self.dropout_layer = Dropout(p=dropout_rate)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(self.n_atom_jumps):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)

        # concatenate graph features to embedding vector
        batch_size = len(np.unique(data.batch.cpu().detach().numpy()))
        graph_attr = data.graph_attr.reshape((batch_size, -1))
        out = torch.cat((out, graph_attr), dim=1)

        out = F.relu(self.lin1(out))
        out = self.dropout_layer(out)
        out = self.lin2(out)
        return out.view(-1)


### Additional models ###


# MY CUSTOM NETWORKS
class GilmerNet_light(torch.nn.Module):
    """Copy of the Gilmer model."""

    def __init__(
        self,
        n_node_features,
        n_edge_features,
        dim=64,
        set2set_steps=3,
        n_atom_jumps=3,
        aggr_function="mean",
    ):
        super().__init__()

        self.n_atom_jumps = n_atom_jumps

        self.lin0 = torch.nn.Linear(n_node_features, dim)

        nn = Sequential(Linear(n_edge_features, 128), ReLU(), Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr=aggr_function)
        self.gru = GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=set2set_steps)
        self.lin1 = torch.nn.Linear(2 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, 1)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(self.n_atom_jumps):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out.view(-1)


class Wrapper_GilmerNet(pl.LightningModule):
    """Wrapper for the model that is passed to pytorch_lightning."""

    def __init__(self, batch_size, **model_kwargs):
        super().__init__()

        # Saving hyperparameters
        self.save_hyperparameters()

        self.model = GilmerNet_light(**model_kwargs)

        # Log model architechture

        self.loss_module = F.mse_loss
        self.batch_size = batch_size

    def forward(self, data, mode="train"):
        x = self.model(data)
        x = x.squeeze(dim=-1)

        preds = x.float()
        data.y = data.y.float()

        loss = self.loss_module(x, data.y)

        return loss, preds

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=0.001
        )  # High lr because of small dataset and small model
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.7, patience=5, min_lr=0.00001
                ),
                "monitor": "val_loss",
            },
        }

    def training_step(self, batch, batch_idx):
        loss, _ = self.forward(batch)
        self.log("train_loss", loss, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _ = self.forward(batch)
        self.log("val_loss", loss, batch_size=self.batch_size)

    def test_step(self, batch, batch_idx):
        loss, _ = self.forward(batch)
        self.log("test_loss", loss, batch_size=self.batch_size)

    def predict(self, data):
        x = self.model(data)
        x = x.squeeze(dim=-1)
        preds = x.float()
        return preds


class MolecularNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MolecularNN, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.double()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out).squeeze()
        return out
