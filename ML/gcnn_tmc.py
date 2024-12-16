"""Module containing a GCNN."""

import pytorch_lightning as pl
import torch
import torch_geometric.nn as geom_nn
from torch import nn


class edge_NN(nn.Module):
    def __init__(self, c_in, c_hidden, c_out):
        super().__init__()
        self.NN = nn.Sequential(
            nn.Linear(c_in, c_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(c_hidden, c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.NN(x)
        # print(x.shape)
        return x


class GNNModel(nn.Module):
    def __init__(
        self,
        c_in,
        c_hidden,
        c_out,
        c_edge_in,
        c_edge_hidden,
        num_layers=2,
        layer_name="edge_conv",
        dp_rate=0.1,
        **kwargs,
    ):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        gnn_layer = geom_nn.NNConv

        layers = []
        in_channels, out_channels = c_in, c_hidden
        print(in_channels, out_channels)
        for l_idx in range(num_layers - 1):
            layers += [
                gnn_layer(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    nn=edge_NN(
                        c_in=c_edge_in,
                        c_hidden=c_edge_hidden,
                        c_out=in_channels * out_channels,
                    ),
                    **kwargs,
                ),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate),
            ]
            in_channels = c_hidden
        layers += [
            gnn_layer(
                in_channels=in_channels,
                out_channels=c_out,
                nn=edge_NN(
                    c_in=c_edge_in, c_hidden=c_edge_hidden, c_out=in_channels * c_out
                ),
                **kwargs,
            )
        ]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index, edge_attr):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        for lay in self.layers:
            # For graph layers, we need to add the "edge_index" tensor as additional input
            # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
            # we can simply check the class type.
            if isinstance(lay, geom_nn.MessagePassing):
                # print(x.shape)
                # print(edge_attr.shape)
                x = lay(x, edge_index, edge_attr)
                # print(x.shape)
            else:
                x = lay(x)
        return x


class GraphGNNModel(nn.Module):
    def __init__(
        self,
        c_in,
        c_hidden,
        c_out,
        c_edge_in,
        c_edge_hidden,
        dp_rate_linear=0.5,
        **kwargs,
    ):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of output features (usually number of classes)
            dp_rate_linear - Dropout rate before the linear layer (usually much higher than inside the GNN)
            kwargs - Additional arguments for the GNNModel object
        """
        super().__init__()
        self.GNN = GNNModel(
            c_in=c_in,
            c_hidden=c_hidden,
            c_out=c_hidden,  # Not our prediction output yet!
            c_edge_in=c_edge_in,
            c_edge_hidden=c_edge_hidden,
            **kwargs,
        )
        self.NN = nn.Sequential(
            nn.Dropout(dp_rate_linear),
            nn.Linear(c_hidden, c_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dp_rate_linear),
            nn.Linear(c_hidden, c_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dp_rate_linear),
            nn.Linear(c_hidden, c_hidden),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Dropout(dp_rate_linear), nn.Linear(c_hidden, c_out)
        )

    def forward(self, x, edge_index, edge_attr, batch_idx):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
            batch_idx - Index of batch element for each node
        """
        x = self.GNN(x, edge_index, edge_attr)
        x = geom_nn.global_mean_pool(x, batch_idx)  # Average pooling
        x = self.NN(x)
        x = self.head(x)
        return x


class GraphLevelGNN(pl.LightningModule):
    """Top level model that is used by pytorch_lightning."""

    def __init__(self, batch_size, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()

        self.model = GraphGNNModel(**model_kwargs)
        self.loss_module = nn.L1Loss()
        self.batch_size = batch_size

    def forward(self, data, mode="train"):
        x, edge_index, edge_attr, batch_idx = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )
        x = self.model(x, edge_index, edge_attr, batch_idx)
        x = x.squeeze(dim=-1)

        preds = x.float()
        data.y = data.y.float()

        loss = self.loss_module(x, data.y)

        return loss, preds

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=1e-4, weight_decay=0.1
        )  # High lr because of small dataset and small model
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, _ = self.forward(batch, mode="train")
        self.log("train_loss", loss, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _ = self.forward(batch, mode="val")
        self.log("val_loss", loss, batch_size=self.batch_size)

    def test_step(self, batch, batch_idx):
        loss, preds = self.forward(batch, mode="test")
        self.log("test_loss", loss, batch_size=self.batch_size)

    def predict(self, data, mode="test"):
        x, edge_index, batch_idx = data.x, data.edge_index, data.batch
        x = self.model(x, edge_index, batch_idx)
        x = x.squeeze(dim=-1)
        preds = x.float()
        return preds
