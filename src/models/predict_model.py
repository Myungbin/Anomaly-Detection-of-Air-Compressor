import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=input_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        encoded, _ = self.encoder(x)
        latent = self.fc(encoded)
        decoded, _ = self.decoder(latent)
        return decoded


class High_Dim_LSTM_Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, latent_size):
        super(High_Dim_LSTM_Autoencoder, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size,
                               num_layers, batch_first=True)
        self.latent = nn.Linear(hidden_size, latent_size)
        self.decoder = nn.LSTM(latent_size, hidden_size,
                               num_layers, batch_first=True)
        self.output = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        encoded, _ = self.encoder(x)
        encoded = self.latent(encoded[:, -1, :])
        decoded, _ = self.decoder(
            encoded.unsqueeze(1).repeat(1, x.shape[1], 1))
        output = self.output(decoded)
        return output


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super(AutoEncoder, self).__init__()

        self.Encoder = nn.Sequential(
            nn.Linear(input_dim, 1000),
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(),
            nn.Linear(1000, 2000),
            nn.BatchNorm1d(2000),
            nn.LeakyReLU(),
            nn.Linear(2000, latent_dim),
        )
        self.Decoder = nn.Sequential(
            nn.Linear(latent_dim, 2000),
            nn.BatchNorm1d(2000),
            nn.LeakyReLU(),
            nn.Linear(2000, 1000),
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(),
            nn.Linear(1000, input_dim),
        )

    def forward(self, x):
        x = self.Encoder(x)
        x = self.Decoder(x)
        return x
