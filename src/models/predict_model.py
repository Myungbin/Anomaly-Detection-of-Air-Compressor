import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=20):
        super(AutoEncoder, self).__init__()

        self.Encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, latent_dim),
        )
        self.Decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, input_dim),
            nn.Tanh()
        )

        self.skip = nn.Linear(input_dim, latent_dim)
                
    def forward(self, x):
        encoded = self.Encoder(x)
        # skip = self.skip(x) 
        # add =  encoded + skip
        decoded = self.Decoder(encoded) 
        return decoded


class DeepAutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super(DeepAutoEncoder, self).__init__()

        self.Encoder = nn.Sequential(
            nn.SiLU(),
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, latent_dim),
        )
        self.Decoder = nn.Sequential(
            nn.SiLU(),
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )
        self.skip = nn.Linear(input_dim, latent_dim)
        
    def forward(self, x):
        encoded = self.Encoder(x)
        skip = self.skip(x) 
        add =  encoded + skip
        decoded = self.Decoder(add) 
        return decoded


class SingleAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SingleAutoEncoder, self).__init__()

        self.Encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.Decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
        )
        
    def forward(self, x):
        x = self.Encoder(x)
        x = self.Decoder(x)
        return x


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VariationalAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar


class EncoderLSTMDecoderCNN(nn.Module):
    def __init__(self, input_channels, output_size, hidden_size, kernel_size):
        super(EncoderLSTMDecoderCNN, self).__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.LSTM(input_channels, hidden_size)
        self.decoder = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Conv1d(hidden_size, output_size, kernel_size),
        )

    def forward(self, input):
        encoder_output, (hidden, cell) = self.encoder(input)
        hidden = hidden.permute(1, 0, 2)
        decoder_output = self.decoder(hidden)
        decoder_output = decoder_output.permute(0, 2, 1)
        return decoder_output


class Conv1DAutoencoder(nn.Module):
    def __init__(self):
        super(Conv1DAutoencoder, self).__init__()
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=128, out_channels=1, kernel_size=3, padding=1),
            nn.Flatten(start_dim=1),
            nn.Tanh()
        )

        
    def forward(self, x):
        x = torch.unsqueeze(x, 1) # add channel dimension
        x = self.encoder(x)
        x = self.decoder(x)
        return x
