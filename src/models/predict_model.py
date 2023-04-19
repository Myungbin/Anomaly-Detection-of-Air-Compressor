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
        )
        self.skip = nn.Linear(input_dim, latent_dim)

    def forward(self, x):
        encoded = self.Encoder(x)
        skip = self.skip(x)
        add = encoded + skip
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
            nn.Conv1d(in_channels=1, out_channels=64,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=32,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=16,
                      kernel_size=3, padding=1),
            nn.ReLU()
        )
        # decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=16, out_channels=32,
                               kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=32, out_channels=64,
                               kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=64, out_channels=1,
                               kernel_size=3, padding=1),
            nn.Flatten(start_dim=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = torch.unsqueeze(x, 1)  # add channel dimension
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size, padding=padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = self.relu(x)
        x = x + residual
        return x


class ResidualConv1DAutoencoder(nn.Module):
    def __init__(self):
        super(ResidualConv1DAutoencoder, self).__init__()
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            ResidualBlock(128, 128, kernel_size=3, padding=1),
            nn.Conv1d(in_channels=128, out_channels=64,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            ResidualBlock(64, 64, kernel_size=3, padding=1),
            nn.Conv1d(in_channels=64, out_channels=32,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            ResidualBlock(32, 32, kernel_size=3, padding=1)
        )
        # decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=32, out_channels=64,
                               kernel_size=3, padding=1),
            nn.ReLU(),
            ResidualBlock(64, 64, kernel_size=3, padding=1),
            nn.ConvTranspose1d(in_channels=64, out_channels=128,
                               kernel_size=3, padding=1),
            nn.ReLU(),
            ResidualBlock(128, 128, kernel_size=3, padding=1),
            nn.ConvTranspose1d(in_channels=128, out_channels=1,
                               kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = torch.unsqueeze(x, 1)  # add channel dimension
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.squeeze(1)  # remove channel dimension

        return x


class SelfAttention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        attention_weights = torch.softmax(torch.matmul(
            query, key.transpose(-2, -1)) / (x.shape[-1] ** 0.5), dim=-1)
        attended_values = torch.matmul(attention_weights, value)

        return attended_values


class SelfAttentionConv1DAutoencoder(nn.Module):
    def __init__(self):
        super(SelfAttentionConv1DAutoencoder, self).__init__()
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.self_attention_enc = SelfAttention(16, 8)  # Self-Attention layer for encoder
        # decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=64, out_channels=1, kernel_size=3, padding=1),
            nn.Tanh()
        )
        self.self_attention_dec = SelfAttention(1, 1)  # Self-Attention layer for decoder

    def forward(self, x):
        x = torch.unsqueeze(x, 1)  # add channel dimension
        x = self.encoder(x)
        x = x.permute(0, 2, 1)  # Permute to apply self-attention (B, C, L) -> (B, L, C)
        x = self.self_attention_enc(x)
        x = x.permute(0, 2, 1)  # Permute back (B, L, C) -> (B, C, L)
        x = self.decoder(x)
        x = x.permute(0, 2, 1)  # Permute to apply self-attention (B, C, L) -> (B, L, C)
        x = self.self_attention_dec(x)
        x = x.permute(0, 2, 1)  # Permute back (B, L, C) -> (B, C, L)
        x = x.squeeze(1)  # remove channel dimension
        return x


class DilationConv1DAutoencoder(nn.Module):
    def __init__(self):
        super(DilationConv1DAutoencoder, self).__init__()
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=4, dilation=4),
            nn.ReLU()
        )
        # decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=16, out_channels=32, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=64, out_channels=1, kernel_size=3, padding=1, dilation=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = torch.unsqueeze(x, 1)  # add channel dimension
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.squeeze(1)  # remove channel dimension
        return x


class Conv1DLSTMAutoencoder(nn.Module):
    def __init__(self):
        super(Conv1DLSTMAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # LSTM-based bottleneck
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=3, batch_first=True)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=128, out_channels=1, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = torch.unsqueeze(x, 1)  # add channel dimension
        x = self.encoder(x)
        x = x.permute(0, 2, 1)  # Permute to apply LSTM (B, C, L) -> (B, L, C)

        # Apply LSTM in the bottleneck
        x, _ = self.lstm(x)
        x = x.permute(0, 2, 1)  # Permute back (B, L, C) -> (B, C, L)

        x = self.decoder(x)
        x = x.squeeze(1)  # remove channel dimension
        return x


class PowerfulConv1DLSTMAutoencoder(nn.Module):
    def __init__(self):
        super(PowerfulConv1DLSTMAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # LSTM-based bottleneck
        self.lstm = nn.LSTM(input_size=32, hidden_size=32, num_layers=2, batch_first=True, dropout=0.5)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=128, out_channels=1, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = torch.unsqueeze(x, 1)  # add channel dimension
        x = self.encoder(x)
        x = x.permute(0, 2, 1)  # Permute to apply LSTM (B, C, L) -> (B, L, C)

        # Apply LSTM in the bottleneck
        x, _ = self.lstm(x)
        x = x.permute(0, 2, 1)  # Permute back (B, L, C) -> (B, C, L)

        x = self.decoder(x)
        x = x.squeeze(1)  # remove channel dimension
        return x    
    
    

class ResidualConv1DLSTMAutoencoder(nn.Module):
    def __init__(self):
        super(ResidualConv1DLSTMAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, padding=1),
            ResidualBlock(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            ResidualBlock(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            ResidualBlock(32, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(input_size=32, hidden_size=32, num_layers=3, batch_first=True, dropout=0.5)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            ResidualBlock(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            ResidualBlock(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=128, out_channels=1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = torch.unsqueeze(x, 1)  # add channel dimension
        x = self.encoder(x)
        x = x.permute(0, 2, 1)  # (B, C, L) -> (B, L, C)

        # Apply LSTM in the bottleneck
        x, _ = self.lstm(x)
        x = x.permute(0, 2, 1)  # (B, L, C) -> (B, C, L)

        x = self.decoder(x)
        x = x.squeeze(1)  # remove channel dimension
        return x    
    