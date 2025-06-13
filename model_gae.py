import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, features, hidden_features, dropout_p=0.2):
        super().__init__()
        self.fc1    = nn.Linear(features, hidden_features)
        self.fc2    = nn.Linear(hidden_features, features)
        self.activate = nn.ReLU()
        self.dropout  = nn.Dropout(dropout_p)

    def forward(self, x):
        identity = x

        out = self.fc1(x)
        out = self.activate(out)
        out = self.fc2(out)

        out += identity
        return out

class Encoder(nn.Module):
    def __init__(self, input_dim=38, latent_dim=1, dropout_p=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            ResBlock(input_dim, 64, dropout_p), nn.ReLU(),
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 20), nn.ReLU(),
            nn.Linear(20, latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim=1, output_dim=38, dropout_p=0.2):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 20), nn.ReLU(),
            nn.Linear(20, 32), nn.ReLU(),
            nn.Linear(32, output_dim), nn.ReLU(),
            ResBlock(output_dim, 64, dropout_p)
        )

    def forward(self, z):
        return self.decoder(z)

class GAE(nn.Module):
    def __init__(self, input_dim, latent_dim, code_dim):
        super(GAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)
        self.regressor = nn.Linear(latent_dim, code_dim)

    def forward(self, x):
        bn = self.encoder(x)
        recon = self.decoder(bn)
        z = self.regressor(bn)
        return z, recon

    def predict(self, x):
        x = self.encoder(x)
        return self.regressor(x)
