import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, features)
        self.activate = nn.ReLU()

    def forward(self, x):
        identity = x

        out = self.fc1(x)
        out = self.activate(out)
        out = self.fc2(out)

        out += identity
        return out

class Encoder(nn.Module):
    def __init__(self, input_dim=38, latent_dim=1):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            ResBlock(input_dim, 128), nn.ReLU(),
            ResBlock(input_dim, 128), nn.ReLU(),
            nn.Linear(input_dim, 16), nn.ReLU(),
            nn.Linear(16, 4), nn.ReLU(),
            nn.Linear(4, latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim=1, output_dim=38):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 4), nn.ReLU(),
            nn.Linear(4, 16), nn.ReLU(),
            nn.Linear(16, output_dim), nn.ReLU(),
            ResBlock(output_dim, 128), nn.ReLU(),
            ResBlock(output_dim, 128)
        )

    def forward(self, z):
        return self.decoder(z)

class GAE(nn.Module):
    def __init__(self, input_dim=38, latent_dim=1):
        super(GAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def forward(self, x):
        bn = self.encoder(x)
        y = self.decoder(bn)
        return bn, y

    def encode(self, x):
        return self.encoder(x)