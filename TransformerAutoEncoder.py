import torch
import torch.nn as nn

class VariableEncoder(nn.Module):
    def __init__(self, var_label, num_vars=38, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        self.var_label = var_label

        self.var_name_embedding = nn.Embedding(num_vars, d_model)
        self.var_value_embedding = nn.Linear(1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """
        x: (B, 38)
        Returns: (B, 38, d_model)
        """
        # Embed value and name for attention
        var_lab = torch.tile(self.var_label, (x.size(0), 1))    # (B, 38, d_model)
        name_emb = self.var_name_embedding(var_lab)             # (B, 38, d_model)
        value_emb = self.var_value_embedding(x.unsqueeze(-1))   # (B, 38, d_model)
        encoder_input = name_emb + value_emb                    # (B, 38, d_model)

        memory = self.transformer_encoder(encoder_input)        # (B, 38, d_model)
        return memory


class VariableDecoder(nn.Module):
    def __init__(self, var_label, num_vars=38, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        self.var_label = var_label

        self.var_name_embedding = nn.Embedding(num_vars, d_model)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, memory):
        """
        memory: Output from encoder (B, 38, d_model)
        Returns: (B, 38, d_model)
        """
        var_lab = torch.tile(self.var_label, (memory.size(0), 1))       # (B, 38, d_model)
        name_emb = self.var_name_embedding(var_lab)                     # (B, 38, d_model)

        output = self.transformer_decoder(tgt=name_emb, memory=memory)  # (B, 38, d_model)
        return output


class TransformerAE(nn.Module):
    def __init__(self, var_label, num_vars=38, d_model=128, latent_dim = 8, nhead=8, num_layers=2):
        super().__init__()
        self.var_label = var_label

        self.encoder = VariableEncoder(var_label=self.var_label, num_vars=num_vars, d_model=d_model, nhead=nhead, num_layers=num_layers)
        self.decoder = VariableDecoder(var_label=self.var_label, num_vars=num_vars, d_model=d_model, nhead=nhead, num_layers=num_layers)

        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.latent_proj = nn.Linear(d_model, latent_dim)
        self.regressor = nn.Linear(latent_dim, 1)

        self.expand_proj = nn.Linear(latent_dim, d_model * num_vars)
        self.output_layer = nn.Linear(d_model, 1)


    def forward(self, x):
        """
        x: (B, 38)
        Returns: (B, 38)
        """
        memory = self.encoder(x)                                          # (B, 38, d_model)
        memory_pooled = self.pooling(memory.transpose(1, 2)).squeeze(-1)  # (B, d_model)
        bn = self.latent_proj(memory_pooled)                              # (B, latent_dim)

        latent = self.regressor(bn).squeeze(-1)

        expand = self.expand_proj(bn).view(x.size(0), 38, -1)             # (B, 38, d_model)
        output = self.decoder(expand)                                     # (B, 38, d_model)
        recons = self.output_layer(output).squeeze(-1)                    # (B, 38)
        return latent, recons
    
    def predict(self, x):
        memory = self.encoder(x)                                          # (B, 38, d_model)
        memory_pooled = self.pooling(memory.transpose(1, 2)).squeeze(-1)  # (B, d_model)
        bn = self.latent_proj(memory_pooled)                              # (B, latent_dim)

        latent = self.regressor(bn)
        return latent
