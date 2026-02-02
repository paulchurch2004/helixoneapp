# 📘 MODULE 3: DEEP LEARNING POUR LA FINANCE
## Réseaux de Neurones Appliqués aux Marchés

---

## 📚 SOURCES PRINCIPALES
- **NYU FRE-7871 Advanced Deep Learning**: https://engineering.nyu.edu/sites/default/files/2023-07/Syllabus_FRE_7871_Perry.pdf
- **Deep Learning Book (Goodfellow)**: https://www.deeplearningbook.org/
- **Stanford CS231n**: http://cs231n.stanford.edu/

---

## 🎯 APPLICATIONS EN FINANCE

| Application | Architecture | Input | Output |
|-------------|--------------|-------|--------|
| Price prediction | LSTM/Transformer | Time series | Returns |
| LOB modeling | CNN/Attention | Order book snapshots | Mid-price move |
| News analysis | BERT/GPT | Text | Sentiment/signal |
| Option pricing | MLP/Physics-informed | Greeks, vol surface | Price |
| Risk assessment | Autoencoder | Historical scenarios | Anomaly score |

---

## 🧠 ARCHITECTURES FONDAMENTALES

### 1. LSTM pour Time Series

```python
import torch
import torch.nn as nn

class FinancialLSTM(nn.Module):
    """
    LSTM for financial time series with attention
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout, bidirectional=False
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softmax(dim=1)
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)
        
        # Attention weights
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        
        # Weighted sum
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden_dim)
        
        output = self.fc(context)
        return output, attn_weights.squeeze(-1)
```

### 2. Transformer pour Séquences Financières

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class FinancialTransformer(nn.Module):
    """
    Transformer encoder for financial time series
    """
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=3, 
                 dim_feedforward=256, dropout=0.1, max_len=500):
        super().__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_fc = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output_fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, x, mask=None):
        # x: (batch, seq_len, input_dim)
        x = self.input_fc(x) * np.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Causal mask for autoregressive prediction
        if mask is None:
            mask = self._generate_causal_mask(x.size(1)).to(x.device)
        
        x = self.transformer_encoder(x, mask)
        
        # Use last position for prediction
        x = self.output_fc(x[:, -1, :])
        return x
    
    def _generate_causal_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask
```

### 3. CNN pour Order Book

```python
class LOBNet(nn.Module):
    """
    CNN for Limit Order Book data
    Based on DeepLOB architecture
    """
    def __init__(self, num_levels=10, num_features=4):
        super().__init__()
        
        # Input: (batch, seq_len, num_levels, features)
        # features: bid_price, bid_volume, ask_price, ask_volume
        
        # Convolutional layers (spatial)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2))
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(4, 1))
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(4, 1))
        
        # Inception modules
        self.inception1 = InceptionModule(32, 64)
        self.inception2 = InceptionModule(64, 64)
        self.inception3 = InceptionModule(64, 64)
        
        # LSTM for temporal
        self.lstm = nn.LSTM(64, 64, num_layers=1, batch_first=True)
        
        # Output
        self.fc = nn.Linear(64, 3)  # Up, Down, Stationary
    
    def forward(self, x):
        # x: (batch, seq_len, num_levels, features)
        batch_size, seq_len = x.shape[:2]
        
        # Process each timestep
        outputs = []
        for t in range(seq_len):
            xt = x[:, t:t+1, :, :]  # (batch, 1, levels, features)
            xt = xt.permute(0, 1, 3, 2)  # (batch, 1, features, levels)
            
            # CNN
            xt = F.leaky_relu(self.conv1(xt))
            xt = F.leaky_relu(self.conv2(xt))
            xt = F.leaky_relu(self.conv3(xt))
            
            # Inception
            xt = self.inception1(xt)
            xt = self.inception2(xt)
            xt = self.inception3(xt)
            
            xt = xt.mean(dim=(-1, -2))  # Global pooling
            outputs.append(xt)
        
        # Stack and LSTM
        x = torch.stack(outputs, dim=1)  # (batch, seq_len, 64)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels//4, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels//4, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels//4, kernel_size=5, padding=2)
        self.pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels//4, kernel_size=1)
        )
    
    def forward(self, x):
        return torch.cat([
            self.conv1(x), self.conv2(x), self.conv3(x), self.pool(x)
        ], dim=1)
```

### 4. Autoencoder pour Anomaly Detection

```python
class VariationalAutoencoder(nn.Module):
    """
    VAE for market regime detection and anomaly scoring
    """
    def __init__(self, input_dim, latent_dim=10, hidden_dims=[64, 32]):
        super().__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU()
            ])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        hidden_dims_rev = hidden_dims[::-1]
        prev_dim = latent_dim
        for h_dim in hidden_dims_rev:
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU()
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var
    
    def loss_function(self, x, x_recon, mu, log_var, beta=1.0):
        """
        VAE loss = Reconstruction + KL divergence
        """
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + beta * kl_loss
    
    def anomaly_score(self, x):
        """
        Compute anomaly score (reconstruction error)
        """
        with torch.no_grad():
            x_recon, mu, log_var = self.forward(x)
            score = F.mse_loss(x_recon, x, reduction='none').mean(dim=1)
        return score
```

---

## 🏋️ TECHNIQUES D'ENTRAÎNEMENT

### 1. Data Augmentation pour Finance

```python
def augment_financial_data(X, y, noise_level=0.01, shift_range=5):
    """
    Data augmentation for financial time series
    """
    augmented_X = [X]
    augmented_y = [y]
    
    # Add noise
    noisy_X = X + np.random.normal(0, noise_level * X.std(), X.shape)
    augmented_X.append(noisy_X)
    augmented_y.append(y)
    
    # Time shift (window sliding)
    for shift in range(-shift_range, shift_range + 1):
        if shift == 0:
            continue
        shifted_X = np.roll(X, shift, axis=0)
        augmented_X.append(shifted_X[abs(shift):])
        augmented_y.append(y[abs(shift):])
    
    return np.vstack(augmented_X), np.hstack(augmented_y)
```

### 2. Training Loop avec Early Stopping

```python
def train_model(model, train_loader, val_loader, epochs=100, 
                lr=1e-3, patience=20, device='cuda'):
    """
    Training loop with validation-based early stopping
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            output = model(X_batch)
            loss = F.mse_loss(output.squeeze(), y_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                val_loss += F.mse_loss(output.squeeze(), y_batch).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train={train_loss:.6f}, Val={val_loss:.6f}")
    
    model.load_state_dict(torch.load('best_model.pt'))
    return model, history
```

---

## 📊 INTERPRÉTABILITÉ

### Attention Visualization

```python
def visualize_attention(model, X_sample, feature_names):
    """
    Visualize attention weights for interpretability
    """
    model.eval()
    with torch.no_grad():
        output, attention = model(X_sample.unsqueeze(0))
    
    attention = attention.squeeze().numpy()
    
    plt.figure(figsize=(12, 4))
    plt.imshow(attention.reshape(1, -1), aspect='auto', cmap='Reds')
    plt.colorbar(label='Attention Weight')
    plt.xlabel('Time Step')
    plt.title('Attention Weights Over Time')
    plt.show()
```

---

## 🔗 RÉFÉRENCES
1. Zhang, Z. et al. (2019). DeepLOB: Deep Convolutional Neural Networks for Limit Order Books
2. Sirignano, J. & Cont, R. (2019). Universal Features of Price Formation in Financial Markets
3. Buehler, H. et al. (2019). Deep Hedging
