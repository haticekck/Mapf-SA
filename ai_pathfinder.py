import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
import pickle
from torch.utils.data import Dataset, DataLoader

class PathPlannerCNN(nn.Module):
    def __init__(self, hidden_dim=256, num_actions=9):
        super().__init__()
        
        # CNN Encoder
        self.conv1 = nn.Conv2d(5, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # CNN çıktısını hidden state'e project et
        self.enc_to_hidden = nn.Linear(128, hidden_dim)
        self.enc_to_cell   = nn.Linear(128, hidden_dim)
        
        # Action embedding (önceki aksiyonu LSTM'e ver)
        self.action_embed = nn.Embedding(num_actions, 32)
        
        # LSTM: input = action_embed (32d)
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_dim, num_actions)
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
    
    def encode_grid(self, grid_features):
        """Grid → 128d encoding"""
        x = F.relu(self.conv1(grid_features))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.global_pool(x)
        return x.view(x.size(0), -1)  # (batch, 128)
    
    def init_hidden(self, encoding, num_layers=2):
        """
        CNN encoding'den LSTM başlangıç state'i üret
        Returns: (h0, c0) her biri (num_layers, batch, hidden_dim)
        """
        batch = encoding.size(0)
        h0 = torch.tanh(self.enc_to_hidden(encoding))  # (batch, hidden)
        c0 = torch.tanh(self.enc_to_cell(encoding))
        
        # num_layers katman için tekrarla
        h0 = h0.unsqueeze(0).repeat(num_layers, 1, 1)  # (2, batch, hidden)
        c0 = c0.unsqueeze(0).repeat(num_layers, 1, 1)
        return h0, c0
    
    def forward(self, grid_features, action_sequences):
        """
        Teacher forcing ile training forward pass
        
        Args:
            grid_features:    (batch, 5, H, W)
            action_sequences: (batch, max_len)  ← önceki aksiyonlar (shifted)
        
        Returns:
            logits: (batch, max_len, num_actions)
        """
        # 1. Grid encode
        encoding = self.encode_grid(grid_features)  # (batch, 128)
        
        # 2. LSTM hidden state'i CNN'den başlat
        h0, c0 = self.init_hidden(encoding)
        
        # 3. Action sequence embed (önceki aksiyonlar input olarak)
        emb = self.action_embed(action_sequences)  # (batch, max_len, 32)
        
        # 4. LSTM decode
        lstm_out, _ = self.lstm(emb, (h0, c0))  # (batch, max_len, hidden)
        
        # 5. Logits
        logits = self.fc(lstm_out)  # (batch, max_len, num_actions)
        return logits
    
    def predict_path(self, grid_features, start, max_length=150):
        """
        Autoregressive inference: adım adım tahmin
        """
        self.eval()
        with torch.no_grad():
            grid_tensor = torch.FloatTensor(grid_features).unsqueeze(0)
            encoding = self.encode_grid(grid_tensor)
            h, c = self.init_hidden(encoding)
            
            # Başlangıç aksiyonu: 8 (START token olarak kullan)
            prev_action = torch.LongTensor([[8]])
            
            path = [start]
            current = start
            height, width = grid_features.shape[1], grid_features.shape[2]
            
            action_to_delta = {
                0: (0, -1), 1: (0, 1), 2: (1, 0), 3: (-1, 0),
                4: (1, -1), 5: (-1, -1), 6: (1, 1), 7: (-1, 1),
                8: (0, 0),
            }
            
            for _ in range(max_length):
                emb = self.action_embed(prev_action)    # (1, 1, 32)
                out, (h, c) = self.lstm(emb, (h, c))   # Stateful!
                logits = self.fc(out.squeeze(1))        # (1, 9)
                action = torch.argmax(logits, dim=-1).item()
                
                if action == 8:
                    break
                
                dx, dy = action_to_delta[action]
                nx, ny = current[0] + dx, current[1] + dy
                
                if 0 <= nx < width and 0 <= ny < height:
                    path.append((nx, ny))
                    current = (nx, ny)
                else:
                    break
                
                prev_action = torch.LongTensor([[action]])
        
        return path


class AIPathFinder:
    def __init__(self, grid, model_path: Optional[str] = None):
        self.grid = grid
        self.model = PathPlannerCNN(hidden_dim=256, num_actions=9)
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Model yüklendi: {model_path}")
    
    def find_path(self, start, goal, grid=None) -> Optional[List[Tuple[int, int]]]:
        from data_collection import TrainingDataCollector
        
        if grid is None:
            grid = self.grid
        
        collector = TrainingDataCollector()
        features = collector.grid_to_features(grid, start, goal)
        path = self.model.predict_path(features, start, max_length=200)
        
        if len(path) > 0 and path[-1] == goal:
            return path
        else:
            from a_star import AStarPathFinder
            fallback = AStarPathFinder(grid)
            return fallback.find_path(start, goal)
    
    def find_initial_paths(self, grid, agents):
        paths = []
        success_count = 0
        
        for start, goal in agents:
            path = self.find_path(start, goal)
            if path is None:
                paths.append([start])
            else:
                paths.append(path)
                if path[-1] == goal:
                    success_count += 1
        
        print(f"\nBaşarı Oranı: {success_count}/{len(agents)} (%{success_count*100/len(agents):.1f})")
        return paths

class PathDataset(Dataset):
        def __init__(self, features, actions):
            self.features = features
            self.actions = actions
            self.max_len = 150
        
        def __len__(self):
            return len(self.features)
        
        def __getitem__(self, idx):
            feature = torch.FloatTensor(self.features[idx])
            
            action_seq = self.actions[idx][:self.max_len]
            # Pad with GOAL (8)
            pad_len = self.max_len - len(action_seq)
            action_seq = action_seq + [8] * pad_len
            
            targets = torch.LongTensor(action_seq)           # (max_len,)
            
            # --- Teacher forcing input ---
            # Input: [START=8, a0, a1, ..., a_(T-2)]  (shifted right by 1)
            inp = [8] + action_seq[:-1]
            inputs = torch.LongTensor(inp)                   # (max_len,)
            
            return feature, inputs, targets

def train_model(training_data_path: str,
                epochs: int = 50,
                batch_size: int = 32,
                lr: float = 0.001,
                save_path: str = "ai_pathfinder_model.pth"):
    
    print(f"Training data yükleniyor: {training_data_path}")
    with open(training_data_path, 'rb') as f:
        data = pickle.load(f)
    
    dataset = PathDataset(data['features'], data['actions'])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    model = PathPlannerCNN(hidden_dim=256, num_actions=9)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # --- GOAL token'a daha düşük ağırlık ver (bias engelle) ---
    action_counts = [0] * 9
    for acts in data['actions']:
        for a in acts:
            action_counts[a] += 1
    total = sum(action_counts)
    # Inverse frequency weighting, GOAL'u yarıya indir
    weights = [total / (9 * max(c, 1)) for c in action_counts]
    weights[8] *= 0.3  # GOAL bias'ını kır
    class_weights = torch.FloatTensor(weights)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
    
    print(f"\nTraining başlıyor")
    print(f"  Samples: {len(dataset)}, Epochs: {epochs}, Batch: {batch_size}")
    
    import os
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for features, inputs, targets in dataloader:
            # inputs: teacher forcing (shifted actions)
            logits = model(features, inputs)          # (batch, max_len, 9)
            
            loss = criterion(
                logits.view(-1, 9),
                targets.view(-1)
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Grad clipping
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path)
        
        elif (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path)
    
    print(f"\nEn iyi loss: {best_loss:.4f}")
    print(f"Model kaydedildi: {save_path}")


if __name__ == "__main__":
    train_model(
        training_data_path="training_data/astar_training_aug.pkl",
        epochs=50,
        batch_size=32,
        save_path="models/ai_pathfinder.pth"
    )
    