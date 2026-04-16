import numpy as np
import pickle
import os
from typing import List, Tuple, Dict
from a_star import AStarPathFinder
from dijkstra import DijkstraPathFinder
import random

class TrainingDataCollector:
    """
    AI model için training data toplama
    
    Data format: (grid, start, goal) → expert_path
    """
    def __init__(self, output_dir: str = "training_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def grid_to_features(self, grid: List[List[str]], start: Tuple[int, int], goal: Tuple[int, int]) -> np.ndarray:
        """
        Grid'i neural network için feature'lara çevir
        
        Returns:
            (channels, height, width) tensor
        """
        height = len(grid)
        width = len(grid[0])
        
        # 5 channel:
        # 0: Obstacle map (1 = engel, 0 = boş)
        # 1: Start position (1 = start, 0 = diğer)
        # 2: Goal position (1 = goal, 0 = diğer)
        # 3: Distance to goal (normalized)
        # 4: Distance to start (normalized)
        
        features = np.zeros((5, height, width), dtype=np.float32)
        
        # Channel 0: Obstacles
        for y in range(height):
            for x in range(width):
                if grid[y][x] == '@':
                    features[0, y, x] = 1.0
        
        # Channel 1: Start
        features[1, start[1], start[0]] = 1.0
        
        # Channel 2: Goal
        features[2, goal[1], goal[0]] = 1.0
        
        # Channel 3: Distance to goal (Manhattan)
        for y in range(height):
            for x in range(width):
                dist = abs(x - goal[0]) + abs(y - goal[1])
                features[3, y, x] = dist / (width + height)  # Normalize
        
        # Channel 4: Distance to start
        for y in range(height):
            for x in range(width):
                dist = abs(x - start[0]) + abs(y - start[1])
                features[4, y, x] = dist / (width + height)  # Normalize
        
        return features
    
    def path_to_actions(self, path: List[Tuple[int, int]]) -> List[int]:
        """
        Path'i action sequence'e çevir
        """
        actions = []
        
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            
            dx = x2 - x1
            dy = y2 - y1
            
            # Action mapping
            action_map = {
                (0, -1): 0,   # N
                (0, 1): 1,    # S
                (1, 0): 2,    # E
                (-1, 0): 3,   # W
                (1, -1): 4,   # NE
                (-1, -1): 5,  # NW
                (1, 1): 6,    # SE
                (-1, 1): 7,   # SW
            }
            
            actions.append(action_map.get((dx, dy), 8))
        
        # Son adım: GOAL
        actions.append(8)
        
        return actions
    
    def collect_from_scenarios(self, 
                               map_file: str, 
                               scenario_file: str,
                               num_samples: int = 1000) -> Dict:
        """
        Scenario dosyalarından training data topla
        
        Args:
            use_algorithm: "astar"
        
        Returns:
            Dictionary: {
                'features': List of input features,
                'actions': List of action sequences,
                'metadata': Problem bilgileri
            }
        """
        from mapf_sa import read_map_file, read_scenario_file
        
        print(f"Training data toplanıyor")
        
        # Grid ve agentları yükle
        grid = read_map_file(map_file)
        agents = read_scenario_file(scenario_file, num_samples)
        
        # Pathfinder seç
        pathfinder = AStarPathFinder(grid)
        
        # Data toplama
        features_list = []
        actions_list = []
        metadata_list = []
        
        for i, (start, goal) in enumerate(agents):
            #if i % 100 == 0:
            #    print(f"  İşlenen: {i}/{len(agents)}")
            
            # Expert path bul
            expert_path = pathfinder.find_path(start, goal)
            
            if expert_path is None:
                continue
            
            # Features çıkar
            features = self.grid_to_features(grid, start, goal)
            
            # Actions çıkar
            actions = self.path_to_actions(expert_path)
            
            features_list.append(features)
            actions_list.append(actions)
            metadata_list.append({
                'start': start,
                'goal': goal,
                'path_length': len(expert_path),
                'map_file': map_file
            })
        
        print(f"\nToplanan sample sayısı: {len(features_list)}")
        
        return {
            'features': features_list,
            'actions': actions_list,
            'metadata': metadata_list
        }
    
    def save_dataset(self, data: Dict, filename: str = "training_data.pkl"):
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"\nDataset kaydedildi: {filepath}")
        print(f"   Samples: {len(data['features'])}")
        
        # İstatistikler
        path_lengths = [m['path_length'] for m in data['metadata']]
        print(f"   Ortalama path uzunluğu: {np.mean(path_lengths):.2f}")
        print(f"   Min/Max: {min(path_lengths)}/{max(path_lengths)}")
    
    def load_dataset(self, filename: str = "training_data.pkl") -> Dict:
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        print(f"  Dataset yüklendi: {filepath}")
        print(f"   Samples: {len(data['features'])}")
        
        return data
    
    def augment_data(self, data: Dict) -> Dict:
        """
        Data augmentation: Flip, rotate, etc.
        
        Daha fazla training data için
        """
        augmented_features = []
        augmented_actions = []
        augmented_metadata = []
        
        for features, actions, metadata in zip(data['features'], data['actions'], data['metadata']):
            # Original
            augmented_features.append(features)
            augmented_actions.append(actions)
            augmented_metadata.append(metadata)
            
            # Horizontal flip
            flipped_features = np.flip(features, axis=2).copy()
            flipped_actions = self.flip_actions_horizontal(actions)
            augmented_features.append(flipped_features)
            augmented_actions.append(flipped_actions)
            augmented_metadata.append({**metadata, 'augmentation': 'h_flip'})
            
            # Vertical flip
            flipped_features = np.flip(features, axis=1).copy()
            flipped_actions = self.flip_actions_vertical(actions)
            augmented_features.append(flipped_features)
            augmented_actions.append(flipped_actions)
            augmented_metadata.append({**metadata, 'augmentation': 'v_flip'})
        
        print(f"\n Data augmentation: {len(data['features'])} → {len(augmented_features)} samples")
        
        return {
            'features': augmented_features,
            'actions': augmented_actions,
            'metadata': augmented_metadata
        }
    
    def flip_actions_horizontal(self, actions: List[int]) -> List[int]:
        """Action'ları horizontal flip'e göre değiştir"""
        # E ↔ W, NE ↔ NW, SE ↔ SW
        flip_map = {
            0: 0,  # N → N
            1: 1,  # S → S
            2: 3,  # E → W
            3: 2,  # W → E
            4: 5,  # NE → NW
            5: 4,  # NW → NE
            6: 7,  # SE → SW
            7: 6,  # SW → SE
            8: 8,  # GOAL → GOAL
        }
        return [flip_map[a] for a in actions]
    
    def flip_actions_vertical(self, actions: List[int]) -> List[int]:
        """Action'ları vertical flip'e göre değiştir"""
        # N ↔ S, NE ↔ SE, NW ↔ SW
        flip_map = {
            0: 1,  # N → S
            1: 0,  # S → N
            2: 2,  # E → E
            3: 3,  # W → W
            4: 6,  # NE → SE
            5: 7,  # NW → SW
            6: 4,  # SE → NE
            7: 5,  # SW → NW
            8: 8,  # GOAL → GOAL
        }
        return [flip_map[a] for a in actions]


def main():
    import glob
    
    collector = TrainingDataCollector(output_dir="training_data")
    
    all_features = []
    all_actions = []
    all_metadata = []
    
    scen_files = glob.glob("data/scen-even-maze/*.scen")
    print(f"Bulunan scen dosyası sayısı: {len(scen_files)}")
    
    for scen_file in scen_files:
        print(f"\nİşleniyor: {scen_file}")
        data = collector.collect_from_scenarios(
            map_file="data/maze-32-32-4.map",
            scenario_file=scen_file,
            num_samples=2000,
        )
        all_features.extend(data['features'])
        all_actions.extend(data['actions'])
        all_metadata.extend(data['metadata'])
    
    combined_data = {
        'features': all_features,
        'actions': all_actions,
        'metadata': all_metadata
    }
    
    print(f"\nToplam sample: {len(all_features)}")
    
    # Augmentation
    augmented = collector.augment_data(combined_data)
    collector.save_dataset(augmented, "astar_training_1000_aug.pkl")


if __name__ == "__main__":
    main()
