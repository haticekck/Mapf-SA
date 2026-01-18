import random
import math
import copy
from typing import List, Tuple, Set
from a_star import AStarPathFinder


class SimulatedAnnealingMAPF:
    def __init__(self, grid: List[List[str]], 
                 initial_temp: float = 1000.0,
                 cooling_rate: float = 0.995,
                 min_temp: float = 1.0,
                 conflict_penalty: float = 1000.0,
                 iterations_per_temp: int = 100):

        self.grid = grid
        self.height = len(grid)
        self.width = len(grid[0]) if grid else 0
        self.pathfinder = AStarPathFinder(grid)
        
        # SA parametreleri
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.conflict_penalty = conflict_penalty
        self.iterations_per_temp = iterations_per_temp
        
    def calculate_cost(self, paths: List[List[Tuple[int, int]]]) -> float:        
        # Maliyet = Toplam yol uzunluğu + Çakışma sayısı * ceza
        total_cost = 0.0
        
        # Toplam yol uzunluğu (sum of costs)
        for path in paths:
            total_cost += len(path)
        
        # Çakışmaları tespit et ve cezalandır
        conflicts = self.detect_conflicts(paths)
        total_cost += len(conflicts) * self.conflict_penalty
        
        return total_cost
    
    """def detect_conflicts(self, paths: List[List[Tuple[int, int]]]) -> List[Tuple]:
        # Return: Çakışma listesi: [(agent1_id, agent2_id, time, conflict_type), ...]
        conflicts = []
        
        # Tüm yolları aynı uzunluğa getirme (bekleme hareketleri ile)
        max_length = max(len(path) for path in paths) if paths else 0
        normalized_paths = []
        for path in paths:
            normalized = path + [path[-1]] * (max_length - len(path))
            normalized_paths.append(normalized)
        
        # Her zaman adımında çakışmaları kontrol et
        for t in range(max_length):
            # Vertex conflicts (aynı anda aynı pozisyon)
            for i in range(len(normalized_paths)):
                for j in range(i + 1, len(normalized_paths)):
                    if normalized_paths[i][t] == normalized_paths[j][t]:
                        conflicts.append((i, j, t, "vertex"))
            
            # Edge conflicts (swap - yer değiştirme)
            if t < max_length - 1: # Son adımda kenar çakışması olmaz
                for i in range(len(normalized_paths)):
                    for j in range(i + 1, len(normalized_paths)):
                        pos_i_now = normalized_paths[i][t]
                        pos_i_next = normalized_paths[i][t + 1]
                        pos_j_now = normalized_paths[j][t]
                        pos_j_next = normalized_paths[j][t + 1]
                        
                        # İki agent yer değiştiriyor mu?
                        if pos_i_now == pos_j_next and pos_i_next == pos_j_now:
                            conflicts.append((i, j, t, "edge"))
        
        return conflicts"""

    #detect_conflicts fonksiyonunun agentlerin çıkış yaptıktan sonraki durumları göz önüne alacak şekilde güncellenmiş hali
    def detect_conflicts(self, paths: List[List[Tuple[int, int]]]) -> List[Tuple]:
        # Return: Çakışma listesi: [(agent1_id, agent2_id, time, conflict_type), ...]
        conflicts = []

        exit_times = []
        for path in paths:
            exit_times.append(len(path) - 1)

        # Tüm yolları aynı uzunluğa getirme (bekleme hareketleri ile)
        max_length = max(len(path) for path in paths) if paths else 0
        normalized_paths = []
        for path in paths:
            normalized = path + [path[-1]] * (max_length - len(path))
            normalized_paths.append(normalized)
        
        # Her zaman adımında çakışmaları kontrol et
        for t in range(max_length):
            # Vertex conflicts (aynı anda aynı pozisyon)
            for i in range(len(normalized_paths)):
                if t >= exit_times[i]:
                    continue

                for j in range(i + 1, len(normalized_paths)):
                    if t >= exit_times[j]:
                        continue

                    if normalized_paths[i][t] == normalized_paths[j][t]:
                        conflicts.append((i, j, t, "vertex"))
            
            # Edge conflicts (swap - yer değiştirme)
            if t < max_length - 1: # Son adımda kenar çakışması olmaz
                for i in range(len(normalized_paths)):
                    if t >= exit_times[i]-1:
                        continue

                    for j in range(i + 1, len(normalized_paths)):
                        if t >= exit_times[j]-1:
                            continue

                        pos_i_now = normalized_paths[i][t]
                        pos_i_next = normalized_paths[i][t + 1]
                        pos_j_now = normalized_paths[j][t]
                        pos_j_next = normalized_paths[j][t + 1]
                        
                        # İki agent yer değiştiriyor mu?
                        if pos_i_now == pos_j_next and pos_i_next == pos_j_now:
                            conflicts.append((i, j, t, "edge"))
        
        return conflicts
    
    def generate_neighbor(self, paths: List[List[Tuple[int, int]]], 
                         agents_info: List[Tuple[Tuple[int, int], Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
        #komşu çözüm yaparken mevcut çözümü korumak için derin kopya
        neighbor = copy.deepcopy(paths)
        
        # Rastgele bir strateji seç
        strategy = random.choice(['replan_segment', 'wait_action'])
        
        # Rastgele bir agent seç
        agent_idx = random.randint(0, len(paths) - 1)
        path = neighbor[agent_idx]
        
        if len(path) < 2:
            return neighbor
        
        if strategy == 'replan_segment':# Yolun bir kısmını yeniden planlam
            # Rastgele bir segment seçme
            # segmentlengthi ajana göre yap
            segment_length = min(random.randint(3, 30), len(path) - 1)
            #min(random.randint(3, 8), len(path) - 1)
            #random.randint(3, (len(path) - 2))
            # Segment için başlangıç ve bitiş indeksleri
            start_idx = random.randint(0, len(path) - segment_length - 1)
            end_idx = min(start_idx + segment_length, len(path) - 1)
            
            start_pos = path[start_idx]
            end_pos = path[end_idx]
            
            # A* kullanarak bu iki nokta arasında yeni bir mini yol oluşturuyoruz.
            new_segment = self.pathfinder.find_path(start_pos, end_pos)
            
            if new_segment is not None:
                neighbor[agent_idx] = path[:start_idx] + new_segment + path[end_idx + 1:]
        
        elif strategy == 'wait_action':
            # Rastgele bir noktada bekle hareketi ekleme
            if random.random() < 0.5 and len(path) > 2:
                # Bekleme ekleme
                insert_idx = random.randint(1, len(path) - 1)
                wait_pos = path[insert_idx]
                #neighbor[agent_idx] = path[:insert_idx] + [wait_pos] + path[insert_idx:]
                k = random.randint(1, 3)

                new_path = path[:insert_idx]
                for _ in range(k):
                    new_path.append(wait_pos)
                new_path += path[insert_idx:]

                neighbor[agent_idx] = new_path
            else:
                # Ardışık aynı pozisyonları bul ve birini çıkar
                if random.random() < 0.05:
                    for i in range(len(path) - 1):
                        if path[i] == path[i + 1]:
                            neighbor[agent_idx] = path[:i] + path[i + 1:]
                            break
        
        """elif strategy == 'full_replan':
            # Tüm yolu yeniden planla
            start, goal = agents_info[agent_idx]
            # A* kullanarak bu iki nokta arasında yeni bir mini yol oluşturuyoruz.
            new_path = self.pathfinder.find_path(start, goal)
            if new_path is not None:
                neighbor[agent_idx] = new_path"""
        
        return neighbor
    
    def optimize(self, initial_paths: List[List[Tuple[int, int]]], 
                agents_info: List[Tuple[Tuple[int, int], Tuple[int, int]]],
                verbose: bool = True) -> Tuple[List[List[Tuple[int, int]]], List[float]]:

        current_solution = copy.deepcopy(initial_paths)
        current_cost = self.calculate_cost(current_solution)
        #yukarıda current_solution deep copy olduğundan calculate_cost fonksiyonu onda pathleri eşitliyor fakat initial_paths'deki path uzunlukları aynı kalıyor
        
        #besti ilkle initialize etme
        best_solution = copy.deepcopy(current_solution)
        best_cost = current_cost
        
        temperature = self.initial_temp
        cost_history = [current_cost]
        
        iteration = 0
        
        if verbose:
            print(f"Başlangıç maliyeti: {current_cost:.2f}")
            print(f"Başlangıç çakışmaları: {len(self.detect_conflicts(current_solution))}")
        
        while temperature > self.min_temp:
            for _ in range(self.iterations_per_temp):
                # Komşu çözüm üret
                neighbor = self.generate_neighbor(current_solution, agents_info)
                neighbor_cost = self.calculate_cost(neighbor)
                
                # Delta hesapla
                delta = neighbor_cost - current_cost
                
                # Kabul kriteri delta 0ken olasılık 1 randomdan büyük.
                # delta ile random kısmını ayır elif
                if delta < 0 or random.random() < math.exp(-delta / temperature):
                    current_solution = neighbor
                    current_cost = neighbor_cost

                elif random.random() < math.exp(-delta / temperature): 
                    current_solution = neighbor
                    current_cost = neighbor_cost
                    
                    # En iyi çözümü güncelle
                if current_cost < best_cost:
                    best_solution = copy.deepcopy(current_solution)
                    best_cost = current_cost
                    if verbose:
                        conflicts = len(self.detect_conflicts(best_solution))
                        print(f"İter {iteration}: Yeni en iyi maliyet={best_cost:.2f}, Çakışma={conflicts}, Temp={temperature:.2f}")
                
                iteration += 1 #toplam iterasyon sayısı
                cost_history.append(current_cost)
            
            # soğutma
            temperature *= self.cooling_rate
        
        if verbose:
            final_conflicts = len(self.detect_conflicts(best_solution))
            print(f"\nOptimizasyon tamamlandı.")
            print(f"Final maliyet: {best_cost:.2f}")
            print(f"Final çakışma sayısı: {final_conflicts}")
        
        return best_solution, cost_history