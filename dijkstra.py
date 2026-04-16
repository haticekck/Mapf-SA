import heapq
from typing import List, Tuple, Set, Optional, Dict

class DijkstraPathFinder:
    def __init__(self, grid: List[List[str]]):
        self.grid = grid
        self.height = len(grid)
        self.width = len(grid[0]) if grid else 0
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[Tuple[int, int], float]]:
        x, y = pos
        neighbors = []
        
        # 8 yönlü hareket
        directions = [
            (0, -1, 1.0),           # N
            (0, 1, 1.0),            # S
            (1, 0, 1.0),            # E
            (-1, 0, 1.0),           # W
            (1, -1, 1.41421356),    # NE
            (-1, -1, 1.41421356),   # NW
            (1, 1, 1.41421356),     # SE
            (-1, 1, 1.41421356)     # SW
        ]
        
        for dx, dy, cost in directions:
            nx, ny = x + dx, y + dy
            
            # Sınırları kontrol et
            if 0 <= nx < self.width and 0 <= ny < self.height:
                # Duvar kontrolü
                if self.grid[ny][nx] != '@':
                    # Diagonal harekette köşe kontrolü
                    if dx != 0 and dy != 0:
                        if self.grid[y][nx] != '@' and self.grid[ny][x] != '@':
                            neighbors.append(((nx, ny), cost))
                    else:
                        neighbors.append(((nx, ny), cost))
        
        return neighbors
    
    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        # Başlangıç ve hedef geçerli mi kontrol et
        if not (0 <= start[0] < self.width and 0 <= start[1] < self.height):
            return None
        if not (0 <= goal[0] < self.width and 0 <= goal[1] < self.height):
            return None
        if self.grid[start[1]][start[0]] == '@' or self.grid[goal[1]][goal[0]] == '@':
            return None
        
        # Başlangıç hedefle aynıysa
        if start == goal:
            return [start]
        
        # Priority queue: (distance, counter, position)
        counter = 0
        open_set = [(0, counter, start)]
        counter += 1
        
        # Mesafeler ve önceki nodlar
        distances = {start: 0}
        came_from = {}
        closed_set: Set[Tuple[int, int]] = set()

        explored_nodes = 0
        
        while open_set:
            current_dist, _, current = heapq.heappop(open_set)
            
            # Hedefe ulaştık mı?
            if current == goal:
                # Yolu reconstruct et
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                
                return path
            
            if current in closed_set:
                continue
                
            closed_set.add(current)
            explored_nodes += 1
            
            # Komşuları incele
            for neighbor, move_cost in self.get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                tentative_distance = current_dist + move_cost
                
                # Daha iyi yol bulundu mu?
                if neighbor not in distances or tentative_distance < distances[neighbor]:
                    distances[neighbor] = tentative_distance
                    came_from[neighbor] = current
                    heapq.heappush(open_set, (tentative_distance, counter, neighbor))
                    counter += 1
        
        # Yol bulunamadı
        return None
    
    def find_initial_paths(self, agents: List[Tuple[Tuple[int, int], Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
        print("DİJKSTRA İLE İLK YOLLAR BULUNUYOR")
        
        paths = []
        total_explored = 0
        
        for i, (start, goal) in enumerate(agents):
            print(f"\nAgent {i}: {start} -> {goal}")
            path = self.find_path(start, goal)
            
            if path is None:
                print(f"UYARI: Agent {i} için yol bulunamadı!")
                paths.append([start])
            else:
                paths.append(path)

        print(f"Tüm agentlar için yollar bulundu!")
        return paths


class DijkstraWithHeatmap(DijkstraPathFinder):
    def find_all_distances(self, start: Tuple[int, int]) -> Dict[Tuple[int, int], float]:
        # Priority queue
        counter = 0
        open_set = [(0, counter, start)]
        counter += 1
        
        distances = {start: 0}
        closed_set: Set[Tuple[int, int]] = set()
        
        while open_set:
            current_dist, _, current = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
            
            closed_set.add(current)
            
            # Komşuları incele
            for neighbor, move_cost in self.get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                tentative_distance = current_dist + move_cost
                
                if neighbor not in distances or tentative_distance < distances[neighbor]:
                    distances[neighbor] = tentative_distance
                    heapq.heappush(open_set, (tentative_distance, counter, neighbor))
                    counter += 1
        
        return distances
    
    def find_path_to_nearest_goal(self, start: Tuple[int, int], 
                                   goals: List[Tuple[int, int]]) -> Optional[List[Tuple[int, int]]]:
        """
        Birden fazla hedef varsa, en yakınına git
        
        Kullanım: Pickup/delivery senaryoları
        """
        # Tüm mesafeleri hesapla
        distances = self.find_all_distances(start)
        
        # En yakın hedefi bul
        nearest_goal = None
        min_distance = float('inf')
        
        for goal in goals:
            if goal in distances and distances[goal] < min_distance:
                min_distance = distances[goal]
                nearest_goal = goal
        
        if nearest_goal is None:
            return None
        
        # O hedefe git
        return self.find_path(start, nearest_goal)


# Performans karşılaştırması için
def compare_astar_dijkstra(grid, agents):
    """A* ve Dijkstra'yı karşılaştır"""
    from a_star import AStarPathFinder
    import time
    
    # A* ile
    print("\n--- A* ile çözüm ---")
    astar = AStarPathFinder(grid)
    start_time = time.time()
    astar_paths = astar.find_initial_paths(agents)
    astar_time = time.time() - start_time
    
    astar_soc = sum(len(p) for p in astar_paths)
    
    # Dijkstra ile
    print("\n--- Dijkstra ile çözüm ---")
    dijkstra = DijkstraPathFinder(grid)
    start_time = time.time()
    dijkstra_paths = dijkstra.find_initial_paths(agents)
    dijkstra_time = time.time() - start_time
    
    dijkstra_soc = sum(len(p) for p in dijkstra_paths)
    
    # Karşılaştırma
    print("\n" + "="*70)
    print("SONUÇLAR")
    print("="*70)
    print(f"{'Metrik':<30} {'A*':<20} {'Dijkstra':<20}")
    print("-"*70)
    print(f"{'Süre (saniye)':<30} {astar_time:<20.3f} {dijkstra_time:<20.3f}")
    print(f"{'SOC (adım)':<30} {astar_soc:<20} {dijkstra_soc:<20}")
    print(f"{'Hız farkı':<30} {'1.0x':<20} {f'{dijkstra_time/astar_time:.2f}x':<20}")
    print("="*70)
    
    return {
        'astar': {'paths': astar_paths, 'time': astar_time, 'soc': astar_soc},
        'dijkstra': {'paths': dijkstra_paths, 'time': dijkstra_time, 'soc': dijkstra_soc}
    }
