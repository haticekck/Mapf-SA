import heapq
from typing import List, Tuple, Set, Optional

class AStarPathFinder:
    def __init__(self, grid: List[List[str]]):
        self.grid = grid #(@ = duvar, . = boş)
        self.height = len(grid)
        self.width = len(grid[0]) if grid else 0
        
    def heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])
        # Diagonal movement cost = sqrt(2)(Octile distance), straight = 1
        # min(dx, dy) diagonal hareket
        # max(dx, dy) - min(dx, dy) düz hareket
        #a*'ın heuristiği (Çapraz Hareket Sayısı * Çapraz Maliyet) + (Düz Hareket Sayısı * Düz Maliyet)
        return max(dx, dy) + (1.41421356 - 1) * min(dx, dy)
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[Tuple[int, int], float]]:
        x, y = pos
        neighbors = []

        directions = [
            (0, -1, 1.0),   # N
            (0, 1, 1.0),    # S
            (1, 0, 1.0),    # E
            (-1, 0, 1.0),   # W
            (1, -1, 1.41421356),  # NE
            (-1, -1, 1.41421356), # NW
            (1, 1, 1.41421356),   # SE
            (-1, 1, 1.41421356)   # SW
        ]
        
        for dx, dy, cost in directions:
            nx, ny = x + dx, y + dy
            
            # Sınırları kontrol etme
            if 0 <= nx < self.width and 0 <= ny < self.height:
                # Duvar kontrolü
                if self.grid[ny][nx] != '@':
                    # Diagonal harekette köşe kontrolü
                    if dx != 0 and dy != 0:
                        # Diagonal hareket için iki yanın da açık olması gerekir
                        if self.grid[y][nx] != '@' and self.grid[ny][x] != '@':
                            neighbors.append(((nx, ny), cost))
                    else:
                        neighbors.append(((nx, ny), cost))
        
        return neighbors
    
    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        # Başlangıç ve hedef geçerliliği
        if self.grid[start[1]][start[0]] == '@' or self.grid[goal[1]][goal[0]] == '@':
            return None
        
        # Başlangıç hedefle aynıysa
        if start == goal:
            return [start]
        
        # Priority queue: (f_score, counter, position, deger_g)
        counter = 0
        open_set = [(self.heuristic(start, goal), counter, start, 0)]
        counter += 1
        

        came_from = {}
        g_score = {start: 0} #başlangıç maliyeti
        closed_set: Set[Tuple[int, int]] = set() # Ziyaret edilen nodlar
        
        while open_set:
            _, _, current, current_g = heapq.heappop(open_set)
            
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
            
            # Komşuları incele
            for neighbor, move_cost in self.get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                deger_g = current_g + move_cost
                
                # Daha iyi yol bulunduysa
                # Bu komşuya daha önce gitmemişsek, ya da şimdi daha kısa bir yol bulduysak güncelle.
                if neighbor not in g_score or deger_g < g_score[neighbor]:
                    g_score[neighbor] = deger_g
                    f_score = deger_g + self.heuristic(neighbor, goal) #f(n) = g(n) + h(n)
                    came_from[neighbor] = current
                    heapq.heappush(open_set, (f_score, counter, neighbor, deger_g))
                    counter += 1

        return None
    
    def find_initial_paths(self, agents: List[Tuple[Tuple[int, int], Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
        paths = []
        for i, (start, goal) in enumerate(agents):
            path = self.find_path(start, goal)
            if path is None:
                print(f"UYARI: Agent {i} için yol bulunamadı! ({start} -> {goal})")
                paths.append([start])  # En azından başlangıç noktası
            else:
                paths.append(path)
        return paths