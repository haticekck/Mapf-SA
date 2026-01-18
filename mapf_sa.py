import sys
from typing import List, Tuple, Dict
from a_star import AStarPathFinder
from simulated_annealing import SimulatedAnnealingMAPF
#from data_splitter import Split_by_Bucket as ScenarioDataSplitter
from metrics_visualization import (
    plot_all_metrics, 
    plot_sa_convergence,
    plot_soc_over_time,
    plot_auc_over_time
)

def read_map_file(filename: str) -> List[List[str]]:
    """
    .map dosyasını okur ve grid döndürür
    
    Returns:
        Grid listesi
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Header bilgilerini oku
    map_type = None
    height = 0
    width = 0
    
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('type'):
            map_type = line.split()[1]
        elif line.startswith('height'):
            height = int(line.split()[1])
        elif line.startswith('width'):
            width = int(line.split()[1])
        elif line.startswith('map'):
            # Harita başlıyor
            grid = []
            for j in range(i + 1, i + 1 + height):
                if j < len(lines):
                    row = list(lines[j].strip())
                    grid.append(row)
            return grid
    
    return []

def read_scenario_file(filename: str, num_agents: int = None) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    .scen dosyasını okur ve agent başlangıç/hedef pozisyonlarını döndürür
    
    Format: bucket map_name width height start_x start_y goal_x goal_y optimal_length
    
    Args:
        filename: Scenario dosya adı
        num_agents: Kaç agent alınacak (None ise hepsi)
        
    Returns:
        [(start, goal), ...] listesi
    """
    agents = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # İlk satır header olabilir, atla
    start_idx = 0
    if lines[0].strip().startswith('version'):
        start_idx = 1
    
    for line in lines[start_idx:]:
        parts = line.strip().split()
        if len(parts) >= 9:
            # Kolonlar: bucket, map, width, height, start_x, start_y, goal_x, goal_y, length
            start_x = int(parts[4])
            start_y = int(parts[5])
            goal_x = int(parts[6])
            goal_y = int(parts[7])
            
            agents.append(((start_x, start_y), (goal_x, goal_y)))
            
            if num_agents is not None and len(agents) >= num_agents:
                break
    
    return agents

def visualize_solution(grid: List[List[str]], 
                       paths: List[List[Tuple[int, int]]],
                       timestep: int = 0) -> None:
    """
    Belirli bir zaman adımında çözümü görselleştirir
    """
    # Grid kopyası oluştur
    vis_grid = [row[:] for row in grid]
    
    # Her agentı yerleştir
    for agent_id, path in enumerate(paths):
        if timestep < len(path):
            x, y = path[timestep]
            if 0 <= x < len(vis_grid[0]) and 0 <= y < len(vis_grid):
                vis_grid[y][x] = str(agent_id % 10)  # 0-9 arası numara
    
    # Yazdır
    print(f"\n--- Timestep {timestep} ---")
    for row in vis_grid:
        print(''.join(row))

def calculate_path_cost(path: List[Tuple[int, int]]) -> float:
    """
    Bir yolun gerçek maliyetini hesaplar (diagonal hareketler dahil)
    
    Args:
        path: [(x, y), ...] pozisyon listesi
        
    Returns:
        Toplam maliyet (düz hareket=1.0, diagonal=1.41421356)
    """
    if len(path) <= 1:
        return 0.0
    
    total_cost = 0.0
    
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        
        dx = x2 - x1
        dy = y2 - y1
        
        # Hareket tipini belirle
        if dx == 0 and dy == 0:
            # Bekle hareketi (aynı yerde kal)
            cost = 0.0
        elif abs(dx) <= 1 and abs(dy) <= 1:
            # Komşu hücreye hareket
            if dx != 0 and dy != 0:
                # Diagonal hareket
                cost = 1.41421356
            else:
                # Düz hareket (N, S, E, W)
                cost = 1.0
        else:
            # Hatalı hareket (olmamalı)
            print(f"UYARI: Geçersiz hareket tespit edildi: ({x1},{y1}) -> ({x2},{y2})")
            cost = 0.0
        
        total_cost += cost
    
    return total_cost


def save_solution(filename: str, paths: List[List[Tuple[int, int]]], time_elapsed: float) -> None:
    """Çözümü dosyaya kaydeder"""
    # Her agent için maliyetleri hesapla
    individual_costs = [calculate_path_cost(path) for path in paths]
    total_sum_of_costs = sum(individual_costs)
    auc = calculate_auc(paths)
    norm_auc = calculate_normalized_auc(paths)
    
    with open(filename, 'w') as f:
        f.write(f"Agents: {len(paths)}\n")
        f.write(f"Time elapsed: {time_elapsed:.2f} seconds\n")
        f.write(f"Makespan: {max(len(p) for p in paths)}\n")
        f.write(f"Sum of costs (steps): {sum(len(p) - 1 for p in paths)}\n")
        f.write(f"Sum of costs (actual): {total_sum_of_costs:.5f}\n")
        f.write(f"AUC: {auc}\n")
        f.write(f"Normalized AUC: {norm_auc:.5f}\n\n")
        
        for i, path in enumerate(paths):
            path_cost = individual_costs[i]
            f.write(f"Agent {i}: cost: {path_cost:.5f}\n")
            
            for t, (x, y) in enumerate(path):
                # Her adımın maliyetini de göster
                """if t < len(path) - 1:
                    x_next, y_next = path[t + 1]
                    dx = x_next - x
                    dy = y_next - y
                    
                    if dx == 0 and dy == 0:
                        move_type = "WAIT"
                        step_cost = 0.0
                    elif dx != 0 and dy != 0:
                        move_type = "DIAG"
                        step_cost = 1.41421356
                    else:
                        move_type = "STRAIGHT"
                        step_cost = 1.0
                    
                    f.write(f"  t={t}: ({x}, {y}) -> {move_type} (cost: {step_cost:.2f})\n")
                else:
                    f.write(f"  t={t}: ({x}, {y}) -> GOAL\n")"""
                f.write(f"  t={t}: ({x}, {y})\n")
            
            f.write("\n")

def calculate_makespan(paths: List[List[Tuple[int, int]]]) -> int:
    """
    Makespan hesaplar (en uzun yolun uzunluğu)
    
    Returns:
        Timestep cinsinden makespan
    """
    if not paths:
        return 0
    
    return max(len(path) - 1 for path in paths)

def calculate_auc(paths: List[List[Tuple[int, int]]]) -> float:
    """
    AUC (Area Under Curve) hesaplar
    
    AUC = Her timestep'te aktif agent sayısının toplamı
    
    Düşük AUC = Agentler hızlı bitiriyor (iyi)
    Yüksek AUC = Agentler uzun süre aktif (kötü)
    """
    if not paths:
        return 0.0
    
    max_length = max(len(path) for path in paths)
    auc = 0.0
    
    for t in range(max_length):
        # Bu timestep'te kaç agent aktif?
        active_agents = sum(1 for path in paths if t < len(path))
        auc += active_agents
    
    return auc


def calculate_normalized_auc(paths: List[List[Tuple[int, int]]]) -> float:
    """
    Normalize edilmiş AUC (0-1 arası)
    
    Normalize = AUC / (num_agents × makespan)
    """
    if not paths:
        return 0.0
    
    auc = calculate_auc(paths)
    num_agents = len(paths)
    makespan = calculate_makespan(paths)
    
    max_possible_auc = num_agents * makespan
    
    return auc / max_possible_auc if max_possible_auc > 0 else 0.0

def calculate_success_rate(paths: List[List[Tuple[int, int]]], conflicts: List[Dict]) -> float:
    
    if len(conflicts) == 0:
        return 1.0  # Başarılı!
    else:
        return 0.0  # Başarısız

def calculate_success_rate_multiple_runs(results: List[Dict]) -> float:
    successful_runs = sum(1 for r in results if r['conflicts'] == 0)
    total_runs = len(results)
    
    return successful_runs / total_runs if total_runs > 0 else 0.0

def main():
    # Dosya adları
    import time
    #map_file = "data/random-32-32-20.map"
    #scenario_file = "data/scen-even 4/random-32-32-20-even-1.scen"

    #map_file = "data/empty-8-8.map"
    #scenario_file = "data/scen-even 5/empty-8-8-even-1.scen"

    map_file = "data/maze-32-32-4.map"
    scenario_file = "data/scen-even 3/maze-32-32-4-even-1.scen"
    
    # Bucket veya zorluk bazlı seçim
    #use_bucket = False # True: bucket kullan, False: tüm dosyadan al
    #selected_bucket = 5  # Hangi bucket'tan agent alınacak
    num_agents = 50
    
    print("=" * 60)
    print("MULTI-AGENT PATH FINDING - Simulated Annealing + A*")
    print("=" * 60)
    
    # Haritayı oku
    #print(f"\n1. Harita okunuyor: {map_file}")
    grid = read_map_file(map_file)
    #print(f"   Harita boyutu: {len(grid[0])}x{len(grid)}")
    
    # Senaryoyu oku
    #print(f"\n2. Senaryo okunuyor: {scenario_file}")
    
    """if use_bucket:
        # Bucket bazlı seçim
        splitter = ScenarioDataSplitter(scenario_file)
        splitter.parse_scenario_file()
        agents = splitter.get_scenarios_by_bucket(bucket=selected_bucket, limit=num_agents)
        print(f"   Bucket {selected_bucket} kullanılıyor")
    else:
        # Tüm dosyadan rastgele al"""
    
    agents = read_scenario_file(scenario_file, num_agents)
    
    #print(f"   Agent sayısı: {len(agents)}")
    
    #for i, (start, goal) in enumerate(agents):
        #print(f"   Agent {i}: {start} -> {goal}")
    
    # A* ile başlangıç yollarını bul
    #print(f"\n3. A* ile başlangıç yolları bulunuyor...")
    pathfinder = AStarPathFinder(grid)
    initial_paths = pathfinder.find_initial_paths(agents)
    
    initial_cost = sum(len(p) for p in initial_paths)
    print(f"Toplam başlangıç maliyeti: {initial_cost}")
    
    # Simulated Annealing ile optimize et
    #print(f"\n4. Simulated Annealing ile optimizasyon başlıyor...")
    start = time.perf_counter()
    sa_optimizer = SimulatedAnnealingMAPF(
        grid=grid,
        initial_temp=1000.0,
        cooling_rate=0.995,
        min_temp=1.0,
        conflict_penalty=1000.0,
        iterations_per_temp=100
    )
    
    optimized_paths, cost_history = sa_optimizer.optimize(
        initial_paths, 
        agents,
        verbose=True
    )

    """results = []
    for run in range(3):
        paths, _ = sa_optimizer.optimize(initial_paths, agents)
        conflicts = len(sa_optimizer.detect_conflicts(paths))
        results.append({'paths': paths, 'conflicts': conflicts})

    success_rate = calculate_success_rate_multiple_runs(results)"""

    
    # Sonuçları göster
    print("\n" + "=" * 60)
    print("SONUÇLAR")
    print("=" * 60)

    end = time.perf_counter()
    time_elapsed = end - start
    print(f"Çözüm süresi: {time_elapsed:.2f} saniye")

    conflicts = sa_optimizer.detect_conflicts(optimized_paths)
    makespan = max(len(p) for p in optimized_paths)
    sum_of_costs_steps = sum(len(p) - 1 for p in optimized_paths)  # Adım sayısı
    sum_of_costs_actual = sum(calculate_path_cost(p) for p in optimized_paths)  # Gerçek maliyet
    auc = calculate_auc(optimized_paths)
    norm_auc = calculate_normalized_auc(optimized_paths)
    success_rate = calculate_success_rate(optimized_paths, conflicts)
    
    print(f"Makespan: {makespan}")
    print(f"Sum of costs (steps): {sum_of_costs_steps}")
    print(f"Sum of costs (actual): {sum_of_costs_actual:.5f}")
    print(f"Çakışma sayısı: {len(conflicts)}")
    print(f"AUC: {auc}")
    print(f"Normalized AUC: {norm_auc:.5f}")
    print(f"Success Rate: {success_rate}")
    
    if conflicts:
        print("\nKalan çakışmalar:")
        for agent1, agent2, time, ctype in conflicts[:10]:  # İlk 10'unu göster
            print(f"  t={time}: Agent {agent1} <-> Agent {agent2} ({ctype})")
        if len(conflicts) > 10:
            print(f"  ... ve {len(conflicts) - 10} çakışma daha")
    
    # Çözümü kaydet
    output_file = "solutions/solution_maze_1.txt"
    save_solution(output_file, optimized_paths, time_elapsed)
    print(f"\nÇözüm '{output_file}' dosyasına kaydedildi.")
    
    # Grafikleri çiz
    print("\n" + "=" * 60)
    print("GRAFİKLER ÇİZİLİYOR")
    print("=" * 60)
    
    # Tüm metriklerin grafikleri
    plot_all_metrics(optimized_paths, output_dir="metrics_plots_maze", show=False)
    
    # SA convergence grafiği
    plot_sa_convergence(cost_history, save_path="metrics_plots_maze/sa_convergence.png", show=False)

    print("\nTüm grafikler 'metrics_plots_maze/' klasörüne kaydedildi!")
    

if __name__ == "__main__":
    main()