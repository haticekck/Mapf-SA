import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import re

def load_map(map_file):
    """MAPF standart .map dosyasını yükle"""
    with open(map_file) as f:
        lines = f.readlines()
    
    height = int([l for l in lines if l.startswith("height")][0].split()[1])
    width = int([l for l in lines if l.startswith("width")][0].split()[1])
    
    map_start = lines.index("map\n") + 1
    grid = []
    
    for i in range(height):
        row = lines[map_start + i].strip()
        grid.append([0 if c == '.' else 1 for c in row])
    
    return np.array(grid)

def load_paths(result_file):
    """Result dosyasından path'leri yükle"""
    paths = {}
    current_agent = None
    
    with open(result_file) as f:
        for line in f:
            if line.startswith("Agent"):
                current_agent = int(line.split()[1].replace(":", ""))
                paths[current_agent] = []
            elif "t=" in line and current_agent is not None:
                match = re.findall(r"\((\d+), (\d+)\)", line)
                if match:
                    x, y = map(int, match[0])
                    paths[current_agent].append((x, y))
    
    # Boş path'leri filtrele
    paths = {k: v for k, v in paths.items() if len(v) > 0}
    
    print(f"Yüklenen ajan sayısı: {len(paths)}")
    if len(paths) == 0:
        print("HATA: Hiç path yüklenemedi!")
    
    return paths

def check_collision(paths, t):
    """Vertex ve Edge çarpışma kontrolü - Goal'deki ajanlar hariç"""
    # Vertex collision kontrolü
    pos = {}
    conflicts = set()
    
    for agent, path in paths.items():
        # Goal'e ulaşmış mı kontrol et
        if t >= len(path) - 1:
            continue  # Goal'de, çarpışma sayma
            
        p = path[t]
            
        if p in pos:
            conflicts.add(agent)
            conflicts.add(pos[p])
        else:
            pos[p] = agent
    
    # Edge collision kontrolü
    if t > 0:
        for a1, p1 in paths.items():
            if t >= len(p1) - 1:  # a1 goal'de
                continue
            for a2, p2 in paths.items():
                if a1 >= a2 or t >= len(p2) - 1:  # a2 goal'de
                    continue
                
                pos1_prev = p1[t-1]
                pos1_curr = p1[t]
                pos2_prev = p2[t-1]
                pos2_curr = p2[t]
                
                if pos1_prev == pos2_curr and pos1_curr == pos2_prev:
                    conflicts.add(a1)
                    conflicts.add(a2)
    
    return conflicts

def visualize_mapf(map_file, result_file, speed=200):
    """MAPF görselleştirme"""
    
    # Verileri yükle
    grid = load_map(map_file)
    paths = load_paths(result_file)
    
    if not paths:
        print("Görselleştirme yapılamıyor - path verisi yok!")
        return
    
    max_t = max(len(p) for p in paths.values())
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown', 'pink']
    
    # Figure oluştur
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Haritayı göster
    ax.imshow(grid, cmap='gray_r')
    
    # Başlangıç ve hedef noktaları
    for i, agent_id in enumerate(paths):
        path = paths[agent_id]
        color = colors[i % len(colors)]
        start_x, start_y = path[0]
        goal_x, goal_y = path[-1]
        
        # Başlangıç: kare
        ax.plot(start_x, start_y, 's', color=color, markersize=12, 
                markeredgecolor='white', markeredgewidth=2, alpha=0.6)
        # Hedef: yıldız
        ax.plot(goal_x, goal_y, '*', color=color, markersize=18, 
                markeredgecolor='white', markeredgewidth=2, alpha=0.6)
    
    # Ajan noktaları
    agents = []
    for i, agent_id in enumerate(paths):
        color = colors[i % len(colors)]
        dot, = ax.plot([], [], 'o', color=color, markersize=12,
                      markeredgecolor='black', markeredgewidth=2)
        agents.append(dot)
    
    # Bilgi metni
    info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                       verticalalignment='top', fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.grid(True, alpha=0.3)
    ax.set_title('MAPF Görselleştirme', fontsize=14, fontweight='bold')
    
    def animate(t):
        conflicts = check_collision(paths, t)
        
        # Her ajanı güncelle - AYNI HAREKET MANTIĞI
        for i, agent_id in enumerate(paths):
            path = paths[agent_id]

            if t >= len(path) - 1:
                agents[i].set_data([], [])  # Haritadan çıkar
                continue
            if t < len(path):
                x, y = path[t]
                agents[i].set_data([x], [y])  # Aynı şekilde [x], [y]
                
                if agent_id in conflicts:
                    agents[i].set_color('red')
                    agents[i].set_markersize(16)
                else:
                    agents[i].set_color(colors[i % len(colors)])
                    agents[i].set_markersize(12)
        
        # Bilgi güncelle
        if conflicts:
            info_text.set_text(f't = {t} / {max_t-1}\n⚠️ ÇARPIŞMA!')
            info_text.set_bbox(dict(boxstyle='round', facecolor='red', alpha=0.8))
            info_text.set_color('white')
        else:
            info_text.set_text(f't = {t} / {max_t-1}\n✓ Çarpışma yok')
            info_text.set_bbox(dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            info_text.set_color('black')
        
        return agents + [info_text]
    
    anim = animation.FuncAnimation(fig, animate, frames=max_t,
                                  interval=speed, blit=True, repeat=True)
    
    plt.tight_layout()
    plt.show()
    
    return anim


# Kullanım
if __name__ == "__main__":
    visualize_mapf("data/maze-32-32-4.map", "solutions/solution_maze_1.txt", speed=1000)