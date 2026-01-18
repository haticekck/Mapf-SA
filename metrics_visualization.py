import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict

def calculate_path_cost(path: List[Tuple[int, int]]) -> float:
    """Bir yolun gerçek maliyetini hesaplar"""
    if len(path) <= 1:
        return 0.0
    
    total_cost = 0.0
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            cost = 0.0  # Wait
        elif dx != 0 and dy != 0:
            cost = 1.41421356  # Diagonal
        else:
            cost = 1.0  # Straight
        
        total_cost += cost
    
    return total_cost


def calculate_soc_over_time(paths: List[List[Tuple[int, int]]]) -> Tuple[List[int], List[float]]:
    """
    Zamana göre kümülatif SOC hesaplar
    
    Returns:
        (timesteps, cumulative_soc)
    """
    if not paths:
        return [], []
    
    max_length = max(len(path) for path in paths)
    timesteps = list(range(max_length))
    cumulative_soc = []
    
    for t in range(max_length):
        total_cost = 0.0
        
        for path in paths:
            # Bu agent t zamanına kadar ne kadar yol aldı?
            if t < len(path):
                # t'ye kadar olan yolun maliyeti
                partial_path = path[:t+1]
                total_cost += calculate_path_cost(partial_path)
            else:
                # Agent hedefe ulaşmış, tüm yolun maliyeti
                total_cost += calculate_path_cost(path)
        
        cumulative_soc.append(total_cost)
    
    return timesteps, cumulative_soc


def calculate_active_agents_over_time(paths: List[List[Tuple[int, int]]]) -> Tuple[List[int], List[int]]:
    """
    Zamana göre aktif agent sayısını hesaplar (AUC için)
    
    Returns:
        (timesteps, active_agents)
    """
    if not paths:
        return [], []
    
    max_length = max(len(path) for path in paths)
    timesteps = list(range(max_length))
    active_agents = []
    
    for t in range(max_length):
        # Bu timestep'te kaç agent aktif?
        active = sum(1 for path in paths if t < len(path))
        active_agents.append(active)
    
    return timesteps, active_agents


def calculate_cumulative_auc(active_agents: List[int]) -> List[float]:
    """
    Kümülatif AUC hesaplar (her timestep'e kadar olan alan)
    
    Args:
        active_agents: Her timestep'teki aktif agent sayısı
        
    Returns:
        Kümülatif AUC değerleri
    """
    cumulative_auc = []
    total = 0.0
    
    for active in active_agents:
        total += active
        cumulative_auc.append(total)
    
    return cumulative_auc


def plot_soc_over_time(paths: List[List[Tuple[int, int]]], 
                       save_path: str = "soc_over_time.png",
                       show: bool = True):
    """
    Zamana göre kümülatif SOC grafiği çizer
    """
    timesteps, cumulative_soc = calculate_soc_over_time(paths)
    
    plt.figure(figsize=(12, 6))
    plt.plot(timesteps, cumulative_soc, linewidth=2, color='#2E86AB', marker='o', 
             markersize=3, markevery=max(1, len(timesteps)//20))
    
    plt.xlabel('Timestep', fontsize=12, fontweight='bold')
    plt.ylabel('Cumulative Sum of Costs', fontsize=12, fontweight='bold')
    plt.title('Sum of Costs Over Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Final değeri göster
    if cumulative_soc:
        final_soc = cumulative_soc[-1]
        plt.axhline(y=final_soc, color='red', linestyle='--', alpha=0.5, 
                   label=f'Final SOC: {final_soc:.2f}')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"SOC grafiği kaydedildi: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_auc_over_time(paths: List[List[Tuple[int, int]]], 
                      save_path: str = "auc_over_time.png",
                      show: bool = True):
    """
    Zamana göre aktif agent sayısı ve kümülatif AUC grafiği çizer
    """
    timesteps, active_agents = calculate_active_agents_over_time(paths)
    cumulative_auc = calculate_cumulative_auc(active_agents)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Üst grafik: Aktif agent sayısı
    ax1.fill_between(timesteps, active_agents, alpha=0.3, color='#A23B72', label='Active Agents')
    ax1.plot(timesteps, active_agents, linewidth=2, color='#A23B72', marker='s', 
            markersize=3, markevery=max(1, len(timesteps)//20))
    ax1.set_xlabel('Timestep', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Active Agents', fontsize=12, fontweight='bold')
    ax1.set_title('Active Agents Over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend()
    
    # Alt grafik: Kümülatif AUC
    ax2.fill_between(timesteps, cumulative_auc, alpha=0.3, color='#F18F01', label='Cumulative AUC')
    ax2.plot(timesteps, cumulative_auc, linewidth=2, color='#F18F01', marker='D', 
            markersize=3, markevery=max(1, len(timesteps)//20))
    ax2.set_xlabel('Timestep', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative AUC', fontsize=12, fontweight='bold')
    ax2.set_title('Area Under Curve (AUC) Over Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Final AUC değerini göster
    if cumulative_auc:
        final_auc = cumulative_auc[-1]
        ax2.axhline(y=final_auc, color='red', linestyle='--', alpha=0.5, 
                   label=f'Final AUC: {final_auc:.2f}')
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"AUC grafiği kaydedildi: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_combined_metrics(paths: List[List[Tuple[int, int]]], 
                         save_path: str = "combined_metrics.png",
                         show: bool = True):
    """
    SOC ve AUC'yi aynı grafikte gösterir (2 Y ekseni)
    """
    timesteps_soc, cumulative_soc = calculate_soc_over_time(paths)
    timesteps_auc, active_agents = calculate_active_agents_over_time(paths)
    cumulative_auc = calculate_cumulative_auc(active_agents)
    
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # Sol Y ekseni: SOC
    color1 = '#2E86AB'
    ax1.set_xlabel('Timestep', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Cumulative Sum of Costs', fontsize=12, fontweight='bold', color=color1)
    ax1.plot(timesteps_soc, cumulative_soc, linewidth=2.5, color=color1, 
            marker='o', markersize=4, markevery=max(1, len(timesteps_soc)//15), 
            label='Cumulative SOC')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Sağ Y ekseni: AUC
    ax2 = ax1.twinx()
    color2 = '#F18F01'
    ax2.set_ylabel('Cumulative AUC', fontsize=12, fontweight='bold', color=color2)
    ax2.plot(timesteps_auc, cumulative_auc, linewidth=2.5, color=color2, 
            marker='s', markersize=4, markevery=max(1, len(timesteps_auc)//15), 
            label='Cumulative AUC')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Başlık
    plt.title('SOC and AUC Over Time', fontsize=14, fontweight='bold', pad=20)
    
    # Legend'ları birleştir
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Birleşik metrik grafiği kaydedildi: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_agent_completion_timeline(paths: List[List[Tuple[int, int]]], 
                                   save_path: str = "agent_timeline.png",
                                   show: bool = True):
    """
    Her agentin ne zaman başlayıp ne zaman bitirdiğini gösteren timeline
    """
    fig, ax = plt.subplots(figsize=(14, max(8, len(paths) * 0.5)))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(paths)))
    
    for i, path in enumerate(paths):
        start_time = 0
        end_time = len(path) - 1
        
        # Agent çizgisi
        ax.barh(i, end_time - start_time, left=start_time, height=0.8, 
               color=colors[i], alpha=0.7, edgecolor='black', linewidth=1)
        
        # Agent etiketleri
        ax.text(-2, i, f'Agent {i}', va='center', ha='right', fontsize=10, fontweight='bold')
        
        # Bitiş zamanı
        ax.text(end_time + 1, i, f't={end_time}', va='center', fontsize=9)
    
    ax.set_xlabel('Timestep', fontsize=12, fontweight='bold')
    ax.set_ylabel('Agent ID', fontsize=12, fontweight='bold')
    ax.set_title('Agent Completion Timeline', fontsize=14, fontweight='bold')
    ax.set_yticks(range(len(paths)))
    ax.set_yticklabels([f'{i}' for i in range(len(paths))])
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    
    # Makespan çizgisi
    makespan = max(len(path) - 1 for path in paths)
    ax.axvline(x=makespan, color='red', linestyle='--', linewidth=2, 
              label=f'Makespan: {makespan}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Agent timeline grafiği kaydedildi: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_all_metrics(paths: List[List[Tuple[int, int]]], 
                    output_dir: str = "metrics_plots",
                    show: bool = False):
    """
    Tüm metriklerin grafiklerini çizer ve kaydeder
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("GRAFİKLER ÇİZİLİYOR...")
    print("="*60)
    
    # 1. SOC Over Time
    plot_soc_over_time(paths, 
                      save_path=os.path.join(output_dir, "soc_over_time.png"),
                      show=show)
    
    # 2. AUC Over Time
    plot_auc_over_time(paths, 
                      save_path=os.path.join(output_dir, "auc_over_time.png"),
                      show=show)
    
    # 3. Combined Metrics
    plot_combined_metrics(paths, 
                         save_path=os.path.join(output_dir, "combined_metrics.png"),
                         show=show)
    
    # 4. Agent Timeline
    plot_agent_completion_timeline(paths, 
                                  save_path=os.path.join(output_dir, "agent_timeline.png"),
                                  show=show)
    
    print("="*60)
    print(f"Tüm grafikler '{output_dir}' klasörüne kaydedildi.")
    print("="*60)


def plot_sa_convergence(cost_history: List[float], 
                       save_path: str = "sa_convergence.png",
                       show: bool = True):
    """
    Simulated Annealing'in öğrenme eğrisini gösterir
    """
    iterations = list(range(len(cost_history)))
    
    plt.figure(figsize=(14, 7))
    plt.plot(iterations, cost_history, linewidth=1, color='#6A4C93', alpha=0.6)
    
    # Moving average (yumuşatılmış)
    window_size = max(1, len(cost_history) // 50)
    if window_size > 1:
        smoothed = np.convolve(cost_history, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(cost_history)), smoothed, 
                linewidth=3, color='#C1121F', label='Moving Average')
    
    plt.xlabel('Iteration', fontsize=12, fontweight='bold')
    plt.ylabel('Cost', fontsize=12, fontweight='bold')
    plt.title('Simulated Annealing Convergence', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()
    
    # Best cost çizgisi
    best_cost = min(cost_history)
    plt.axhline(y=best_cost, color='green', linestyle='--', alpha=0.5, 
               label=f'Best Cost: {best_cost:.2f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"SA convergence grafiği kaydedildi: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


# Örnek kullanım
if __name__ == "__main__":
    # Test için örnek paths
    example_paths = [
        [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)],  # Agent 0: 4 adım
        [(5, 5), (6, 6), (7, 7), (8, 8)],          # Agent 1: 3 adım
        [(10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15)]  # Agent 2: 5 adım
    ]
    
    print("Örnek grafikler çiziliyor...")
    plot_all_metrics(example_paths, output_dir="example_plots", show=True)