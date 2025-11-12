#!/usr/bin/env python3
"""
最终评估脚本 - 完全独立运行

使用final_result文件夹内的所有资源
"""
import sys
import os

# 确保使用当前文件夹的模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(os.path.dirname(current_dir), 'CybORG'))

import pickle
from statistics import mean, stdev
from train_selector_phase2 import SelectorPhase2Trainer


def run_final_evaluation(num_episodes=100):
    """
    最终评估：9个标准场景
    
    Args:
        num_episodes: 每个场景的测试轮数
    """
    print("\n" + "="*100)
    print("最终评估 - Selector + MoE混合专家模型")
    print("="*100)
    print(f"测试场景: 9个标准场景 (B_line/Meander/Sleep × 30/50/100步)")
    print(f"测试轮数: {num_episodes}轮/场景")
    print("="*100 + "\n")
    
    # 9个标准场景
    test_scenarios = [
        {'name': 'B_line-30', 'scenario': 'B_line', 'steps': 30},
        {'name': 'B_line-50', 'scenario': 'B_line', 'steps': 50},
        {'name': 'B_line-100', 'scenario': 'B_line', 'steps': 100},
        {'name': 'Meander-30', 'scenario': 'Meander', 'steps': 30},
        {'name': 'Meander-50', 'scenario': 'Meander', 'steps': 50},
        {'name': 'Meander-100', 'scenario': 'Meander', 'steps': 100},
        {'name': 'Sleep-30', 'scenario': 'Sleep', 'steps': 30},
        {'name': 'Sleep-50', 'scenario': 'Sleep', 'steps': 50},
        {'name': 'Sleep-100', 'scenario': 'Sleep', 'steps': 100},
    ]
    
    # 初始化训练器
    print("正在初始化智能体...")
    trainer = SelectorPhase2Trainer()
    
    # 加载Selector
    selector_path = os.path.join(current_dir, 'checkpoints', 'selector_qtable.pkl')
    with open(selector_path, 'rb') as f:
        selector_data = pickle.load(f)
    trainer.selector.q_table = selector_data['q_table']
    trainer.selector.epsilon = 0.0  # 评估模式
    print(f"智能体加载完成 (Selector Q表: {len(trainer.selector.q_table)}个状态)\n")
    
    print("="*100)
    print("开始评估...")
    print("="*100 + "\n")
    results = {}
    
    for config in test_scenarios:
        name = config['name']
        scenario = config['scenario']
        max_steps = config['steps']
        
        print(f"  评估 {name}...", end=' ', flush=True)
        
        rewards = []
        for ep in range(num_episodes):
            ep_reward, steps, selections = trainer.run_episode(
                scenario=scenario,
                max_steps=max_steps,
                eval_mode=True,
                use_consistency_reward=False
            )
            rewards.append(ep_reward)
            
            if (ep + 1) % 25 == 0:
                print(f"\r  评估 {name} ({ep+1}/{num_episodes})...", end=' ', flush=True)
        
        results[name] = {
            'mean': mean(rewards),
            'std': stdev(rewards) if len(rewards) > 1 else 0,
            'min': min(rewards),
            'max': max(rewards),
            'rewards': rewards
        }
        
        print(f"\r  评估 {name} ({num_episodes}/{num_episodes})... ✓ "
              f"平均: {mean(rewards):7.2f} ± {stdev(rewards) if len(rewards) > 1 else 0:5.2f}")
    
    # ============================================================================
    # 结果汇总
    # ============================================================================
    print("\n" + "="*100)
    print("评估结果汇总")
    print("="*100)
    print(f"{'场景':<15} {'平均奖励':>12} {'标准差':>10} {'最小值':>10} {'最大值':>10}")
    print("-" * 70)
    
    for config in test_scenarios:
        name = config['name']
        r = results[name]
        print(f"{name:<15} {r['mean']:12.2f} {r['std']:10.2f} {r['min']:10.2f} {r['max']:10.2f}")
    
    
    # ============================================================================
    # 分类得分
    # ============================================================================
    print("\n" + "="*100)
    print("分类得分")
    print("="*100)
    
    bline_scores = [results[f'B_line-{s}']['mean'] for s in [30, 50, 100]]
    meander_scores = [results[f'Meander-{s}']['mean'] for s in [30, 50, 100]]
    sleep_scores = [results[f'Sleep-{s}']['mean'] for s in [30, 50, 100]]
    
    print(f"B_line平均:   {mean(bline_scores):7.2f}")
    print(f"Meander平均:  {mean(meander_scores):7.2f}")
    print(f"Sleep平均:    {mean(sleep_scores):7.2f}")
    print(f"\n总体平均:     {mean(bline_scores + meander_scores + sleep_scores):7.2f}")
    
    print("\n" + "="*100)
    print("评估完成！")
    print("="*100 + "\n")
    
    # 保存结果
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(current_dir, f'evaluation_results_{timestamp}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump({
            'results': results,
            'num_episodes': num_episodes,
            'timestamp': timestamp
        }, f)
    print(f"[保存] 结果已保存到: {save_path}\n")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='最终评估脚本')
    parser.add_argument('--episodes', type=int, default=100,
                       help='每个场景的测试轮数 (默认: 100)')
    
    args = parser.parse_args()
    
    run_final_evaluation(num_episodes=args.episodes)

