#!/usr/bin/env python3
"""
评估DeceptiveRedAgent场景下的性能对比
专门测试Cardiff的fingerprint机制弱点
"""
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
final_result_dir = os.path.join(parent_dir, 'final_result')
cardiff_dir = os.path.join(parent_dir, 'cardiff')

sys.path.insert(0, parent_dir)
sys.path.insert(0, final_result_dir)
sys.path.insert(0, cardiff_dir)
sys.path.insert(0, os.path.join(parent_dir, 'CybORG'))

import pickle
import torch
from statistics import mean, stdev
import datetime
import matplotlib.pyplot as plt
import numpy as np

from CybORG import CybORG
from CybORG.Agents.Wrappers import ChallengeWrapper
from deceptive_red_agent import DeceptiveRedAgent

# Our model components
from hardcoded_selector import HardcodedSelector
from subnet_observation_utils import SubnetObservationExtractor
from train_mixed_scenarios import MixedScenarioTrainer

# Cardiff components
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2
from Agents.MainAgent import MainAgent as CardiffAgent

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300


def evaluate_both_models(num_episodes=100):
    """
    同时评估两个模型
    """
    print("="*100)
    print("DeceptiveRedAgent场景对比实验")
    print("="*100)
    print("Red Agent: DeceptiveRedAgent")
    print("  • 前3步：B_line模式（扫描2次）")
    print("  • 第4步+：Meander模式（随机游走）")
    print("\n预期:")
    print("  • Cardiff: 会被误导为B_line，加载错误模型")
    print("  • Our Model: 基于威胁，不受影响")
    print("="*100 + "\n")
    
    all_results = {}
    
    # ========================================================================
    # 评估我们的模型
    # ========================================================================
    print("\n[1/2] 评估 Our Selector+MoE 模型")
    print("-"*100)
    
    os.chdir(final_result_dir)
    trainer = MixedScenarioTrainer()
    trainer.agent._initialize_experts(145)
    ok = trainer.load_mixed_checkpoints('user_enterprise_expert_metadata', 
                                        'operational_expert_metadata')
    if not ok:
        raise RuntimeError("无法加载checkpoints")
    
    os.chdir(current_dir)
    selector = HardcodedSelector()
    obs_extractor = SubnetObservationExtractor()
    agent = trainer.agent
    
    print("模型加载完成\n")
    
    our_results = {}
    
    for max_steps in [30, 50, 100]:
        name = f'Deceptive-{max_steps}'
        print(f"测试 {name}...")
        
        rewards = []
        for ep in range(num_episodes):
            path = os.path.join(parent_dir, 'CybORG/CybORG/Shared/Scenarios/Scenario2.yaml')
            cyborg = CybORG(path, 'sim', agents={'Red': DeceptiveRedAgent})
            env = ChallengeWrapper(env=cyborg, agent_name='Blue')
            
            observation = env.reset()
            episode_reward = 0
            
            for step in range(max_steps):
                true_state = cyborg.get_agent_state('True')
                cyborg_obs_dict = trainer.parse_observation_to_dict(true_state)
                
                selected_subnet = selector.select_subnet(cyborg_obs_dict)
                subnet_obs = obs_extractor.extract_subnet_observation(cyborg_obs_dict, selected_subnet)
                
                expert = agent.subnet_experts[selected_subnet]
                with torch.no_grad():
                    subnet_obs_tensor = torch.FloatTensor(subnet_obs).unsqueeze(0).to(expert.device)
                    shared_features = expert.shared_layers(subnet_obs_tensor)
                    action_probs = expert.actor_head(shared_features)
                    action_dist = torch.distributions.Categorical(torch.softmax(action_probs, dim=-1))
                    local_action = action_dist.sample()
                
                global_action = expert.action_indices[local_action.item()]
                observation, reward, done, info = env.step(global_action)
                episode_reward += reward
            
            rewards.append(episode_reward)
        
        our_results[name] = {
            'mean': mean(rewards),
            'std': stdev(rewards) if len(rewards) > 1 else 0,
            'rewards': rewards
        }
        print(f"  ✓ {name}: {mean(rewards):7.2f} ± {stdev(rewards) if len(rewards) > 1 else 0:5.2f}")
    
    all_results['Our'] = our_results
    
    # ========================================================================
    # 评估Cardiff模型
    # ========================================================================
    print("\n[2/2] 评估 Cardiff Champion 模型")
    print("-"*100)
    
    os.chdir(cardiff_dir)
    cardiff_agent = CardiffAgent()
    os.chdir(current_dir)
    
    print("Cardiff模型加载完成\n")
    
    cardiff_results = {}
    
    for max_steps in [30, 50, 100]:
        name = f'Deceptive-{max_steps}'
        print(f"测试 {name}...")
        
        rewards = []
        for ep in range(num_episodes):
            os.chdir(cardiff_dir)
            path = os.path.join(parent_dir, 'CybORG/CybORG/Shared/Scenarios/Scenario2.yaml')
            cyborg = CybORG(path, 'sim', agents={'Red': DeceptiveRedAgent})
            env = ChallengeWrapper2(env=cyborg, agent_name='Blue')
            
            observation = env.reset()
            episode_reward = 0
            
            for step in range(max_steps):
                action = cardiff_agent.get_action(observation, env.get_action_space('Blue'))
                observation, reward, done, info = env.step(action)
                episode_reward += reward
            
            rewards.append(episode_reward)
            cardiff_agent.end_episode()
            os.chdir(current_dir)
        
        cardiff_results[name] = {
            'mean': mean(rewards),
            'std': stdev(rewards) if len(rewards) > 1 else 0,
            'rewards': rewards
        }
        print(f"  ✓ {name}: {mean(rewards):7.2f} ± {stdev(rewards) if len(rewards) > 1 else 0:5.2f}")
    
    all_results['Cardiff'] = cardiff_results
    
    # ========================================================================
    # 对比分析
    # ========================================================================
    print("\n" + "="*100)
    print("对比结果分析")
    print("="*100)
    print(f"{'场景':<18} {'Our Model':>12} {'Cardiff':>12} {'差异':>12} {'优胜方':>15}")
    print("-" * 75)
    
    for scenario in ['Deceptive-30', 'Deceptive-50', 'Deceptive-100']:
        our_mean = all_results['Our'][scenario]['mean']
        cardiff_mean = all_results['Cardiff'][scenario]['mean']
        diff = our_mean - cardiff_mean
        winner = "Our Model ✅" if our_mean > cardiff_mean else "Cardiff"
        
        print(f"{scenario:<18} {our_mean:12.2f} {cardiff_mean:12.2f} {diff:+12.2f} {winner:>15}")
    
    # 总体
    our_avg = mean([all_results['Our'][s]['mean'] for s in ['Deceptive-30', 'Deceptive-50', 'Deceptive-100']])
    cardiff_avg = mean([all_results['Cardiff'][s]['mean'] for s in ['Deceptive-30', 'Deceptive-50', 'Deceptive-100']])
    
    print("-" * 75)
    print(f"{'总体平均':<18} {our_avg:12.2f} {cardiff_avg:12.2f} {our_avg - cardiff_avg:+12.2f} {'Our Model ✅' if our_avg > cardiff_avg else 'Cardiff':>15}")
    
    # 保存结果
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('results', exist_ok=True)
    
    with open(f'results/deceptive_comparison_{timestamp}.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    
    # 生成图表
    print("\n生成对比图表...")
    
    scenarios_short = ['30 steps', '50 steps', '100 steps']
    our_means_list = [all_results['Our'][f'Deceptive-{s}']['mean'] for s in [30, 50, 100]]
    cardiff_means_list = [all_results['Cardiff'][f'Deceptive-{s}']['mean'] for s in [30, 50, 100]]
    our_stds_list = [all_results['Our'][f'Deceptive-{s}']['std'] for s in [30, 50, 100]]
    cardiff_stds_list = [all_results['Cardiff'][f'Deceptive-{s}']['std'] for s in [30, 50, 100]]
    
    x = np.arange(len(scenarios_short))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars1 = ax.bar(x - width/2, our_means_list, width, yerr=our_stds_list, capsize=5,
                   label='Our Selector+MoE', color='#2c3e50', 
                   edgecolor='black', linewidth=1.5, alpha=0.9)
    bars2 = ax.bar(x + width/2, cardiff_means_list, width, yerr=cardiff_stds_list, capsize=5,
                   label='Cardiff Champion', color='#e74c3c', 
                   edgecolor='black', linewidth=1.5, alpha=0.9)
    
    ax.set_xlabel('Episode Length', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Reward', fontsize=14, fontweight='bold')
    ax.set_title('Performance on DeceptiveRedAgent\n(First 3 steps: B_line, Then: Meander)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios_short, fontsize=13)
    ax.legend(fontsize=12)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom' if height > 0 else 'top',
                   fontsize=10, fontweight='bold')
    
    # 标注优胜方
    for i in range(len(scenarios_short)):
        diff = our_means_list[i] - cardiff_means_list[i]
        if diff > 0:
            ax.text(i, min(our_means_list[i], cardiff_means_list[i]) - 3,
                   f'Our +{diff:.1f}', ha='center', fontsize=9, 
                   fontweight='bold', color='green')
        else:
            ax.text(i, min(our_means_list[i], cardiff_means_list[i]) - 3,
                   f'Cardiff +{abs(diff):.1f}', ha='center', fontsize=9, 
                   fontweight='bold', color='red')
    
    plt.tight_layout()
    plt.savefig('results/deceptive_comparison.png')
    print(f"✓ 图表已保存: results/deceptive_comparison.png")
    
    print("\n实验完成！")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=100)
    args = parser.parse_args()
    
    evaluate_both_models(num_episodes=args.episodes)

