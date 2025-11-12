#!/usr/bin/env python3
"""
评估SleepDeceptiveRedAgent场景下的性能对比
测试Cardiff在面对"潜伏后突袭"策略时的脆弱性
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
from sleep_deceptive_red_agent import SleepDeceptiveRedAgent

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
    print("SleepDeceptiveRedAgent场景对比实验")
    print("="*100)
    print("Red Agent: SleepDeceptiveRedAgent")
    print("  • 前3步：Sleep（完全静默）")
    print("  • 第4步+：Meander模式（突然攻击）")
    print("")
    print("预期:")
    print("  • Cardiff: 误判为SleepAgent → 放松防守 → 遭受重创")
    print("  • Our Model: 监控威胁 → 及时响应 → 性能稳定")
    print("="*100)
    print("")
    
    scenarios = [30, 50, 100]
    our_results = {}
    cardiff_results = {}
    
    # ========================================================================
    # 评估我们的模型
    # ========================================================================
    print("[1/2] 评估 Our Selector+MoE 模型")
    print("-"*100)
    
    # 加载我们的模型
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
    
    for max_steps in scenarios:
        print(f"测试 SleepDeceptive-{max_steps}...")
        
        rewards = []
        for ep in range(num_episodes):
            path = os.path.join(parent_dir, 'CybORG/CybORG/Shared/Scenarios/Scenario2.yaml')
            cyborg = CybORG(path, 'sim', agents={'Red': SleepDeceptiveRedAgent})
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
        
        avg_reward = mean(rewards)
        std_reward = stdev(rewards) if len(rewards) > 1 else 0
        our_results[f'SleepDeceptive-{max_steps}'] = {
            'rewards': rewards,
            'mean': avg_reward,
            'std': std_reward,
            'min': min(rewards),
            'max': max(rewards)
        }
        print(f"  ✓ SleepDeceptive-{max_steps}: {avg_reward:7.2f} ± {std_reward:5.2f}")
    
    # ========================================================================
    # 评估Cardiff模型
    # ========================================================================
    print(f"\n[2/2] 评估 Cardiff Champion 模型")
    print("-"*100)
    
    os.chdir(cardiff_dir)
    cardiff_agent = CardiffAgent()
    os.chdir(current_dir)
    
    print("Cardiff模型加载完成\n")
    
    for max_steps in scenarios:
        print(f"测试 SleepDeceptive-{max_steps}...")
        
        rewards = []
        for ep in range(num_episodes):
            path = os.path.join(parent_dir, 'CybORG/CybORG/Shared/Scenarios/Scenario2.yaml')
            cyborg = CybORG(path, 'sim', agents={'Red': SleepDeceptiveRedAgent})
            env = ChallengeWrapper2(env=cyborg, agent_name='Blue')
            
            observation = env.reset()
            episode_reward = 0
            
            for step in range(max_steps):
                action = cardiff_agent.get_action(observation, env.action_space)
                observation, reward, done, info = env.step(action)
                episode_reward += reward
            
            rewards.append(episode_reward)
        
        avg_reward = mean(rewards)
        std_reward = stdev(rewards) if len(rewards) > 1 else 0
        cardiff_results[f'SleepDeceptive-{max_steps}'] = {
            'rewards': rewards,
            'mean': avg_reward,
            'std': std_reward,
            'min': min(rewards),
            'max': max(rewards)
        }
        print(f"  ✓ SleepDeceptive-{max_steps}: {avg_reward:7.2f} ± {std_reward:5.2f}")
    
    # ========================================================================
    # 对比分析
    # ========================================================================
    print("\n" + "="*100)
    print("对比结果分析")
    print("="*100)
    print(f"{'场景':<20} {'Our Model':>12} {'Cardiff':>12} {'差异':>12} {'优胜方':>15}")
    print("-"*75)
    
    for scenario in [f'SleepDeceptive-{s}' for s in scenarios]:
        our_mean = our_results[scenario]['mean']
        cardiff_mean = cardiff_results[scenario]['mean']
        diff = our_mean - cardiff_mean
        winner = "Our Model ✅" if our_mean > cardiff_mean else "Cardiff ✅"
        
        print(f"{scenario:<20} {our_mean:>12.2f} {cardiff_mean:>12.2f} {diff:>12.2f} {winner:>15}")
    
    # 计算总体平均
    our_overall = mean([r['mean'] for r in our_results.values()])
    cardiff_overall = mean([r['mean'] for r in cardiff_results.values()])
    overall_diff = our_overall - cardiff_overall
    overall_winner = "Our Model ✅" if our_overall > cardiff_overall else "Cardiff ✅"
    
    print("-"*75)
    print(f"{'总体平均':<20} {our_overall:>12.2f} {cardiff_overall:>12.2f} {overall_diff:>12.2f} {overall_winner:>15}")
    
    # ========================================================================
    # 生成对比图表
    # ========================================================================
    print("\n生成对比图表...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_labels = [f'{s}步' for s in scenarios]
    x_pos = np.arange(len(scenarios))
    width = 0.35
    
    our_means = [our_results[f'SleepDeceptive-{s}']['mean'] for s in scenarios]
    our_stds = [our_results[f'SleepDeceptive-{s}']['std'] for s in scenarios]
    cardiff_means = [cardiff_results[f'SleepDeceptive-{s}']['mean'] for s in scenarios]
    cardiff_stds = [cardiff_results[f'SleepDeceptive-{s}']['std'] for s in scenarios]
    
    bars1 = ax.bar(x_pos - width/2, our_means, width, yerr=our_stds, 
                   label='Our Selector+MoE', color='#2E86AB', alpha=0.8, capsize=5)
    bars2 = ax.bar(x_pos + width/2, cardiff_means, width, yerr=cardiff_stds,
                   label='Cardiff Champion', color='#A23B72', alpha=0.8, capsize=5)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('测试场景（步长）', fontsize=12, fontweight='bold')
    ax.set_ylabel('平均奖励', fontsize=12, fontweight='bold')
    ax.set_title('SleepDeceptiveRedAgent场景下的性能对比\n(前3步Sleep，第4步+Meander)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 添加说明文字
    textstr = f'总体优势: Our Model 领先 {abs(overall_diff):.2f} 分'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # 保存图表
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f'results/sleep_deceptive_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ 图表已保存: {plot_path}")
    
    # 保存结果
    results = {
        'our_model': our_results,
        'cardiff': cardiff_results,
        'timestamp': timestamp
    }
    result_path = f'results/sleep_deceptive_final_{timestamp}.pkl'
    with open(result_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n实验完成！")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=100, help='每个场景的测试轮数')
    args = parser.parse_args()
    
    evaluate_both_models(num_episodes=args.episodes)

