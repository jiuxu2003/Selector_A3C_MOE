#!/usr/bin/env python3
"""
混合场景训练：50% B_line-30 + 50% Meander-50
奖励重塑后的全新训练

重要特性：
1. ✅ 混合场景训练（B_line和Meander均衡）
2. ✅ 统一学习率 8e-4（无调度，简单直接）
3. ✅ Critic loss weight: 0.02
4. ✅ 动作分布监控：每200 episode
5. ✅ 基于子网选择分布优化训练效率

训练配置：
- 场景: 50% B_line-30步 + 50% Meander-50步
- Episodes: 15,000
- 所有子网: 从零初始化
- Learning rate: 8e-4（统一，无调度）
- Critic loss weight: 0.02
- 监控间隔: 每200 episodes
- 保存间隔: 每500 episodes

场景选择理由：
- B_line-30: User 55.5%, Enterprise 41.4% → Enterprise/Op充分训练
- Meander-50: 更长episode，更多Red攻击，更激烈对抗
- 混合比例: 1:1 → 平衡训练所有子网
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'CybORG'))

import numpy as np
import pickle
import torch
import torch.nn.functional as F
import random
from statistics import mean, stdev
from collections import defaultdict, Counter

from CybORG import CybORG
from CybORG.Agents import B_lineAgent, RedMeanderAgent
from CybORG.Agents.Wrappers import ChallengeWrapper
from CybORG.Shared import Results

# Phase 9L组件
from hardcoded_selector import HardcodedSelector
from subnet_observation_utils import SubnetObservationExtractor
from subnet_reward_utils import SubnetRewardCalculator
from god_view_reward_utils import get_god_view_metrics, get_all_hostnames, get_action_type

# 使用v2版本
from simple_hierarchical_a3c_v2 import SimpleHierarchicalA3CAgent


class MixedScenarioTrainer:
    """混合场景训练器（70% Meander + 30% B_line）"""
    
    def __init__(self, critic_loss_weight=0.02, learning_rate=8e-4):
        self.agent = SimpleHierarchicalA3CAgent()
        self.hardcoded_selector = HardcodedSelector()
        self.obs_extractor = SubnetObservationExtractor()
        self.reward_calculator = SubnetRewardCalculator()
        
        # 设置Critic loss weight
        self.agent.critic_loss_weight = critic_loss_weight
        
        # 学习率配置
        self.learning_rate = learning_rate
        
        # 训练统计
        self.episode_count = 0
        self.episode_rewards = []
        self.subnet_reward_history = defaultdict(list)
        
        # 场景统计
        self.scenario_counts = {'Meander': 0, 'B_line': 0}
        self.scenario_rewards = {'Meander': [], 'B_line': []}
        
        # 动作分布统计（每200 episode重置）
        self.action_distribution = {
            'User': Counter(),
            'Enterprise': Counter(),
            'Operational': Counter()
        }
        self.last_print_episode = 0
        self.cross_target_stats = {
            'User': {'ok': 0, 'bad': 0},
            'Enterprise': {'ok': 0, 'bad': 0},
            'Operational': {'ok': 0, 'bad': 0}
        }
        
    def select_scenario(self):
        """选择训练场景：50% B_line-30, 50% Meander-50（1:1比例）"""
        if random.random() < 0.5: 
            return 'B_line'
        else:
            return 'Meander'
    
    def create_environment(self, scenario='Meander'):
        """创建指定场景的环境"""
        from CybORG.Agents import SleepAgent
        
        # 使用父目录的CybORG
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        path = os.path.join(parent_dir, 'CybORG/CybORG/Shared/Scenarios/Scenario2.yaml')
        
        if scenario == 'Meander':
            cyborg = CybORG(path, 'sim', agents={'Red': RedMeanderAgent})
        elif scenario == 'Sleep':
            cyborg = CybORG(path, 'sim', agents={'Red': SleepAgent})
        else:  # B_line
            cyborg = CybORG(path, 'sim', agents={'Red': B_lineAgent})
        
        env = ChallengeWrapper(env=cyborg, agent_name='Blue')
        return env, cyborg
    
    def parse_observation_to_dict(self, raw_obs):
        """将CybORG观察对象转换为字典格式（为hardcoded_selector使用）"""
        obs_dict = {}
        
        if isinstance(raw_obs, dict):
            source_data = raw_obs
        elif hasattr(raw_obs, 'data') and isinstance(raw_obs.data, dict):
            source_data = raw_obs.data
        else:
            return obs_dict
        
        for hostname, host_info in source_data.items():
            if hostname == 'success':
                continue
            
            if not isinstance(host_info, dict):
                continue
            
            # 转换为hardcoded_selector期望的格式：将列表转为数字
            obs_dict[hostname] = {
                'Sessions': self._count_red_sessions(host_info),
                'Privileged Access': self._check_privileged(host_info),
                'Processes': self._count_suspicious_processes(host_info),
                'System info': host_info.get('System info', {})
            }
        
        return obs_dict
    
    def _count_red_sessions(self, host_info):
        """统计Red session数量"""
        if 'Sessions' not in host_info:
            return 0
        sessions = host_info['Sessions']
        if not isinstance(sessions, list):
            return 0
        red_count = sum(1 for s in sessions if isinstance(s, dict) and s.get('Agent') == 'Red')
        return red_count
    
    def _check_privileged(self, host_info):
        """检查是否有特权访问（返回1或0）"""
        if 'Sessions' not in host_info:
            return 0
        sessions = host_info['Sessions']
        if not isinstance(sessions, list):
            return 0
        for s in sessions:
            if isinstance(s, dict) and s.get('Agent') == 'Red':
                username = s.get('Username', '')
                if username in ['SYSTEM', 'root']:
                    return 1
        return 0
    
    def _count_suspicious_processes(self, host_info):
        """统计可疑进程数量"""
        if 'Processes' not in host_info:
            return 0
        processes = host_info['Processes']
        if not isinstance(processes, list):
            return 0
        suspicious_count = 0
        for proc in processes:
            if isinstance(proc, dict):
                proc_name = proc.get('Process Name', '')
                if 'red' in proc_name.lower() or 'malicious' in proc_name.lower():
                    suspicious_count += 1
        return suspicious_count
    
    def extract_subnet_observation(self, cyborg_obs_dict, selected_subnet):
        """提取子网观察"""
        return self.obs_extractor.extract_subnet_observation(cyborg_obs_dict, selected_subnet)
    
    def run_episode(self, scenario='Meander', max_steps=30, op_focus=False, eval_mode=False):
        """运行单个episode"""
        # 创建环境
        env, cyborg = self.create_environment(scenario)
        
        observation = env.reset()
        done = False
        step = 0
        
        total_reward = 0.0
        subnet_rewards = defaultdict(float)
        subnet_actions = defaultdict(list)
        
        # God-view metrics追踪
        prev_metrics_dict = {}
        last_analyse_state = {}
        
        # 记录最后的威胁分数
        last_threat_scores = {'User': 0.0, 'Enterprise': 0.0, 'Operational': 0.0}
        
        if eval_mode:
            for subnet_name in ['User', 'Enterprise', 'Operational']:
                self.agent.subnet_experts[subnet_name].eval()
        
        while not done and step < max_steps:
            # 获取真实状态
            true_state = cyborg.get_agent_state('True')
            cyborg_obs_dict = self.parse_observation_to_dict(true_state)
            
            selected_subnet, threat_scores = self.hardcoded_selector.select_subnet_with_scores(cyborg_obs_dict)
            last_threat_scores = threat_scores  # 保存最后的威胁分数
            
            # 提取子网观察
            subnet_obs = self.extract_subnet_observation(cyborg_obs_dict, selected_subnet)
            
            forced_sleep = op_focus and selected_subnet in ['User', 'Enterprise']
            if forced_sleep:
                global_action_idx = 0
                local_action_idx = None
            else:
                if eval_mode:
                    with torch.no_grad():
                        global_action_idx = self.agent.get_action_with_subnet_obs(
                            subnet_obs,
                            selected_subnet,
                            explore=False
                        )
                else:
                    global_action_idx = self.agent.get_action_with_subnet_obs(
                        subnet_obs,
                        selected_subnet,
                        explore=True
                    )
                local_action_idx = self.agent.last_decision['subnet_action']
            
            # 执行动作
            next_observation, env_reward, done, info = env.step(global_action_idx)
            
            # 获取动作对象和类型
            action_obj = cyborg.get_last_action('Blue')
            action_type = get_action_type(action_obj)
            target_host = getattr(action_obj, 'hostname', None)
            if target_host is not None:
                allowed_hosts = self.reward_calculator.get_subnet_hosts(selected_subnet)
                if target_host in allowed_hosts:
                    self.cross_target_stats[selected_subnet]['ok'] += 1
                else:
                    self.cross_target_stats[selected_subnet]['bad'] += 1
            
            # 计算子网塑形奖励（使用正确的参数顺序）
            shaped_reward = self.reward_calculator.calculate_subnet_shaped_reward(
                action_obj, action_type, cyborg, selected_subnet,
                prev_metrics_dict, last_analyse_state
            )
            
            subnet_rewards[selected_subnet] += shaped_reward
            subnet_actions[selected_subnet].append(action_type)
            
            if not eval_mode and not (op_focus and selected_subnet in ['User', 'Enterprise']):
                expert = self.agent.subnet_experts[selected_subnet]
                if not hasattr(expert, 'experience_buffer'):
                    expert.experience_buffer = []
                subnet_obs_tensor = torch.tensor(subnet_obs, dtype=torch.float32, device=self.agent.device)
                next_subnet_obs_tensor = torch.tensor(
                    self.extract_subnet_observation(
                        self.parse_observation_to_dict(cyborg.get_agent_state('True')),
                        selected_subnet
                    ),
                    dtype=torch.float32,
                    device=self.agent.device
                )
                with torch.no_grad():
                    action_logits, value = expert(subnet_obs_tensor.unsqueeze(0))
                    action_probs = torch.softmax(action_logits, dim=-1)
                    action_dist = torch.distributions.Categorical(action_probs)
                    log_prob = action_dist.log_prob(torch.tensor(local_action_idx, device=self.agent.device))
                expert.experience_buffer.append({
                    'state': subnet_obs_tensor,
                    'action': local_action_idx,
                    'reward': shaped_reward,
                    'next_state': next_subnet_obs_tensor,
                    'done': done,
                    'log_prob': log_prob,
                    'value': value.item()
                })
                if len(expert.experience_buffer) >= expert.expert_batch_size:
                    optimizer = self.agent.subnet_optimizers[selected_subnet]
                    self.agent._train_expert(expert, optimizer)
                    expert.experience_buffer.clear()
            
            total_reward += env_reward
            observation = next_observation
            
            # 更新God-view metrics（遍历所有主机）
            for hostname in get_all_hostnames(cyborg):
                metrics = get_god_view_metrics(cyborg, hostname)
                if metrics:
                    prev_metrics_dict[hostname] = metrics
            
            # 更新Analyse状态
            if action_type == 'Analyse':
                target_host = getattr(action_obj, 'hostname', None)
                if target_host:
                    curr_metrics = get_god_view_metrics(cyborg, target_host)
                    if curr_metrics:
                        last_analyse_state[target_host] = {
                            'sessions': curr_metrics.get('red_session_count', 0),
                            'access': curr_metrics.get('access_level', 'None'),
                            'malware': curr_metrics.get('malware_files', 0)
                        }
            
            step += 1
        
        # 更新动作分布统计
        for subnet_name in ['User', 'Enterprise', 'Operational']:
            if subnet_name in subnet_actions:
                self.action_distribution[subnet_name].update(subnet_actions[subnet_name])
        
        # 更新场景统计
        self.scenario_counts[scenario] += 1
        self.scenario_rewards[scenario].append(total_reward)
        
        if eval_mode:
            for subnet_name in ['User', 'Enterprise', 'Operational']:
                self.agent.subnet_experts[subnet_name].train()
        return total_reward, subnet_rewards, step, scenario, last_threat_scores
    
    def print_action_distribution(self, episode_start, episode_end):
        """打印动作分布统计"""
        print(f"\n{'='*80}")
        print(f"动作分布统计 [Episodes {episode_start+1}-{episode_end}]")
        print(f"{'='*80}")
        
        for subnet_name in ['User', 'Enterprise', 'Operational']:
            counter = self.action_distribution[subnet_name]
            total = sum(counter.values())
            
            if total == 0:
                print(f"\n{subnet_name}子网: 未被选择")
                continue
            
            print(f"\n{subnet_name}子网 (总动作数: {total}):")
            
            # 按动作类型分组统计（Monitor已移除）
            action_groups = {
                'Sleep': 0,
                'Analyse': 0,
                'Remove': 0,
                'Restore': 0,
                'Decoy': 0
            }
            
            for action_type, count in counter.items():
                if 'Sleep' in action_type:
                    action_groups['Sleep'] += count
                elif 'Analyse' in action_type:
                    action_groups['Analyse'] += count
                elif 'Remove' in action_type:
                    action_groups['Remove'] += count
                elif 'Restore' in action_type:
                    action_groups['Restore'] += count
                elif 'Decoy' in action_type:
                    action_groups['Decoy'] += count
            
            # 打印分组统计
            sorted_groups = sorted(action_groups.items(), key=lambda x: x[1], reverse=True)
            for group_name, group_count in sorted_groups:
                pct = 100.0 * group_count / total
                if group_count > 0:
                    # 标记异常情况
                    warning = ""
                    if pct > 70:
                        warning = " ⚠️ 过度依赖！"
                    elif pct > 50:
                        warning = " ⚠️ 占比较高"
                    
                    print(f"  {group_name:12}: {group_count:6} ({pct:5.1f}%){warning}")
            
            # 打印详细动作（只显示前10个最常见的）
            print(f"  详细分布（Top 10）:")
            for action_type, count in counter.most_common(10):
                pct = 100.0 * count / total
                print(f"    {action_type:30}: {count:6} ({pct:5.1f}%)")
            ok = self.cross_target_stats[subnet_name]['ok']
            bad = self.cross_target_stats[subnet_name]['bad']
            total_targets = ok + bad
            if total_targets > 0:
                bad_pct = 100.0 * bad / total_targets
                print(f"  目标主机归属校验: OK={ok}, BAD={bad} ({bad_pct:5.1f}% 非本子网)")
        
        print(f"{'='*80}\n")
        
        # 重置计数器
        for subnet_name in self.action_distribution:
            self.action_distribution[subnet_name].clear()
        for subnet_name in self.cross_target_stats:
            self.cross_target_stats[subnet_name]['ok'] = 0
            self.cross_target_stats[subnet_name]['bad'] = 0
    
    def print_scenario_stats(self):
        """打印场景统计"""
        print(f"\n场景分布统计:")
        total = sum(self.scenario_counts.values())
        for scenario, count in self.scenario_counts.items():
            pct = 100.0 * count / total if total > 0 else 0
            avg_reward = mean(self.scenario_rewards[scenario]) if self.scenario_rewards[scenario] else 0
            print(f"  {scenario:10}: {count:5} episodes ({pct:5.1f}%) | 平均奖励: {avg_reward:7.2f}")
    
    def train(self, total_episodes=15000, save_interval=500, action_dist_interval=200, start_episode=0):
        """训练主循环"""
        print(f"\n{'='*80}")
        print(f"混合场景训练 - 50% B_line-50 + 50% Meander-100")
        print(f"{'='*80}")
        if start_episode > 0:
            print(f"[恢复训练] 从Episode {start_episode}继续")
            print(f"剩余Episodes: {total_episodes - start_episode}")
        else:
            print(f"总Episodes: {total_episodes}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Critic loss weight: {self.agent.critic_loss_weight}")
        print(f"场景混合比例: 50% B_line-50 + 50% Meander-100（1:1均衡训练）")
        print(f"动作分布打印间隔: 每{action_dist_interval} episodes")
        print(f"Checkpoint保存间隔: 每{save_interval} episodes")
        print(f"{'='*80}\n")
        
        # 初始化Agent（需要在第一个环境上）
        if not self.agent.subnet_experts:
            env, _ = self.create_environment('Meander')
            action_space_size = env.get_action_space('Blue')
            self.agent._initialize_experts(action_space_size)
            print(f"[初始化] Agent已创建，动作空间大小: {action_space_size}\n")
        
        for ep in range(start_episode, total_episodes):
            # 选择场景
            scenario = self.select_scenario()
            
            # 根据场景设置步数：B_line-50, Meander-100
            max_steps = 50 if scenario == 'B_line' else 100
            
            # 运行episode
            ep_reward, subnet_rewards, steps, actual_scenario, threat_scores = self.run_episode(scenario, max_steps=max_steps)
            
            self.episode_count += 1
            self.episode_rewards.append(ep_reward)
            
            for subnet_name, reward in subnet_rewards.items():
                self.subnet_reward_history[subnet_name].append(reward)
            
            # 每10个episode打印一次简要统计
            if (ep + 1) % 10 == 0:
                recent_rewards = self.episode_rewards[-10:]
                avg_reward = mean(recent_rewards)
                
                # 计算子网平均奖励
                subnet_avgs = {}
                for subnet_name in ['User', 'Enterprise', 'Operational']:
                    if subnet_name in self.subnet_reward_history:
                        recent_subnet = self.subnet_reward_history[subnet_name][-10:]
                        if recent_subnet:
                            subnet_avgs[subnet_name] = mean(recent_subnet)
                
                # 场景标记（带步数）
                scenario_label = f"[{ 'M100' if actual_scenario == 'Meander' else 'B50'}]"
                
                # 格式化威胁分数
                threat_str = f"U={threat_scores.get('User', 0):.1f} E={threat_scores.get('Enterprise', 0):.1f} O={threat_scores.get('Operational', 0):.1f}"
                
                print(f"Ep {ep+1:5}/{total_episodes} {scenario_label} | "
                      f"环境: {avg_reward:7.2f} | "
                      f"User: {subnet_avgs.get('User', 0):6.2f} | "
                      f"Ent: {subnet_avgs.get('Enterprise', 0):6.2f} | "
                      f"Op: {subnet_avgs.get('Operational', 0):6.2f} | "
                      f"威胁: {threat_str}")
            
            # 每action_dist_interval个episode打印动作分布
            if (ep + 1) % action_dist_interval == 0:
                self.print_action_distribution(self.last_print_episode, ep + 1)
                self.print_scenario_stats()
                self.last_print_episode = ep + 1
            
            # 每save_interval个episode保存checkpoint
            if (ep + 1) % save_interval == 0:
                self.save_checkpoint(f"mixed_scenarios_ep{ep+1}")
                self.print_detailed_stats(ep + 1)
        
        # 训练结束，保存最终checkpoint
        self.save_checkpoint("mixed_scenarios_final")
        self.print_detailed_stats(total_episodes, final=True)
    
    def print_detailed_stats(self, current_ep, final=False):
        """打印详细统计"""
        prefix = "最终" if final else f"Episode {current_ep}"
        
        print(f"\n{'='*80}")
        print(f"{prefix}统计")
        print(f"{'='*80}")
        
        # 环境奖励统计
        if self.episode_rewards:
            recent = self.episode_rewards[-100:]
            print(f"环境奖励 (最近100 episodes):")
            print(f"  平均: {mean(recent):7.2f}")
            if len(recent) > 1:
                print(f"  标准差: {stdev(recent):7.2f}")
            print(f"  范围: [{min(recent):7.2f}, {max(recent):7.2f}]")
        
        # 子网奖励统计
        print(f"\n子网奖励 (最近100 episodes):")
        for subnet_name in ['User', 'Enterprise', 'Operational']:
            if subnet_name in self.subnet_reward_history:
                recent = self.subnet_reward_history[subnet_name][-100:]
                if recent:
                    print(f"  {subnet_name:12}: {mean(recent):7.2f} ± {stdev(recent) if len(recent) > 1 else 0:6.2f}")
        
        # 场景分布统计
        self.print_scenario_stats()
        
        print(f"{'='*80}\n")
    
    def load_checkpoint(self, checkpoint_name):
        """加载checkpoint恢复训练"""
        checkpoint_dir = 'checkpoints'
        
        # 加载训练统计
        stats_path = f"{checkpoint_dir}/{checkpoint_name}.pkl"
        if not os.path.exists(stats_path):
            print(f"[错误] 找不到checkpoint: {stats_path}")
            return False
        
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
        
        self.episode_count = stats['episode']
        self.episode_rewards = stats['episode_rewards']
        self.subnet_reward_history = defaultdict(list, stats['subnet_reward_history'])
        self.scenario_counts = stats['scenario_counts']
        self.scenario_rewards = stats['scenario_rewards']
        self.learning_rate = stats.get('learning_rate', self.learning_rate)
        
        # 加载各子网expert（使用直观命名）
        checkpoint_files = {
            'User': os.path.join(checkpoint_dir, 'user_expert_weights.pth'),
            'Enterprise': os.path.join(checkpoint_dir, 'enterprise_expert_weights.pth'),
            'Operational': os.path.join(checkpoint_dir, 'operational_expert_weights.pth')
        }
        
        for subnet_name in ['User', 'Enterprise', 'Operational']:
            checkpoint_path = checkpoint_files[subnet_name]
            if not os.path.exists(checkpoint_path):
                print(f"[错误] 找不到{subnet_name}的checkpoint: {checkpoint_path}")
                return False
            
            checkpoint = torch.load(checkpoint_path, map_location=self.agent.device, weights_only=False)
            
            expert = self.agent.subnet_experts[subnet_name]
            optimizer = self.agent.subnet_optimizers[subnet_name]
            
            # 兼容加载：仅加载形状匹配的参数（例如Enterprise动作头已变更）
            model_state = expert.state_dict()
            saved_state = checkpoint['expert_state_dict']
            compatible_state = {}
            skipped = []
            for k, v in saved_state.items():
                if k in model_state and isinstance(v, torch.Tensor) and isinstance(model_state[k], torch.Tensor):
                    if v.shape == model_state[k].shape:
                        compatible_state[k] = v
                    else:
                        skipped.append(k)
                elif k in model_state and not isinstance(v, torch.Tensor):
                    # 非Tensor（如元数据）直接跳过
                    skipped.append(k)
            
            # 先用当前模型参数填充，再更新兼容参数
            new_state = {**model_state, **compatible_state}
            expert.load_state_dict(new_state, strict=False)
            if skipped:
                print(f"[警告] {subnet_name} 局部加载: 跳过不兼容权重 {len(skipped)} 项，例如 {skipped[:3]}")
            
            # 加载优化器，如失败则重置
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                print(f"[警告] {subnet_name} 优化器状态加载失败，使用新优化器: {e}")
                # 依据当前学习率重建优化器
                optimizer.param_groups[0]['lr'] = self.learning_rate
        
        print(f"[恢复] 从Episode {self.episode_count}恢复训练")
        print(f"  已完成episodes: {self.episode_count}")
        print(f"  平均奖励: {np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else 0:.2f}")
        print(f"  场景分布: Meander={self.scenario_counts['Meander']}, B_line={self.scenario_counts['B_line']}")
        return True
    
    def save_checkpoint(self, name):
        """保存checkpoint"""
        checkpoint_dir = 'checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 保存agent
        for subnet_name in ['User', 'Enterprise', 'Operational']:
            expert = self.agent.subnet_experts[subnet_name]
            optimizer = self.agent.subnet_optimizers[subnet_name]
            
            checkpoint = {
                'expert_state_dict': expert.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'episode': self.episode_count,
                'learning_rate': self.learning_rate,
            }
            
            filename = f"{checkpoint_dir}/{name}_{subnet_name}.pth"
            torch.save(checkpoint, filename)
        
        # 保存训练统计
        stats = {
            'episode': self.episode_count,
            'episode_rewards': self.episode_rewards,
            'subnet_reward_history': dict(self.subnet_reward_history),
            'scenario_counts': self.scenario_counts,
            'scenario_rewards': self.scenario_rewards,
            'critic_loss_weight': self.agent.critic_loss_weight,
            'learning_rate': self.learning_rate,
        }
        
        with open(f"{checkpoint_dir}/{name}.pkl", 'wb') as f:
            pickle.dump(stats, f)
        
        print(f"[保存] Checkpoint: {name}")

    def train_op_focus(self, total_episodes=10000, save_interval=500, action_dist_interval=200, start_episode=0):
        print(f"\n{'='*80}")
        print(f"混合场景训练（OP-FOCUS） - 50% B_line-100 + 50% Meander-100")
        print(f"{'='*80}")
        if start_episode > 0:
            print(f"[恢复训练] 从Episode {start_episode}继续")
            print(f"剩余Episodes: {total_episodes - start_episode}")
        else:
            print(f"总Episodes: {total_episodes}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Critic loss weight: {self.agent.critic_loss_weight}")
        print(f"场景混合比例: 50% B_line-100 + 50% Meander-100（1:1均衡训练）")
        print(f"动作分布打印间隔: 每{action_dist_interval} episodes")
        print(f"Checkpoint保存间隔: 每{save_interval} episodes")
        print(f"{'='*80}\n")

        if not self.agent.subnet_experts:
            env, _ = self.create_environment('Meander')
            action_space_size = env.get_action_space('Blue')
            self.agent._initialize_experts(action_space_size)
            print(f"[初始化] Agent已创建，动作空间大小: {action_space_size}\n")

        for ep in range(start_episode, total_episodes):
            scenario = self.select_scenario()
            max_steps = 100

            ep_reward, subnet_rewards, steps, actual_scenario, threat_scores = self.run_episode(
                scenario, max_steps=max_steps, op_focus=True
            )

            self.episode_count += 1
            self.episode_rewards.append(ep_reward)

            for subnet_name, reward in subnet_rewards.items():
                self.subnet_reward_history[subnet_name].append(reward)

            if (ep + 1) % 10 == 0:
                recent_rewards = self.episode_rewards[-10:]
                avg_reward = mean(recent_rewards)

                subnet_avgs = {}
                for subnet_name in ['User', 'Enterprise', 'Operational']:
                    if subnet_name in self.subnet_reward_history:
                        recent_subnet = self.subnet_reward_history[subnet_name][-10:]
                        if recent_subnet:
                            subnet_avgs[subnet_name] = mean(recent_subnet)

                scenario_label = f"[{ 'M100' if actual_scenario == 'Meander' else 'B100'}]"
                threat_str = f"U={threat_scores.get('User', 0):.1f} E={threat_scores.get('Enterprise', 0):.1f} O={threat_scores.get('Operational', 0):.1f}"
                print(f"Ep {ep+1:5}/{total_episodes} {scenario_label} | "
                      f"环境: {avg_reward:7.2f} | "
                      f"User: {subnet_avgs.get('User', 0):6.2f} | "
                      f"Ent: {subnet_avgs.get('Enterprise', 0):6.2f} | "
                      f"Op: {subnet_avgs.get('Operational', 0):6.2f} | "
                      f"威胁: {threat_str}")

            if (ep + 1) % action_dist_interval == 0:
                self.print_action_distribution(self.last_print_episode, ep + 1)
                self.print_scenario_stats()
                self.last_print_episode = ep + 1

            if (ep + 1) % save_interval == 0:
                self.save_checkpoint(f"op_focus_ep{ep+1}")
                self.print_detailed_stats(ep + 1)

        self.save_checkpoint("op_focus_final")
        self.print_detailed_stats(total_episodes, final=True)


    def load_mixed_checkpoints(self, ue_ckpt, op_ckpt):
        # 使用当前文件所在目录的checkpoints文件夹
        current_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_dir = os.path.join(current_dir, 'checkpoints')
        stats_path = f"{checkpoint_dir}/{ue_ckpt}.pkl"
        if os.path.exists(stats_path):
            with open(stats_path, 'rb') as f:
                stats = pickle.load(f)
            self.episode_rewards = stats.get('episode_rewards', [])
            self.scenario_counts = stats.get('scenario_counts', {'Meander': 0, 'B_line': 0})
            self.scenario_rewards = stats.get('scenario_rewards', {'Meander': [], 'B_line': []})
            self.learning_rate = stats.get('learning_rate', self.learning_rate)
        # 新的文件名映射（直观命名）
        checkpoint_files = {
            'User': os.path.join(checkpoint_dir, 'user_expert_weights.pth'),
            'Enterprise': os.path.join(checkpoint_dir, 'enterprise_expert_weights.pth'),
            'Operational': os.path.join(checkpoint_dir, 'operational_expert_weights.pth')
        }
        
        for subnet_name in ['User', 'Enterprise', 'Operational']:
            checkpoint_path = checkpoint_files[subnet_name]
            if not os.path.exists(checkpoint_path):
                print(f"[错误] 找不到{subnet_name}的checkpoint: {checkpoint_path}")
                return False
            checkpoint = torch.load(checkpoint_path, map_location=self.agent.device, weights_only=False)
            expert = self.agent.subnet_experts[subnet_name]
            optimizer = self.agent.subnet_optimizers[subnet_name]
            model_state = expert.state_dict()
            saved_state = checkpoint['expert_state_dict']
            compatible_state = {}
            for k, v in saved_state.items():
                if k in model_state and isinstance(v, torch.Tensor) and isinstance(model_state[k], torch.Tensor) and v.shape == model_state[k].shape:
                    compatible_state[k] = v
            new_state = {**model_state, **compatible_state}
            expert.load_state_dict(new_state, strict=False)
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception:
                optimizer.param_groups[0]['lr'] = self.learning_rate
        return True


    def evaluate_bline100(self, episodes=50):
        if not self.agent.subnet_experts:
            env, _ = self.create_environment('B_line')
            action_space_size = env.get_action_space('Blue')
            self.agent._initialize_experts(action_space_size)
        rewards = []
        subnet_rewards_hist = {'User': [], 'Enterprise': [], 'Operational': []}
        last_threat = {'User': 0.0, 'Enterprise': 0.0, 'Operational': 0.0}
        for i in range(episodes):
            ep_reward, subnet_rewards, steps, actual_scenario, threat_scores = self.run_episode('B_line', max_steps=100, op_focus=False, eval_mode=True)
            rewards.append(ep_reward)
            for k in subnet_rewards_hist.keys():
                subnet_rewards_hist[k].append(subnet_rewards.get(k, 0.0))
            last_threat = threat_scores
            
            if (i + 1) % 10 == 0:
                recent_rewards = rewards[-10:]
                avg_reward = mean(recent_rewards)
                subnet_avgs = {}
                for k in ['User', 'Enterprise', 'Operational']:
                    recent_subnet = subnet_rewards_hist[k][-10:]
                    if recent_subnet:
                        subnet_avgs[k] = mean(recent_subnet)
                scenario_label = "[B100]"
                threat_str = f"U={last_threat.get('User', 0):.1f} E={last_threat.get('Enterprise', 0):.1f} O={last_threat.get('Operational', 0):.1f}"
                print(f"Eval {i+1:5}/{episodes} {scenario_label} | "
                      f"环境: {avg_reward:7.2f} | "
                      f"User: {subnet_avgs.get('User', 0):6.2f} | "
                      f"Ent: {subnet_avgs.get('Enterprise', 0):6.2f} | "
                      f"Op: {subnet_avgs.get('Operational', 0):6.2f} | "
                      f"威胁: {threat_str}")
        print(f"\n评估(B_line-100) {episodes} episodes")
        print(f"环境奖励: 平均={mean(rewards):.2f} 范围=[{min(rewards):.2f}, {max(rewards):.2f}]")
        print(f"Operational奖励: 平均={mean(subnet_rewards_hist['Operational']) if subnet_rewards_hist['Operational'] else 0.0:.2f}")

    def evaluate_scenario(self, scenario, steps=100, episodes=50):
        if not self.agent.subnet_experts:
            env, _ = self.create_environment(scenario)
            action_space_size = env.get_action_space('Blue')
            self.agent._initialize_experts(action_space_size)
        rewards = []
        subnet_rewards_hist = {'User': [], 'Enterprise': [], 'Operational': []}
        last_threat = {'User': 0.0, 'Enterprise': 0.0, 'Operational': 0.0}
        for i in range(episodes):
            ep_reward, subnet_rewards, steps_used, actual_scenario, threat_scores = self.run_episode(scenario, max_steps=steps, op_focus=False, eval_mode=True)
            rewards.append(ep_reward)
            for k in subnet_rewards_hist.keys():
                subnet_rewards_hist[k].append(subnet_rewards.get(k, 0.0))
            last_threat = threat_scores
            if (i + 1) % 10 == 0:
                recent_rewards = rewards[-10:]
                avg_reward = mean(recent_rewards)
                subnet_avgs = {}
                for k in ['User', 'Enterprise', 'Operational']:
                    recent_subnet = subnet_rewards_hist[k][-10:]
                    if recent_subnet:
                        subnet_avgs[k] = mean(recent_subnet)
                tag = ('B' if scenario == 'B_line' else 'M') + str(steps)
                threat_str = f"U={last_threat.get('User', 0):.1f} E={last_threat.get('Enterprise', 0):.1f} O={last_threat.get('Operational', 0):.1f}"
                print(f"Eval {i+1:5}/{episodes} [{tag}] | "
                      f"环境: {avg_reward:7.2f} | "
                      f"User: {subnet_avgs.get('User', 0):6.2f} | "
                      f"Ent: {subnet_avgs.get('Enterprise', 0):6.2f} | "
                      f"Op: {subnet_avgs.get('Operational', 0):6.2f} | "
                      f"威胁: {threat_str}")
        tag = ('B' if scenario == 'B_line' else 'M') + str(steps)
        print(f"\n评估({scenario}-{steps}) {episodes} episodes")
        print(f"环境奖励: 平均={mean(rewards):.2f} 范围=[{min(rewards):.2f}, {max(rewards):.2f}]")
        print(f"Operational奖励: 平均={mean(subnet_rewards_hist['Operational']) if subnet_rewards_hist['Operational'] else 0.0:.2f}")


def main():
    import sys
    trainer = MixedScenarioTrainer(critic_loss_weight=0.02, learning_rate=8e-4)

    args = sys.argv[1:]
    mode = 'normal'
    checkpoint_name = None
    add_extra = None
    if len(args) > 0:
        if args[0] == 'op_focus':
            mode = 'op_focus'
            checkpoint_name = args[1] if len(args) > 1 else None
        elif args[0] == 'eval_mix':
            mode = 'eval_mix'
        elif args[0] == 'eval_mix6':
            mode = 'eval_mix6'
        else:
            checkpoint_name = args[0]

    # parse optional add=<N> to only add N episodes when resuming
    if len(args) > 1:
        for a in args[1:]:
            if isinstance(a, str) and a.startswith('add='):
                try:
                    add_extra = int(a.split('=', 1)[1])
                except Exception:
                    add_extra = None
            if isinstance(a, str) and a.startswith('n='):
                try:
                    os.environ['EVAL_N'] = a.split('=', 1)[1]
                except Exception:
                    pass

    start_episode = 0
    if checkpoint_name and mode != 'eval_mix':
        print(f"[加载] 尝试从checkpoint恢复: {checkpoint_name}")
        from CybORG import CybORG
        from CybORG.Agents import RedMeanderAgent
        from CybORG.Agents.Wrappers import ChallengeWrapper
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        path = os.path.join(parent_dir, 'CybORG/CybORG/Shared/Scenarios/Scenario2.yaml')
        cyborg = CybORG(path, 'sim', agents={'Red': RedMeanderAgent})
        env = ChallengeWrapper(env=cyborg, agent_name='Blue')
        action_space_size = env.get_action_space('Blue')
        trainer.agent._initialize_experts(action_space_size)
        if trainer.load_checkpoint(checkpoint_name):
            start_episode = trainer.episode_count
        else:
            print(f"[警告] 加载checkpoint失败，从零开始训练")

    if mode == 'op_focus':
        if checkpoint_name and add_extra is not None:
            target_total = start_episode + add_extra
        else:
            target_total = (start_episode + 10000) if checkpoint_name else 10000
        trainer.train_op_focus(total_episodes=target_total, save_interval=500, action_dist_interval=200, start_episode=start_episode)
    elif mode == 'eval_mix':
        if len(args) < 3:
            print("用法: python train_mixed_scenarios.py eval_mix <ue_ckpt> <op_ckpt> [n=Episodes]")
            return
        ue_ckpt = args[1]
        op_ckpt = args[2]
        from CybORG import CybORG
        from CybORG.Agents import RedMeanderAgent
        from CybORG.Agents.Wrappers import ChallengeWrapper
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        path = os.path.join(parent_dir, 'CybORG/CybORG/Shared/Scenarios/Scenario2.yaml')
        cyborg = CybORG(path, 'sim', agents={'Red': RedMeanderAgent})
        env = ChallengeWrapper(env=cyborg, agent_name='Blue')
        action_space_size = env.get_action_space('Blue')
        trainer.agent._initialize_experts(action_space_size)
        ok = trainer.load_mixed_checkpoints(ue_ckpt, op_ckpt)
        if not ok:
            return
        n = int(os.environ.get('EVAL_N', '50'))
        trainer.evaluate_bline100(episodes=n)
    elif mode == 'eval_mix6':
        if len(args) < 3:
            print("用法: python train_mixed_scenarios.py eval_mix6 <ue_ckpt> <op_ckpt> [n=Episodes]")
            return
        ue_ckpt = args[1]
        op_ckpt = args[2]
        from CybORG import CybORG
        from CybORG.Agents import RedMeanderAgent
        from CybORG.Agents.Wrappers import ChallengeWrapper
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        path = os.path.join(parent_dir, 'CybORG/CybORG/Shared/Scenarios/Scenario2.yaml')
        cyborg = CybORG(path, 'sim', agents={'Red': RedMeanderAgent})
        env = ChallengeWrapper(env=cyborg, agent_name='Blue')
        action_space_size = env.get_action_space('Blue')
        trainer.agent._initialize_experts(action_space_size)
        ok = trainer.load_mixed_checkpoints(ue_ckpt, op_ckpt)
        if not ok:
            return
        n = int(os.environ.get('EVAL_N', '100'))
        scenarios = [('B_line', 30), ('B_line', 50), ('B_line', 100), ('Meander', 30), ('Meander', 50), ('Meander', 100)]
        for sc, st in scenarios:
            trainer.evaluate_scenario(sc, steps=st, episodes=n)
    else:
        target_total = (start_episode + 15000) if checkpoint_name else 15000
        trainer.train(total_episodes=target_total, save_interval=500, action_dist_interval=200, start_episode=start_episode)


if __name__ == '__main__':
    main()
