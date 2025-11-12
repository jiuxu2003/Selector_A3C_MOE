#!/usr/bin/env python3
"""
Phase 9L: 简化分层A3C智能体 v2
核心改进：
1. 移除Q-learning Selector → 使用硬编码威胁分数选择器
2. 子网观察空间分离：User(24维), Enterprise(16维), Operational(20维)
3. 子网奖励隔离：每个Expert只关心自己子网的奖励/惩罚
4. Critic Loss权重：0.5 → 0.015
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'CybORG'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
from collections import deque, defaultdict
from statistics import mean
from typing import List

from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from CybORG.Shared import Results


class SubnetA3CExpert(nn.Module):
    """
    Phase 9L: 子网A3C专家 v2
    
    核心改进：
    1. 支持子网独立观察空间维度（不再是全局52维）
    2. 每个子网有自己的观察维度：User(24), Enterprise(16), Operational(20)
    """
    
    def __init__(self, subnet_name: str, host_indices: List[int], action_dim: int, 
                 obs_dim: int = None, device='cuda'):
        super(SubnetA3CExpert, self).__init__()
        
        self.subnet_name = subnet_name
        self.host_indices = host_indices
        self.action_dim = action_dim
        self.device = device
        
        # Phase 9L: 使用子网独立观察空间维度
        # 如果未指定，则使用旧的计算方式（向后兼容）
        if obs_dim is not None:
            self.input_dim = obs_dim
        else:
            # 旧方式：监控主机数 × 4特征
            self.input_dim = len(host_indices) * 4
        
        # Phase 3B: 匹配数据复杂度的网络架构（还原原始设计）
        # 共享底层 + 独立head（适合A3C）
        self.shared_layers = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        self.actor_head = nn.Linear(32, action_dim)
        self.critic_head = nn.Linear(32, 1)
        
        self.to(device)
        self.apply(self._init_weights)
        
        # Phase 4A: 差异化子网超参数
        self._setup_subnet_hyperparams()
        
        # 子网特定的防御策略参数
        self._setup_defense_strategy()
        
        # Phase 5: action_indices会在_initialize_experts中设置
        # 此时先初始化为空，稍后调用setup_action_types
        self.action_indices = []
        self.action_types = {}
        
    
    def _setup_subnet_hyperparams(self):
        """
        Phase 9F: 统一超参数（动作集已简化且对等）
        
        核心改进：
        1. 三个子网动作数相近（14-17个）→ 学习率统一
        2. 参考PPO/A3C经验：lr=3e-4是常见选择
        3. 保留适度的熵系数差异（鼓励不同探索策略）
        """
        # Phase 9L: 统一学习率为8e-4
        # 所有子网使用相同的学习率以保证训练一致性
        self.expert_lr = 8e-4  # 统一学习率
        
        # 统一批次大小
        self.expert_batch_size = 64  # 标准batch size
        
        # Phase 9K: 统一并提高熵系数（所有子网都需要探索复杂场景）
        # 混合场景（30/50/100步 + Meander/B-line）需要更多探索
        self.entropy_coef = 0.05  # 统一提高到0.05（从0.01-0.03）
        self.reward_scale = 1.0
    
    def _setup_action_mapping(self):
        """Phase 5修复：基于真实全局索引的动作类型映射"""
        # Phase 5: 使用action_indices（全局索引）来精确分类动作类型
        # 不要用本地索引，而是根据全局索引判断动作类型！
        
        self.action_types = {}
        
        # 全局动作范围（基于CybORG验证结果）
        # 注意：Monitor(1)已移除，因为环境会自动执行
        GLOBAL_ACTION_RANGES = {
            'Sleep': [0],
            'Analyse': list(range(2, 15)),       # 2-14
            'Remove': list(range(15, 28)),       # 15-27
            'DecoyApache': list(range(28, 41)),  # 28-40
            'DecoyFemitter': list(range(41, 54)), # 41-53
            'DecoyHarakaSMTP': list(range(54, 67)), # 54-66
            'DecoySmss': list(range(67, 80)),    # 67-79
            'DecoySSHD': list(range(80, 93)),    # 80-92
            'DecoySvchost': list(range(93, 106)), # 93-105
            'DecoyTomcat': list(range(106, 119)), # 106-118
            'DecoyVsftpd': list(range(119, 132)), # 119-131
            'Restore': list(range(132, 145))      # 132-144
        }
        
        # 将全局索引转换为本地索引
        for action_type, global_indices in GLOBAL_ACTION_RANGES.items():
            local_indices = []
            
            # 检查哪些全局索引在当前专家的action_indices中
            for local_idx, global_idx in enumerate(self.action_indices):
                if global_idx in global_indices:
                    local_indices.append(local_idx)
            
            self.action_types[action_type] = local_indices
        
        # Phase 7: 为启发式筛选暴露各类动作列表
        self.sleep_actions = self.action_types.get('Sleep', [])
        self.analyse_actions = self.action_types.get('Analyse', [])
        self.remove_actions = self.action_types.get('Remove', [])
        self.restore_actions = self.action_types.get('Restore', [])
        
        # 所有Decoy类型合并
        self.decoy_actions = []
        for key in self.action_types:
            if key.startswith('Decoy'):
                self.decoy_actions.extend(self.action_types[key])
        
        # 根据子网策略设置优先动作类型
        if self.subnet_name == 'User':
            self.preferred_actions = ['Sleep', 'Remove']
            self.avoided_actions = ['Restore']  # 成本高
        elif self.subnet_name == 'Enterprise': 
            self.preferred_actions = ['Analyse', 'Remove', 'Restore']
            self.avoided_actions = []  # 平衡策略
        else:  # Operational
            self.preferred_actions = ['Restore', 'Analyse', 'Remove'] 
            self.avoided_actions = ['Sleep']  # 不能懈怠
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0)
    
    def _setup_defense_strategy(self):
        """设置子网特定的防御策略"""
        if self.subnet_name == 'User':
            self.defense_strategy = "高容忍度"
            self.threat_threshold = 0.4  # 高阈值，容忍更多威胁
            self.action_preference = "Remove"  # 优先Remove
            self.preferred_actions = ['Sleep', 'Remove']
            self.avoided_actions = ['Restore']  # 成本高
        elif self.subnet_name == 'Enterprise':
            self.defense_strategy = "平衡策略"
            self.threat_threshold = 0.2  # 中等阈值
            self.action_preference = "Analyse+Remove"  # 先分析再Remove
            self.preferred_actions = ['Analyse', 'Remove', 'Restore']
            self.avoided_actions = []  # 平衡策略
        else:  # Operational
            self.defense_strategy = "低容忍度"
            self.threat_threshold = 0.1  # 极低阈值，快速响应
            self.action_preference = "Restore"  # 优先Restore确保安全
            self.preferred_actions = ['Restore', 'Analyse', 'Remove']
            self.avoided_actions = ['Sleep']  # 禁止懈怠
    
    def extract_subnet_state(self, full_observation):
        """提取子网状态"""
        if isinstance(full_observation, np.ndarray):
            full_obs = full_observation
        else:
            full_obs = np.array(full_observation)
        
        obs_reshaped = full_obs.reshape(13, 4)
        
        subnet_state = []
        for host_idx in self.host_indices:
            subnet_state.extend(obs_reshaped[host_idx])
        
        # Phase 7: 保存子网状态供_get_candidate_actions使用
        self.last_subnet_state = np.array(subnet_state)
        
        return torch.FloatTensor(subnet_state).to(self.device)
    
    def forward(self, state):
        """还原原始forward逻辑（共享层+独立head）"""
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        shared_features = self.shared_layers(state)
        action_logits = self.actor_head(shared_features)
        state_value = self.critic_head(shared_features)
        
        return action_logits, state_value.squeeze(-1)
    
    def get_action_and_value(self, state):
        action_logits, state_value = self.forward(state)
        
        # 获取候选动作（威胁门槛筛选）
        threat_level = self._assess_threat_level(state)
        candidate_actions = self._get_candidate_actions(threat_level)
        
        # 创建严格的动作掩码
        if len(candidate_actions) < self.action_dim:
            # 只有候选动作可以被选择
            action_mask = torch.zeros_like(action_logits, dtype=torch.bool)
            for action_idx in candidate_actions:
                if action_idx < len(action_mask):
                    action_mask[action_idx] = True
            
            # 应用掩码：非候选动作设置为极小logit
            masked_logits = action_logits.clone()
            masked_logits[~action_mask] = -1e8  # 极小值确保不被选择
            
            # 对候选动作应用轻微策略偏置
            for action_idx in candidate_actions:
                if action_idx < len(masked_logits):
                    bias = self._get_action_bias(action_idx, threat_level)
                    masked_logits[action_idx] += bias
            
            action_probs = F.softmax(masked_logits, dim=-1)
        else:
            # 所有动作都是候选（正常情况）
            action_probs = F.softmax(action_logits, dim=-1)
        
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        return action, log_prob, state_value
    
    
    def _get_candidate_actions(self, threat_level):
        """
        Phase 7: 基于动作类型的智能筛选
        
        核心思路：
        1. 根据威胁状态判断允许的动作类型
        2. 从action_types中筛选对应的本地索引
        3. A3C从筛选后的动作中学习
        """
        candidates = set()
        
        # 根据威胁状态判断允许的动作类型
        allowed_types = []
        
        if self.subnet_name == 'Operational':
            # 运维子网：最关键，严格筛选
            # Phase 7修复：降低威胁门限，让入侵快速触发强力防御
            if threat_level >= 0.15:  # 高威胁（任何主机入侵都>=0.25）
                # ❌ 不能Sleep/Decoy（有威胁立即处理）
                # ✅ 只能Remove/Restore - 直接行动！
                allowed_types = ['Remove', 'Restore']
            elif threat_level >= 0.05:  # 中等威胁（扫描）
                # ❌ 不能Decoy（有威胁时不部署诱饵）
                # ✅ 可以Analyse（侦查威胁）
                # ✅ 可以Remove/Restore（处理威胁）
                allowed_types = ['Analyse', 'Remove', 'Restore']
            else:  # 无威胁
                # ✅ 可以Sleep（监控状态）
                # ✅ 可以Analyse（主动侦查）
                # ✅ 可以Remove（预防性清理）
                # ✅ 可以Decoy（无威胁时才部署诱饵）
                allowed_types = ['Sleep', 'Analyse', 'Remove',
                                'DecoySSHD', 'DecoyApache', 'DecoyTomcat', 'DecoyHarakaSMTP']
                
        elif self.subnet_name == 'Enterprise':
            # 企业子网：重要，平衡防御
            if threat_level >= 0.5:  # 高威胁
                # ❌ 不能Sleep（高威胁不休息）
                # ❌ 不能Decoy（有威胁时，不部署诱饵）
                # ✅ 可以Analyse（评估威胁）
                # ✅ 可以Remove/Restore（处理威胁）
                allowed_types = ['Analyse', 'Remove', 'Restore']
            elif threat_level >= 0.2:  # 中等威胁
                # ❌ 不能Decoy（有威胁时，优先防御）
                # ✅ 可以Analyse/Remove/Restore（主动防御）
                # ✅ 可以Sleep（有时需要观察）
                allowed_types = ['Analyse', 'Remove', 'Restore', 'Sleep']
            else:  # 低威胁
                # ✅ 几乎全开放（预防和监控）
                allowed_types = ['Sleep', 'Analyse', 'Remove', 
                                'DecoySSHD', 'DecoyApache', 'DecoyTomcat', 'DecoyHarakaSMTP']
                
        else:  # User
            # 用户子网：价值低，容忍度高
            if threat_level >= 0.7:  # 极高威胁
                # ❌ 不能Decoy（极高威胁时，优先处理）
                # ✅ 可以Remove/Restore（处理威胁）
                # ✅ 可以Analyse（评估威胁）
                # ✅ 可以Sleep（有时容忍也是策略）
                allowed_types = ['Remove', 'Restore', 'Analyse', 'Sleep']
            elif threat_level >= 0.3:  # 中等威胁
                # ✅ 相对宽松，可以用Decoy（价值低，可以诱饵）
                # ✅ 可以Remove/Restore/Analyse（主动防御）
                # ✅ 可以Sleep（容忍策略）
                allowed_types = ['Remove', 'Restore', 'Analyse', 'Sleep',
                                'DecoySSHD', 'DecoyApache', 'DecoyTomcat']
            else:  # 低威胁
                # ✅ 全开放（预防和诱饵都可以）
                allowed_types = ['Sleep', 'Analyse', 'Remove', 'Restore',
                                'DecoySSHD', 'DecoyApache', 'DecoyTomcat', 'DecoyHarakaSMTP']
        
        # 从action_types中收集对应的本地索引
        for action_type in allowed_types:
            if action_type in self.action_types:
                candidates.update(self.action_types[action_type])
        
        # 确保有候选动作
        if not candidates:
            # 保底：返回所有动作
            candidates = set(range(self.action_dim))
        
        return list(candidates)
    
    def _get_action_bias(self, action_idx, threat_level):
        """为候选动作提供轻微的策略偏置"""
        bias = 0.0
        
        # 确定动作类型
        action_type = self._get_action_type(action_idx)
        
        if self.subnet_name == 'User':
            # 轻微偏好低成本动作和诱饵
            if action_type == 'Sleep' and threat_level < 0.3:
                bias += 0.2  # 低威胁时偏好休眠
            elif action_type.startswith('Decoy'):
                bias += 0.4  # 强烈偏好诱饵动作（低成本高效果）
            elif action_type == 'Remove':
                bias += 0.2
        elif self.subnet_name == 'Enterprise':
            # 轻微偏好分析和平衡防御
            if action_type == 'Analyse':
                bias += 0.25
            elif action_type == 'Remove':
                bias += 0.15
            elif action_type.startswith('Decoy'):
                bias += 0.1  # 适度使用诱饵
        else:  # Operational
            # 基于威胁级别的偏好
            if threat_level >= 0.05:
                # 有威胁时：偏好积极防御
                if action_type == 'Restore':
                    bias += 0.4
                elif action_type in ['Remove', 'Analyse']:
                    bias += 0.3
            else:
                # 无威胁时：适度偏好分析，允许Sleep
                if action_type == 'Analyse':
                    bias += 0.2  # 保持警惕
                elif action_type == 'Sleep':
                    bias += 0.1  # 允许但不鼓励
        
        return bias
    
    def _assess_threat_level(self, state):
        """
        威胁评估：基于子网重要性和红队动作威胁级别量化
        威胁分数 = 子网重要性权重 × 动作威胁级别
        """
        # 子网重要性权重
        subnet_weights = {
            'User': 0.3,      # 低价值子网
            'Enterprise': 0.6, # 中价值子网  
            'Operational': 1.0 # 高价值子网
        }
        
        # 红队动作威胁级别映射（基于Activity和Compromised状态推断）
        threat_levels = {
            'no_activity': 0,        # 无活动
            'scan_activity': 2,      # 扫描活动 (DiscoverRemoteSystems)
            'service_scan': 4,       # 服务扫描 (DiscoverNetworkServices)
            'exploit_attempt': 6,    # 利用尝试 (ExploitRemoteService)
            'privilege_escalation': 8, # 权限提升 (PrivilegeEscalate)
            'impact_action': 10      # 影响操作 (Impact)
        }
        
        # Phase 9L: 处理新的观察格式（num_hosts × 4 + 4统计）
        num_hosts = len(self.host_indices)
        
        # 检查是否是Phase 9L格式（包含额外的4维统计）
        total_size = state.numel()
        expected_size_with_stats = num_hosts * 4 + 4
        
        if total_size == expected_size_with_stats:
            # Phase 9L格式：提取前num_hosts×4维（主机信息）
            host_features = state.view(-1)[:num_hosts * 4]
            state_reshaped = host_features.view(num_hosts, 4)
        else:
            # 旧格式：直接reshape
            state_reshaped = state.view(num_hosts, 4)
        
        total_threat_score = 0.0
        subnet_weight = subnet_weights[self.subnet_name]
        
        for host_state in state_reshaped:
            activity = host_state[0]      # 活动状态 (0/1)
            scanned = host_state[1]       # 扫描状态 (0/1)
            privilege = host_state[2]     # 特权状态 (0/1/2)
            compromised = host_state[3]   # 入侵状态 (0/1)
            
            # 根据状态推断威胁级别（按优先级从高到低）
            host_threat_level = 0
            
            # 最高威胁：完全入侵
            if compromised > 0:
                host_threat_level = threat_levels['impact_action']  # 10
            # 次高威胁：权限提升
            elif privilege > 0:
                host_threat_level = threat_levels['privilege_escalation']  # 8
            # 中等威胁：有活动+被扫描（正在被利用）
            elif activity > 0 and scanned > 0:
                host_threat_level = threat_levels['exploit_attempt']  # 6
            # 低威胁：仅被扫描
            elif scanned > 0:
                host_threat_level = threat_levels['scan_activity']  # 2
            
            # 计算该主机的威胁分数
            host_threat_score = subnet_weight * host_threat_level
            total_threat_score += host_threat_score
        
        # 归一化：最大可能威胁 = 子网主机数 × 子网权重 × 最高威胁级别(10)
        max_possible_threat = num_hosts * subnet_weight * 10.0
        normalized_threat = total_threat_score / max_possible_threat if max_possible_threat > 0 else 0.0
        
        return min(normalized_threat, 1.0)  # 确保在[0,1]范围内
    
    def shape_reward(self, base_reward, action_taken, threat_level, prev_threat=None):
        """
        Phase 1增强：基于威胁量化的A3C奖励塑造
        1. 保持原始环境奖励为主要信号
        2. 根据威胁门槛和子网策略添加奖励调整
        3. 鼓励A3C学会威胁感知的防御决策
        4. 新增：威胁降低奖励（鼓励有效防御）
        """
        shaped_reward = base_reward  # 环境奖励是主要信号
        
        # 确定动作类型
        action_type = self._get_action_type(action_taken)
        
        # 威胁门槛策略奖励
        strategy_bonus = 0.0
        
        # Phase 7关键修复：完全移除威胁变化奖励
        # 原因：威胁变化受Red Agent随机性影响，引入巨大方差
        #       导致Critic Loss爆炸（User:20万, Enterprise:3.3万）
        # 
        # ❌ 威胁降低≠动作有效（可能是Red没攻击）
        # ❌ 威胁上升≠动作无效（可能是Red同时攻击多台）
        # ✅ 只用确定性的strategy bonus，基于动作类型
        #
        # if prev_threat is not None:
        #     threat_reduction = prev_threat - threat_level
        #     strategy_bonus += threat_reduction * coef  # 移除！
        
        if self.subnet_name == 'User':
            # Phase 7: User子网奖励强化（适度，避免掩盖环境信号）
            # 目标：引导而非主导
            if threat_level < 0.3:  # 低威胁时
                if action_type in ['Sleep', 'DecoyApache', 'DecoySSHD', 'DecoyTomcat']:
                    strategy_bonus += 0.8  # 适度提高（从0.5）
                elif action_type == 'Remove':
                    strategy_bonus += 0.5  # 适度提高（从0.3）
            else:  # 高威胁时
                if action_type == 'Remove':
                    strategy_bonus += 1.5  # 适度提高（从1.0）
                elif action_type == 'Restore':
                    strategy_bonus += 1.0  # 适度提高（从0.5）
                elif action_type == 'Analyse':
                    strategy_bonus += 0.8  # 适度提高（从0.5）
                    
        elif self.subnet_name == 'Enterprise':
            # Phase 7: Enterprise子网奖励强化（适度，避免掩盖环境信号）
            # 环境信号中等强度，适度补偿即可
            if threat_level < 0.2:  # 低威胁
                if action_type == 'Analyse':
                    strategy_bonus += 0.5  # 适度提高（从0.3）
                elif action_type.startswith('Decoy'):
                    strategy_bonus += 0.4  # 适度提高（从0.2）
            elif threat_level < 0.5:  # 中等威胁
                if action_type in ['Analyse', 'Remove']:
                    strategy_bonus += 0.8  # 适度提高（从0.5）
                elif action_type == 'Restore':
                    strategy_bonus += 0.6  # 适度提高（从0.3）
            else:  # 高威胁
                if action_type == 'Restore':
                    strategy_bonus += 1.2  # 适度提高（从0.8）
                elif action_type in ['Remove', 'Analyse']:
                    strategy_bonus += 1.0  # 适度提高（从0.6）
                    
        else:  # Operational子网
            # 高价值子网：Phase 3B保守调整
            if threat_level >= 0.1:  # 高威胁（极低门槛）
                if action_type == 'Restore':
                    strategy_bonus += 0.30  # 最高奖励
                elif action_type in ['Remove', 'Analyse']:
                    strategy_bonus += 0.24
                elif action_type == 'Sleep':
                    strategy_bonus += -0.30  # 严重惩罚
            elif threat_level >= 0.05:  # 中等威胁
                if action_type in ['Analyse', 'Remove']:
                    strategy_bonus += 0.16
                elif action_type == 'Sleep':
                    strategy_bonus += -0.10
            else:  # 无威胁
                if action_type == 'Analyse':
                    strategy_bonus += 0.08
                # 无威胁时允许Sleep，不额外奖励或惩罚
        
        # Phase 7: 临时关闭奖励塑造，只用环境信号！
        # 让我们看清楚A3C是否真的在学习
        # final_reward = (shaped_reward + strategy_bonus) * self.reward_scale
        final_reward = shaped_reward  # 只用base_reward
        return final_reward
    
    def _get_action_type(self, action_idx):
        """确定动作类型"""
        for action_type, indices in self.action_types.items():
            if action_idx in indices:
                return action_type
        return 'Unknown'


class SimpleSelector:
    """
    简化选择器
    核心逻辑：红方攻击哪个子网，蓝方就调用哪个子网专家
    """
    
    def __init__(self):
        # Q-table: 简单的state → action映射，使用随机初始化避免偏置
        self.q_table = defaultdict(lambda: {
            'User': np.random.uniform(-0.1, 0.1), 
            'Enterprise': np.random.uniform(-0.1, 0.1), 
            'Operational': np.random.uniform(-0.1, 0.1)
        })
        
        # 学习参数
        self.learning_rate = 0.1  # 降低学习率，避免Q值震荡
        self.epsilon = 1.0  # 从完全探索开始
        self.epsilon_decay = 0.9995  # 更慢的衰减率
        self.min_epsilon = 0.01  # 降低最小探索率到1%
        
        # 历史记录
        self.last_selection = None
        self.last_state_key = None
        
        # 统计
        self.total_updates = 0
        self.correct_selections = 0
        self.valid_decisions = 0  # 有攻击时的决策次数（必须初始化！）
        
        print(f"[选择器] 学习率={self.learning_rate}, 初始探索率={self.epsilon}")
    
    def get_state_key(self, observation):
        """生成简单的状态键"""
        obs_reshaped = observation.reshape(13, 4)
        
        # 计算每个子网的威胁级别
        subnet_threats = {}
        # ⚠️ 修复：根据Scenario2.yaml中Blue.INT.Hosts的实际顺序
        subnet_ranges = {
            'Enterprise': range(0, 4),    # Defender, Enterprise0-2
            'Operational': range(4, 8),   # Op_Host0-2, Op_Server0
            'User': range(8, 13)          # User0-4
        }
        
        for subnet_name, host_range in subnet_ranges.items():
            threat_level = 0
            for host_idx in host_range:
                host_state = obs_reshaped[host_idx]
                activity = host_state[:2]
                compromised = host_state[2:]
                
                # 简单威胁计分
                if not np.array_equal(activity, [0, 0]):  # 有Activity
                    threat_level += 1
                if not np.array_equal(compromised, [0, 0]):  # 被攻陷
                    threat_level += 2
            
            subnet_threats[subnet_name] = min(threat_level, 3)  # 限制在0-3
        
        # 生成状态键
        state_key = f"U{subnet_threats['User']}_E{subnet_threats['Enterprise']}_O{subnet_threats['Operational']}"
        return state_key, subnet_threats
    
    def select_subnet(self, observation):
        """选择子网"""
        state_key, subnet_threats = self.get_state_key(observation)
        
        # epsilon-greedy选择
        if np.random.random() < self.epsilon:
            # 探索：真正的随机选择
            selected_subnet = np.random.choice(['User', 'Enterprise', 'Operational'])
        else:
            # 利用：选择Q值最高的（确保state存在）
            if state_key not in self.q_table:
                # 新状态，使用默认初始化
                self.q_table[state_key] = {k: np.random.uniform(-0.1, 0.1) for k in ['User', 'Enterprise', 'Operational']}
            q_values = self.q_table[state_key]
            # 找到最大Q值
            max_q = max(q_values.values())
            # 如果有多个最大值，随机选择一个
            best_actions = [k for k, v in q_values.items() if v == max_q]
            selected_subnet = np.random.choice(best_actions)
        
        # 记录选择
        self.last_selection = selected_subnet
        self.last_state_key = state_key
        
        return selected_subnet
    
    def train_selector(self, reward, attacked_subnet):
        """训练选择器"""
        if self.last_selection is None or self.last_state_key is None:
            return
        
        self.total_updates += 1
        
        # 计算选择器奖励
        if attacked_subnet is not None:
            # 有攻击时：选择正确 → +1，选择错误 → -1
            selector_reward = 1.0 if self.last_selection == attacked_subnet else -1.0
            if self.last_selection == attacked_subnet:
                self.correct_selections += 1
            # 只有检测到攻击时才计入有效决策统计
            self.valid_decisions = getattr(self, 'valid_decisions', 0) + 1
        else:
            # 无攻击时：无奖励
            selector_reward = 0.0
        
        # Q-learning更新（使用温和的更新策略）
        current_q = self.q_table[self.last_state_key][self.last_selection]
        # TD-error based update with clipping to prevent extreme values
        td_error = selector_reward - current_q
        new_q = current_q + self.learning_rate * td_error
        # 限制Q值范围在[-2, 2]之间，避免数值不稳定
        new_q = np.clip(new_q, -2.0, 2.0)
        self.q_table[self.last_state_key][self.last_selection] = new_q
        
        # 仅在有效更新时衰减探索率（避免过快衰减）
        if attacked_subnet is not None:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
      
        return {
            'selected_subnet': self.last_selection,
            'attacked_subnet': attacked_subnet,
            'selector_reward': selector_reward,
            'q_change': new_q - current_q,
            'current_epsilon': self.epsilon
        }


class SimpleHierarchicalA3CAgent(BaseAgent):
    """
    简化分层A3C智能体
    双层结构：选择器 + 专家
    """
    
    # Phase 9F: 简化动作集，移除Decoy
    # 理念：统一动作集（Analyse + Remove + Restore），让MoE体现在策略差异上
    # 关键：每个子网只操作自己子网的主机
    # 
    # 动作索引规律（CybORG按主机顺序排列）:
    # - Analyse: 2-14 → Defender(2), Enterprise0-2(3-5), Op_Host0-2(6-8), Op_Server0(9), User0-4(10-14)
    # - Remove:  15-27 → 同样顺序
    # - Restore: 132-144 → 同样顺序
    CORRECT_ACTION_ALLOCATION = {
        'Enterprise': [
            3, 4, 5,              # Analyse (Defender, Enterprise0-2)
            16, 17, 18,           # Remove (Defender, Enterprise0-2)
            133, 134, 135         # Restore (Defender, Enterprise0-2)
        ],
        'Operational': [
            6, 7, 8, 9,           # Analyse (Op_Host0-2, Op_Server0)
            19, 20, 21, 22,       # Remove (Op_Host0-2, Op_Server0)
            136, 137, 138, 139    # Restore (Op_Host0-2, Op_Server0)
        ],
        'User': [
            # Phase 9K: 移除User0动作（persistent foothold无法清除）
            # 10, 23, 140,          # User0 - 已删除
            11, 12, 13, 14,       # Analyse (User1-4)
            24, 25, 26, 27,       # Remove (User1-4)
            141, 142, 143, 144    # Restore (User1-4)
        ],
        'Global': [0]  # Sleep（所有子网都可用），Monitor(1)已移除
    }
    
    def __init__(self, use_gpu: bool = True, enable_sleep_detection: bool = False):
        super().__init__()
        
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        
        # 子网配置
        # ⚠️ 修复：根据Scenario2.yaml中Blue.INT.Hosts的实际顺序
        # Defender(0), Enterprise0-2(1-3), Op_Host0-2(4-6), Op_Server0(7), User0-4(8-12)
        self.subnet_config = {
            'Enterprise': {'host_indices': [1, 2, 3], 'value': 'Medium'},  # Defender, Enterprise0-2
            'Operational': {'host_indices': [4, 5, 6, 7], 'value': 'High'},   # Op_Host0-2, Op_Server0
            'User': {'host_indices': [9, 10, 11, 12], 'value': 'Low'}      # User0-4
        }
        
        # 简化选择器
        self.selector = SimpleSelector()
        
        # 子网专家
        self.subnet_experts = {}
        self.subnet_optimizers = {}
        
        # 训练参数 - Phase 4A: 学习率调度
        self.gamma = 0.99
        self.gae_lambda = 0.95
        
        # Phase 4A: 学习率调度参数（根据动作空间大小调整）
        # 原则：动作空间越大，学习率越高（需要更快探索）
        #       子网越关键，学习率越低（需要更稳定策略）
        self.initial_lr = {
            'User': 1e-4,           # 最大动作空间(57)，需要快速探索
            'Enterprise': 5e-5,     # 中等动作空间(46)，平衡策略
            'Operational': 3e-5     # 最关键子网，需要稳定精确
        }
        self.min_lr = {
            'User': 1e-5,           # 最终仍保持较高学习能力
            'Enterprise': 5e-6,     # 中等
            'Operational': 1e-6     # 最终最稳定
        }
        self.lr_schedule_steps = 4000  # Phase 4B: 延长学习率衰减周期
        
        # 统计
        self.episode_count = 0
        self.total_steps = 0
        self.selector_accuracy = deque(maxlen=100)
        
        print(f"[初始化] 层次化A3C智能体")
        print(f"   设备: {self.device}")
    
    def _initialize_experts(self, action_space_size):
        """Phase 9L: 使用子网独立观察空间维度初始化专家"""
        if self.subnet_experts:
            return
        
        # Phase 9L: 子网观察空间维度
        obs_dims = {
            'User': 20,        # 4台主机×4 + 4统计（排除User0）
            'Enterprise': 16,  # 3台主机×4 + 4统计
            'Operational': 20  # 4台主机×4 + 4统计
        }
        
        # 为每个子网创建专家，使用正确的动作映射
        self.global_to_local_action = {}  # 全局动作索引 → (子网名, 本地动作索引)
        self.local_to_global_action = {}  # (子网名, 本地索引) → 全局动作索引
        
        for subnet_name, config in self.subnet_config.items():
            # 获取该子网的真实动作（包含Global动作）
            subnet_actions = self.CORRECT_ACTION_ALLOCATION[subnet_name] + self.CORRECT_ACTION_ALLOCATION['Global']
            action_dim = len(subnet_actions)
            
            # 建立映射关系
            self.local_to_global_action[subnet_name] = {}
            for local_idx, global_idx in enumerate(subnet_actions):
                self.global_to_local_action[global_idx] = (subnet_name, local_idx)
                self.local_to_global_action[subnet_name][local_idx] = global_idx
            
            # Phase 9L: 创建专家（传入obs_dim）
            expert = SubnetA3CExpert(
                subnet_name=subnet_name,
                host_indices=config['host_indices'],
                action_dim=action_dim,
                obs_dim=obs_dims[subnet_name],  # Phase 9L: 子网独立观察维度
                device=self.device
            )
            
            # 保存真实动作索引到专家
            expert.action_indices = subnet_actions
            
            # Phase 5: 现在可以设置动作类型映射了
            expert._setup_action_mapping()
            
            # Phase 4A: 使用专家自己的学习率
            optimizer = optim.Adam(expert.parameters(), lr=expert.expert_lr)
            
            self.subnet_experts[subnet_name] = expert
            self.subnet_optimizers[subnet_name] = optimizer
            
    
    def get_action(self, observation, action_space):
        """获取动作（Phase 9K及之前版本）"""
        self.total_steps += 1
        
        # 初始化
        if isinstance(action_space, int):
            self._initialize_experts(action_space)
        
        # 选择器选择子网
        selected_subnet = self.selector.select_subnet(observation)
        
        # 专家选择动作
        expert = self.subnet_experts[selected_subnet]
        subnet_state = expert.extract_subnet_state(observation)
        
        with torch.no_grad():
            local_action_idx, log_prob, value = expert.get_action_and_value(subnet_state)
        
        # Phase 5修复：使用正确的映射转换为全局动作
        global_action = self.local_to_global_action[selected_subnet][local_action_idx.item()]
        
        # Phase 9K: User0动作已从动作空间中移除，不再需要运行时过滤
        
        # 保存决策
        self.last_decision = {
            'selected_subnet': selected_subnet,
            'subnet_action': local_action_idx.item(),  # 本地索引
            'global_action': global_action,  # 全局索引
            'subnet_state': subnet_state,
            'log_prob': log_prob,
            'value': value.item(),
            'observation': observation
        }
        
        return global_action
    
    def get_action_with_subnet_obs(self, subnet_obs, subnet_name, explore=True):
        """
        Phase 9L: 使用子网专属观察选择动作
        
        Args:
            subnet_obs: 子网观察向量（24/16/20维）
            subnet_name: 'User' / 'Enterprise' / 'Operational'
            explore: 是否探索（当前未使用，保留接口）
            
        Returns:
            全局动作索引
        """
        self.total_steps += 1
        
        # 获取子网Expert
        expert = self.subnet_experts[subnet_name]
        
        # 转换为tensor
        subnet_state = torch.FloatTensor(subnet_obs).unsqueeze(0).to(self.device)
        
        # 选择动作
        with torch.no_grad():
            local_action_idx, log_prob, value = expert.get_action_and_value(subnet_state)
        
        # 转换为全局动作索引
        global_action = self.local_to_global_action[subnet_name][local_action_idx.item()]
        
        # 保存决策（用于训练）
        self.last_decision = {
            'selected_subnet': subnet_name,
            'subnet_action': local_action_idx.item(),
            'global_action': global_action,
            'subnet_state': subnet_state,
            'log_prob': log_prob,
            'value': value.item(),
            'observation': subnet_obs  # Phase 9L: 保存子网观察
        }
        
        return global_action
    
    def train(self, results: Results, attacked_subnet=None):
        """训练 - 现在接受外部提供的攻击检测信息"""
        if hasattr(self, 'last_decision'):
            # 如果没有提供攻击检测，尝试从观察变化中检测
            if attacked_subnet is None:
                attacked_subnet = self._detect_attacked_subnet(
                    self.last_decision['observation'], 
                    results.observation if hasattr(results.observation, 'reshape') else None
                )
            
            # 训练选择器
            selector_info = self.selector.train_selector(results.reward, attacked_subnet)
            
            # 记录选择器准确率（只在有攻击时记录）
            if selector_info and 'selector_reward' in selector_info and attacked_subnet is not None:
                self.selector_accuracy.append(1 if selector_info['selector_reward'] > 0 else 0)
            
            # 训练激活的专家
            selected_subnet = self.last_decision['selected_subnet']
            expert = self.subnet_experts[selected_subnet]
            
            if not hasattr(expert, 'experience_buffer'):
                expert.experience_buffer = []
                expert.prev_threat_level = None  # 记录上一步威胁级别
            
            # 应用奖励塑造 - Phase 1增强
            threat_level = expert._assess_threat_level(self.last_decision['subnet_state'])
            prev_threat = getattr(expert, 'prev_threat_level', None)
            shaped_reward = expert.shape_reward(
                results.reward, 
                self.last_decision['subnet_action'], 
                threat_level,
                prev_threat  # 传入上一步威胁级别
            )
            expert.prev_threat_level = threat_level  # 更新威胁级别
            
            expert.experience_buffer.append({
                'state': self.last_decision['subnet_state'],
                'action': self.last_decision['subnet_action'],
                'reward': shaped_reward,  # 使用塑造后的奖励
                'original_reward': results.reward,  # 保留原始奖励用于调试
                'log_prob': self.last_decision['log_prob'],
                'value': self.last_decision['value'],
                'done': results.done,
                'threat_level': threat_level
            })
            
            # 批量训练 - Phase 4A: 使用专家自己的批量大小
            if len(expert.experience_buffer) >= expert.expert_batch_size or results.done:
                self._train_expert(expert, self.subnet_optimizers[selected_subnet])
                expert.experience_buffer.clear()
    
    def _detect_attacked_subnet(self, prev_obs, current_obs):
        """检测被攻击的子网
        
        ⚠️ 注意：训练时不应使用此函数，应直接从env.get_last_action('Red')获取
        此函数仅用于推理阶段的fallback
        """
        if current_obs is None or len(prev_obs) != 52 or len(current_obs) != 52:
            return None
        
        prev_state = prev_obs.reshape(13, 4)
        current_state = current_obs.reshape(13, 4)
        
        # 检测状态变化
        for host_idx in range(13):
            if not np.array_equal(prev_state[host_idx], current_state[host_idx]):
                # ✅ 修复：根据Scenario2.yaml实际主机顺序
                # Defender(0), Enterprise0-2(1-3), Op_Host0-2(4-6), Op_Server0(7), User0-4(8-12)
                if 8 <= host_idx <= 12:  # User0-4
                    return 'User'
                elif 0 <= host_idx <= 3:  # Defender, Enterprise0-2
                    return 'Enterprise'
                elif 4 <= host_idx <= 7:  # Op_Host0-2, Op_Server0
                    return 'Operational'
        
        return None
    
    def _update_learning_rate(self, expert_name, optimizer):
        """Phase 4A: Cosine Annealing学习率调度"""
        progress = min(1.0, self.episode_count / self.lr_schedule_steps)
        
        # Cosine annealing
        initial_lr = self.initial_lr[expert_name]
        min_lr = self.min_lr[expert_name]
        lr = min_lr + (initial_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def _train_expert(self, expert, optimizer):
        """训练专家"""
        if len(expert.experience_buffer) < 2:
            return
        
        # Phase 4A: 更新学习率
        current_lr = self._update_learning_rate(expert.subnet_name, optimizer)
        
        # 准备数据
        states = torch.stack([exp['state'] for exp in expert.experience_buffer])
        actions = torch.tensor([exp['action'] for exp in expert.experience_buffer], 
                              dtype=torch.long, device=self.device)
        rewards = torch.tensor([exp['reward'] for exp in expert.experience_buffer], 
                              dtype=torch.float32, device=self.device)
        old_log_probs = torch.stack([exp['log_prob'] for exp in expert.experience_buffer])
        old_values = torch.tensor([exp['value'] for exp in expert.experience_buffer], 
                                 dtype=torch.float32, device=self.device)
        dones = torch.tensor([exp['done'] for exp in expert.experience_buffer], 
                            dtype=torch.bool, device=self.device)
        
        # 计算回报和优势
        returns, advantages = self._compute_gae(rewards, old_values, dones)
        
        # 前向传播
        action_logits, current_values = expert(states)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # 计算损失
        action_dist = torch.distributions.Categorical(action_probs)
        new_log_probs = action_dist.log_prob(actions)
        
        actor_loss = -(new_log_probs * advantages.detach()).mean()
        # Phase 7修复：使用Huber Loss，对异常值更鲁棒
        # MSE对异常值敏感：(200-50)²=22,500
        # Huber对异常值线性惩罚：约150
        critic_loss = F.smooth_l1_loss(current_values, returns)
        entropy_loss = -action_dist.entropy().mean()
        
        # Phase 9K: Critic Loss weight (移除归一化后)
        # Critic Loss ≈ 10-13, Actor Loss ≈ 0.3
        # Phase 9L: 降低Critic Loss权重
        # 原因：避免Critic过度主导，让Actor有更多探索空间
        critic_loss_weight = 0.015  # 从0.5降低到0.015
        
        # Phase 4A: 使用专家自己的entropy系数
        total_loss = actor_loss + critic_loss_weight * critic_loss + expert.entropy_coef * entropy_loss
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(expert.parameters(), 0.5)
        optimizer.step()
        
        # 记录训练统计
        if not hasattr(expert, 'train_count'):
            expert.train_count = 0
            expert.total_actor_loss = 0
            expert.total_critic_loss = 0
        
        expert.train_count += 1
        expert.total_actor_loss += actor_loss.item()
        expert.total_critic_loss += critic_loss.item()
        
        # 每100次训练打印一次统计
        if expert.train_count % 100 == 0:
            avg_actor_loss = expert.total_actor_loss / 100
            avg_critic_loss = expert.total_critic_loss / 100
            print(f"  [{expert.subnet_name} A3C] Train #{expert.train_count}: "
                  f"Actor Loss={avg_actor_loss:.4f}, Critic Loss={avg_critic_loss:.4f}, "
                  f"Batch Size={len(states)}")
            expert.total_actor_loss = 0
            expert.total_critic_loss = 0
    
    def _compute_gae(self, rewards, values, dones):
        """计算GAE - Phase 1优化"""
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        
        gae = 0
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[step + 1]
            
            if dones[step]:
                next_value = 0
            
            delta = rewards[step] + self.gamma * next_value - values[step]
            gae = delta + self.gamma * self.gae_lambda * gae * (1 - dones[step].float())
            
            advantages[step] = gae
            returns[step] = gae + values[step]
        
        # Phase 9C修正：减弱归一化，保留更多信号强度
        # 原本的归一化会让advantages压缩到[-1,1]，导致梯度太小
        if advantages.std() > 1e-8:
            # 只减去均值，不除以标准差
            advantages = advantages - advantages.mean()
        
        return returns, advantages
    
    def end_episode(self, display_episode: int = None):
        """
        结束episode
        
        Args:
            display_episode: 用于显示的episode编号（如果为None则使用内部计数器）
        """
        # 训练所有专家
        for subnet_name, expert in self.subnet_experts.items():
            if hasattr(expert, 'experience_buffer') and expert.experience_buffer:
                self._train_expert(expert, self.subnet_optimizers[subnet_name])
                expert.experience_buffer.clear()
        
        self.episode_count += 1
        
        # 打印统计
        if self.episode_count % 50 == 0:
            selector_acc = mean(self.selector_accuracy) if self.selector_accuracy else 0
            episode_to_show = display_episode if display_episode is not None else self.episode_count
            print(f"\nEpisode {episode_to_show} - 选择器准确率: {selector_acc:.1%}")
    
    def set_initial_values(self, action_space, observation):
        pass
    
    def save(self, filepath: str):
        """保存模型（简化接口）"""
        self.save_model(filepath)
    
    def save_model(self, filepath: str, episode: int = None, training_stats: dict = None):
        """
        保存完整模型（专家网络+Selector+训练统计）
        
        Args:
            filepath: 基础文件路径（不含扩展名）
            episode: 当前episode数
            training_stats: 训练统计数据
        
        Returns:
            bool: 是否保存成功
        """
        try:
            if not self.subnet_experts:
                print("[Warning] No experts to save")
                return False
            
            # 保存专家网络（.pth文件）
            for subnet_name, expert in self.subnet_experts.items():
                expert_file = f"{filepath}_{subnet_name}.pth"
                torch.save({
                    'expert_state_dict': expert.state_dict(),
                    'optimizer_state_dict': self.subnet_optimizers[subnet_name].state_dict(),
                    'subnet_name': subnet_name,
                    # 保存超参数
                    'expert_lr': expert.expert_lr,
                    'expert_batch_size': expert.expert_batch_size,
                    'entropy_coef': expert.entropy_coef,
                    'reward_scale': expert.reward_scale,
                    'train_count': getattr(expert, 'train_count', 0)
                }, expert_file)
            
            # 保存选择器（单独的.pkl文件）
            selector_file = f"{filepath}_selector.pkl"
            with open(selector_file, 'wb') as f:
                pickle.dump({
                    'q_table': dict(self.selector.q_table),
                    'epsilon': self.selector.epsilon,
                    'min_epsilon': self.selector.min_epsilon,
                    'epsilon_decay': self.selector.epsilon_decay,
                    'learning_rate': self.selector.learning_rate,
                    'total_updates': self.selector.total_updates,
                    'correct_selections': self.selector.correct_selections,
                    'valid_decisions': getattr(self.selector, 'valid_decisions', 0)
                }, f)
            
            # 保存主checkpoint（包含episode和训练统计）
            main_checkpoint = {
                'episode': episode,
                'training_stats': training_stats or {},
                'episode_count': self.episode_count,
                'total_steps': self.total_steps,
                # 记录文件路径，方便load_model找到对应文件
                'filepath': filepath
            }
            
            with open(f"{filepath}.pkl", 'wb') as f:
                pickle.dump(main_checkpoint, f)
            
            print(f"[Success] Model saved successfully")
            return True
            
        except Exception as e:
            print(f"[Error] Failed to save model: {e}")
            return False
    
    def load_model(self, filepath: str):
        """
        加载模型（兼容新旧格式）
        
        新格式: filepath.pkl + filepath_*.pth + filepath_selector.pkl
        旧格式: filepath_*.pth + filepath_selector.pkl (无主.pkl文件)
        """
        # 先初始化专家（如果还没初始化）
        if not self.subnet_experts:
            self._initialize_experts(145)  # 默认动作空间
        
        # 尝试加载主checkpoint（新格式）
        main_file = f"{filepath}.pkl"
        training_stats = {}
        episode = 0
        
        if os.path.exists(main_file):
            try:
                with open(main_file, 'rb') as f:
                    main_data = pickle.load(f)
                    episode = main_data.get('episode', 0)
                    training_stats = main_data.get('training_stats', {})
                    self.episode_count = main_data.get('episode_count', 0)
                    self.total_steps = main_data.get('total_steps', 0)
                    print(f"  [Loaded] Main checkpoint (episode={episode})")
            except Exception as e:
                print(f"  [Warning] Failed to load main checkpoint: {e}")
        else:
            print(f"  [Info] No main checkpoint found, loading legacy format")
        
        # 加载每个专家网络
        loaded_experts = 0
        for subnet_name in self.subnet_experts.keys():
            expert_file = f"{filepath}_{subnet_name}.pth"
            if os.path.exists(expert_file):
                checkpoint = torch.load(expert_file, map_location=self.device)
                expert = self.subnet_experts[subnet_name]
                
                # 加载网络权重
                expert.load_state_dict(checkpoint['expert_state_dict'])
                self.subnet_optimizers[subnet_name].load_state_dict(checkpoint['optimizer_state_dict'])
                
                # 恢复超参数（如果有保存）
                if 'expert_lr' in checkpoint:
                    expert.expert_batch_size = checkpoint['expert_batch_size']
                    expert.entropy_coef = checkpoint['entropy_coef']
                    expert.reward_scale = checkpoint['reward_scale']
                    expert.train_count = checkpoint.get('train_count', 0)
                    
                    # Phase 9L: 强制统一学习率为8e-4（覆盖checkpoint中的旧值）
                    expert.expert_lr = 8e-4
                    
                    # 更新优化器的学习率
                    for param_group in self.subnet_optimizers[subnet_name].param_groups:
                        param_group['lr'] = expert.expert_lr
                
                # 确保训练统计属性被初始化
                if not hasattr(expert, 'total_actor_loss'):
                    expert.total_actor_loss = 0
                if not hasattr(expert, 'total_critic_loss'):
                    expert.total_critic_loss = 0
                
                print(f"  [Loaded] {subnet_name} expert (lr={expert.expert_lr:.1e})")
                loaded_experts += 1
            else:
                print(f"  [Warning] Expert file not found: {expert_file}")
        
        if loaded_experts == 0:
            print(f"  [Error] No expert files found for {filepath}")
            return False
        
        # 加载选择器
        selector_file = f"{filepath}_selector.pkl"
        if os.path.exists(selector_file):
            with open(selector_file, 'rb') as f:
                selector_data = pickle.load(f)
                self.selector.q_table = selector_data['q_table']
                self.selector.epsilon = selector_data.get('epsilon', self.selector.epsilon)
                self.selector.min_epsilon = selector_data.get('min_epsilon', self.selector.min_epsilon)
                self.selector.epsilon_decay = selector_data.get('epsilon_decay', self.selector.epsilon_decay)
                self.selector.learning_rate = selector_data.get('learning_rate', self.selector.learning_rate)
                self.selector.total_updates = selector_data.get('total_updates', 0)
                self.selector.correct_selections = selector_data.get('correct_selections', 0)
                self.selector.valid_decisions = selector_data.get('valid_decisions', 0)
                print(f"  [Loaded] Selector (epsilon={self.selector.epsilon:.4f}, updates={self.selector.total_updates})")
        else:
            print(f"  [Warning] Selector file not found: {selector_file}")
        
        return True


# 测试
if __name__ == "__main__":
    print("[Test] Hierarchical A3C Agent")
    
    agent = SimpleHierarchicalA3CAgent(use_gpu=True)
    
    # 模拟测试
    dummy_obs = np.random.random(52).astype(np.float32)
    action = agent.get_action(dummy_obs, 145)
    
    print(f"测试动作: {action}")
    print(f"选择子网: {agent.last_decision['selected_subnet']}")
    
    print("[Success] Test completed")
