"""
阶段2：Selector训练（基于Blue Observation）

前提：三个A3C experts已训练完成并冻结
目标：训练selector仅基于Blue observation（52维）选择子网，最大化环境奖励
"""
import sys
import os
import torch
import numpy as np
import pickle
from collections import defaultdict
from statistics import mean

from train_mixed_scenarios import MixedScenarioTrainer


class QLearningSelector:
    """
    Q-learning Selector（基于Blue Observation）
    
    关键改进：
    1. 输入：52维Blue observation（不是True State）
    2. 目标：最大化环境奖励（不是识别准确率）
    3. 探索：ε-greedy + OP bonus
    """
    
    def __init__(self, learning_rate=0.1, epsilon=0.6, epsilon_decay=0.9973, min_epsilon=0.05, gamma=0.99,
                 enterprise_unlock_step=7, operational_unlock_step=15):
        self.q_table = defaultdict(lambda: {
            'User': 0.0,
            'Enterprise': 0.0,
            'Operational': 0.0
        })
        
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        
        # 阶段性子网解锁（基于红队攻击路径）
        self.enterprise_unlock_step = enterprise_unlock_step  # Enterprise最早可能被攻击的步数
        self.operational_unlock_step = operational_unlock_step  # Operational最早可能被攻击的步数
        
        # 方案1：奖励归一化 - 场景baseline（使用硬编码selector的真实评估结果）
        # 来源：evaluations/mix6_UE-mixed_scenarios_final_OP-op_focus_final_100ep/results.txt
        self.scene_baselines = {
            'B_line_30': -14.41,    # 真实评估：-14.41 (范围 [-19.20, -9.70])
            'B_line_50': -24.94,    # 真实评估：-24.94 (范围 [-30.60, -19.90])
            'B_line_100': -51.64,   # 真实评估：-51.64 (范围 [-59.60, -38.90])
            'Meander_30': -10.96,   # 真实评估：-10.96 (范围 [-14.50, -7.60])
            'Meander_50': -22.13,   # 真实评估：-22.13 (范围 [-27.80, -16.50])
            'Meander_100': -52.13   # 真实评估：-52.13 (范围 [-62.20, -41.90])
        }
        
        # 动态更新baseline（滑动平均）
        self.scene_reward_history = defaultdict(list)
        self.baseline_update_weight = 0.1  # 滑动平均权重
        
        # 子网主机索引（在52维向量中的位置）
        # 基于Scenario2.yaml: Defender(0), Enterprise0-2(1-3), Op_Host0-2(4-6), Op_Server0(7), User0-4(8-12)
        self.subnet_host_indices = {
            'User': [8, 9, 10, 11, 12],         # User0-4 (5台主机)
            'Enterprise': [1, 2, 3],            # Enterprise0-2 (3台，排除Defender[0])
            'Operational': [4, 5, 6, 7]         # Op_Host0-2, Op_Server0 (4台)
        }
        
        # 统计（累计）
        self.selection_counts = {'User': 0, 'Enterprise': 0, 'Operational': 0}
        self.total_selections = 0
        self.phase_locked_counts = 0  # 因阶段锁定而强制选User的次数
        
        # 滑动窗口统计（最近N次选择）
        self.recent_selections = []  # 存储最近的选择记录
        self.recent_window_size = 200  # 窗口大小
        
        # 一致性统计（选择是否与True State一致）
        self.consistency_history = []  # 存储最近的一致性记录（True/False）
        self.total_consistency_correct = 0
        self.total_consistency_checks = 0
    
    def get_state_key(self, blue_obs_52dim, scenario='B_line', max_steps=100, current_step=0):
        """
        从52维Blue observation构造状态键（极简版：阶段+威胁）
        
        关键改进：
        - 移除场景和配置（通过分场景训练解决）
        - 移除步数分桶（阶段信息已足够）
        - 仅保留阶段编码（early/mid/late）+ 威胁等级
        
        状态表示：阶段_威胁等级
        返回格式："mid_U2_E1_O0" 表示中期，威胁User=2, Ent=1, Op=0
        状态空间：3阶段 × 4³威胁 = 192个状态（每场景）
        """
        obs_reshaped = blue_obs_52dim.reshape(13, 4)
        
        # 计算威胁等级（改进：分级而非简单累加）
        subnet_threats = {}
        for subnet_name, host_indices in self.subnet_host_indices.items():
            threat_score = 0
            for idx in host_indices:
                # **关键修复**：排除User0（idx=8），与hardcoded selector保持一致
                # User0是Red的永久入口点，不可防御
                if idx == 8:  # User0
                    continue
                
                host_obs = obs_reshaped[idx]
                
                # Activity: [0,0]=None, [1,0]=Scan, [1,1]=Exploit
                activity_bits = host_obs[0:2]
                if activity_bits[1] == 1:  # Exploit [1,1]
                    threat_score += 2
                elif activity_bits[0] == 1:  # Scan [1,0]
                    threat_score += 1
                
                # Compromised: [0,0]=No, [1,0]=Unknown, [0,1]=User, [1,1]=Privileged
                comp_bits = host_obs[2:4]
                if comp_bits[0] == 1 and comp_bits[1] == 1:  # Privileged [1,1]
                    threat_score += 3  # 特权入侵更严重
                elif comp_bits[0] == 0 and comp_bits[1] == 1:  # User [0,1]
                    threat_score += 1
                elif comp_bits[0] == 1 and comp_bits[1] == 0:  # Unknown [1,0]
                    threat_score += 1
            
            # 威胁分级（折中：4级，平衡泛化和区分度）
            # 0-3的等级在学习效率和区分能力间取得平衡
            if threat_score == 0:
                level = 0  # 无威胁
            elif threat_score <= 2:
                level = 1  # 轻微威胁（观察）
            elif threat_score <= 5:
                level = 2  # 中等威胁（需要处理）
            else:
                level = 3  # 严重威胁（优先处理）
            
            subnet_threats[subnet_name] = level
        
        # 阶段编码（基于绝对步数，足够表达时序信息）
        # 不需要步数分桶 - 阶段已经包含了关键时序特征
        if current_step < 10:
            phase = 'early'    # 0-9步：Red初始入侵
        elif current_step < 20:
            phase = 'mid'      # 10-19步：横向移动
        else:
            phase = 'late'     # 20+步：深入攻击
        
        # 极简状态键：阶段_威胁等级（移除步数分桶）
        # 优势：状态空间从2112减少到192（每场景），更容易充分探索
        state_key = (f"{phase}_"
                    f"U{subnet_threats['User']}_"
                    f"E{subnet_threats['Enterprise']}_"
                    f"O{subnet_threats['Operational']}")
        return state_key
    
    def record_consistency(self, is_consistent):
        """
        记录一致性结果
        
        Args:
            is_consistent: 是否与True State一致
        """
        self.consistency_history.append(is_consistent)
        if len(self.consistency_history) > self.recent_window_size:
            self.consistency_history.pop(0)
        
        if is_consistent:
            self.total_consistency_correct += 1
        self.total_consistency_checks += 1
    
    def get_consistency_rate(self):
        """
        获取一致率统计
        
        Returns:
            {'recent': 最近N次一致率, 'total': 总体一致率}
        """
        recent_rate = 0.0
        if self.consistency_history:
            recent_rate = sum(self.consistency_history) / len(self.consistency_history) * 100
        
        total_rate = 0.0
        if self.total_consistency_checks > 0:
            total_rate = self.total_consistency_correct / self.total_consistency_checks * 100
        
        return {
            'recent': recent_rate,
            'total': total_rate,
            'count': len(self.consistency_history)
        }
    
    def select_subnet(self, blue_obs_52dim, scenario='B_line', max_steps=100, current_step=0):
        """
        选择子网（ε-greedy策略 + OP bonus）
        
        已移除阶段锁定：所有子网从第0步即可选择
        让Q-learning通过奖励信号自己学习何时选择哪个子网
        
        Args:
            blue_obs_52dim: 52维Blue observation numpy array
            scenario: 场景名称
            max_steps: 最大步数
            current_step: 当前回合步数（保留参数以便未来调整）
        
        Returns:
            'User' / 'Enterprise' / 'Operational'
        """
        state_key = self.get_state_key(blue_obs_52dim, scenario, max_steps, current_step)
        
        # 所有子网始终可选（阶段锁定已移除）
        available_subnets = ['User', 'Enterprise', 'Operational']
        
        # ε-greedy探索（均匀探索，让Q值自然学习）
        if np.random.random() < self.epsilon:
            # 均匀随机选择（移除OP bonus，因为实际威胁分布不均）
            selected = np.random.choice(available_subnets)
        else:
            # 贪心选择（在可选子网中选最大Q值）
            # 如果遇到未见过的状态，初始化它
            if state_key not in self.q_table:
                self.q_table[state_key] = {'User': 0.0, 'Enterprise': 0.0, 'Operational': 0.0}
            
            q_values = self.q_table[state_key]
            available_q = {k: v for k, v in q_values.items() if k in available_subnets}
            max_q = max(available_q.values())
            best_actions = [k for k, v in available_q.items() if v == max_q]
            selected = np.random.choice(best_actions)
        
        # 统计
        self.selection_counts[selected] += 1
        self.total_selections += 1
        if len(available_subnets) == 1:  # 被阶段锁定
            self.phase_locked_counts += 1
        
        # 滑动窗口统计
        self.recent_selections.append({
            'subnet': selected,
            'locked': len(available_subnets) == 1
        })
        if len(self.recent_selections) > self.recent_window_size:
            self.recent_selections = self.recent_selections[-self.recent_window_size:]
        
        return selected
    
    def normalize_reward(self, raw_reward, scenario='B_line', max_steps=100):
        """
        方案1：奖励归一化
        
        将绝对奖励转为相对改进，解决不同场景奖励尺度冲突
        
        Args:
            raw_reward: 原始环境奖励
            scenario: 场景名称
            max_steps: 最大步数
        
        Returns:
            归一化后的奖励（相对改进百分比）
        """
        scene_key = f"{scenario}_{max_steps}"
        baseline = self.scene_baselines.get(scene_key, raw_reward)
        
        # 动态更新baseline（滑动平均）
        self.scene_reward_history[scene_key].append(raw_reward)
        if len(self.scene_reward_history[scene_key]) > 100:
            # 保持最近100个
            self.scene_reward_history[scene_key] = self.scene_reward_history[scene_key][-100:]
        
        # 更新baseline
        recent_avg = np.mean(self.scene_reward_history[scene_key])
        self.scene_baselines[scene_key] = (
            (1 - self.baseline_update_weight) * baseline + 
            self.baseline_update_weight * recent_avg
        )
        
        # 相对改进 = (实际奖励 - baseline) / |baseline|
        if abs(baseline) > 1e-6:
            normalized = (raw_reward - baseline) / abs(baseline)
        else:
            normalized = 0.0
        
        # 裁剪到合理范围
        normalized = np.clip(normalized, -2.0, 2.0)
        
        return normalized
    
    def update(self, state_key, action, raw_reward, next_state_key, done, scenario='B_line', max_steps=100, current_step=0):
        """
        Q-learning更新（带时序奖励加权）
        
        Q(s,a) ← Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]
        
        关键改进：时序奖励加权
        - 早期（0-10步）防御User/Enterprise：权重×2.0
        - 中期（11-20步）防御：权重×1.5
        - 后期（20+步）：正常权重
        - 后期Op威胁出现：额外惩罚-20（说明前期防御失败）
        
        Args:
            raw_reward: 原始环境奖励
            current_step: 当前步数
        """
        # 直接使用原始奖励（移除时序加权，让Q-learning自然学习）
        # 时序加权虽然理论上合理，但实践中可能干扰学习：
        # 1. 负奖励也被放大（早期失败 → -10 × 2 = -20）
        # 2. Op惩罚过重（成功防御 → +8 - 20 = -12）
        # 3. 让我们先用简单的原始奖励，看Q-learning能否自然学到正确策略
        reward = raw_reward
        
        current_q = self.q_table[state_key][action]
        
        if done:
            target_q = reward
        else:
            next_max_q = max(self.q_table[next_state_key].values())
            target_q = reward + self.gamma * next_max_q
        
        # 更新Q值
        new_q = current_q + self.learning_rate * (target_q - current_q)
        
        # Q值裁剪（调整范围以适应原始奖励）
        new_q = np.clip(new_q, -200.0, 50.0)
        
        self.q_table[state_key][action] = new_q
    
    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def get_statistics(self):
        """获取选择统计（累计）"""
        if self.total_selections == 0:
            return {
                'User': {'count': 0, 'percentage': 0.0},
                'Enterprise': {'count': 0, 'percentage': 0.0},
                'Operational': {'count': 0, 'percentage': 0.0},
                'total': 0
            }
        
        return {
            'User': {
                'count': self.selection_counts['User'],
                'percentage': self.selection_counts['User'] / self.total_selections * 100
            },
            'Enterprise': {
                'count': self.selection_counts['Enterprise'],
                'percentage': self.selection_counts['Enterprise'] / self.total_selections * 100
            },
            'Operational': {
                'count': self.selection_counts['Operational'],
                'percentage': self.selection_counts['Operational'] / self.total_selections * 100
            },
            'total': self.total_selections
        }
    
    def get_recent_statistics(self):
        """获取最近N次的选择统计（滑动窗口）"""
        if len(self.recent_selections) == 0:
            return {
                'User': {'count': 0, 'percentage': 0.0},
                'Enterprise': {'count': 0, 'percentage': 0.0},
                'Operational': {'count': 0, 'percentage': 0.0},
                'locked_count': 0,
                'locked_rate': 0.0,
                'total': 0
            }
        
        # 统计最近的选择
        recent_counts = {'User': 0, 'Enterprise': 0, 'Operational': 0}
        locked_count = 0
        
        for record in self.recent_selections:
            recent_counts[record['subnet']] += 1
            if record['locked']:
                locked_count += 1
        
        total = len(self.recent_selections)
        
        return {
            'User': {
                'count': recent_counts['User'],
                'percentage': recent_counts['User'] / total * 100
            },
            'Enterprise': {
                'count': recent_counts['Enterprise'],
                'percentage': recent_counts['Enterprise'] / total * 100
            },
            'Operational': {
                'count': recent_counts['Operational'],
                'percentage': recent_counts['Operational'] / total * 100
            },
            'locked_count': locked_count,
            'locked_rate': locked_count / total * 100,
            'total': total
        }


class SelectorPhase2Trainer:
    """
    阶段2训练器：冻结Experts，训练Selector
    """
    
    def __init__(self):
        # 1. 初始化并加载冻结的experts
        self.base_trainer = MixedScenarioTrainer()
        
        # 先创建环境以初始化experts
        env, _ = self.base_trainer.create_environment('B_line')
        action_space_size = env.get_action_space('Blue')
        self.base_trainer.agent._initialize_experts(action_space_size)
        
        # 加载checkpoints
        ok = self.base_trainer.load_mixed_checkpoints(
            'user_enterprise_expert_metadata',
            'operational_expert_metadata'
        )
        
        if not ok:
            raise RuntimeError("无法加载experts checkpoints")
        
        # 冻结experts
        for subnet_name in ['User', 'Enterprise', 'Operational']:
            expert = self.base_trainer.agent.subnet_experts[subnet_name]
            expert.eval()
            for param in expert.parameters():
                param.requires_grad = False
        
        # 2. 初始化Q-learning selector
        self.selector = QLearningSelector(
            learning_rate=0.1,
            epsilon=0.8,
            epsilon_decay=0.9982,
            min_epsilon=0.1,
            gamma=0.99,
            enterprise_unlock_step=0,
            operational_unlock_step=0
        )
        
        # 3. 子网主机索引（用于提取子网观察）
        self.subnet_host_indices = {
            'User': [8, 9, 10, 11, 12],
            'Enterprise': [1, 2, 3],
            'Operational': [4, 5, 6, 7]
        }
    
    def extract_subnet_obs_from_blue(self, blue_obs_52dim, subnet_name):
        """
        从52维Blue observation提取子网观察
        
        User: 5主机 × 4维 = 20维
        Enterprise: 3主机 × 4维 = 12维 → padding到16维
        Operational: 4主机 × 4维 = 16维 → padding到20维
        """
        obs_reshaped = blue_obs_52dim.reshape(13, 4)
        host_indices = self.subnet_host_indices[subnet_name]
        
        subnet_obs = []
        for idx in host_indices:
            subnet_obs.extend(obs_reshaped[idx])
        
        # Padding到目标维度
        target_dims = {'User': 20, 'Enterprise': 16, 'Operational': 20}
        target_dim = target_dims[subnet_name]
        current_dim = len(subnet_obs)
        
        if current_dim < target_dim:
            subnet_obs = np.pad(subnet_obs, (0, target_dim - current_dim), mode='constant')
        
        return np.array(subnet_obs, dtype=np.float32)
    
    def get_most_threatened_subnet_from_true_state(self, cyborg):
        """
        从True State判断当前哪个子网威胁最大
        
        **关键修复**：直接使用hardcoded_selector的完整逻辑
        包括特权访问、可疑进程和子网权重
        
        Args:
            cyborg: CybORG环境实例
            
        Returns:
            'User' / 'Enterprise' / 'Operational'
        """
        # 获取True State并转换为hardcoded_selector期望的格式
        true_state = cyborg.get_agent_state('True')
        cyborg_obs_dict = self.base_trainer.parse_observation_to_dict(true_state)
        
        # 使用hardcoded_selector的完整威胁评分逻辑
        from hardcoded_selector import HardcodedSelector
        temp_selector = HardcodedSelector()
        selected_subnet, threat_scores = temp_selector.select_subnet_with_scores(cyborg_obs_dict)
        
        return selected_subnet
    
    def run_episode(self, scenario='B_line', max_steps=100, eval_mode=False, use_consistency_reward=True, consistency_weight=0.5):
        """
        运行一个episode，训练selector
        
        Args:
            scenario: 'B_line' or 'Meander'
            max_steps: 最大步数
            eval_mode: 是否评估模式（禁用探索）
            use_consistency_reward: 是否使用一致性奖励（训练时建议True）
            consistency_weight: 一致性奖励权重（0-1之间）
        
        Returns:
            (episode_reward, steps_taken, subnet_selections)
        """
        env, cyborg = self.base_trainer.create_environment(scenario)
        observation = env.reset()  # 52维Blue observation
        
        done = False
        step = 0
        episode_reward = 0
        trajectory = []
        subnet_selections = {'User': 0, 'Enterprise': 0, 'Operational': 0}
        
        while not done and step < max_steps:
            # 1. 获取True State并转换为dict（与A3C训练时一致）
            true_state = cyborg.get_agent_state('True')
            cyborg_obs_dict = self.base_trainer.parse_observation_to_dict(true_state)
            
            # 2. Hardcoded selector做实际决策（保证环境正确防御）
            from hardcoded_selector import HardcodedSelector
            temp_hardcoded = HardcodedSelector()
            teacher_subnet, threat_scores = temp_hardcoded.select_subnet_with_scores(cyborg_obs_dict)
            
            # 3. Q-learning selector做预测（用于学习和对比）
            state_key = self.selector.get_state_key(observation, scenario, max_steps, current_step=step)
            
            if eval_mode:
                # 评估模式：Q-learning做实际决策，贪心选择
                old_epsilon = self.selector.epsilon
                self.selector.epsilon = 0.0
                student_subnet = self.selector.select_subnet(observation, scenario, max_steps, current_step=step)
                self.selector.epsilon = old_epsilon
                selected_subnet = student_subnet  # 评估时用Q-learning的决策
            else:
                # 训练模式：让Q-learning做预测，但用hardcoded的决策与环境交互
                student_subnet = self.selector.select_subnet(observation, scenario, max_steps, current_step=step)
                selected_subnet = teacher_subnet  # 训练时用hardcoded的决策保证环境正确
            
            subnet_selections[selected_subnet] += 1
            
            # 4. 获取子网观察（使用True State观察，与A3C训练时一致）
            subnet_obs = self.base_trainer.obs_extractor.extract_subnet_observation(
                cyborg_obs_dict, selected_subnet
            )
            
            # 5. Expert选择动作
            with torch.no_grad():
                global_action = self.base_trainer.agent.get_action_with_subnet_obs(
                    subnet_obs,
                    selected_subnet,
                    explore=False  # Expert不探索
                )
            
            # 5. 执行动作（在正确防御的轨迹上）
            next_observation, env_reward, done, info = env.step(global_action)
            
            # 6. 计算一致性奖励（训练Q-learning）
            q_learning_reward = env_reward  # 默认使用环境奖励
            
            if use_consistency_reward and not eval_mode:
                # 比较Q-learning的预测和hardcoded的选择
                is_consistent = (student_subnet == teacher_subnet)
                self.selector.record_consistency(is_consistent)
                
                # 纯一致性奖励（行为克隆目标）
                if is_consistent:
                    q_learning_reward = +10  # 预测正确
                else:
                    q_learning_reward = -10  # 预测错误
            
            # 7. 记录轨迹（记录Q-learning的预测和奖励）
            next_state_key = self.selector.get_state_key(next_observation, scenario, max_steps, current_step=step+1)
            trajectory.append({
                'state_key': state_key,
                'action': student_subnet if not eval_mode else selected_subnet,  # 训练时记录Q-learning的预测
                'reward': q_learning_reward,
                'next_state_key': next_state_key,
                'done': done,
                'scenario': scenario,
                'max_steps': max_steps,
                'current_step': step
            })
            
            # episode奖励使用环境奖励（反映真实防御效果）
            episode_reward += env_reward
            observation = next_observation
            step += 1
        
        # 6. 训练：更新Q-table（带时序奖励加权）
        if not eval_mode:
            for t in trajectory:
                self.selector.update(
                    t['state_key'],
                    t['action'],
                    t['reward'],  # 原始奖励，update内部会进行时序加权
                    t['next_state_key'],
                    t['done'],
                    t['scenario'],
                    t['max_steps'],
                    t['current_step']  # 传入步数用于时序加权
                )
        
        return episode_reward, step, subnet_selections
    
    def train(self, total_episodes=40000, eval_interval=200, save_interval=2000):
        """
        训练selector（分阶段课程学习）
        
        Args:
            total_episodes: 总训练轮数（将被分阶段配置覆盖）
            eval_interval: 打印统计间隔
            save_interval: 保存checkpoint间隔
        """
        print("\n" + "=" * 80)
        print("双Selector独立训练策略 (行为克隆模式)")
        print("=" * 80)
        print("**训练模式**：")
        print("  - Hardcoded selector做实际决策 → 保证环境正确防御")
        print("  - Q-learning做预测 → 学习模仿hardcoded的决策")
        print("  - 一致性奖励：预测对+10，预测错-10")
        print("")
        print("状态空间：阶段_威胁等级 (极简版)")
        print("预计状态数：3阶段 x 4^3威胁 = 192个状态 (每场景)")
        print("  - 阶段: early(0-9步), mid(10-19步), late(20+步)")
        print("  - 威胁: 4级(0=无, 1=轻微, 2=中等, 3=严重)")
        print("  - 示例: 'mid_U2_E1_O0' = 中期,User威胁2,Ent威胁1,Op无威胁")
        print("")
        print("训练策略（优化版）：")
        print("  1. **两个独立的selector** - 各自针对一种攻击模式")
        print("  2. B_line selector: 100,000轮 (30/50/100 → 30k/30k/40k)")
        print("  3. Meander selector: 100,000轮 (30/50/100 → 30k/30k/40k，从零开始)")
        print("  4. 探索率独立衰减 → 每个selector: 0.8→0.1 (100k轮)")
        print("  5. 子网锁定移除 → 从第0步即可选择所有子网")
        print("")
        print("总训练轮数：200,000 (2 selectors × 100k)")
        print("探索率衰减：0.9982 (每100轮)")
        print("=" * 80)
        
        # 双Selector训练配置（每个100000轮，独立训练）
        training_phases = [
            {
                'selector_name': 'B_line',
                'description': 'B_line Selector训练（线性攻击专家）',
                'stages': [
                    {'episodes': 30000, 'scenario': 'B_line', 'max_steps': 30, 'desc': 'B30'},
                    {'episodes': 30000, 'scenario': 'B_line', 'max_steps': 50, 'desc': 'B50'},
                    {'episodes': 40000, 'scenario': 'B_line', 'max_steps': 100, 'desc': 'B100'},
                ]
            },
            {
                'selector_name': 'Meander',
                'description': 'Meander Selector训练（迂回攻击专家）',
                'stages': [
                    {'episodes': 30000, 'scenario': 'Meander', 'max_steps': 30, 'desc': 'M30'},
                    {'episodes': 30000, 'scenario': 'Meander', 'max_steps': 50, 'desc': 'M50'},
                    {'episodes': 40000, 'scenario': 'Meander', 'max_steps': 100, 'desc': 'M100'},
                ]
            }
        ]
        
        # 训练两个独立的selector
        for phase_idx, phase in enumerate(training_phases, 1):
            selector_name = phase['selector_name']
            
            print(f"\n{'='*100}")
            print(f"开始训练 {selector_name} Selector ({phase['description']})")
            print(f"{'='*100}")
            
            # 初始化新的selector（探索率独立重置）
            self.selector = QLearningSelector(
                learning_rate=0.1,
                epsilon=0.8,           # 从0.8开始（增加初始探索）
                epsilon_decay=0.9982,  # 100000轮衰减到0.1
                min_epsilon=0.1,       # 最终保留10%探索
                gamma=0.99,
                enterprise_unlock_step=0,   # 移除锁定
                operational_unlock_step=0   # 移除锁定
            )
            print(f"[初始化] 新Selector - 探索率ε={self.selector.epsilon}, 衰减率={self.selector.epsilon_decay}, 子网锁定已移除")
            
            phase_episode_rewards = []
            phase_episode = 0  # 当前phase内的episode计数
            
            # 训练当前selector的所有阶段
            for stage_idx, stage in enumerate(phase['stages'], 1):
                scenario = stage['scenario']
                max_steps = stage['max_steps']
                stage_episodes = stage['episodes']
                stage_desc = stage['desc']
                
                print(f"\n{'='*80}")
                print(f"[{selector_name}] 阶段{stage_idx}: {stage_desc} (场景={scenario}, 步数={max_steps})")
                print(f"训练轮数: {stage_episodes}")
                print(f"{'='*80}")
                
                stage_episode_rewards = []
                
                for episode in range(stage_episodes):
                    # 运行episode（纯一致性奖励训练）
                    ep_reward, ep_steps, subnet_sels = self.run_episode(
                        scenario, max_steps, 
                        eval_mode=False,
                        use_consistency_reward=True,
                        consistency_weight=1.0  # 不需要衰减，纯一致性
                    )
                    stage_episode_rewards.append(ep_reward)
                    phase_episode_rewards.append(ep_reward)
                    
                    phase_episode += 1
                    
                    # 每100轮衰减探索率
                    if phase_episode % 100 == 0:
                        self.selector.decay_epsilon()
                    
                    # 简单进度提示（每50轮）
                    if phase_episode % 50 == 0 and phase_episode % eval_interval != 0:
                        print(f"[进度] [{selector_name}] Episode {phase_episode}/45000 (阶段{stage_idx}: {episode+1}/{stage_episodes}) - 探索率: {self.selector.epsilon:.3f}")
                    
                    # 详细统计（每eval_interval轮）
                    if phase_episode % eval_interval == 0:
                        recent_rewards = phase_episode_rewards[-eval_interval:]
                        avg_reward = mean(recent_rewards)
                        std_reward = np.std(recent_rewards)
                        min_reward = min(recent_rewards)
                        max_reward = max(recent_rewards)
                        
                        # Q表大小
                        q_table_size = len(self.selector.q_table)
                        
                        # 最近N轮的选择分布
                        recent_stats = self.selector.get_recent_statistics()
                        
                        # 一致率统计
                        consistency_stats = self.selector.get_consistency_rate()
                        
                        print(f"\n[{selector_name}] Episode {phase_episode}/45000 (阶段{stage_idx}: {episode+1}/{stage_episodes})")
                        print(f"  环境奖励: {avg_reward:.2f} ± {std_reward:.2f} (范围: [{min_reward:.2f}, {max_reward:.2f}]) [真实防御效果]")
                        print(f"  Q表大小: {q_table_size}个状态")
                        print(f"  探索率ε: {self.selector.epsilon:.3f}")
                        print(f"  一致率: {consistency_stats['recent']:.1f}% (最近{consistency_stats['count']}次) | 总体: {consistency_stats['total']:.1f}% [Q训练用+10/-10]")
                        print(f"  选择分布(最近{recent_stats['total']}轮): "
                              f"User {recent_stats['User']['percentage']:.1f}% | "
                              f"Ent {recent_stats['Enterprise']['percentage']:.1f}% | "
                              f"Op {recent_stats['Operational']['percentage']:.1f}%")
                    
                    # 保存checkpoint
                    if phase_episode % save_interval == 0:
                        self.save_checkpoint(f"{selector_name}_selector_ep{phase_episode}")
                
                # 当前stage完成统计
                stage_avg = mean(stage_episode_rewards)
                stage_std = np.std(stage_episode_rewards)
                print(f"\n[完成] [{selector_name}] 阶段{stage_idx}({stage_desc})完成！")
                print(f"  阶段平均奖励: {stage_avg:.2f} ± {stage_std:.2f}")
                print(f"  当前Q表大小: {len(self.selector.q_table)}个状态")
                print(f"  当前探索率: {self.selector.epsilon:.3f}")
            
            # 当前selector训练完成
            phase_avg = mean(phase_episode_rewards)
            phase_std = np.std(phase_episode_rewards)
            
            print(f"\n{'='*100}")
            print(f"[完成] {selector_name} Selector训练完成！")
            print(f"{'='*100}")
            print(f"  总训练轮数: {phase_episode}")
            print(f"  平均奖励: {phase_avg:.2f} ± {phase_std:.2f}")
            print(f"  最终Q表大小: {len(self.selector.q_table)}个状态")
            print(f"  最终探索率: {self.selector.epsilon:.3f}")
            
            # 最终统计
            final_stats = self.selector.get_statistics()
            print(f"  最终选择分布:")
            print(f"    User: {final_stats['User']['percentage']:.1f}% ({final_stats['User']['count']}次)")
            print(f"    Enterprise: {final_stats['Enterprise']['percentage']:.1f}% ({final_stats['Enterprise']['count']}次)")
            print(f"    Operational: {final_stats['Operational']['percentage']:.1f}% ({final_stats['Operational']['count']}次)")
            
            # 一致率统计
            consistency_stats = self.selector.get_consistency_rate()
            print(f"  最终一致率: {consistency_stats['total']:.1f}%")
            
            # 保存最终checkpoint
            self.save_checkpoint(f"{selector_name}_selector_final")
        
        print("\n" + "=" * 100)
        print("全部训练完成！")
        print("=" * 100)
        print("已训练两个独立的selector:")
        print("  1. B_line_selector_final.pkl")
        print("  2. Meander_selector_final.pkl")
    
    def save_checkpoint(self, checkpoint_name):
        """保存selector checkpoint"""
        selector_data = {
            'q_table': dict(self.selector.q_table),
            'learning_rate': self.selector.learning_rate,
            'epsilon': self.selector.epsilon,
            'epsilon_decay': self.selector.epsilon_decay,
            'min_epsilon': self.selector.min_epsilon,
            'gamma': self.selector.gamma,
            'selection_counts': self.selector.selection_counts,
            'total_selections': self.selector.total_selections
        }
        
        save_path = f'checkpoints/{checkpoint_name}.pkl'
        os.makedirs('checkpoints', exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(selector_data, f)
        
        print(f"\n[保存] Selector checkpoint: {save_path}")
    
    def evaluate(self, num_episodes=100):
        """
        在六场景各100轮评估selector
        """
        print("\n" + "=" * 80)
        print("评估Selector（六场景各100轮）")
        print("=" * 80)
        
        scenarios = [
            ('B_line', 30),
            ('B_line', 50),
            ('B_line', 100),
            ('Meander', 30),
            ('Meander', 50),
            ('Meander', 100)
        ]
        
        for scenario, steps in scenarios:
            rewards = []
            subnet_selections_total = {'User': 0, 'Enterprise': 0, 'Operational': 0}
            
            for ep in range(num_episodes):
                ep_reward, ep_steps, subnet_sels = self.run_episode(scenario, steps, eval_mode=True)
                rewards.append(ep_reward)
                
                for subnet in subnet_selections_total:
                    subnet_selections_total[subnet] += subnet_sels[subnet]
            
            avg_reward = mean(rewards)
            std_reward = np.std(rewards)
            min_reward = min(rewards)
            max_reward = max(rewards)
            
            total_selections = sum(subnet_selections_total.values())
            sel_dist = {k: v/total_selections*100 for k, v in subnet_selections_total.items()}
            
            print(f"\n{scenario}-{steps}步 ({num_episodes} episodes):")
            print(f"  环境奖励: {avg_reward:.2f} ± {std_reward:.2f} (范围: [{min_reward:.2f}, {max_reward:.2f}])")
            print(f"  选择分布: User {sel_dist['User']:.1f}% | "
                  f"Ent {sel_dist['Enterprise']:.1f}% | "
                  f"Op {sel_dist['Operational']:.1f}%")


def main():
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = 'train'
    
    # 快速验证模式
    if len(sys.argv) > 2 and sys.argv[2] == 'quick':
        print("\n[快速验证模式] 仅运行100轮...")
        trainer = SelectorPhase2Trainer()
        trainer.train(total_episodes=100, eval_interval=20, save_interval=100)
        print("\n[快速验证] 完成！如需完整训练，请运行: python train_selector_phase2.py train")
        return
    
    trainer = SelectorPhase2Trainer()
    
    if mode == 'train':
        # 训练90000轮（3阶段 × 30000轮）
        trainer.train(total_episodes=90000, eval_interval=200, save_interval=2000)
        
        # 训练完成后评估
        print("\n训练完成，开始评估...")
        trainer.evaluate(num_episodes=100)
    
    elif mode == 'eval':
        # 仅评估（需要先加载checkpoint）
        # TODO: 实现加载逻辑
        trainer.evaluate(num_episodes=100)
    
    else:
        print("用法:")
        print("  python train_selector_phase2.py train  # 训练selector")
        print("  python train_selector_phase2.py eval   # 评估selector")


if __name__ == '__main__':
    main()
