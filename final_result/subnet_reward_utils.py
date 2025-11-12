"""
子网奖励隔离工具
每个Expert只关心自己子网的奖励/惩罚
"""

import numpy as np
from typing import Dict, List, Optional
from god_view_reward_utils import get_god_view_metrics, shape_reward_with_god_view


class SubnetRewardCalculator:
    """
    子网奖励计算器
    
    功能：
    1. 计算子网专属的环境原始奖励（只看本子网主机）
    2. 计算子网专属的God-view塑形奖励（只奖励本子网的防御行为）
    3. 组合奖励：final_reward = original_reward * 0.8 + shaped_reward
    """
    
    def __init__(self):
        """初始化子网奖励计算器"""
        # 子网主机映射
        self.subnet_hosts = {
            'User': ['User0', 'User1', 'User2', 'User3', 'User4'],
            'Enterprise': ['Enterprise0', 'Enterprise1', 'Enterprise2'],
            'Operational': ['Op_Server0', 'Op_Host0', 'Op_Host1', 'Op_Host2']
        }
        
        # 主机价值映射（基于CybORG的Scenario2.yaml）
        self.host_confidentiality = {
            # Operational子网
            'Op_Server0': 10.0,    # High
            'Op_Host0': 0.1,       # Low
            'Op_Host1': 0.1,       # Low
            'Op_Host2': 0.1,       # Low
            
            # Enterprise子网
            'Enterprise0': 1.0,    # Medium
            'Enterprise1': 1.0,    # Medium
            'Enterprise2': 0.1,    # Low
            
            # User子网
            'User0': 0.1,          # Low
            'User1': 0.1,          # Low
            'User2': 0.1,          # Low
            'User3': 0.1,          # Low
            'User4': 0.1,          # Low
        }
        
        self.host_availability = {
            # Operational子网
            'Op_Server0': 10.0,    # High
            'Op_Host0': 0.1,       # Low
            'Op_Host1': 0.1,       # Low
            'Op_Host2': 0.1,       # Low
            
            # Enterprise子网
            'Enterprise0': 0.1,    # Low
            'Enterprise1': 0.1,    # Low
            'Enterprise2': 0.1,    # Low
            
            # User子网
            'User0': 0.1,          # Low
            'User1': 0.1,          # Low
            'User2': 0.1,          # Low
            'User3': 0.1,          # Low
            'User4': 0.1,          # Low
        }
    
    def get_subnet_hosts(self, subnet_name: str) -> List[str]:
        """
        获取子网的主机列表
        
        Args:
            subnet_name: 'User' / 'Enterprise' / 'Operational'
            
        Returns:
            主机名列表
        """
        return self.subnet_hosts.get(subnet_name, [])
    
    def calculate_subnet_original_reward(self, env, subnet_name: str) -> float:
        """
        计算子网的环境原始奖励（只看本子网主机）
        
        基于CybORG的Hybrid奖励计算器逻辑：
        - 机密性惩罚：Red获得特权session → 负奖励
        - 可用性惩罚：OTService被破坏 → 负奖励
        
        Args:
            env: CybORG环境对象
            subnet_name: 'User' / 'Enterprise' / 'Operational'
            
        Returns:
            子网原始奖励（通常为负值或0）
        """
        reward = 0.0
        subnet_hosts = self.get_subnet_hosts(subnet_name)
        
        try:
            true_state = env.get_agent_state('True')
            
            for hostname in subnet_hosts:
                if hostname not in true_state:
                    continue
                
                host_state = true_state[hostname]
                
                # 1. 机密性惩罚（特权session）
                if 'Sessions' in host_state:
                    for session in host_state['Sessions']:
                        if session.get('Agent') == 'Red':
                            username = session.get('Username', '')
                            # 只计算特权session（SYSTEM/root）
                            if username in ['SYSTEM', 'root']:
                                confidentiality_value = self.host_confidentiality.get(hostname, 0.1)
                                reward -= confidentiality_value
                                break  # 每台主机只计一次
                
                # 2. 可用性惩罚（OTService破坏）
                # 检查OTService进程是否运行
                has_ot_service = False
                if 'Processes' in host_state:
                    for proc in host_state['Processes']:
                        if proc.get('Process Name') == 'OTService':
                            has_ot_service = True
                            break
                
                # 如果主机应该有OTService但没有运行，则惩罚
                # （Op_Server0应该有OTService）
                if hostname == 'Op_Server0' and not has_ot_service:
                    availability_value = self.host_availability.get(hostname, 0.1)
                    reward -= availability_value
        
        except Exception as e:
            print(f"[Warning] calculate_subnet_original_reward error for {subnet_name}: {e}")
        
        return reward
    
    def calculate_subnet_shaped_reward(self, 
                                      action,
                                      action_type: str,
                                      env,
                                      subnet_name: str,
                                      prev_metrics_dict: Dict,
                                      last_analyse_state: Optional[Dict] = None) -> float:
        """
        计算子网的God-view塑形奖励（只奖励本子网的防御行为）
        
        核心逻辑：
        - 如果action.hostname属于本子网 → 计算塑形奖励
        - 如果action.hostname不属于本子网 → 奖励为0（不关心）
        
        Args:
            action: CybORG动作对象
            action_type: 动作类型字符串
            env: CybORG环境对象
            subnet_name: 'User' / 'Enterprise' / 'Operational'
            prev_metrics_dict: 上一步的威胁指标字典
            last_analyse_state: Analyse状态字典（用于重复检测）
            
        Returns:
            子网塑形奖励
        """
        subnet_hosts = self.get_subnet_hosts(subnet_name)
        
        # 检查动作目标是否属于本子网
        target_hostname = getattr(action, 'hostname', None)
        
        if target_hostname is None:
            # Sleep动作，检查本子网是否有威胁扩散
            return self._calculate_sleep_reward_for_subnet(env, subnet_name, prev_metrics_dict)
        
        if target_hostname not in subnet_hosts:
            # 动作不在本子网，奖励为0
            return 0.0
        
        # 动作在本子网，复用现有的God-view塑形逻辑
        shaped_reward = shape_reward_with_god_view(
            action, action_type, env, prev_metrics_dict, last_analyse_state
        )
        
        return shaped_reward
    
    def _calculate_sleep_reward_for_subnet(self, 
                                           env,
                                           subnet_name: str,
                                           prev_metrics_dict: Dict) -> float:
        """
        计算Sleep动作在子网的奖励
        
        逻辑：
        - 如果本子网有威胁扩散 → -1.0
        - 如果本子网无威胁变化 → -0.3（浪费时间）
        
        Args:
            env: CybORG环境对象
            subnet_name: 子网名称
            prev_metrics_dict: 上一步的威胁指标
            
        Returns:
            Sleep奖励
        """
        subnet_hosts = self.get_subnet_hosts(subnet_name)
        
        # 检查本子网是否有威胁扩散
        threat_increased = False
        
        for hostname in subnet_hosts:
            current_metrics = get_god_view_metrics(env, hostname)
            if current_metrics is None:
                continue
            
            prev_metrics = prev_metrics_dict.get(hostname, {
                'red_session_count': 0,
                'access_level': 'None'
            })
            
            # 检查威胁是否增加
            if (current_metrics['red_session_count'] > prev_metrics['red_session_count'] or
                self._access_level_increased(prev_metrics['access_level'], 
                                            current_metrics['access_level'])):
                threat_increased = True
                break
        
        if threat_increased:
            return -1.0  # 严重惩罚：Sleep期间本子网威胁扩散
        else:
            return -0.3  # 轻微惩罚：浪费时间
    
    def _access_level_increased(self, prev_level: str, curr_level: str) -> bool:
        """检查访问级别是否提升"""
        level_order = {'None': 0, 'User': 1, 'Privileged': 2}
        return level_order.get(curr_level, 0) > level_order.get(prev_level, 0)
    
    def calculate_subnet_reward(self,
                               action,
                               action_type: str,
                               env,
                               subnet_name: str,
                               prev_metrics_dict: Dict,
                               last_analyse_state: Optional[Dict] = None) -> float:
        """
        计算子网的训练奖励（只用子网塑形奖励）
        
        核心逻辑：
        - 训练时：只使用子网的God-view塑形奖励
        - 显示时：环境原始奖励单独打印（不参与训练）
        
        Args:
            action: CybORG动作对象
            action_type: 动作类型字符串
            env: CybORG环境对象
            subnet_name: 'User' / 'Enterprise' / 'Operational'
            prev_metrics_dict: 上一步的威胁指标字典
            last_analyse_state: Analyse状态字典
            
        Returns:
            子网塑形奖励（用于训练）
        """
        # 只使用子网塑形奖励进行训练
        # 环境原始奖励仅用于console显示，不参与训练
        subnet_shaped_reward = self.calculate_subnet_shaped_reward(
            action, action_type, env, subnet_name, 
            prev_metrics_dict, last_analyse_state
        )
        
        return subnet_shaped_reward


# ============================================================================
# 单元测试
# ============================================================================

def test_subnet_reward_calculator():
    """测试子网奖励计算器（不依赖CybORG环境）"""
    calculator = SubnetRewardCalculator()
    
    # 测试1: 获取子网主机列表
    user_hosts = calculator.get_subnet_hosts('User')
    assert len(user_hosts) == 5, f"User子网应有5台主机，实际{len(user_hosts)}"
    assert 'User0' in user_hosts and 'User4' in user_hosts
    print(f"✓ 测试1通过: User子网主机列表 = {user_hosts}")
    
    ent_hosts = calculator.get_subnet_hosts('Enterprise')
    assert len(ent_hosts) == 3, f"Enterprise子网应有3台主机，实际{len(ent_hosts)}"
    print(f"✓ 测试2通过: Enterprise子网主机列表 = {ent_hosts}")
    
    op_hosts = calculator.get_subnet_hosts('Operational')
    assert len(op_hosts) == 4, f"Operational子网应有4台主机，实际{len(op_hosts)}"
    assert 'Op_Server0' in op_hosts
    print(f"✓ 测试3通过: Operational子网主机列表 = {op_hosts}")
    
    # 测试2: 主机价值映射
    assert calculator.host_confidentiality['Op_Server0'] == 10.0
    assert calculator.host_confidentiality['Enterprise0'] == 1.0
    assert calculator.host_confidentiality['User0'] == 0.1
    print(f"✓ 测试4通过: 主机价值映射正确")
    
    # 测试3: 访问级别提升检测
    assert calculator._access_level_increased('None', 'User') == True
    assert calculator._access_level_increased('User', 'Privileged') == True
    assert calculator._access_level_increased('Privileged', 'User') == False
    assert calculator._access_level_increased('None', 'None') == False
    print(f"✓ 测试5通过: 访问级别提升检测正确")
    
    print("\n✅ 所有测试通过！")
    print("\n注意：完整的奖励计算测试需要CybORG环境，将在集成测试中进行。")


if __name__ == '__main__':
    print("=" * 80)
    print("子网奖励计算器单元测试")
    print("=" * 80)
    test_subnet_reward_calculator()

