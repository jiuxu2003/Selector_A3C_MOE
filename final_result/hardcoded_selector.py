"""
硬编码子网选择器 - Phase 9L
基于威胁分数逻辑直接判断调用哪个子网Expert
"""

from typing import Dict, Tuple
import numpy as np


class HardcodedSelector:
    """
    基于威胁分数的硬编码子网选择器
    
    核心逻辑：计算各子网的威胁分数，选择威胁最严重的子网进行防御
    
    威胁分数计算考虑：
    1. Red session数量
    2. 特权session（SYSTEM/root）
    3. 可疑进程数量
    4. 子网价值权重（Op > Enterprise > User）
    """
    
    def __init__(self):
        """初始化硬编码选择器"""
        # 子网主机映射
        self.subnet_hosts = {
            'User': ['User0', 'User1', 'User2', 'User3', 'User4'],
            'Enterprise': ['Enterprise0', 'Enterprise1', 'Enterprise2'],
            'Operational': ['Op_Server0', 'Op_Host0', 'Op_Host1', 'Op_Host2']
        }
        
        # 统计信息
        self.selection_counts = {'User': 0, 'Enterprise': 0, 'Operational': 0}
        self.total_selections = 0
    
    def select_subnet(self, observation: Dict) -> str:
        """
        根据观察计算各子网威胁分数，选择最需要防御的子网
        
        Phase 9L改进：严格的优先级逻辑
        - 只要高优先级子网有威胁，就优先防御它
        - 优先级：Operational > Enterprise > User
        
        决策规则：
        1. 如果Op有威胁 → 选Op
        2. 否则，如果Enterprise有威胁 → 选Enterprise
        3. 否则，如果User有威胁 → 选User
        4. 都无威胁 → 默认巡逻Op
        
        Args:
            observation: CybORG观察字典（包含所有主机信息）
            
        Returns:
            'User' / 'Enterprise' / 'Operational'
        """
        # 计算各子网威胁分数
        threat_scores = {
            'User': self._calculate_user_threat(observation),
            'Enterprise': self._calculate_enterprise_threat(observation),
            'Operational': self._calculate_operational_threat(observation)
        }
        
        # Phase 9L改进：严格的优先级逻辑
        # 按优先级顺序检查，只要有威胁就选择
        if threat_scores['Operational'] > 0:
            selected_subnet = 'Operational'
        elif threat_scores['Enterprise'] > 0:
            selected_subnet = 'Enterprise'
        elif threat_scores['User'] > 0:
            selected_subnet = 'User'
        else:
            # 所有子网都安全，默认巡逻User（Red的入口点）
            selected_subnet = 'User'
        
        # 更新统计
        self.selection_counts[selected_subnet] += 1
        self.total_selections += 1
        
        return selected_subnet
    
    def select_subnet_with_scores(self, observation: Dict) -> Tuple[str, Dict[str, float]]:
        """
        选择子网并返回详细的威胁分数（用于调试和分析）
        
        Phase 9L改进：严格的优先级逻辑
        
        Args:
            observation: CybORG观察字典
            
        Returns:
            (selected_subnet, threat_scores)
        """
        threat_scores = {
            'User': self._calculate_user_threat(observation),
            'Enterprise': self._calculate_enterprise_threat(observation),
            'Operational': self._calculate_operational_threat(observation)
        }
        
        # Phase 9L改进：严格的优先级逻辑
        # 按优先级顺序检查，只要有威胁就选择
        if threat_scores['Operational'] > 0:
            selected_subnet = 'Operational'
        elif threat_scores['Enterprise'] > 0:
            selected_subnet = 'Enterprise'
        elif threat_scores['User'] > 0:
            selected_subnet = 'User'
        else:
            # 所有子网都安全，默认巡逻User（Red的入口点）
            selected_subnet = 'User'
        
        self.selection_counts[selected_subnet] += 1
        self.total_selections += 1
        
        return selected_subnet, threat_scores
    
    def _calculate_user_threat(self, obs: Dict) -> float:
        """
        User子网威胁评分逻辑
        
        威胁指标：
        - Red session数量：每个+1.0
        - 特权session（SYSTEM/root）：+5.0
        - 可疑进程数量：每个+0.5
        
        **重要**：User0被排除，因为：
        - User0是Red的persistent foothold（无法清除）
        - User0的动作已从动作空间移除（Phase 9K）
        - 不应该因为无法防御的威胁被惩罚
        
        Args:
            obs: 观察字典
            
        Returns:
            威胁分数
        """
        threat = 0.0
        
        for host in self.subnet_hosts['User']:
            # 跳过User0（不可防御的Red入口点）
            if host == 'User0':
                continue
                
            if host not in obs:
                continue
            
            host_info = obs[host]
            
            # Red session: 每个+1.0
            red_sessions = host_info.get('Sessions', 0)
            if isinstance(red_sessions, (int, float)):
                threat += red_sessions * 1.0
            
            # 特权session: +5.0（严重威胁）
            privileged = host_info.get('Privileged Access', 0)
            if privileged > 0:
                threat += 5.0
            
            # 可疑进程: 每个+0.5
            suspicious = host_info.get('Processes', 0)
            if isinstance(suspicious, (int, float)):
                threat += suspicious * 0.5
        
        return threat
    
    def _calculate_enterprise_threat(self, obs: Dict) -> float:
        """
        Enterprise子网威胁评分（权重更高）
        
        威胁指标：
        - Red session数量：每个+1.5（权重更高）
        - 特权session：+8.0（更严重）
        - 可疑进程数量：每个+0.8
        
        Args:
            obs: 观察字典
            
        Returns:
            威胁分数
        """
        threat = 0.0
        
        for host in self.subnet_hosts['Enterprise']:
            if host not in obs:
                continue
            
            host_info = obs[host]
            
            # Enterprise价值更高，权重×1.5
            red_sessions = host_info.get('Sessions', 0)
            if isinstance(red_sessions, (int, float)):
                threat += red_sessions * 1.5
            
            # 特权session: +8.0（更严重）
            privileged = host_info.get('Privileged Access', 0)
            if privileged > 0:
                threat += 8.0
            
            # 可疑进程: 每个+0.8
            suspicious = host_info.get('Processes', 0)
            if isinstance(suspicious, (int, float)):
                threat += suspicious * 0.8
        
        return threat
    
    def _calculate_operational_threat(self, obs: Dict) -> float:
        """
        Operational子网威胁评分（最高权重）
        
        威胁指标：
        - Red session数量：每个+2.0（最高权重）
        - Op_Server0特权session：+15.0（灾难性）
        - 其他主机特权session：+10.0
        - 可疑进程数量：每个+1.0
        
        Args:
            obs: 观察字典
            
        Returns:
            威胁分数
        """
        threat = 0.0
        
        for host in self.subnet_hosts['Operational']:
            if host not in obs:
                continue
            
            host_info = obs[host]
            
            # Op子网最关键，权重×2.0
            red_sessions = host_info.get('Sessions', 0)
            if isinstance(red_sessions, (int, float)):
                threat += red_sessions * 2.0
            
            # 特权session: Op_Server0最严重
            privileged = host_info.get('Privileged Access', 0)
            if privileged > 0:
                if host == 'Op_Server0':
                    threat += 15.0  # 灾难性威胁
                else:
                    threat += 10.0
            
            # 可疑进程: 每个+1.0
            suspicious = host_info.get('Processes', 0)
            if isinstance(suspicious, (int, float)):
                threat += suspicious * 1.0
        
        return threat
    
    def get_statistics(self) -> Dict:
        """
        获取选择器统计信息
        
        Returns:
            统计字典，包含各子网选择次数和比例
        """
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
    
    def reset_statistics(self):
        """重置统计信息"""
        self.selection_counts = {'User': 0, 'Enterprise': 0, 'Operational': 0}
        self.total_selections = 0


# ============================================================================
# 单元测试
# ============================================================================

def test_hardcoded_selector():
    """测试硬编码选择器的基本功能"""
    selector = HardcodedSelector()
    
    # 测试1: 空观察（所有子网安全）→ 默认User
    empty_obs = {}
    result = selector.select_subnet(empty_obs)
    assert result == 'User', f"空观察应选择User，实际选择了{result}"
    print("✓ 测试1通过: 空观察 → User")
    
    # 测试2: User子网有威胁
    user_threat_obs = {
        'User0': {'Sessions': 2, 'Privileged Access': 0, 'Processes': 1},
        'User1': {'Sessions': 0, 'Privileged Access': 0, 'Processes': 0}
    }
    result = selector.select_subnet(user_threat_obs)
    assert result == 'User', f"User威胁应选择User，实际选择了{result}"
    print("✓ 测试2通过: User威胁 → User")
    
    # 测试3: Enterprise子网威胁更高（权重×1.5）
    enterprise_threat_obs = {
        'User0': {'Sessions': 1, 'Privileged Access': 0, 'Processes': 0},
        'Enterprise0': {'Sessions': 1, 'Privileged Access': 0, 'Processes': 0}
    }
    result = selector.select_subnet(enterprise_threat_obs)
    assert result == 'Enterprise', f"Enterprise威胁应选择Enterprise，实际选择了{result}"
    print("✓ 测试3通过: Enterprise威胁 → Enterprise")
    
    # 测试4: Op子网特权session（最高优先级）
    op_threat_obs = {
        'User0': {'Sessions': 3, 'Privileged Access': 0, 'Processes': 2},
        'Enterprise0': {'Sessions': 2, 'Privileged Access': 0, 'Processes': 1},
        'Op_Server0': {'Sessions': 1, 'Privileged Access': 1, 'Processes': 0}
    }
    result = selector.select_subnet(op_threat_obs)
    assert result == 'Operational', f"Op特权威胁应选择Operational，实际选择了{result}"
    print("✓ 测试4通过: Op特权威胁 → Operational")
    
    # 测试5: 威胁分数详细输出
    result, scores = selector.select_subnet_with_scores(op_threat_obs)
    print(f"✓ 测试5通过: 威胁分数 = {scores}")
    assert scores['Operational'] > max(scores['Enterprise'], scores['User'])
    print(f"  威胁分数排序正确: Op({scores['Operational']:.1f}) > Ent({scores['Enterprise']:.1f}), User({scores['User']:.1f})")
    
    # 测试6: 统计信息
    stats = selector.get_statistics()
    print(f"✓ 测试6通过: 统计信息 = {stats}")
    assert stats['total'] == 5
    
    print("\n✅ 所有测试通过！")


if __name__ == '__main__':
    print("=" * 80)
    print("硬编码Selector单元测试")
    print("=" * 80)
    test_hardcoded_selector()

