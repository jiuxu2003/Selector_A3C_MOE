"""
子网观察空间提取工具
将全局52维观察空间分离为子网独立观察空间
"""

import numpy as np
from typing import Dict, List


class SubnetObservationExtractor:
    """
    子网观察空间提取器
    
    功能：
    1. 从全局观察中提取子网专属观察
    2. 归一化观察值
    3. 支持三个子网：User(20维), Enterprise(16维), Operational(20维)
    """
    
    def __init__(self):
        """初始化观察提取器"""
        # 子网主机映射
        # User0被排除（Red的persistent foothold，无法防御）
        self.subnet_hosts = {
            'User': ['User1', 'User2', 'User3', 'User4'],
            'Enterprise': ['Enterprise0', 'Enterprise1', 'Enterprise2'],
            'Operational': ['Op_Server0', 'Op_Host0', 'Op_Host1', 'Op_Host2']
        }
        
        # 观察空间维度
        self.obs_dims = {
            'User': 20,        # 4台主机×4维 + 4维统计（排除User0）
            'Enterprise': 16,  # 3台主机×4维 + 4维统计
            'Operational': 20  # 4台主机×4维 + 4维统计
        }
        
        # 归一化参数（基于经验值）
        self.normalization = {
            'sessions': 5.0,           # 最大session数
            'privileged': 1.0,         # 0或1
            'processes': 10.0,         # 最大可疑进程数
            'access': 4.0,             # 访问级别 0-4
            'total_sessions': 20.0,    # 子网总session数
            'total_privileged': 5.0,   # 子网总特权session数
            'total_processes': 30.0,   # 子网总可疑进程数
            'availability': 1.0        # 可用性 0-1
        }
    
    def extract_user_observation(self, obs_dict: Dict) -> np.ndarray:
        """
        提取User子网观察空间（20维）
        
        结构：
        - User1-4主机信息：4台 × 4维 = 16维（排除User0）
          每台主机：[sessions, privileged, processes, access]
        - User子网统计：4维
          [total_sessions, total_privileged, total_processes, availability]
        
        Args:
            obs_dict: CybORG观察字典
            
        Returns:
            np.ndarray[20]，归一化后的观察向量
        """
        observation = []
        
        # 统计变量
        total_sessions = 0
        total_privileged = 0
        total_processes = 0
        available_hosts = 0
        
        # 1. 提取各主机信息（16维 - 排除User0）
        for hostname in self.subnet_hosts['User']:
            if hostname in obs_dict:
                host_info = obs_dict[hostname]
                
                # Sessions
                sessions = host_info.get('Sessions', 0)
                if isinstance(sessions, (int, float)):
                    observation.append(min(sessions / self.normalization['sessions'], 1.0))
                    total_sessions += sessions
                else:
                    observation.append(0.0)
                
                # Privileged Access
                privileged = host_info.get('Privileged Access', 0)
                observation.append(float(privileged > 0))
                if privileged > 0:
                    total_privileged += 1
                
                # Processes (suspicious)
                processes = host_info.get('Processes', 0)
                if isinstance(processes, (int, float)):
                    observation.append(min(processes / self.normalization['processes'], 1.0))
                    total_processes += processes
                else:
                    observation.append(0.0)
                
                # Access Level
                access = host_info.get('System info', {}).get('access_level', 0)
                if isinstance(access, (int, float)):
                    observation.append(min(access / self.normalization['access'], 1.0))
                else:
                    observation.append(0.0)
                
                # 主机可用性
                if host_info.get('System info', {}).get('OSType') is not None:
                    available_hosts += 1
            else:
                # 主机不可见，填充0
                observation.extend([0.0, 0.0, 0.0, 0.0])
        
        # 2. 子网统计信息（4维）
        observation.append(min(total_sessions / self.normalization['total_sessions'], 1.0))
        observation.append(min(total_privileged / self.normalization['total_privileged'], 1.0))
        observation.append(min(total_processes / self.normalization['total_processes'], 1.0))
        observation.append(available_hosts / len(self.subnet_hosts['User']))
        
        return np.array(observation, dtype=np.float32)
    
    def extract_enterprise_observation(self, obs_dict: Dict) -> np.ndarray:
        """
        提取Enterprise子网观察空间（16维）
        
        结构：
        - Enterprise0-2主机信息：3台 × 4维 = 12维
        - Enterprise子网统计：4维
        
        Args:
            obs_dict: CybORG观察字典
            
        Returns:
            np.ndarray[16]，归一化后的观察向量
        """
        observation = []
        
        # 统计变量
        total_sessions = 0
        total_privileged = 0
        total_processes = 0
        available_hosts = 0
        
        # 1. 提取各主机信息（12维）
        for hostname in self.subnet_hosts['Enterprise']:
            if hostname in obs_dict:
                host_info = obs_dict[hostname]
                
                # Sessions
                sessions = host_info.get('Sessions', 0)
                if isinstance(sessions, (int, float)):
                    observation.append(min(sessions / self.normalization['sessions'], 1.0))
                    total_sessions += sessions
                else:
                    observation.append(0.0)
                
                # Privileged Access
                privileged = host_info.get('Privileged Access', 0)
                observation.append(float(privileged > 0))
                if privileged > 0:
                    total_privileged += 1
                
                # Processes
                processes = host_info.get('Processes', 0)
                if isinstance(processes, (int, float)):
                    observation.append(min(processes / self.normalization['processes'], 1.0))
                    total_processes += processes
                else:
                    observation.append(0.0)
                
                # Access Level
                access = host_info.get('System info', {}).get('access_level', 0)
                if isinstance(access, (int, float)):
                    observation.append(min(access / self.normalization['access'], 1.0))
                else:
                    observation.append(0.0)
                
                # 主机可用性
                if host_info.get('System info', {}).get('OSType') is not None:
                    available_hosts += 1
            else:
                observation.extend([0.0, 0.0, 0.0, 0.0])
        
        # 2. 子网统计信息（4维）
        observation.append(min(total_sessions / self.normalization['total_sessions'], 1.0))
        observation.append(min(total_privileged / self.normalization['total_privileged'], 1.0))
        observation.append(min(total_processes / self.normalization['total_processes'], 1.0))
        observation.append(available_hosts / len(self.subnet_hosts['Enterprise']))
        
        return np.array(observation, dtype=np.float32)
    
    def extract_operational_observation(self, obs_dict: Dict) -> np.ndarray:
        """
        提取Operational子网观察空间（20维）
        
        结构：
        - Op_Server0, Op_Host0-2主机信息：4台 × 4维 = 16维
        - Operational子网统计：4维
        
        Args:
            obs_dict: CybORG观察字典
            
        Returns:
            np.ndarray[20]，归一化后的观察向量
        """
        observation = []
        
        # 统计变量
        total_sessions = 0
        total_privileged = 0
        total_processes = 0
        available_hosts = 0
        
        # 1. 提取各主机信息（16维）
        for hostname in self.subnet_hosts['Operational']:
            if hostname in obs_dict:
                host_info = obs_dict[hostname]
                
                # Sessions
                sessions = host_info.get('Sessions', 0)
                if isinstance(sessions, (int, float)):
                    observation.append(min(sessions / self.normalization['sessions'], 1.0))
                    total_sessions += sessions
                else:
                    observation.append(0.0)
                
                # Privileged Access
                privileged = host_info.get('Privileged Access', 0)
                observation.append(float(privileged > 0))
                if privileged > 0:
                    total_privileged += 1
                
                # Processes
                processes = host_info.get('Processes', 0)
                if isinstance(processes, (int, float)):
                    observation.append(min(processes / self.normalization['processes'], 1.0))
                    total_processes += processes
                else:
                    observation.append(0.0)
                
                # Access Level
                access = host_info.get('System info', {}).get('access_level', 0)
                if isinstance(access, (int, float)):
                    observation.append(min(access / self.normalization['access'], 1.0))
                else:
                    observation.append(0.0)
                
                # 主机可用性
                if host_info.get('System info', {}).get('OSType') is not None:
                    available_hosts += 1
            else:
                observation.extend([0.0, 0.0, 0.0, 0.0])
        
        # 2. 子网统计信息（4维）
        observation.append(min(total_sessions / self.normalization['total_sessions'], 1.0))
        observation.append(min(total_privileged / self.normalization['total_privileged'], 1.0))
        observation.append(min(total_processes / self.normalization['total_processes'], 1.0))
        observation.append(available_hosts / len(self.subnet_hosts['Operational']))
        
        return np.array(observation, dtype=np.float32)
    
    def extract_subnet_observation(self, obs_dict: Dict, subnet_name: str) -> np.ndarray:
        """
        根据子网名称提取对应的观察空间
        
        Args:
            obs_dict: CybORG观察字典
            subnet_name: 子网名称 ('User', 'Enterprise', 'Operational')
            
        Returns:
            对应子网的观察向量
        """
        if subnet_name == 'User':
            return self.extract_user_observation(obs_dict)
        elif subnet_name == 'Enterprise':
            return self.extract_enterprise_observation(obs_dict)
        elif subnet_name == 'Operational':
            return self.extract_operational_observation(obs_dict)
        else:
            raise ValueError(f"Unknown subnet name: {subnet_name}")
    
    def extract_all_subnets(self, obs_dict: Dict) -> Dict[str, np.ndarray]:
        """
        提取所有子网的观察空间
        
        Args:
            obs_dict: CybORG观察字典
            
        Returns:
            {'User': ndarray[20], 'Enterprise': ndarray[16], 'Operational': ndarray[20]}
        """
        return {
            'User': self.extract_user_observation(obs_dict),
            'Enterprise': self.extract_enterprise_observation(obs_dict),
            'Operational': self.extract_operational_observation(obs_dict)
        }


# ============================================================================
# 单元测试
# ============================================================================

def test_observation_extractor():
    """测试观察空间提取器"""
    extractor = SubnetObservationExtractor()
    
    # 测试观察字典
    test_obs = {
        'User0': {'Sessions': 2, 'Privileged Access': 1, 'Processes': 3, 
                  'System info': {'access_level': 2, 'OSType': 'Windows'}},
        'User1': {'Sessions': 0, 'Privileged Access': 0, 'Processes': 0,
                  'System info': {'access_level': 0, 'OSType': 'Windows'}},
        'User2': {'Sessions': 1, 'Privileged Access': 0, 'Processes': 1,
                  'System info': {'access_level': 1, 'OSType': 'Windows'}},
        'Enterprise0': {'Sessions': 3, 'Privileged Access': 1, 'Processes': 5,
                       'System info': {'access_level': 3, 'OSType': 'Linux'}},
        'Enterprise1': {'Sessions': 0, 'Privileged Access': 0, 'Processes': 0,
                       'System info': {'access_level': 0, 'OSType': 'Linux'}},
        'Op_Server0': {'Sessions': 1, 'Privileged Access': 1, 'Processes': 2,
                      'System info': {'access_level': 4, 'OSType': 'Linux'}},
        'Op_Host0': {'Sessions': 0, 'Privileged Access': 0, 'Processes': 0,
                    'System info': {'access_level': 0, 'OSType': 'Linux'}}
    }
    
    # 测试1: User观察空间
    user_obs = extractor.extract_user_observation(test_obs)
    assert user_obs.shape == (24,), f"User观察维度应为24，实际为{user_obs.shape}"
    assert np.all(user_obs >= 0.0) and np.all(user_obs <= 1.0), "观察值应在[0,1]范围内"
    print(f"✓ 测试1通过: User观察空间 shape={user_obs.shape}, 范围=[{user_obs.min():.2f}, {user_obs.max():.2f}]")
    
    # 测试2: Enterprise观察空间
    ent_obs = extractor.extract_enterprise_observation(test_obs)
    assert ent_obs.shape == (16,), f"Enterprise观察维度应为16，实际为{ent_obs.shape}"
    assert np.all(ent_obs >= 0.0) and np.all(ent_obs <= 1.0), "观察值应在[0,1]范围内"
    print(f"✓ 测试2通过: Enterprise观察空间 shape={ent_obs.shape}, 范围=[{ent_obs.min():.2f}, {ent_obs.max():.2f}]")
    
    # 测试3: Operational观察空间
    op_obs = extractor.extract_operational_observation(test_obs)
    assert op_obs.shape == (20,), f"Operational观察维度应为20，实际为{op_obs.shape}"
    assert np.all(op_obs >= 0.0) and np.all(op_obs <= 1.0), "观察值应在[0,1]范围内"
    print(f"✓ 测试3通过: Operational观察空间 shape={op_obs.shape}, 范围=[{op_obs.min():.2f}, {op_obs.max():.2f}]")
    
    # 测试4: 提取所有子网
    all_obs = extractor.extract_all_subnets(test_obs)
    assert len(all_obs) == 3, f"应返回3个子网观察，实际为{len(all_obs)}"
    assert 'User' in all_obs and 'Enterprise' in all_obs and 'Operational' in all_obs
    print(f"✓ 测试4通过: 提取所有子网观察")
    
    # 测试5: 空观察
    empty_obs = {}
    user_empty = extractor.extract_user_observation(empty_obs)
    assert user_empty.shape == (24,), "空观察应返回正确维度"
    assert np.all(user_empty == 0.0), "空观察应全为0"
    print(f"✓ 测试5通过: 空观察处理正确")
    
    # 测试6: 观察值细节检查
    print(f"\n观察值细节:")
    print(f"  User前4维（User0）: {user_obs[:4]}")
    print(f"  User后4维（统计）: {user_obs[-4:]}")
    print(f"  Enterprise前4维（Ent0）: {ent_obs[:4]}")
    print(f"  Operational前4维（Op_Server0）: {op_obs[:4]}")
    
    print("\n✅ 所有测试通过！")


if __name__ == '__main__':
    print("=" * 80)
    print("子网观察空间提取器单元测试")
    print("=" * 80)
    test_observation_extractor()

