"""
SleepDeceptiveRedAgent - 潜伏式欺骗攻击

设计目标：
- 前3步：Sleep（伪装成SleepAgent，无威胁）
- 第4步开始：Meander模式（突然发起全面攻击）

核心思想：
- Cardiff在前几步观察Red Agent行为判定威胁级别
  - 连续Sleep → 判定为SleepAgent → 放松防守
- SleepDeceptiveRedAgent前3步完全静默（0威胁）
- 第4步开始突然切换到Meander（高强度随机攻击）
- Cardiff已经放松警惕 → 猝不及防

预期效果：
- Cardiff模型：误判为Sleep → 放松防守 → 遭受重创
- 我们的模型：始终监控威胁 → 发现攻击立即响应 → 性能稳定

攻击特点：
1. 极致隐蔽性：前3步零动作，完全潜伏
2. 突然性：第4步突然全面攻击
3. 欺骗性：利用Cardiff的威胁判定机制
"""
import random
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'CybORG'))

from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from CybORG.Shared import Results
from CybORG.Shared.Actions import (PrivilegeEscalate, ExploitRemoteService, 
                                   DiscoverRemoteSystems, Impact, 
                                   DiscoverNetworkServices, Sleep)


class SleepDeceptiveRedAgent(BaseAgent):
    """
    潜伏式欺骗Red Agent
    
    攻击阶段：
    1. 步骤0-2：Sleep模式（完全静默，伪装成SleepAgent）
    2. 步骤3+：Meander模式（突然发起随机游走攻击）
    """
    
    def __init__(self):
        # Meander状态
        self.scanned_subnets = []
        self.scanned_ips = []
        self.exploited_ips = []
        self.escalated_hosts = []
        self.host_ip_map = {}
        self.last_host = None
        self.last_ip = None
        
        # 阶段标记
        self.sleeping = True  # True: Sleep模式, False: Meander模式
        self.total_steps = 0
    
    def train(self, results: Results):
        """允许agent学习策略"""
        pass
    
    def get_action(self, observation, action_space):
        """
        前3步：Sleep（完全静默）
        第4步开始：Meander模式（突然攻击）
        """
        self._process_success(observation)
        
        session = list(action_space['session'].keys())[0] if 'session' in action_space else 0
        
        # 始终优先Impact
        if 'Op_Server0' in self.escalated_hosts:
            self.last_host = 'Op_Server0'
            return Impact(agent='Red', hostname='Op_Server0', session=session)
        
        # 前3步：完全Sleep（伪装成SleepAgent）
        if self.total_steps < 3:
            self.total_steps += 1
            return Sleep()
        
        # 第4步：切换到Meander模式（突然攻击）
        if self.total_steps == 3:
            self.sleeping = False
        
        self.total_steps += 1
        return self._get_meander_action(observation, action_space, session)
    
    def _get_meander_action(self, observation, action_space, session):
        """
        Meander模式：随机游走攻击
        """
        # 扫描子网
        for subnet in action_space["subnet"]:
            if not action_space["subnet"][subnet] or subnet in self.scanned_subnets:
                continue
            self.scanned_subnets.append(subnet)
            return DiscoverRemoteSystems(subnet=subnet, agent='Red', session=session)
        
        # 随机扫描IP
        addresses = [i for i in action_space["ip_address"]]
        random.shuffle(addresses)
        
        for address in addresses:
            if not action_space["ip_address"][address] or address in self.scanned_ips:
                continue
            self.scanned_ips.append(address)
            return DiscoverNetworkServices(ip_address=address, agent='Red', session=session)
        
        # 随机提权
        hostnames = [x for x in action_space['hostname'].keys()]
        random.shuffle(hostnames)
        
        for hostname in hostnames:
            if not action_space["hostname"][hostname]:
                continue
            if hostname in self.escalated_hosts:
                continue
            if hostname in self.host_ip_map and self.host_ip_map[hostname] not in self.exploited_ips:
                continue
            self.escalated_hosts.append(hostname)
            self.last_host = hostname
            return PrivilegeEscalate(hostname=hostname, agent='Red', session=session)
        
        # 随机利用漏洞
        for address in addresses:
            if not action_space["ip_address"][address] or address in self.exploited_ips:
                continue
            self.exploited_ips.append(address)
            self.last_ip = address
            return ExploitRemoteService(ip_address=address, agent='Red', session=session)
        
        # 无动作可用
        return Sleep()
    
    def _process_success(self, observation):
        """处理动作成功/失败反馈"""
        if self.last_ip is not None:
            if observation.get('success') == True:
                try:
                    hostname = [value['System info']['Hostname'] for key, value in observation.items()
                               if key != 'success' and 'System info' in value
                               and 'Hostname' in value['System info']][0]
                    self.host_ip_map[hostname] = self.last_ip
                except:
                    pass
            else:
                if self.last_ip in self.exploited_ips:
                    self.exploited_ips.remove(self.last_ip)
            self.last_ip = None
        
        if self.last_host is not None:
            if observation.get('success') == False:
                if self.last_host in self.escalated_hosts:
                    self.escalated_hosts.remove(self.last_host)
                if self.last_host in self.host_ip_map and self.host_ip_map[self.last_host] in self.exploited_ips:
                    self.exploited_ips.remove(self.host_ip_map[self.last_host])
            self.last_host = None
    
    def end_episode(self):
        """重置episode"""
        self.scanned_subnets = []
        self.scanned_ips = []
        self.exploited_ips = []
        self.escalated_hosts = []
        self.host_ip_map = {}
        self.last_host = None
        self.last_ip = None
        
        self.sleeping = True
        self.total_steps = 0
    
    def set_initial_values(self, action_space, observation):
        """设置初始值（BaseAgent接口）"""
        pass


# ============================================================================
# 测试
# ============================================================================
if __name__ == '__main__':
    print("="*80)
    print("SleepDeceptiveRedAgent 测试")
    print("="*80)
    print("\n设计说明:")
    print("  • 前3步：Sleep（完全静默）→ Cardiff误判为SleepAgent")
    print("  • 第4步+：Meander模式（突然全面攻击）")
    print("\n攻击特点:")
    print("  1. 极致隐蔽：前3步零威胁")
    print("  2. 突然性：第4步突然攻击")
    print("  3. 欺骗性：利用威胁判定弱点")
    print("\n预期效果:")
    print("  • Cardiff: 误判为Sleep → 放松防守 → 猝不及防")
    print("  • Our Model: 监控威胁 → 及时响应 → 性能稳定")
    print("\n✅ Agent创建成功")
    
    agent = SleepDeceptiveRedAgent()
    print(f"  初始状态: 潜伏模式={agent.sleeping}, 步数={agent.total_steps}")
    
    # 模拟步数增长
    agent.total_steps = 3
    agent.sleeping = False
    print(f"  第3步后: 潜伏模式={agent.sleeping}, 步数={agent.total_steps}")
    print(f"\n✅ SleepDeceptiveRedAgent准备就绪！")

