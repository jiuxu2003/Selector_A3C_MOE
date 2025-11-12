"""
DeceptiveRedAgent - 欺骗性攻击模式

设计目标：
- 前3步：模仿B_lineAgent（触发Cardiff识别为B_line）
- 第4步开始：切换到MeanderAgent（随机游走）

核心思想：
- Cardiff在第3步后通过fingerprint识别Red Agent类型
  - scan_count == 2 → 判定为B_line，加载B_line模型
  - scan_count == 3 → 判定为Meander，加载Meander模型
- DeceptiveRedAgent故意在前3步扫描2次（伪装成B_line）
- 第4步开始切换到Meander模式（随机游走）
- Cardiff会错误地加载B_line模型，但实际攻击是Meander模式

预期效果：
- Cardiff模型：错误识别 → 策略不匹配 → 性能下降
- 我们的模型：基于当前威胁 → 不依赖识别 → 性能稳定
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


class DeceptiveRedAgent(BaseAgent):
    """
    欺骗性Red Agent
    
    攻击阶段：
    1. 步骤0-2：B_line模式（扫描2次，伪装成B_line）
    2. 步骤3+：Meander模式（随机游走，真实攻击）
    """
    
    def __init__(self):
        # B_line状态（前3步）
        self.bline_step = 0
        self.initial_ip = None
        self.last_subnet = None
        self.last_ip_address = None
        
        # Meander状态（第4步开始）
        self.scanned_subnets = []
        self.scanned_ips = []
        self.exploited_ips = []
        self.escalated_hosts = []
        self.host_ip_map = {}
        self.last_host = None
        self.last_ip = None
        
        # 阶段标记
        self.deception_phase = True  # True: B_line模式, False: Meander模式
        self.total_steps = 0
    
    def train(self, results: Results):
        """允许agent学习策略"""
        pass
    
    def get_action(self, observation, action_space):
        """
        前3步：B_line模式（精确扫描2次）
        第4步开始：Meander模式（随机游走）
        """
        self._process_success(observation)
        
        session = list(action_space['session'].keys())[0] if 'session' in action_space else 0
        
        # 始终优先Impact
        if 'Op_Server0' in self.escalated_hosts:
            self.last_host = 'Op_Server0'
            return Impact(agent='Red', hostname='Op_Server0', session=session)
        
        # 前3步：B_line模式（关键：扫描恰好2次）
        if self.total_steps < 3:
            action = self._get_bline_initial_actions(observation, action_space, session)
            self.total_steps += 1
            return action
        
        # 第4步开始：切换到Meander模式
        if self.total_steps == 3:
            self.deception_phase = False
            # print(f"  [DeceptiveRedAgent] 第{self.total_steps}步：切换到Meander模式（欺骗成功）")
        
        self.total_steps += 1
        return self._get_meander_action(observation, action_space, session)
    
    def _get_bline_initial_actions(self, observation, action_space, session):
        """
        B_line风格的前3步（精确控制扫描次数）
        
        目标：让Cardiff的fingerprint检测到scan_count == 2
        """
        # 步骤0：扫描User子网
        if self.bline_step == 0:
            self.initial_ip = observation.get('User0', {}).get('Interface', [{}])[0].get('IP Address')
            if self.initial_ip:
                self.last_subnet = observation['User0']['Interface'][0]['Subnet']
                self.bline_step += 1
                return DiscoverRemoteSystems(session=session, agent='Red', subnet=self.last_subnet)
        
        # 步骤1：扫描一个随机IP（第2次扫描）
        elif self.bline_step == 1:
            hosts = [value for key, value in observation.items() if key != 'success']
            if hosts:
                get_ip = lambda x: x.get('Interface', [{}])[0].get('IP Address')
                interfaces = [get_ip(x) for x in hosts if get_ip(x) and get_ip(x) != self.initial_ip]
                if interfaces:
                    self.last_ip_address = random.choice(interfaces)
                    self.bline_step += 1
                    return DiscoverNetworkServices(session=session, agent='Red', ip_address=self.last_ip_address)
        
        # 步骤2：利用漏洞（不扫描，保持scan_count=2）
        elif self.bline_step == 2:
            if self.last_ip_address:
                self.bline_step += 1
                return ExploitRemoteService(session=session, agent='Red', ip_address=self.last_ip_address)
        
        # 回退：如果出错，返回Sleep
        return Sleep()
    
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
        self.bline_step = 0
        self.initial_ip = None
        self.last_subnet = None
        self.last_ip_address = None
        
        self.scanned_subnets = []
        self.scanned_ips = []
        self.exploited_ips = []
        self.escalated_hosts = []
        self.host_ip_map = {}
        self.last_host = None
        self.last_ip = None
        
        self.deception_phase = True
        self.total_steps = 0
    
    def set_initial_values(self, action_space, observation):
        """设置初始值（BaseAgent接口）"""
        pass


# ============================================================================
# 测试
# ============================================================================
if __name__ == '__main__':
    print("="*80)
    print("DeceptiveRedAgent 测试")
    print("="*80)
    print("\n设计说明:")
    print("  • 前3步：B_line模式（扫描2次）→ Cardiff识别为B_line")
    print("  • 第4步+：Meander模式（随机游走）→ 实际攻击模式")
    print("\n预期效果:")
    print("  • Cardiff: 加载B_line模型，但面对Meander攻击 → 策略不匹配")
    print("  • Our Model: 基于威胁动态防御 → 不受影响")
    print("\n✅ Agent创建成功")
    
    agent = DeceptiveRedAgent()
    print(f"  初始状态: 欺骗模式={agent.deception_phase}, 步数={agent.total_steps}")
    
    # 模拟步数增长
    agent.total_steps = 3
    agent.deception_phase = False
    print(f"  第3步后: 欺骗模式={agent.deception_phase}, 步数={agent.total_steps}")
    print(f"\n✅ DeceptiveRedAgent准备就绪！")

