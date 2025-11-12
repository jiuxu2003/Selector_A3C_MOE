"""
Phase 8E: 基于上帝视角的奖励塑形工具
使用env.get_agent_state('True')获取完整真实状态
"""

def get_god_view_metrics(env, hostname):
    """
    使用上帝视角提取主机的关键威胁指标
    
    Args:
        env: CybORG环境对象
        hostname: 目标主机名（如'User1', 'Op_Server0'等）
    
    Returns:
        dict: {
            'has_red_session': bool,      # 是否有Red session
            'red_session_count': int,     # Red session数量
            'access_level': str,          # 'None'/'User'/'Privileged'
            'malware_files': int,         # 恶意文件数量
            'malware_processes': int      # 恶意进程数量
        }
        或None（如果主机不存在）
    """
    try:
        true_state = env.get_agent_state('True')
        
        if hostname not in true_state:
            return None
        
        host_state = true_state[hostname]
        
        metrics = {
            'has_red_session': False,
            'red_session_count': 0,
            'access_level': 'None',
            'malware_files': 0,
            'malware_processes': 0
        }
        
        # 检查Red Sessions
        if 'Sessions' in host_state:
            for session in host_state['Sessions']:
                if session.get('Agent') == 'Red':
                    metrics['has_red_session'] = True
                    metrics['red_session_count'] += 1
                    
                    # 根据Username判断Access Level
                    username = session.get('Username', '')
                    if username in ['SYSTEM', 'root']:
                        metrics['access_level'] = 'Privileged'
                    elif metrics['access_level'] != 'Privileged':  # 不覆盖更高级别
                        metrics['access_level'] = 'User'
        
        # 检查恶意文件
        if 'Files' in host_state:
            for file in host_state['Files']:
                density = file.get('Density', 0)
                if density >= 0.9:  # CybORG恶意文件标志
                    metrics['malware_files'] += 1
        
        # 检查恶意进程
        if 'Processes' in host_state:
            for proc in host_state['Processes']:
                proc_name = proc.get('Process Name', '')
                # Red的进程名通常包含'Red'或特定恶意标记
                if 'Red' in proc_name or 'RedAbstractSession' in proc_name:
                    metrics['malware_processes'] += 1
        
        return metrics
        
    except Exception as e:
        print(f"[Warning] get_god_view_metrics error for {hostname}: {e}")
        return None


def get_all_hostnames(env):
    """
    获取环境中所有主机名（排除'success'等非主机key）
    
    Args:
        env: CybORG环境对象
    
    Returns:
        list: 主机名列表
    """
    try:
        true_state = env.get_agent_state('True')
        # 排除'success'这个特殊key
        hostnames = [k for k in true_state.keys() if k != 'success']
        return hostnames
    except Exception as e:
        print(f"[Warning] get_all_hostnames error: {e}")
        return []


def shape_reward_with_god_view(action, action_type, env, prev_metrics_dict, last_analyse_state=None):
    """
    基于上帝视角的结果导向奖励塑形
    
    核心理念：
    - 有效动作（清除威胁）→ 高奖励
    - 无效动作（浪费资源）→ 低奖励或惩罚
    - 预防性动作 → 轻微鼓励
    
    Args:
        action: CybORG动作对象
        action_type: 动作类型字符串（'Restore', 'Remove', 'Analyse'等）
        env: CybORG环境对象
        prev_metrics_dict: 执行动作前的metrics字典 {hostname: metrics}
        last_analyse_state: dict，记录每个主机上次Analyse时的威胁状态
                           {hostname: {'sessions': int, 'access': str, 'malware': int}}
    
    Returns:
        float: 塑形后的奖励值
    """
    
    # 提取目标主机
    target_host = getattr(action, 'hostname', None)
    
    # === 全局动作（无特定目标主机） ===
    if target_host is None:
        if action_type in ['Sleep', 'Monitor']:
            # 检查全局威胁变化
            all_hosts = get_all_hostnames(env)
            
            total_sessions_before = sum(
                prev_metrics_dict.get(h, {}).get('red_session_count', 0)
                for h in all_hosts
            )
            
            # 获取当前状态
            current_metrics = {h: get_god_view_metrics(env, h) for h in all_hosts}
            total_sessions_after = sum(
                (current_metrics.get(h) or {}).get('red_session_count', 0)
                for h in all_hosts
            )
            
            # Phase 9H: 提高惩罚，避免Sleep/Monitor
            if total_sessions_after > total_sessions_before:
                return -1.0  # 威胁扩散时严重惩罚
            else:
                return -0.3  # 浪费时间
        
        elif 'Decoy' in action_type:
            return 0.4  # 从2.0降到0.4（预防性诱捕）
        
        else:
            return -0.4  # 从-2.0降到-0.4（其他无目标动作）
    
    # === 针对特定主机的动作 ===
    
    # 获取动作前后的metrics
    prev_metrics = prev_metrics_dict.get(target_host)
    current_metrics = get_god_view_metrics(env, target_host)
    
    if prev_metrics is None or current_metrics is None:
        # 无法获取metrics，返回中性奖励
        return 0.0
    
    prev = prev_metrics
    curr = current_metrics
    
    # === Restore 动作 ===
    if action_type == 'Restore':
        # Phase 9K: 🎯 只奖励"降低权限"
        # 核心逻辑：Restore的价值在于清除权限提升，不是清除session
        # 只有当主机之前有权限提升时，Restore才有意义
        
        # 最高价值：降低Privileged权限
        if prev['access_level'] == 'Privileged' and curr['access_level'] in ['User', 'None']:
            sessions_cleared = prev['red_session_count'] - curr['red_session_count']
            base_reward = 10.0  # 高奖励：清除特权
            bonus = sessions_cleared * 1.0  # session清除奖励
            return base_reward + bonus  # 10.0-13.0分
        
        # 中等价值：降低User权限
        if prev['access_level'] == 'User' and curr['access_level'] == 'None':
            sessions_cleared = prev['red_session_count'] - curr['red_session_count']
            base_reward = 6.0
            bonus = sessions_cleared * 0.8
            return base_reward + bonus  # 6.0-8.4分
        
        # ⚠️ 关键：在没有权限提升的主机上Restore → 严厉惩罚
        # 如果主机本来就没有权限提升，Restore是浪费（应该用Remove）
        if prev['access_level'] == 'None':
            return -4.0  # 严厉惩罚：主机没有权限提升就Restore
        
        # Restore尝试但效果不明显（权限未降低）
        return -1.0  # 提高惩罚：Restore无效果
    
    # === Remove 动作 ===
    elif action_type == 'Remove':
        # Phase 9K: User0动作已从动作空间移除，不再需要特殊处理
        
        # Phase 9K: 提高Remove奖励，鼓励主动清除
        # 清除了Red session
        sessions_removed = prev['red_session_count'] - curr['red_session_count']
        if sessions_removed > 0:
            base_reward = 5.0  # 提高from 3.2
            bonus = sessions_removed * 1.0  # 提高from 0.8
            return base_reward + bonus  # 5.0-8.0分
        
        # 清除了恶意进程
        processes_removed = prev['malware_processes'] - curr['malware_processes']
        if processes_removed > 0:
            return 4.0  # 提高from 2.8
        
        # Remove无法降低Privileged权限（只能清除user-level shell）
        # 移除这个不合理的奖励
        # if prev['access_level'] == 'Privileged' and curr['access_level'] == 'User':
        #     return 1.2  # 错误！Remove做不到这个
        
        # 无效Remove（目标本来就干净）
        if not prev['has_red_session'] and prev['malware_processes'] == 0:
            return -0.5  # 保持不变
        
        # Remove尝试但无效（包括对privileged shell无效）
        return -0.3  # 保持不变
    
    # === Analyse 动作 ===
    elif action_type == 'Analyse':
        # Phase 9J: 🎯 信息增量逻辑 - 只奖励"新信息"
        # 奖励条件：
        # 1. 首次Analyse该主机 → 根据威胁给奖励
        # 2. 重复Analyse，但发现新威胁/权限提升 → 给奖励
        # 3. 重复Analyse，无新信息 → 惩罚
        
        if last_analyse_state is None:
            last_analyse_state = {}
        
        # 获取当前威胁状态
        curr_sessions = curr.get('red_session_count', 0)
        curr_access = curr.get('access_level', 'None')
        curr_malware = curr.get('malware_files', 0)
        
        # 检查是否首次Analyse
        if target_host not in last_analyse_state:
            # 首次Analyse：根据威胁级别给奖励
            has_threat = (curr_sessions > 0 or curr_malware > 0 or curr_access != 'None')
            
            if has_threat:
                threat_level = 0
                if curr_sessions > 0:
                    threat_level += 0.8  # 发现session
                if curr_malware > 0:
                    threat_level += 0.4  # 发现恶意文件
                if curr_access == 'Privileged':
                    threat_level += 0.4  # 发现特权
                return min(threat_level, 1.2)
            else:
                return 0.12  # 预防性扫描干净主机
        
        # 非首次Analyse：检查信息增量
        else:
            last_state = last_analyse_state[target_host]
            last_sessions = last_state['sessions']
            last_access = last_state['access']
            last_malware = last_state['malware']
            
            # 计算增量
            new_sessions = max(0, curr_sessions - last_sessions)
            access_escalated = (last_access == 'None' and curr_access != 'None') or \
                              (last_access == 'User' and curr_access == 'Privileged')
            new_malware = max(0, curr_malware - last_malware)
            
            # 有信息增量 → 奖励
            if new_sessions > 0 or access_escalated or new_malware > 0:
                info_gain = 0
                if new_sessions > 0:
                    info_gain += 0.8  # 发现新session
                if access_escalated:
                    info_gain += 0.4  # 发现权限提升
                if new_malware > 0:
                    info_gain += 0.4  # 发现新恶意文件
                return min(info_gain, 1.2)
            
            # 无信息增量 → 惩罚浪费
            else:
                # Phase 9K: 提高重复Analyse惩罚
                return -1.0  # 提高from -0.5，强烈阻止重复扫描
    
    # === 其他动作 ===
    else:
        # 默认返回0，保持环境原始奖励
        return 0.0


def get_action_type(action):
    """
    从CybORG动作对象提取动作类型字符串
    
    Args:
        action: CybORG动作对象
    
    Returns:
        str: 动作类型（'Restore', 'Remove', 'Analyse', 'Sleep'等）
    """
    if action is None:
        return 'Unknown'
    
    # 获取类名
    class_name = action.__class__.__name__
    
    # CybORG的动作类名就是动作类型
    # 例如：Restore, Remove, Analyse, DecoyApache等
    return class_name


# ===== 测试函数 =====

def test_god_view_metrics():
    """测试get_god_view_metrics函数"""
    import inspect
    from CybORG import CybORG
    from CybORG.Agents import B_lineAgent
    
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'
    
    env = CybORG(path, 'sim', agents={'Red': B_lineAgent})
    env.reset()
    
    # 让Red行动几步
    for _ in range(10):
        env.step()
    
    # 测试User0（Red起点，应该有session）
    print("=== Testing User0 ===")
    metrics = get_god_view_metrics(env, 'User0')
    print(f"Metrics: {metrics}")
    
    # 测试Op_Server0（应该干净）
    print("\n=== Testing Op_Server0 ===")
    metrics = get_god_view_metrics(env, 'Op_Server0')
    print(f"Metrics: {metrics}")
    
    # 测试所有主机
    print("\n=== All Hostnames ===")
    hostnames = get_all_hostnames(env)
    print(f"Hostnames: {hostnames}")
    
    print("\n✅ Test completed!")


if __name__ == "__main__":
    test_god_view_metrics()

