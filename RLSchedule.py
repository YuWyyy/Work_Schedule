import torch
import numpy as np
from torch import nn, optim
from datetime import datetime
from collections import deque
from Tools.ReadAndSort import work_sort, read_excel_range
from Tools.Writer import show_info

# === 基础参数 ===
file_path = "Data.xlsx"                             # 需要读取的Excel文件名
sheet_name = 1                                      # 数据所在的工作簿序号（0，1，2）
worker_sand_efficiency = 5000                       # 单跨间一天冲砂面积
max_paint_float = 6000.0                            # 涂漆容许浮动范围
start_date = datetime(2024, 8, 19)  # 起始日期
work_data = [19, 20, 21, 22, 23, 25]                # 原方案跨间工作日
team_name = ["富纬", "皓源", "库众", "松宇", "旭丰"]    # 团队名称
wash_file_path = "冲砂方案.xlsx"                      # 冲砂新方案的Excel文件名
paint_file_path = "涂漆方案.xlsx"                     # 涂漆新方案的Excel文件名
insert_cell = 'A1'                                  # 新方案的起始插入位置
max_days = len(work_data)
workshop_num = len(team_name)
model_path = 'best_model.pth'
# ==============

class WorkshopEnv:
    def __init__(self, new_work_list, max_days, workshop_num=5, worker_sand_efficiency=5000):
        self.max_days = max_days
        self.original_work = new_work_list
        self.workshop_num = workshop_num
        self.max_daily = worker_sand_efficiency
        self.max_sand_area = 0
        self.current_day = 0
        self.all_works = []

        for daily_works in self.original_work:
            for work in daily_works:
                self.max_sand_area = max(self.max_sand_area, work.sand_area)
                self.all_works.append(work)
        self.reset()

    def reset(self):
        self.current_day = 0
        for work in self.all_works:
            work.delay_days = 0
            work.assigned = False
        self.unassigned = self._get_initial_unassigned()
        self.factories = [{'remaining': self.max_daily} for _ in range(self.workshop_num)]
        return self._get_state()

    def step(self, priorities):
        if isinstance(priorities, np.ndarray):
            priorities = priorities.tolist()
        priorities = priorities[:len(self.unassigned)]

        sorted_works = sorted(zip(self.unassigned, priorities),
                              key=lambda x: x[1], reverse=True)
        sorted_works = [w for w, _ in sorted_works]

        # 贪心分配
        for work in sorted_works:
            assigned = False
            for factory in self.factories:
                if factory['remaining'] >= work.sand_area:
                    factory['remaining'] -= work.sand_area
                    work.assigned = True
                    assigned = True
                    break
            if assigned:
                self.unassigned.remove(work)

        # 计算奖励
        utilization_penalty = 0.0
        for factory in self.factories:
            used = self.max_daily - factory['remaining']
            utilization_penalty -= abs(used - self.max_daily) / self.max_daily

        delay_penalty = -0.1 * sum(w.delay_days for w in self.unassigned)
        reward = utilization_penalty + delay_penalty

        # 更新延迟
        for work in self.unassigned:
            work.delay_days += 1

        # 进入下一天
        self.current_day += 1
        self._update_unassigned()
        self.factories = [{'remaining': self.max_daily} for _ in range(self.workshop_num)]

        # 检查终止
        done = all(w.assigned for w in self.all_works)
        return self._get_state(), reward, done, {}

    def _get_state(self):
        """获取每个工厂的剩余冲砂能力，以及遗留工件的状态（冲砂量，预计加工时间，延期天数）,共计 125 维 """
        factory_features = [f['remaining'] / self.max_daily for f in self.factories]
        work_features = []
        for w in self.unassigned:
            features = [w.sand_area / self.max_sand_area,
                        w.scheduled_day / self.max_days,
                        w.delay_days / self.max_days
                        ]
            work_features.extend(features)
        max_works = 40
        while len(work_features) <= 3 * max_works:
            work_features.append(0)
        work_features = work_features[:3 * max_works]

        return np.array(factory_features + work_features, dtype=np.float32)

    def _get_initial_unassigned(self):

        return [w for w in self.all_works if w.scheduled_day in [0, 1]]

    def _update_unassigned(self):
        """ 将明日的工件添加到纬分配的列表里 """
        if self.current_day < self.max_days - 1:
            new_day = self.current_day + 1
            new_works = [w for w in self.all_works
                         if w.scheduled_day == new_day and not w.assigned]
            self.unassigned.extend(new_works)


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Actor网络：输出每个工件的优先级均值
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        # Critic网络：输出状态价值
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)


class PPO:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99,
                 clip_epsilon=0.2, batch_size=64, update_epochs=4):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.batch_size = batch_size
        self.update_epochs = update_epochs
        self.memory = deque(maxlen=2048)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)  # 增加 batch 维度
        with torch.no_grad():
            mu, value = self.policy(state)
        mu = mu.squeeze(0)  # 形状: [action_dim]
        dist = torch.distributions.Normal(mu, torch.ones_like(mu))  # 连续动作分布
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()  # 对数概率求和
        return action.cpu().numpy(), value.item(), log_prob.item()

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        # 转换经验为张量
        states, actions, old_log_probs, returns, advantages = zip(*self.memory)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device).view(-1, 1)
        advantages = torch.FloatTensor(advantages).to(self.device).view(-1, 1)

        # 多次更新
        for _ in range(self.update_epochs):
            # 随机打乱数据
            indices = torch.randperm(len(states))

            for i in range(0, len(states), self.batch_size):
                batch_idx = indices[i:i + self.batch_size]
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]

                # 计算新策略
                mu, values = self.policy(batch_states)
                dist = torch.distributions.Normal(mu, torch.ones_like(mu))
                new_log_probs = dist.log_prob(batch_actions).sum(dim=1)
                entropy = dist.entropy().mean()

                # 计算比率和损失
                ratio = (new_log_probs - batch_old_log_probs).exp()
                surr1 = ratio * batch_advantages.squeeze()
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages.squeeze()
                actor_loss = -torch.min(surr1, surr2).mean()

                # 价值损失
                critic_loss = 0.5 * (batch_returns - values).pow(2).mean()

                # 总损失
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                # 更新参数
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

        self.memory.clear()


def train(new_work_list, max_days):
    best_reward = -float('inf')
    # 初始化环境和PPO
    env = WorkshopEnv(new_work_list, max_days)
    state_dim = len(env.reset())
    action_dim = 40  # 最大工件数（需与状态填充长度一致）
    ppo = PPO(state_dim, action_dim)

    max_episodes = 50000
    print_interval = 100

    for ep in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # 选择动作（生成优先级列表）
            priorities, value, log_prob = ppo.select_action(state)

            # 执行动作
            next_state, reward, done, _ = env.step(priorities)

            # 存储经验
            ppo.memory.append((state, priorities, log_prob, value, reward))

            state = next_state
            episode_reward += reward

        # 更新模型
        ppo.update()

        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save({
                'episode': ep,
                'reward': best_reward,
                'model_state': ppo.policy.state_dict(),
                'optimizer': ppo.optimizer.state_dict()
            }, 'best_model.pth')
            print(f" 发现新最佳模型（奖励 {best_reward:.2f}），已保存")

        # 打印训练进度
        if ep % print_interval == 0:
            print(f"Episode {ep}, Reward: {episode_reward:.2f}")

    torch.save({
        'policy_state_dict': ppo.policy.state_dict(),
        'optimizer_state_dict': ppo.optimizer.state_dict(),
    }, 'final_model.pth')
    print("模型已保存至 final_model.pth")


def load_trained_model(model_path, env):
    state_dim = len(env.reset())
    action_dim = 40

    # 重建网络结构
    model = ActorCritic(state_dim, action_dim)

    # 加载参数
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    return model


class ScheduleGenerator:
    def __init__(self, model, env):
        self.model = model
        self.env = env
        # 三维列表结构：schedule[day][factory][work_list]
        self.schedule = []

    def generate(self):
        """生成三维调度方案"""
        state = self.env.reset()
        done = False

        while not done:
            current_day = len(self.schedule)  # 当前天数
            # 初始化当天的车间分配列表
            daily_schedule = [[] for _ in range(self.env.workshop_num)]

            # 模型推理
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                priorities, _ = self.model(state_tensor)
                priorities = priorities.squeeze(0).numpy()

            # 执行分配并记录
            self._assign_works(priorities, daily_schedule, current_day)

            # 保存当天分配结果
            self.schedule.append(daily_schedule)

            # 进入下一天
            next_state, _, done, _ = self.env.step(priorities)
            state = next_state

        return self.schedule

    def _assign_works(self, priorities, daily_schedule, current_day):
        """执行单日分配并填充三维列表"""
        # 获取有效可分配工件（有足够容量的）
        valid_works = [
            (w, idx) for idx, w in enumerate(self.env.unassigned)
            if any(f['remaining'] >= w.sand_area for f in self.env.factories)
        ]

        # 按优先级排序（使用工件的原始索引）
        sorted_indices = np.argsort(-priorities[:len(self.env.unassigned)])
        sorted_works = [self.env.unassigned[i] for i in sorted_indices
                        if i < len(valid_works)]  # 防止索引越界

        # 贪心分配
        for work in sorted_works:
            for factory_id, factory in enumerate(self.env.factories):
                if factory['remaining'] >= work.sand_area:
                    # 记录到对应车间
                    daily_schedule[factory_id].append(work)

                    # 更新工厂状态
                    factory['remaining'] -= work.sand_area
                    work.assigned = True
                    self.env.unassigned.remove(work)
                    break

    def print_schedule(self):
        """打印三维调度方案"""
        for day_idx, day_schedule in enumerate(self.schedule):
            print(f"\n=== Day {day_idx} ===")
            for factory_idx, works in enumerate(day_schedule):
                print(f"Factory {factory_idx} (剩余容量: {self.env.factories[factory_idx]['remaining']}):")
                for work in works:
                    print(f"  - 工件: {work['sand_area']}m², "
                          f"预定日: {work['scheduled_day']}, "
                          f"延迟: {work['delay_days']}天")


work_list, expect_paint = read_excel_range(file_path, sheet_name, workshop_num, max_paint_float)
new_work_list = work_sort(work_list, work_data)
# train(new_work_list, max_days)

env = WorkshopEnv(new_work_list, max_days)
model = load_trained_model("best_model.pth", env)
generator = ScheduleGenerator(model, env)
schedule_result = generator.generate()
show_info(schedule_result, wash_file_path, insert_cell, start_date, team_name)