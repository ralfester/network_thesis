import mesa
import random
from ABM import PersonAgent, setup_economy
from law_enforcement import step_incarceration
from economics import biased_wealth_transfer
from data_export import agents_to_dataframe


class CrimeSocietyModel(mesa.Model):
    def __init__(self, num_agents=100, wage_mode="A", r_w=0.05):
        self.num_agents = num_agents
        self.wage_mode = wage_mode
        self.r_w = r_w
        self.current_step = 0
        self.snapshots = []

        self.agent_list = []  # 🔧 Replaces mesa.time.RandomActivation

        # Setup economy
        initial_wealths, wages = setup_economy(num_agents, mode=wage_mode, r_w=r_w)

        # Create agents
        for i in range(self.num_agents):
            agent = PersonAgent(i, self)
            agent.wealth = initial_wealths[i]
            agent.wage = wages[i]
            agent.incarcerated = False
            agent.desisted = False
            agent.immune = False
            agent.criminal_score = 0
            agent.num_racket_victims = 0
            self.agent_list.append(agent)

        # Initial data snapshot
        self.agent_dataframe = agents_to_dataframe(self.agent_list)
        self.snapshots.append(self.agent_dataframe.assign(step=self.current_step))

    def step(self):
        self.current_step += 1

        agents = self.agent_list.copy()
        random.shuffle(agents)  # 🔁 Custom randomized activation

        # Wage income
        for agent in agents:
            if not agent.incarcerated:
                agent.wealth += agent.wage

        # Wealth exchange
        for _ in range(self.num_agents // 2):
            a, b = random.sample(agents, 2)
            biased_wealth_transfer(a, b)

        # Criminal activity
        for agent in agents:
            if not agent.incarcerated and not agent.desisted:
                agent.step_criminal_activity(agents)

        # Incarceration update
        for agent in agents:
            step_incarceration(agent)

        self.log_statistics()

    def log_statistics(self):
        wealths = [a.wealth for a in self.agent_list if not a.incarcerated]
        num_criminals = sum(
            1
            for a in self.agent_list
            if a.criminal_status != a.criminal_status.NON_CRIMINAL
        )
        print(
            f"Step {self.current_step} — Mean Wealth: {sum(wealths)/len(wealths):.2f}, Criminals: {num_criminals}"
        )

        df = agents_to_dataframe(self.agent_list).assign(step=self.current_step)
        self.snapshots.append(df)
