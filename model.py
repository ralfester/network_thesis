import mesa

from ABM import PersonAgent, setup_economy
from law_enforcement import step_incarceration
from economics import biased_wealth_transfer
from data_export import agents_to_dataframe


class CrimeSocietyModel(mesa.Model):
    def __init__(self, num_agents=100, wage_mode="A", r_w=0.05):
        self.num_agents = num_agents
        self.schedule = mesa.time.RandomActivation(self)
        self.wage_mode = wage_mode
        self.r_w = r_w
        self.current_step = 0
        self.snapshots = []

        #  Setup economy
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
            self.schedule.add(agent)

        #  Initial data snapshot
        self.agent_dataframe = agents_to_dataframe(self.schedule.agents)
        self.snapshots.append(self.agent_dataframe.assign(step=self.current_step))

    def step(self):
        """
        Executes one time step in the simulation.
        """
        self.current_step += 1
        agents = list(self.schedule.agents)

        # time-step wage income
        for agent in agents:
            if not agent.incarcerated:
                agent.wealth += agent.wage

        #  Wealth exchange
        for _ in range(self.num_agents // 2):
            a, b = self.random.sample(agents, 2)
            biased_wealth_transfer(a, b)

        # Criminal activity
        for agent in agents:
            if not agent.incarcerated and not agent.desisted:
                agent.step_criminal_activity(agents)

        # Incarceration time step
        for agent in agents:
            step_incarceration(agent)

        # Log statistics & snapshots
        self.log_statistics()

    def log_statistics(self):
        """
        Prints basic stats and appends agent snapshots.
        """
        wealths = [a.wealth for a in self.schedule.agents if not a.incarcerated]
        num_criminals = sum(
            1
            for a in self.schedule.agents
            if a.criminal_status != a.criminal_status.NON_CRIMINAL
        )
        print(
            f"Step {self.current_step} — Mean Wealth: {sum(wealths)/len(wealths):.2f}, Criminals: {num_criminals}"
        )

        df = agents_to_dataframe(self.schedule.agents).assign(step=self.current_step)
        self.snapshots.append(df)
