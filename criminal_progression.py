from enums import CriminalStatus

S_MAX = 6  # Maximum min sentence (for normalization)


def update_status(agent, s_k, w_k):
    """
    Increment criminal score and update criminal status.
    """
    # Initialize if missing
    if not hasattr(agent, "criminal_score"):
        agent.criminal_score = 0

    # Score increment
    agent.criminal_score += w_k * (s_k / S_MAX) * 100

    score = agent.criminal_score
    degree = len(agent.associates)
    has_connected_criminal = any(
        getattr(agent.model.schedule.agents[aid], "criminal_score", 0) >= 30
        for aid in agent.associates
    )

    # Update status
    if score >= 300 and degree >= 6:
        # Check centrality (placeholder: use degree for now)
        centrality = degree
        max_centrality = max(len(a.associates) for a in agent.model.schedule.agents)
        if centrality == max_centrality:
            agent.criminal_status = CriminalStatus.VORY
    elif score >= 60 and degree >= 2 and has_connected_criminal:
        agent.criminal_status = CriminalStatus.ORGANIZED_CRIMINAL
    elif score >= 30:
        agent.criminal_status = CriminalStatus.PETTY_CRIMINAL
