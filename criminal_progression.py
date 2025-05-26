from enums import CriminalStatus

S_MAX = 6  # Maximum min sentence, max gain and normalizer


def update_status(agent, s_k, w_k):
    """
    Increment criminal score and update criminal status.
    """
    # Initialize a criminal score if missing
    if not hasattr(agent, "criminal_score"):
        agent.criminal_score = 0

    # Score increment logic
    agent.criminal_score += w_k * (s_k / S_MAX) * 100

    score = agent.criminal_score
    degree = len(agent.associates)
    has_connected_criminal = any(
        next(
            (a for a in agent.model.agent_list if a.unique_id == aid), None
        ).criminal_score
        >= 30
        for aid in agent.associates
        if next((a for a in agent.model.agent_list if a.unique_id == aid), None)
        is not None
    )

    # Updates criminal status
    if score >= 300 and degree >= 6:
        # Need a check for centrality (placeholder: use degree for now)
        centrality = degree
        max_centrality = max(len(a.associates) for a in agent.model.agent_list)
        if centrality == max_centrality:
            agent.criminal_status = CriminalStatus.VORY
    elif score >= 60 and degree >= 2 and has_connected_criminal:
        agent.criminal_status = CriminalStatus.ORGANIZED_CRIMINAL
    elif score >= 30:
        agent.criminal_status = CriminalStatus.PETTY_CRIMINAL
