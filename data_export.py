import pandas as pd


def agents_to_dataframe(agent_list):
    data = []
    for agent in agent_list:
        row = {
            "unique_id": agent.unique_id,
            "gender": getattr(agent, "gender", None),
            "age": getattr(agent, "age", None),
            "height": getattr(agent, "height", None),
            "weight": getattr(agent, "weight", None),
            "muscle_mass": getattr(agent, "muscle_mass", None),
            "wealth": getattr(agent, "wealth", None),
            "wage": getattr(agent, "wage", None),
            "nationality": getattr(agent, "nationality", None),
            "language_primary": getattr(agent, "language_primary", None),
            "language_secondary": getattr(agent, "language_secondary", None),
            "criminal_status": getattr(agent.criminal_status, "name", None),
            "incarcerated": getattr(agent, "incarcerated", False),
            "desisted": getattr(agent, "desisted", False),
        }
        data.append(row)
    return pd.DataFrame(data)
