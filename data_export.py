import pandas as pd


def agents_to_dataframe(agent_list):
    """
    Convert a list of PersonAgent objects into a pandas DataFrame.

    Parameters:
        agent_list (list): List of PersonAgent instances.

    Returns:
        pd.DataFrame: DataFrame with agent attributes.
    """
    data = []
    for agent in agent_list:
        data.append({
            "id": agent.unique_id,
            "height": agent.height,
            "weight": agent.weight,
            "wealth": getattr(agent, "wealth", None),
            "wage": getattr(agent, "wage", None),
            "nationality": getattr(agent, "nationality", None),
            "lang_primary": getattr(agent, "language_primary", None),
            "lang_secondary": getattr(agent, "language_secondary", None),
        })

    return pd.DataFrame(data)
