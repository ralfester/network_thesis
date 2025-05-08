RACKET_TRIBUTE = 50

def apply_racketeering(racketeer, victim):
    """
    Executes tribute logic. Escalates on default.
    """
    if not hasattr(victim, "racket_defaults"):
        victim.racket_defaults = 0

    if victim.wealth >= RACKET_TRIBUTE:
        victim.wealth -= RACKET_TRIBUTE
        racketeer.wealth += RACKET_TRIBUTE
        return f"{victim.unique_id} paid tribute to {racketeer.unique_id}"
    else:
        victim.racket_defaults += 1

        if victim.racket_defaults == 1:
            # escalate to robbery
            loot = min(0.2 * victim.wealth, racketeer.wealth)
            victim.wealth -= loot
            racketeer.wealth += loot
            return f"Robbery! {racketeer.unique_id} looted {loot:.2f} from {victim.unique_id}"
        elif victim.racket_defaults >= 2:
            # escalate to murder
            victim.alive = False
            return f"Murder! {racketeer.unique_id} eliminated {victim.unique_id}"
