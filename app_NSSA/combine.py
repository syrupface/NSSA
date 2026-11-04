""" Arithmetic of two channels (e.g. friction from the forward and backward lateral force). The computations are NSFopen's (NSFopen.process.combine, NSFopen >= 2.3.0) and are re-exported here; this
module only adds what the app shows: the default name of a combined channel. """

from NSFopen.process.combine import *            # noqa: F401,F403  (the functions, classes and constants)


def default_name(a_key: str, b_key: str, operation: str) -> str:
    """ A name for the new channel, e.g. 'Friction force ((F - B) / 2)' for the forward and backward of one channel,
        else e.g.
        'Phase (Forward) - Amplitude (Forward)'. """
    a_name, a_dir = (a_key[:-1].rsplit(" (", 1) + [""])[:2] if a_key.endswith(")") else (a_key, "")
    b_name, b_dir = (b_key[:-1].rsplit(" (", 1) + [""])[:2] if b_key.endswith(")") else (b_key, "")
    expression = OPERATIONS[operation]
    if a_name == b_name and a_dir != b_dir:
        short = {"Forward": "F", "Backward": "B"}
        a_tag, b_tag = short.get(a_dir, a_dir or "A"), short.get(b_dir, b_dir or "B")
        text = expression.replace("A", "\0").replace("B", "\1").replace("\0", a_tag).replace("\1", b_tag)
        return f"{a_name} ({text})"
    return expression.replace("A", "\0").replace("B", "\1").replace("\0", a_key).replace("\1", b_key)
