"""Helper to configure SUMO_HOME and add SUMO tools to sys.path.

Importing this module sets environment variable SUMO_HOME from either the
environment or src.params.SUMO_HOME and appends the SUMO tools dir to sys.path.
This keeps SUMO setup centralized and allows imports to remain at the top of files.
"""

import os
import sys

try:
    from src.params import SUMO_HOME as PARAM_SUMO_HOME
except Exception:
    PARAM_SUMO_HOME = None

sumo_home = os.environ.get("SUMO_HOME") or PARAM_SUMO_HOME
if sumo_home:
    tools = os.path.join(sumo_home, "tools")
    if tools not in sys.path:
        sys.path.append(tools)
    os.environ["SUMO_HOME"] = sumo_home
else:
    # Do not raise here; experiments may want to exit cleanly. Some callers expect sys.exit.
    # We simply leave SUMO_HOME unset and let consumers handle it.
    pass
