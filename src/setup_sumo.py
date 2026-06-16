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

try:
    from src.params import USE_LIBSUMO as _USE_LIBSUMO
except Exception:
    _USE_LIBSUMO = False

# Must be set before traci/sumolib is imported (env.py reads this at module load).
# Exception: libsumo has no GUI support — if the caller requested --show 1, fall
# back to TraCI (a separate sumo-gui process) so the visualisation works.
_gui_requested = (
    "--show" in sys.argv
    and sys.argv.index("--show") + 1 < len(sys.argv)
    and sys.argv[sys.argv.index("--show") + 1] == "1"
)
if _USE_LIBSUMO and not _gui_requested:
    os.environ["LIBSUMO_AS_TRACI"] = "1"

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
