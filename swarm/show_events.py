"""Print a one-line-per-event view of `codex exec --json` output (stdin) and append it to the file given as argv[1]."""

import json
import sys
import time

log = open(sys.argv[1], "a")
for line in sys.stdin:
    try:
        event = json.loads(line)
    except ValueError:
        continue
    item = event.get("item", {})
    kind = item.get("type") or event.get("type")
    if kind == "agent_message":
        text = "SAY " + item.get("text", "").replace("\n", " ")[:300]
    elif kind == "command_execution" and event.get("type") == "item.started":
        text = "RUN " + item.get("command", "").replace("\n", " ")[:200]
    elif kind == "turn.completed":
        text = "TURN DONE " + json.dumps(event.get("usage", {}))
    else:
        continue
    stamped = time.strftime("%H:%M:%S ") + text
    print(stamped, flush=True)
    log.write(stamped + "\n")
    log.flush()
