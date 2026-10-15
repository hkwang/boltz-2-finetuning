"""Laptop-side check that the ALCF IRI path the swarm needs works: auth, read, list, upload (small and ~1 MB patch-sized), and a job
submission that runs in reservation R7728155 and writes its output to /eagle.

Run from the root of a gm_hackathon_agentic_ai clone (its venv active, `python scripts/auth/globus_auth.py authenticate --facility alcf`
done): python /path/to/boltz2-finetuning/swarm/iri_probe.py [--no-job]
It imports that repo's mcp/hpc/alcf_server.py, so it exercises the same code the agents' MCP tools use. Prints PASS/FAIL per step.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = "/eagle/GenesisHackathonOct26/hkwang/boltz2-swarm/iri-test"
POLARIS = "55c1c993-1124-47f9-b823-514ba3849a9a"


def report(step: str, ok: bool, detail: str) -> bool:
    print(f"{'PASS' if ok else 'FAIL'}  {step}: {detail[:300]}", flush=True)
    return ok


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-job", action="store_true", help="skip the job submission step")
    args = parser.parse_args()
    sys.path.insert(0, str(Path.cwd() / "mcp" / "hpc"))
    sys.path.insert(0, str(Path.cwd()))
    try:
        import alcf_server as server
    except ImportError as exc:
        print(f"run this from the gm_hackathon_agentic_ai repo root with its venv active ({exc})")
        return 2

    stamp = time.strftime("%Y%m%dT%H%M%S")
    results = []

    text = await server.list_projects()
    results.append(report("auth (list_projects)", "GenesisHackathonOct26" in text, text.replace("\n", " ")))

    text = await server.read_file(f"{ROOT}/probe.txt", mode="view")
    results.append(report("read_file /eagle", "hello from polaris" in text, text.strip()))

    small = f"swarm probe {stamp}\n"
    text = await server.upload_file(f"{ROOT}/uploads/small_{stamp}.txt", small)
    results.append(report("upload_file small", "rror" not in text, text.replace("\n", " ")))
    text = await server.read_file(f"{ROOT}/uploads/small_{stamp}.txt", mode="view")
    results.append(report("upload round trip", small.strip() in text, text.strip()))

    patch = "".join(f"+line {index:06d} " + "x" * 50 + "\n" for index in range(16000))
    started = time.time()
    text = await server.upload_file(f"{ROOT}/uploads/patch_{stamp}.patch", patch)
    results.append(report(f"upload_file {len(patch) / 1e6:.1f} MB ({time.time() - started:.0f} s)", "rror" not in text, text.replace("\n", " ")))

    text = await server.list_directory(f"{ROOT}/uploads")
    results.append(report("list_directory", f"small_{stamp}.txt" in text and f"patch_{stamp}.patch" in text, f"{text.count(stamp)} uploads seen"))

    if not args.no_job:
        command = (f"echo node $(hostname) job $PBS_JOBID; wc -c {ROOT}/uploads/patch_{stamp}.patch; "
                   f"date -u > {ROOT}/jobs/ran_{stamp}.txt; which python3; nvidia-smi -L | head -n 1")
        text = await server.submit_job(
            resource_id=POLARIS, executable="/bin/bash", arguments=f"-c,{command}", name=f"iri-probe-{stamp}", queue_name="R7728155",
            account="GenesisHackathonOct26", duration=300, node_count=1, stdout_path=f"{ROOT}/jobs/probe_{stamp}.out",
            stderr_path=f"{ROOT}/jobs/probe_{stamp}.err", custom_attributes=json.dumps({"filesystems": "home:eagle"}),
        )
        job_id = None
        try:
            job_id = json.loads(text).get("job_id")
        except json.JSONDecodeError:
            pass
        results.append(report("submit_job (R7728155, output on /eagle)", bool(job_id), text.replace("\n", " ")))
        if job_id:
            for _ in range(60):
                status = await server.get_job_status(POLARIS, job_id)
                state = json.loads(status).get("status", {}).get("state", "") if status.startswith("{") else status
                if str(state).lower() in ("completed", "failed", "canceled"):
                    break
                await asyncio.sleep(10)
            results.append(report("job finished", str(state).lower() == "completed", f"state {state}"))
            text = await server.read_file(f"{ROOT}/jobs/probe_{stamp}.out", mode="view")
            results.append(report("job stdout on /eagle", "node x" in text, text.strip().replace("\n", " | ")))
            text = await server.read_file(f"{ROOT}/jobs/ran_{stamp}.txt", mode="view")
            results.append(report("job wrote to /eagle", "UTC" in text or "202" in text, text.strip()))

    print(f"\n{sum(results)}/{len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
