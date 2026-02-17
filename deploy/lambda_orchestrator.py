"""Lambda Labs GPU orchestrator for tournament training.

Manages the lifecycle of Lambda Labs GPU instances for running
distributed tournament training. Handles instance provisioning,
setup, training execution, result collection, and teardown.

Requires:
    - LL_API_KEY environment variable set with your Lambda Labs API key
    - SSH key pair for instance access

Usage:
    # Launch a tournament training run on Lambda Labs
    python -m deploy.lambda_orchestrator \
        --instance-type gpu_1x_a10 \
        --generations 50 \
        --population-size 12 \
        --episodes-per-gen 1000

    # Or with a warm-start checkpoint
    python -m deploy.lambda_orchestrator \
        --instance-type gpu_1x_a100_sxm4 \
        --warmstart-checkpoint data/self_play/self_play_best.pt \
        --generations 100
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Lambda Labs API client
# ---------------------------------------------------------------------------

LAMBDA_API_BASE = "https://cloud.lambdalabs.com/api/v1"


def _api_key() -> str:
    key = os.environ.get("LL_API_KEY")
    if not key:
        raise RuntimeError(
            "LL_API_KEY environment variable not set. "
            "Get your API key from https://cloud.lambdalabs.com/api-keys"
        )
    return key


def _api_call(method: str, endpoint: str, data: Optional[dict] = None) -> dict:
    """Make an API call to Lambda Labs REST API using curl."""
    url = f"{LAMBDA_API_BASE}/{endpoint}"
    cmd = [
        "curl", "-s", "-X", method, url,
        "-H", f"Authorization: Bearer {_api_key()}",
        "-H", "Content-Type: application/json",
    ]
    if data:
        cmd += ["-d", json.dumps(data)]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"API call failed: {result.stderr}")
    return json.loads(result.stdout)


def list_instance_types() -> Dict[str, Any]:
    """List available Lambda Labs instance types and pricing."""
    return _api_call("GET", "instance-types")


def list_running_instances() -> List[Dict]:
    """List currently running instances."""
    resp = _api_call("GET", "instances")
    return resp.get("data", [])


def resolve_region(instance_type: str) -> str:
    """Pick the first region with available capacity for the given instance type."""
    data = list_instance_types().get("data", {})
    info = data.get(instance_type, {})
    regions = info.get("regions_with_capacity_available", [])
    if not regions:
        raise RuntimeError(
            f"No regions with capacity for {instance_type}. "
            "Check availability at https://cloud.lambdalabs.com/instances"
        )
    region = regions[0]
    return region.get("name", region) if isinstance(region, dict) else region


def launch_instance(
    instance_type: str,
    region: Optional[str] = None,
    ssh_key_name: Optional[str] = None,
    name: str = "golf-tournament",
) -> Dict[str, Any]:
    """Launch a new GPU instance.

    If region is None, auto-selects the first region with capacity.
    """
    if not region:
        region = resolve_region(instance_type)
        print(f"  Auto-selected region: {region}")

    payload: Dict[str, Any] = {
        "region_name": region,
        "instance_type_name": instance_type,
        "quantity": 1,
        "name": name,
    }
    if ssh_key_name:
        payload["ssh_key_names"] = [ssh_key_name]

    resp = _api_call("POST", "instance-operations/launch", payload)
    if "error" in resp:
        raise RuntimeError(f"Launch failed: {resp['error']}")
    return resp


def terminate_instance(instance_id: str, retries: int = 4, verify: bool = True) -> Dict[str, Any]:
    """Terminate a running instance with retries and optional verification."""
    last_error = None
    for attempt in range(retries):
        try:
            resp = _api_call("POST", "instance-operations/terminate", {
                "instance_ids": [instance_id],
            })
            if "error" not in resp:
                break
            last_error = resp.get("error")
            print(f"  Terminate attempt {attempt + 1} got error: {last_error}")
        except Exception as e:
            last_error = e
            print(f"  Terminate attempt {attempt + 1} failed: {e}")

        if attempt < retries - 1:
            backoff = 2 ** (attempt + 1)
            print(f"  Retrying in {backoff}s...")
            time.sleep(backoff)
    else:
        raise RuntimeError(
            f"Failed to terminate instance {instance_id} after {retries} attempts: {last_error}"
        )

    if verify:
        time.sleep(10)
        for _ in range(3):
            try:
                info = get_instance(instance_id)
                status = info.get("status", "unknown")
                if status in ("terminated", "unknown"):
                    print(f"  Instance confirmed terminated (status={status})")
                    return resp
                print(f"  Instance still {status}, waiting...")
                time.sleep(10)
            except Exception:
                # Instance not found = terminated
                return resp
        print(f"  WARNING: Instance {instance_id} may still be running")

    return resp


def get_instance(instance_id: str) -> Dict[str, Any]:
    """Get instance details."""
    resp = _api_call("GET", f"instances/{instance_id}")
    return resp.get("data", {})


# ---------------------------------------------------------------------------
# SSH helpers
# ---------------------------------------------------------------------------

def _ssh_cmd(ip: str, command: str, key_path: str, timeout: int = 300) -> subprocess.CompletedProcess:
    """Execute a command on a remote instance via SSH."""
    return subprocess.run(
        [
            "ssh", "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", f"ConnectTimeout=10",
            "-i", key_path,
            f"ubuntu@{ip}",
            command,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _scp_to(ip: str, local_path: str, remote_path: str, key_path: str) -> None:
    """Copy a file to the remote instance."""
    subprocess.run(
        [
            "scp", "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-i", key_path,
            local_path,
            f"ubuntu@{ip}:{remote_path}",
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )


def _scp_from(ip: str, remote_path: str, local_path: str, key_path: str) -> None:
    """Copy a file from the remote instance."""
    subprocess.run(
        [
            "scp", "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-r",
            "-i", key_path,
            f"ubuntu@{ip}:{remote_path}",
            local_path,
        ],
        check=True,
        capture_output=True,
        timeout=300,
    )


# ---------------------------------------------------------------------------
# Instance lifecycle
# ---------------------------------------------------------------------------

def wait_for_instance(instance_id: str, timeout: int = 600) -> str:
    """Wait for instance to become active. Returns IP address."""
    start = time.time()
    while time.time() - start < timeout:
        info = get_instance(instance_id)
        status = info.get("status", "unknown")
        ip = info.get("ip")

        if status == "active" and ip:
            # Verify SSH connectivity
            time.sleep(5)
            return ip
        elif status in ("terminated", "error"):
            raise RuntimeError(f"Instance {instance_id} entered state: {status}")

        print(f"  Instance status: {status} (waiting...)")
        time.sleep(15)

    raise TimeoutError(f"Instance {instance_id} did not become active within {timeout}s")


def setup_instance(ip: str, key_path: str) -> None:
    """Install dependencies and clone the repo on the instance."""
    setup_script = """
set -e

# Install system dependencies
sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip git > /dev/null 2>&1

# Clone repo (or pull if exists)
if [ -d ~/golf ]; then
    cd ~/golf && git pull
else
    git clone https://github.com/$(git config --get remote.origin.url 2>/dev/null || echo 'vgainullin/golf') ~/golf 2>/dev/null || true
fi

# Install Python dependencies
cd ~/golf
pip install -q torch numpy pandas 2>/dev/null
pip install -q -e . 2>/dev/null

echo "SETUP_COMPLETE"
"""
    print("  Setting up instance...")
    result = _ssh_cmd(ip, setup_script, key_path, timeout=600)
    if "SETUP_COMPLETE" not in result.stdout:
        print(f"  Setup output: {result.stdout[-500:]}")
        print(f"  Setup errors: {result.stderr[-500:]}")
        raise RuntimeError("Instance setup failed")
    print("  Instance ready")


# ---------------------------------------------------------------------------
# Training execution
# ---------------------------------------------------------------------------

@dataclass
class TrainingJob:
    instance_type: str = "gpu_1x_a10"
    region: Optional[str] = None
    ssh_key_name: Optional[str] = None
    ssh_key_path: str = "~/.ssh/id_rsa"

    # Tournament config
    population_size: int = 12
    generations: int = 50
    episodes_per_gen: int = 1000
    matches_per_pair: int = 8
    batch_size: int = 512
    warmstart_checkpoint: Optional[str] = None

    # Output
    output_dir: str = "data/tournament_lambda"
    repo_url: Optional[str] = None
    repo_branch: str = "main"


def run_training_job(job: TrainingJob) -> Dict[str, Any]:
    """Launch a Lambda Labs instance, run tournament training, collect results."""
    key_path = os.path.expanduser(job.ssh_key_path)
    if not os.path.exists(key_path):
        raise FileNotFoundError(f"SSH key not found: {key_path}")

    instance_id = None
    ip = None

    try:
        # 1. Launch instance
        print(f"Launching {job.instance_type} instance...")
        resp = launch_instance(
            instance_type=job.instance_type,
            region=job.region,
            ssh_key_name=job.ssh_key_name,
            name="golf-tournament",
        )
        instance_ids = resp.get("data", {}).get("instance_ids", [])
        if not instance_ids:
            raise RuntimeError(f"No instance IDs returned: {resp}")
        instance_id = instance_ids[0]
        print(f"  Instance ID: {instance_id}")

        # 2. Wait for active
        print("Waiting for instance to start...")
        ip = wait_for_instance(instance_id)
        print(f"  Instance IP: {ip}")

        # 3. Setup environment
        setup_instance(ip, key_path)

        # 4. Upload warm-start checkpoint if specified
        if job.warmstart_checkpoint:
            print(f"Uploading checkpoint: {job.warmstart_checkpoint}")
            _scp_to(ip, job.warmstart_checkpoint, "~/golf/warmstart.pt", key_path)

        # 5. Run tournament training
        warmstart_flag = ""
        if job.warmstart_checkpoint:
            warmstart_flag = "--warmstart-checkpoint ~/golf/warmstart.pt"

        train_cmd = f"""
cd ~/golf && python -m src.tournament \
    --population-size {job.population_size} \
    --generations {job.generations} \
    --episodes-per-gen {job.episodes_per_gen} \
    --matches-per-pair {job.matches_per_pair} \
    --batch-size {job.batch_size} \
    --output-dir {job.output_dir} \
    --device auto \
    {warmstart_flag} \
    2>&1 | tee ~/golf/training.log
"""
        print(f"Starting tournament training ({job.generations} generations)...")
        print(f"  Population: {job.population_size} agents")
        print(f"  Episodes/gen: {job.episodes_per_gen}")

        result = _ssh_cmd(ip, train_cmd, key_path, timeout=86400)  # 24h timeout
        print(f"Training output (last 1000 chars):\n{result.stdout[-1000:]}")

        # 6. Download results
        local_output = Path(job.output_dir)
        local_output.mkdir(parents=True, exist_ok=True)
        print("Downloading results...")
        _scp_from(ip, f"~/golf/{job.output_dir}/", str(local_output), key_path)
        _scp_from(ip, "~/golf/training.log", str(local_output / "training.log"), key_path)

        print(f"Results downloaded to: {local_output}")

        return {
            "status": "success",
            "instance_id": instance_id,
            "instance_type": job.instance_type,
            "output_dir": str(local_output),
        }

    except Exception as e:
        print(f"Error: {e}")
        return {"status": "error", "error": str(e), "instance_id": instance_id}

    finally:
        # 7. Terminate instance (with retries and verification)
        if instance_id:
            print(f"Terminating instance {instance_id}...")
            try:
                terminate_instance(instance_id, retries=4, verify=True)
                print("  Instance terminated")
            except Exception as e:
                print(f"  CRITICAL: Failed to terminate instance {instance_id}: {e}")
                print(f"  Manually terminate at https://cloud.lambdalabs.com/instances")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Lambda Labs tournament training orchestrator")

    # Instance
    p.add_argument("--instance-type", default="gpu_1x_a10",
                    help="Lambda Labs instance type (gpu_1x_a10, gpu_1x_a100_sxm4, etc.)")
    p.add_argument("--region", default=None, help="Preferred region")
    p.add_argument("--ssh-key-name", default=None, help="Lambda Labs SSH key name")
    p.add_argument("--ssh-key-path", default="~/.ssh/id_rsa", help="Local SSH private key path")

    # Training
    p.add_argument("--population-size", type=int, default=12)
    p.add_argument("--generations", type=int, default=50)
    p.add_argument("--episodes-per-gen", type=int, default=1000)
    p.add_argument("--matches-per-pair", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--warmstart-checkpoint", default=None)

    # Output
    p.add_argument("--output-dir", default="data/tournament_lambda")

    # Utilities
    p.add_argument("--list-instances", action="store_true", help="List running instances and exit")
    p.add_argument("--list-types", action="store_true", help="List available instance types and exit")
    p.add_argument("--terminate", type=str, default=None, help="Terminate an instance by ID")

    args = p.parse_args(argv)
    return args


def main(argv=None):
    args = parse_args(argv)

    if args.list_types:
        types = list_instance_types()
        print(json.dumps(types, indent=2))
        return

    if args.list_instances:
        instances = list_running_instances()
        if not instances:
            print("No running instances")
        for inst in instances:
            print(f"  {inst.get('id', '?')}: {inst.get('instance_type', {}).get('name', '?')} "
                  f"status={inst.get('status', '?')} ip={inst.get('ip', 'N/A')}")
        return

    if args.terminate:
        print(f"Terminating instance {args.terminate}...")
        terminate_instance(args.terminate)
        print("Done")
        return

    job = TrainingJob(
        instance_type=args.instance_type,
        region=args.region,
        ssh_key_name=args.ssh_key_name,
        ssh_key_path=args.ssh_key_path,
        population_size=args.population_size,
        generations=args.generations,
        episodes_per_gen=args.episodes_per_gen,
        matches_per_pair=args.matches_per_pair,
        batch_size=args.batch_size,
        warmstart_checkpoint=args.warmstart_checkpoint,
        output_dir=args.output_dir,
    )

    result = run_training_job(job)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
