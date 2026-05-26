import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path


def app_root():
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        if (exe_dir.parent / "main.py").exists():
            return exe_dir.parent
        return exe_dir
    return Path(__file__).resolve().parent


def find_python(root):
    env_python = os.getenv("DIARIZATION_PYTHON")
    candidates = []
    if env_python:
        candidates.append(Path(env_python))
    candidates.extend(
        [
            root / ".venv" / "Scripts" / "python.exe",
            root / "venv" / "Scripts" / "python.exe",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return sys.executable if not getattr(sys, "frozen", False) else "python"


def service_url(host=None, port=None):
    host = host or os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    port = port or os.getenv("GRADIO_SERVER_PORT", "3002")
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"
    return f"http://{host}:{port}"


def read_env_file(path):
    values = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        values[key.strip()] = value
    return values


def ensure_docker_env(root):
    env_path = root / "docker.env"
    example_path = root / "docker.env.example"
    if not env_path.exists():
        if not example_path.exists():
            raise FileNotFoundError("docker.env.example was not found")
        shutil.copyfile(example_path, env_path)
        print("Created docker.env from docker.env.example.")
        print("Edit docker.env if you need another port, model, or Hugging Face token.")
    return env_path


def docker_service_url(root):
    env_values = read_env_file(root / "docker.env")
    port = os.getenv("APP_PORT") or env_values.get("APP_PORT") or "3002"
    return service_url(host="127.0.0.1", port=port)


def is_service_up(url):
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return 200 <= response.status < 500
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def wait_for_service(url, timeout_seconds):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if is_service_up(url):
            return True
        time.sleep(1)
    return False


def open_logs(root):
    log_dir = Path(os.getenv("LOG_DIR", str(root / "logs")))
    log_dir.mkdir(exist_ok=True, parents=True)
    stdout_path = log_dir / "launcher-service.out.log"
    stderr_path = log_dir / "launcher-service.err.log"
    return (
        stdout_path.open("a", encoding="utf-8"),
        stderr_path.open("a", encoding="utf-8"),
        stdout_path,
        stderr_path,
    )


def start_service(root, python_exe):
    stdout_file, stderr_file, stdout_path, stderr_path = open_logs(root)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("GRADIO_SERVER_NAME", "127.0.0.1")
    env.setdefault("GRADIO_SERVER_PORT", "3002")
    env.setdefault("LOG_DIR", str(root / "logs"))

    process = subprocess.Popen(
        [python_exe, "-u", "main.py"],
        cwd=str(root),
        env=env,
        stdout=stdout_file,
        stderr=stderr_file,
    )
    return process, stdout_file, stderr_file, stdout_path, stderr_path


def ensure_docker_available():
    if not shutil.which("docker"):
        print("Docker was not found in PATH.")
        print("Install Docker Desktop and enable WSL2/GPU support before running this launcher.")
        return False
    result = subprocess.run(
        ["docker", "info"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        print("Docker Desktop is not running or is not accessible.")
        if result.stderr:
            print(result.stderr.strip())
        return False
    return True


def run_docker_compose(root, env_file, compose_args):
    command = ["docker", "compose", "--env-file", str(env_file), *compose_args]
    print("Running:", " ".join(command))
    return subprocess.run(command, cwd=str(root))


def start_docker_stack(root, no_browser, wait_timeout):
    if not (root / "docker-compose.yml").exists():
        print(f"docker-compose.yml was not found in {root}")
        return 1
    if not ensure_docker_available():
        return 1

    env_file = ensure_docker_env(root)
    url = docker_service_url(root)

    if is_service_up(url):
        print(f"Docker service is already running: {url}")
        if not no_browser:
            webbrowser.open(url)
        return 0

    result = run_docker_compose(root, env_file, ["up", "--build", "-d"])
    if result.returncode != 0:
        print("Docker Compose finished with an error.")
        return result.returncode

    print(f"Waiting for service: {url}")
    if not wait_for_service(url, wait_timeout):
        print(f"Service did not become ready in {wait_timeout} seconds.")
        print("Docker is still running. Check logs with:")
        print(f"docker compose --env-file {env_file} logs -f")
        return 1

    print(f"Service is ready: {url}")
    if not no_browser:
        webbrowser.open(url)
    print("You can close this launcher window. Stop the stack from the UI or with:")
    print(f"docker compose --env-file {env_file} down")
    return 0


def stop_docker_stack(root):
    if not ensure_docker_available():
        return 1
    env_file = ensure_docker_env(root)
    result = run_docker_compose(root, env_file, ["down"])
    return result.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description="Diarization System launcher")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser")
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Start Docker Compose stack instead of local Python service",
    )
    parser.add_argument(
        "--stop-docker",
        action="store_true",
        help="Stop Docker Compose stack and exit",
    )
    parser.add_argument(
        "--wait-timeout",
        type=int,
        default=900,
        help="Seconds to wait until the local service starts",
    )
    args = parser.parse_args(argv)

    root = app_root()

    if args.stop_docker:
        return stop_docker_stack(root)
    if args.docker:
        return start_docker_stack(root, args.no_browser, args.wait_timeout)

    url = service_url()

    if is_service_up(url):
        print(f"Service is already running: {url}")
        if not args.no_browser:
            webbrowser.open(url)
        return 0

    python_exe = find_python(root)
    if not (root / "main.py").exists():
        print(f"main.py was not found in {root}")
        print("Run this launcher from the project folder or rebuild it from the repo root.")
        return 1

    print("Starting Diarization System...")
    print(f"Project folder: {root}")
    print(f"Python: {python_exe}")
    process, stdout_file, stderr_file, stdout_path, stderr_path = start_service(
        root,
        python_exe,
    )
    print(f"Service logs: {root / 'logs' / 'app.log'}")
    print(f"Launcher stdout: {stdout_path}")
    print(f"Launcher stderr: {stderr_path}")

    try:
        if not wait_for_service(url, args.wait_timeout):
            print(f"Service did not start in {args.wait_timeout} seconds.")
            print("Check logs/app.log and logs/launcher-service.err.log.")
            return process.poll() or 1

        print(f"Service is ready: {url}")
        if not args.no_browser:
            webbrowser.open(url)

        print("Keep this window open while you use the service.")
        print("Press Ctrl+C here to stop it.")
        return process.wait()
    except KeyboardInterrupt:
        print("\nStopping service...")
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
        return 0
    finally:
        stdout_file.close()
        stderr_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
