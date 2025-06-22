import os
import sys
import hashlib
from pathlib import Path

import docker


def is_path(arg):
    try:
        return Path(arg).exists()
    except Exception:
        return False


def hash_path(p):
    return hashlib.sha256(str(p).encode()).hexdigest()[:10]


def resolve_mounts(args):
    mounts = {}
    for arg in args:
        if is_path(arg):
            abs_path = Path(arg).resolve()
            if abs_path not in mounts:
                container_path = f"/mnt/{hash_path(abs_path)}{abs_path.suffix}"
                mounts[str(abs_path)] = container_path

    cwd = str(Path.cwd().resolve())
    if cwd not in mounts:
        mounts[cwd] = "/mnt/workspace"

    return mounts


def replace_args_with_container_paths(args, mounts):
    return [
        mounts[str(Path(arg).resolve())] if is_path(arg) else arg
        for arg in args
    ]


def run_container(image_name, container_command, args):
    client = docker.from_env()
    mounts_map = resolve_mounts(args)
    container_args = replace_args_with_container_paths(args, mounts_map)

    mounts = [
        docker.types.Mount(target=container_path, source=host_path, type='bind', read_only=False)
        for host_path, container_path in mounts_map.items()
    ]

    print(f"Running container `{image_name}` with command:")
    print([container_command] + container_args)
    print("Mounts:")
    for m in mounts:
        print(f"  {m['Source']} -> {m['Target']}")

    try:
        container = client.containers.run(
            user=f"{os.getuid()}:{os.getgid()}",
            image=image_name,
            command=[container_command] + container_args,
            mounts=mounts,
            remove=True,
            tty=True,
            stdin_open=True,
            working_dir="/mnt/workspace",
            runtime="nvidia",
            environment={
                "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
                "NVIDIA_VISIBLE_DEVICES": "0"
            },
        )
        print(container.decode())
    except docker.errors.ContainerError as e:
        print(f"Container failed: {e}")
    except docker.errors.ImageNotFound:
        print(f"Image '{image_name}' not found.")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python docker_launch.py [args...]")
        sys.exit(1)

    image = 'sdegno-prod'
    script = '/home/sdegno/Cosmica_V8/exefiles/Cosmica'
    script_args = sys.argv[1:]
    run_container(image, script, script_args)
