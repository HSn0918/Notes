"""How many identical gangs fit in a snapshot? No cluster or scheduler is contacted."""

import argparse

WORKERS_PER_JOB = 8
GPUS_PER_NODE = 8
# Each tuple is one network domain; values are free devices on each node.
SNAPSHOTS = {
    "packed": ((8,) * 8,),
    "fragmented": ((4,) * 16,),
    "split": ((8,) * 4, (8,) * 4),
    "fleet": ((0,) * 1000 + (8,) * 125 + (4,) * 125,),
}


def capacity(domains, gpus_per_worker):
    if not 1 <= gpus_per_worker <= GPUS_PER_NODE:
        raise ValueError("invalid worker shape")
    total = sum(sum(nodes) for nodes in domains)
    slots = [sum(free // gpus_per_worker for free in nodes) for nodes in domains]
    return (total, total // (WORKERS_PER_JOB * gpus_per_worker),
            sum(slots) // WORKERS_PER_JOB, sum(n // WORKERS_PER_JOB for n in slots))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus-per-worker", type=int, choices=range(1, GPUS_PER_NODE + 1),
                        default=GPUS_PER_NODE)
    args = parser.parse_args()
    print("case          free_gpu  pooled_jobs  node_fit  same_domain")
    for case, domains in SNAPSHOTS.items():
        total, pooled, node_fit, same_domain = capacity(domains, args.gpus_per_worker)
        print(f"{case:12} {total:9} {pooled:12} {node_fit:9} {same_domain:12}")


if __name__ == "__main__":
    main()
