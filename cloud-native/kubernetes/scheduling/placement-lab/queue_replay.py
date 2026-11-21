"""Replay a fixed queue on eight identical nodes, with exact job durations."""

from dataclasses import dataclass
from typing import NamedTuple

NODES = 8


@dataclass(frozen=True)
class Job:
    name: str
    arrival: int
    nodes: int
    duration: int


class Running(NamedTuple):
    end: int
    nodes: int


JOBS = (
    Job("occupied", 0, 4, 6),
    Job("large", 1, 8, 4),
    Job("short", 2, 2, 2),
    Job("long", 2, 4, 8),
)


def earliest(job, now, running):
    free = NODES - sum(r.nodes for r in running)
    if free >= job.nodes:
        return now
    for end, nodes in sorted(running):
        free += nodes
        if free >= job.nodes:
            return end
    raise ValueError("job requests more nodes than the cluster has")


def replay(jobs, policy):
    if policy not in {"fifo", "backfill", "greedy"}:
        raise ValueError(policy)
    pending = sorted(jobs, key=lambda j: j.arrival)  # Stable for simultaneous arrivals.
    if any(j.nodes < 1 or j.nodes > NODES or j.duration <= 0 or j.arrival < 0 for j in pending):
        raise ValueError("invalid job shape or time")
    if len({j.name for j in pending}) != len(pending):
        raise ValueError("duplicate job name")
    now, starts, running, ready = 0, {}, [], []
    while pending or ready:
        running = [r for r in running if r.end > now]
        while pending and pending[0].arrival <= now:
            ready.append(pending.pop(0))
        while ready:
            head = ready[0]
            reserved = earliest(head, now, running)
            pick = None
            if reserved == now:
                pick = 0
            elif policy != "fifo":
                for i, job in enumerate(ready[1:], 1):
                    candidate = Running(now + job.duration, job.nodes)
                    if earliest(job, now, running) != now:
                        continue
                    if policy == "backfill" and earliest(head, now, running + [candidate]) > reserved:
                        continue
                    pick = i
                    break
            if pick is None:
                break
            job = ready.pop(pick)
            starts[job.name] = now
            running.append(Running(now + job.duration, job.nodes))
        if pending or ready:
            events = [r.end for r in running]
            if pending:
                events.append(pending[0].arrival)
            now = min(events)
    return starts


if __name__ == "__main__":
    print("policy    waits (occupied, large, short, long)   last_finish")
    for policy in ("fifo", "backfill", "greedy"):
        starts = replay(JOBS, policy)
        waits = [starts[j.name] - j.arrival for j in JOBS]
        end = max(starts[j.name] + j.duration for j in JOBS)
        print(f"{policy:8}  {str(waits):36}  {end}")
