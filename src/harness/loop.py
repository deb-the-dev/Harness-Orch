"""Worker loop: claim a run from the queue, keep the lease alive, run the worker, release.

A worker process serves many organisations, so workers and orchestrators are built per org by the factories passed in.
"""
from __future__ import annotations

import asyncio
import os
import socket
from typing import Callable

from .queue import PostgresQueue
from .runs import LIVE, RunState
from .worker import Orchestrator, Worker


class WorkerLoop:
    def __init__(self, queue: PostgresQueue, worker_for: Callable[[str], Worker], orch_for: Callable[[str], Orchestrator],
                 worker_id: str | None = None, lease_s: float = 30.0, max_attempts: int = 3, retry_backoff_s: float = 2.0):
        self.queue, self.worker_for, self.orch_for = queue, worker_for, orch_for
        self.worker_id = worker_id or f"{socket.gethostname()}-{os.getpid()}"
        self.lease_s, self.max_attempts, self.retry_backoff_s = lease_s, max_attempts, retry_backoff_s

    async def _q(self, fn, *a, **kw):
        return await asyncio.to_thread(fn, *a, **kw)         # database calls must not block the event loop that runs heartbeats

    async def run_once(self) -> bool:
        """Claim and process one run. Returns False when the queue had nothing ready."""
        claim = await self._q(self.queue.claim, self.worker_id, self.lease_s)
        if claim is None:
            return False
        org, run_id = claim.org_id, claim.run_id
        orch, worker = self.orch_for(org), self.worker_for(org)

        if claim.attempts > self.max_attempts:                # a run that keeps crashing workers is stopped, not retried forever
            if orch.runs.view(run_id).state in LIVE:
                orch.runs.transition(run_id, "run.failed", {"reason": "max_attempts", "attempts": claim.attempts - 1})
            await self._q(self.queue.release, run_id, self.worker_id)
            return True

        lost = asyncio.Event()
        task = asyncio.ensure_future(worker.run(run_id))

        async def heartbeat():
            while not task.done():
                await asyncio.sleep(self.lease_s / 3)
                if not await self._q(self.queue.heartbeat, run_id, self.worker_id, self.lease_s):
                    lost.set()                                # another worker may own the run now: stop touching it
                    task.cancel()
                    return

        hb = asyncio.ensure_future(heartbeat())
        try:
            view = await task
        except asyncio.CancelledError:
            if lost.is_set():
                return True                                   # lease lost: leave the queue row to whoever holds it
            raise
        except Exception as e:                                # noqa: BLE001  a bug or an infrastructure fault, not a run outcome
            orch.log.append(org, run_id, "run.worker_error", {"error": f"{type(e).__name__}: {str(e)[:300]}", "attempt": claim.attempts})
            if orch.runs.view(run_id).state == RunState.RUNNING:
                orch.recover(run_id)                          # RUNNING -> QUEUED so the next claim resumes it
            await self._q(self.queue.release, run_id, self.worker_id, self.retry_backoff_s * claim.attempts)
            return True
        finally:
            hb.cancel()
        if view.state == RunState.RUNNING:                    # stopped without finishing or waiting: hand it straight back
            await self._q(self.queue.release, run_id, self.worker_id, 0.0)
        else:                                                 # finished, or waiting for a human, or paused: nothing to do until woken
            await self._q(self.queue.release, run_id, self.worker_id)
        return True

    async def reap(self) -> int:
        """Return runs whose worker stopped heartbeating to the queue, and record the loss in each run's log."""
        pairs = await self._q(self.queue.expire_leases)
        for org, run_id in pairs:
            orch = self.orch_for(org)
            if orch.runs.view(run_id).state == RunState.RUNNING:
                orch.recover(run_id)
        return len(pairs)

    async def run_forever(self, stop: asyncio.Event, poll_s: float = 0.5, reap_every_s: float = 5.0) -> None:
        last_reap = 0.0
        loop = asyncio.get_running_loop()
        while not stop.is_set():
            if loop.time() - last_reap >= reap_every_s:
                await self.reap()
                last_reap = loop.time()
            if not await self.run_once():
                try:
                    await asyncio.wait_for(stop.wait(), poll_s)
                except asyncio.TimeoutError:
                    pass
