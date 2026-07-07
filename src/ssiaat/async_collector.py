"""
Module for asynchronous collection and processing of data with concurrency control.
"""

import asyncio
import contextlib
from typing import (
    TypeVar,
    Generic,
    Iterable,
    AsyncIterable,
    Callable,
    Awaitable,
    List,
    Optional,
    Tuple,
    Union,
    Any,
)
from tqdm.asyncio import tqdm as tqdm_asyncio

T = TypeVar("T")  # Input item type
R = TypeVar("R")  # Result type

class AsyncCollector(Generic[T, R]):
    """
    A utility class to process an iterable of inputs asynchronously using a fixed number of workers.

    It uses an asyncio.Queue to distribute items from the input iterator to multiple
    worker tasks, collects the results, and optionally displays a progress bar.

    Failure contract: an exception raised by the processor never kills a
    worker (so the run can never deadlock on queue.join()); instead the
    offending ``(item, exception)`` pair is appended to ``self.failures``
    and processing continues. ``run()`` returns only the successful,
    non-None results.
    """

    def __init__(
        self,
        iter_input: Union[Iterable[T], AsyncIterable[T]],
        processor: Callable[[T], Awaitable[Optional[R]]],
        *,
        total: Optional[int] = None,
    ) -> None:
        """
        Initialize the AsyncCollector.

        Args:
            iter_input: An iterable (sync or async) containing the items to process.
            processor: An awaitable function/method that takes one item and returns a result.
            total: The total number of items in iter_input. If None, it tries to use len(iter_input).
        """
        self.iter_input: Union[Iterable[T], AsyncIterable[T]] = iter_input
        self.total: Optional[int] = total
        self.processor: Callable[[T], Awaitable[Optional[R]]] = processor
        self.queue: asyncio.Queue[T] = asyncio.Queue()
        self.results: List[R] = []
        self.failures: List[Tuple[T, Exception]] = []

    async def _queue_input(self) -> None:
        """
        Internal method to populate the queue with items from the input iterator.
        Supports both synchronous and asynchronous iterables.
        """
        if hasattr(self.iter_input, "__aiter__"):
            async for item in self.iter_input:  # type: ignore
                await self.queue.put(item)
        else:
            for item in self.iter_input:  # type: ignore
                await self.queue.put(item)

    async def _worker(self, pbar: Optional[Any] = None) -> None:
        """
        Internal worker method that processes items from the queue.

        Exceptions from the processor are recorded in self.failures rather
        than propagated: a raising item must not kill the worker, or the
        remaining queue items would never receive task_done() and run()
        would wait on queue.join() forever.

        Args:
            pbar: A progress bar instance to update after each item is processed.
        """
        while True:
            item = await self.queue.get()
            try:
                r = await self.processor(item)
                if r is not None:
                    self.results.append(r)
            except Exception as e:
                # asyncio.CancelledError is a BaseException, so worker
                # cancellation in run() still propagates normally.
                self.failures.append((item, e))
            finally:
                self.queue.task_done()
                if pbar is not None:
                    pbar.update()

    async def run(self, num_tasks: int = 4, progress: bool = False) -> List[R]:
        """
        Execute the processing of all items.

        Args:
            num_tasks: Number of concurrent worker tasks to run.
            progress: If True, display a progress bar using tqdm.

        Returns:
            A list of results returned by the processor (excluding None values).
            Note: Results are appended in the order they complete.
            Items whose processing raised are available as (item, exception)
            pairs in self.failures.
        """
        total: Optional[int] = self.total
        if total is None:
            try:
                total = len(self.iter_input)  # type: ignore
            except (TypeError, AttributeError):
                total = None

        # Start the task that populates the queue
        init_task = asyncio.create_task(self._queue_input())

        if progress:
            cm = tqdm_asyncio(total=total)
        else:
            cm = contextlib.nullcontext()

        with cm as pbar:
            # Create worker tasks
            tasks = [
                asyncio.create_task(self._worker(pbar=pbar)) for _ in range(num_tasks)
            ]

            # Wait for all input to be queued
            await init_task
            # Wait for all items in the queue to be processed
            await self.queue.join()

            # Cancel workers as they are no longer needed
            for t in tasks:
                t.cancel()
                # Suppress cancellation errors
                with contextlib.suppress(asyncio.CancelledError):
                    await t

        return self.results
