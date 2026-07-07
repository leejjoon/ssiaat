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

        Args:
            pbar: A progress bar instance to update after each item is processed.
        """
        while True:
            item = await self.queue.get()
            try:
                r = await self.processor(item)
                if r is not None:
                    self.results.append(r)
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

if __name__ == '__main__':
    async def process(item: int) -> int:
        """Example processor function."""
        await asyncio.sleep(0.1)
        return item

    async def main() -> None:
        """Main entry point for the example."""
        # Using a range (sync iterable)
        collector = AsyncCollector(range(10), process)
        results = await collector.run(progress=True)
        print(f"Results: {results}")

    asyncio.run(main())
