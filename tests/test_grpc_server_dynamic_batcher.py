import time
import unittest
from concurrent.futures import Future
from threading import Event

from scripts.grpc_server import _DynamicInferBatcher


class _DeferredEngine:
    def __init__(self) -> None:
        self.calls: list[list[dict]] = []
        self.futures: list[Future] = []

    def infer_multi_stream_batch_async(self, items: list[dict]) -> Future:
        future: Future = Future()
        self.calls.append(items)
        self.futures.append(future)
        return future


class _BlockingFirstEngine:
    def __init__(self) -> None:
        self.calls: list[list[dict]] = []
        self.first_call_started = Event()
        self.release_first_call = Event()

    def infer_multi_stream_batch_async(self, items: list[dict]) -> Future:
        self.calls.append(items)
        if len(self.calls) == 1:
            self.first_call_started.set()
            if not self.release_first_call.wait(timeout=2):
                raise TimeoutError("first model call was not released")
        future: Future = Future()
        future.set_result(
            [{"frame_id": item["frame_id"]} for item in items]
        )
        return future


def _wait_for_call_count(engine: _DeferredEngine, expected: int) -> None:
    deadline = time.monotonic() + 2.0
    while len(engine.calls) < expected and time.monotonic() < deadline:
        time.sleep(0.01)
    if len(engine.calls) != expected:
        raise AssertionError(
            f"expected {expected} engine calls, got {len(engine.calls)}"
        )


class DynamicInferBatcherTests(unittest.TestCase):
    def test_overlapping_batches_keep_results_with_their_own_jobs(self) -> None:
        engine = _DeferredEngine()
        batcher = _DynamicInferBatcher(
            engine=engine,
            instance_name="test",
            max_batch_size=2,
            max_wait_ms=10,
            max_queue_size=10,
        )
        try:
            first = batcher.submit_async(
                stream_key="stream-a",
                frames=[("a-0", b"a"), ("a-1", b"b")],
            )
            _wait_for_call_count(engine, 1)

            second = batcher.submit_async(
                stream_key="stream-b",
                frames=[("b-0", b"c")],
            )
            _wait_for_call_count(engine, 2)

            engine.futures[1].set_result([{"frame_id": "b-0"}])
            engine.futures[0].set_result(
                [{"frame_id": "a-0"}, {"frame_id": "a-1"}]
            )

            self.assertEqual(
                [result["frame_id"] for result in second.result(timeout=1)],
                ["b-0"],
            )
            self.assertEqual(
                [result["frame_id"] for result in first.result(timeout=1)],
                ["a-0", "a-1"],
            )
        finally:
            batcher.close()

    def test_aged_job_still_gets_a_fresh_coalescing_window(self) -> None:
        engine = _BlockingFirstEngine()
        batcher = _DynamicInferBatcher(
            engine=engine,
            instance_name="test",
            max_batch_size=2,
            min_batch_size=2,
            idle_wait_ms=10,
            max_wait_ms=80,
            max_queue_size=10,
        )
        try:
            first = batcher.submit_async(
                stream_key="stream-a",
                frames=[("a-0", b"a")],
            )
            self.assertTrue(engine.first_call_started.wait(timeout=1))

            second = batcher.submit_async(
                stream_key="stream-b",
                frames=[("b-0", b"b")],
            )
            time.sleep(0.1)
            engine.release_first_call.set()
            time.sleep(0.02)
            third = batcher.submit_async(
                stream_key="stream-c",
                frames=[("c-0", b"c")],
            )

            _wait_for_call_count(engine, 2)
            self.assertEqual(
                [item["frame_id"] for item in engine.calls[1]],
                ["b-0", "c-0"],
            )
            self.assertEqual(first.result(timeout=1)[0]["frame_id"], "a-0")
            self.assertEqual(second.result(timeout=1)[0]["frame_id"], "b-0")
            self.assertEqual(third.result(timeout=1)[0]["frame_id"], "c-0")
        finally:
            engine.release_first_call.set()
            batcher.close()

    def test_minimum_batch_releases_after_idle_window(self) -> None:
        engine = _DeferredEngine()
        batcher = _DynamicInferBatcher(
            engine=engine,
            instance_name="test",
            max_batch_size=8,
            min_batch_size=4,
            idle_wait_ms=20,
            max_wait_ms=200,
            max_queue_size=16,
        )
        try:
            first = batcher.submit_async(
                stream_key="stream-a",
                frames=[("a-0", b"a"), ("a-1", b"b"), ("a-2", b"c")],
            )
            time.sleep(0.04)
            self.assertEqual(engine.calls, [])

            second = batcher.submit_async(
                stream_key="stream-b",
                frames=[("b-0", b"d")],
            )
            _wait_for_call_count(engine, 1)
            self.assertEqual(len(engine.calls[0]), 4)

            engine.futures[0].set_result(
                [{"frame_id": item["frame_id"]} for item in engine.calls[0]]
            )
            self.assertEqual(len(first.result(timeout=1)), 3)
            self.assertEqual(len(second.result(timeout=1)), 1)
        finally:
            batcher.close()

    def test_hard_deadline_releases_batch_below_minimum(self) -> None:
        engine = _DeferredEngine()
        batcher = _DynamicInferBatcher(
            engine=engine,
            instance_name="test",
            max_batch_size=8,
            min_batch_size=4,
            idle_wait_ms=10,
            max_wait_ms=60,
            max_queue_size=16,
        )
        started_at = time.monotonic()
        try:
            result = batcher.submit_async(
                stream_key="stream-a",
                frames=[("a-0", b"a")],
            )
            _wait_for_call_count(engine, 1)
            self.assertGreaterEqual(time.monotonic() - started_at, 0.04)
            self.assertEqual(len(engine.calls[0]), 1)

            engine.futures[0].set_result([{"frame_id": "a-0"}])
            self.assertEqual(result.result(timeout=1)[0]["frame_id"], "a-0")
        finally:
            batcher.close()


if __name__ == "__main__":
    unittest.main()
