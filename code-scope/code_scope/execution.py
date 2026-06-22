from __future__ import annotations

import multiprocessing as mp
import traceback


def _execute(code: str, test: str, entry_point: str, queue) -> None:
    try:
        namespace = {}
        exec(code + "\n" + test + f"\ncheck({entry_point})\n", namespace)
        queue.put({"passed": True, "error": None})
    except BaseException:
        queue.put({"passed": False, "error": traceback.format_exc(limit=5)})


def check_completion(problem: dict, completion: str, timeout: float) -> dict:
    queue = mp.get_context("spawn").Queue()
    process = mp.get_context("spawn").Process(
        target=_execute,
        args=(problem["prompt"] + completion, problem["test"], problem["entry_point"], queue),
    )
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.kill()
        process.join()
        return {"passed": False, "error": f"timeout after {timeout}s"}
    return queue.get() if not queue.empty() else {"passed": False, "error": "worker exited without result"}
