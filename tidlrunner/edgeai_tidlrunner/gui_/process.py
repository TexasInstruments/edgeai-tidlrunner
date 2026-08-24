# Copyright (c) 2018-2025, Texas Instruments
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Runs tidlrunner-cli as a child process and streams its output line by line."""

import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
from typing import List, Optional, Tuple


def cli_prefix() -> List[str]:
    """The tidlrunner-cli launcher, falling back to the current interpreter."""
    launcher = shutil.which('tidlrunner-cli')
    if launcher:
        return [launcher]
    return [sys.executable, '-m', 'edgeai_tidlrunner.main']


class CommandRunner:
    """Non-blocking wrapper around a single tidlrunner-cli invocation.

    Output lines and the final exit code arrive on ``events`` as
    ``('line', text)`` and ``('exit', returncode)`` tuples.
    """

    def __init__(self) -> None:
        self.events: queue.Queue = queue.Queue()
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        with self._lock:
            return self._process is not None and self._process.poll() is None

    def start(self, argv: List[str], cwd: Optional[str] = None) -> None:
        if self.running:
            raise RuntimeError('a run is already in progress')

        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        env.setdefault('RUNNER_INVOKE_NAME', 'tidlrunner-cli')

        # start_new_session puts the runner and its parallel workers in their own
        # process group, so stop() can signal the whole tree
        process = subprocess.Popen(
            argv,
            cwd=cwd or None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors='replace',
            bufsize=1,
            start_new_session=True,
        )
        with self._lock:
            self._process = process
        threading.Thread(target=self._pump, args=(process,), daemon=True).start()

    def _pump(self, process: subprocess.Popen) -> None:
        try:
            for line in process.stdout:
                self.events.put(('line', line.rstrip('\n')))
        finally:
            process.stdout.close()
            self.events.put(('exit', process.wait()))

    def stop(self) -> None:
        with self._lock:
            process = self._process
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            process.terminate()

    def drain(self, limit: int = 500) -> List[Tuple[str, object]]:
        events = []
        for _ in range(limit):
            try:
                events.append(self.events.get_nowait())
            except queue.Empty:
                break
        return events
