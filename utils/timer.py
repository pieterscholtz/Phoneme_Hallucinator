import time
from typing import Optional

class RuntimeSummarizer:

    def __init__(self):
        self.runtimes = {}
        self.current_tag = None
        self.current_st = None

    def __iadd__(self, other):
        for key, value in other.runtimes.items():
            if key in self.runtimes:
                self.runtimes[key] += value
            else:
                self.runtimes[key] = value
        return self

    def tag(self, tag_name: Optional[str] = None):
        if tag_name and tag_name not in self.runtimes:
            self.runtimes[tag_name] = 0

        if tag_name != self.current_tag:
            # create new start time, end pervious time
            if self.current_tag is not None:
                self.runtimes[self.current_tag] += time.monotonic() - self.current_st
            self.current_st = time.monotonic()
            self.current_tag = tag_name

    def summarize_runtime(self) -> str:
        report = ""
        for tag_name in self.runtimes:
            report += f"\t{tag_name}: {self.runtimes[tag_name]:.6f} seconds\n"
        report += f"Total: {sum(list(self.runtimes.values())):.6f} seconds"
        return report.strip()

    def reset(self) -> None:
        self.runtimes = {}
        self.current_tag = None
        self.current_st = None
