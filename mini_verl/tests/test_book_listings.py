"""Verify every standalone listing exported from verl From Scratch executes cleanly."""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


class TestBookListings(unittest.TestCase):
    def test_all_book_listings_run_and_pass_assertions(self) -> None:
        listings_dir = Path(__file__).resolve().parent.parent / "examples" / "book_listings"
        scripts = sorted(p for p in listings_dir.glob("ch*.py"))
        self.assertEqual(len(scripts), 10)
        for script in scripts:
            with self.subTest(listing=script.name):
                proc = subprocess.run(
                    [sys.executable, str(script)],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(
                    proc.returncode,
                    0,
                    msg=f"{script.name} failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}",
                )


if __name__ == "__main__":
    unittest.main()
