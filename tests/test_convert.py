import unittest
import shutil
import subprocess
import datetime
import math
import tempfile
from typing import Callable
from click.testing import CliRunner
from rrdparse.main import cli

rrdtool_path = shutil.which("rrdtool")


class TestConvert(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_help(self):
        res = CliRunner().invoke(cli, ["convert", "--help"])
        if res.exception:
            raise res.exception
        self.assertEqual(0, res.exit_code)
        self.assertIn("--endian", res.output)

    def _mkdata(self, filename: str, start: float, step: int, name: str, min: float, max: float, count: int,
                fn: Callable[[float], float]):
        opts = ["--start", str(start), "--step", str(step)]
        opts.append(f"DS:{name}:GAUGE:{step*2}:{min}:{max}")
        xff = 0.5
        for s, c in [(1, 576), (6, 432), (24, 540), (288, 450)]:
            for typ in ["AVERAGE", "MIN", "MAX"]:
                opts.append(f"RRA:{typ}:{xff}:{s}:{c}")
        subprocess.call([rrdtool_path, "create", filename, *opts])
        for i in range(start+step, start+step*count, step):
            val = fn(float(i))
            subprocess.call([rrdtool_path, "update", filename, f"{i}:{val}"])

    @unittest.skipUnless(rrdtool_path, "rrdtool not installed")
    def test_init(self):
        def fn(x: float) -> float:
            return math.sin(x/500/math.pi)
        st = int(datetime.datetime(2024, 1, 1).timestamp())
        with tempfile.NamedTemporaryFile(suffix=".rrd") as tf:
            self._mkdata(tf.name, st, 300, "name", -1.0, 1.0, 500, fn)
