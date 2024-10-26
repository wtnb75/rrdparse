import io
import sys
import struct
import math
import array
import fnmatch
from abc import ABCMeta, abstractmethod
from logging import getLogger
from datetime import datetime
from dataclasses import dataclass, field
from typing import Generator

_log = getLogger(__name__)


class BaseIf(metaclass=ABCMeta):
    prefix_map = {
        'little': '<',
        'big': '>',
        'native': '=',
    }

    def prefix(self, endian: str) -> str:
        return self.prefix_map.get(endian, '=')

    @abstractmethod
    def parse_bytes(self, b: bytes, endian: str) -> None:
        pass

    @abstractmethod
    def read(self, fp: io.RawIOBase, endian: str) -> None:
        pass

    @abstractmethod
    def encode_bytes(self, endian: str) -> bytes:
        pass

    def write(self, fp: io.RawIOBase, endian: str):
        return fp.write(self.encode_bytes(endian))


@dataclass
class RrUnion(BaseIf):
    data: bytes
    endian: str

    def __init__(self, d: bytes | float | int, endian: str):
        self.endian = self.prefix(endian)
        if isinstance(d, float):
            self.float_val = d
        elif isinstance(d, int):
            self.int_val = d
        else:
            self.data = d

    @property
    def float_val(self) -> float:
        res = struct.unpack(self.endian+"d", self.data)
        return res[0]

    @float_val.setter
    def float_val(self, v: float):
        self.data = struct.pack(self.endian+"d", v)

    @property
    def int_val(self) -> int:
        res = struct.unpack(self.endian+"Q", self.data)
        return res[0]

    @int_val.setter
    def int_val(self, v: int):
        self.data = struct.pack(self.endian+"Q", v)

    def __str__(self) -> str:
        if self.int_val == 0:
            return "0"
        elif math.isnan(self.float_val):
            return "nan"
        return f"{self.int_val}/{self.float_val}"

    def __repr__(self) -> str:
        if self.int_val == 0:
            return "U0"
        elif math.isnan(self.float_val):
            return "Unan"
        return f"U[{self.int_val}/{self.float_val}]"

    def parse_bytes(self, b: bytes, endian: str) -> None:
        self.endian = self.prefix(endian)
        self.data = b

    def read(self, fp: io.RawIOBase, endian: str) -> None:
        self.parse_bytes(fp.read(8), endian)

    def encode_bytes(self, endian) -> bytes:
        if self.endian == self.prefix(endian):
            return self.data
        return RrUnion(self.int_val, endian).data


FLOAT_COOKIE = 8.642135E130


@dataclass
class Header(BaseIf):
    cookie: str = "RRD"
    version: str = "0003"
    float_cookie: float = FLOAT_COOKIE
    ds_cnt: int = 0
    rra_cnt: int = 0
    pdp_step: int = 0
    param: list[RrUnion] = field(default_factory=list)
    endian: str = '='

    def parse_bytes(self, b: bytes, endian: str):
        pfx = self.prefix(endian)
        offset = 0
        hdr0 = pfx + "4s5s3x4xdQQQ"
        cookie, version, self.float_cookie, self.ds_cnt, self.rra_cnt, self.pdp_step = \
            struct.unpack_from(hdr0, buffer=b, offset=offset)
        self.cookie = cookie.rstrip(b"\0").decode("utf-8")
        self.version = version.rstrip(b"\0").decode("utf-8")
        offset += struct.calcsize(hdr0)
        _log.debug("cookie=%s, version=%s, float_cookie=%s, offset=%s",
                   self.cookie, self.version, self.float_cookie, offset)
        assert self.cookie == "RRD"
        assert self.version in ("0002", "0003")
        assert self.float_cookie == FLOAT_COOKIE
        self.param = []
        hdr1_0 = pfx+"Q"
        hdr1_1 = pfx+"d"
        step = max(struct.calcsize(hdr1_0), struct.calcsize(hdr1_1))
        for _ in range(10):
            self.param.append(RrUnion(b[offset:offset+step], endian))
            offset += step
        _log.debug("header parsed: %s", self)

    def read(self, fp: io.RawIOBase, endian: str):
        self.parse_bytes(fp.read(128), endian)

    def encode_bytes(self, endian) -> bytes:
        pfx = self.prefix(endian)
        hdr0 = pfx+"4s5s3x4xdQQQ"
        res = struct.pack(hdr0, self.cookie.encode("utf-8"), self.version.encode("utf-8"),
                          self.float_cookie, self.ds_cnt, self.rra_cnt, self.pdp_step)
        for i in self.param:
            res += i.encode_bytes(endian)
        assert len(res) == 128
        return res


@dataclass
class DsHeader(BaseIf):
    name: str = ""
    dst: str = ""
    param: list[RrUnion] = field(default_factory=list)

    def parse_bytes(self, b: bytes, endian: str):
        pfx = self.prefix(endian)
        offset = 0
        hdr0 = pfx+"20s20s"
        name, dst = struct.unpack_from(hdr0, buffer=b, offset=offset)
        self.name = name.rstrip(b"\0").decode("utf-8")
        self.dst = dst.rstrip(b"\0").decode("utf-8")
        offset += struct.calcsize(hdr0)
        hdr1_0 = pfx+"Q"
        hdr1_1 = pfx+"d"
        step = max(struct.calcsize(hdr1_0), struct.calcsize(hdr1_1))
        for _ in range(10):
            self.param.append(RrUnion(b[offset:offset+step], endian))
            offset += step
        _log.debug("ds:header parsed: %s", self)

    def read(self, fp: io.RawIOBase, endian: str):
        self.parse_bytes(fp.read(120), endian)

    def encode_bytes(self, endian: str) -> bytes:
        pfx = self.prefix(endian)
        hdr0 = pfx+"20s20s"
        res = struct.pack(hdr0, self.name.encode("utf-8"), self.dst.encode("utf-8"))
        for i in self.param:
            res += i.encode_bytes(endian)
        assert len(res) == 120
        return res

    @property
    def minimal_heartbeat(self) -> int:
        return self.param[0].int_val

    @property
    def minval(self) -> int:
        return self.param[1].float_val

    @property
    def maxval(self) -> int:
        return self.param[2].float_val


@dataclass
class RraHeader(BaseIf):
    name: str = ""
    row_cnt: int = 0
    pdp_cnt: int = 0
    param: list[RrUnion] = field(default_factory=list)

    def parse_bytes(self, b: bytes, endian: str):
        pfx = self.prefix(endian)
        offset = 0
        hdr0 = pfx+"20s4xQQ"
        name, self.row_cnt, self.pdp_cnt = struct.unpack_from(hdr0, buffer=b, offset=offset)
        self.name = name.rstrip(b"\0").decode("utf-8")
        offset += struct.calcsize(hdr0)
        hdr1_0 = pfx+"Q"
        hdr1_1 = pfx+"d"
        step = max(struct.calcsize(hdr1_0), struct.calcsize(hdr1_1))
        for _ in range(10):
            self.param.append(RrUnion(b[offset:offset+step], endian))
            offset += step
        _log.debug("rra:header parsed: %s", self)

    def read(self, fp: io.RawIOBase, endian: str):
        self.parse_bytes(fp.read(120), endian)

    def encode_bytes(self, endian: str) -> bytes:
        pfx = self.prefix(endian)
        hdr0 = pfx+"20s4xQQ"
        res = struct.pack(hdr0, self.name.encode("utf-8"), self.row_cnt, self.pdp_cnt)
        for i in self.param:
            res += i.encode_bytes(endian)
        assert len(res) == 120
        return res

    @property
    def xff(self) -> float:
        return self.param[0].float_val


@dataclass
class LiveHead(BaseIf):
    ts: float = 0.0

    def parse_bytes(self, b: bytes, endian: str):
        pfx = self.prefix(endian)
        if len(b) == 8:
            ts, = struct.unpack(pfx+"Q", b)
            self.ts = float(ts)
        elif len(b) == 16:
            ts, usec = struct.unpack(pfx+"QQ", b)
            self.ts = float(ts)+float(usec)/1000000
        else:
            raise Exception("invalid live header")
        _log.debug("livehead parsed: %s", self)

    def read(self, fp: io.RawIOBase, endian: str, version: str = "0003"):
        if version < "0003":
            self.parse_bytes(fp.read(8), endian)
        else:
            self.parse_bytes(fp.read(16), endian)

    def encode_bytes(self, endian: str, version: str = "0003") -> bytes:
        pfx = self.prefix(endian)
        if version < "0003":
            hdr0 = pfx+"Q"
            return struct.pack(hdr0, int(self.ts))
        else:
            hdr0 = pfx+"QQ"
            return struct.pack(hdr0, int(self.ts), int(self.ts*1000000) % 1000000)

    def write(self, fp: io.RawIOBase, endian: str, version: str = "0003"):
        return fp.write(self.encode_bytes(endian, version))

    def __str__(self) -> str:
        return f"{datetime.fromtimestamp(self.ts).isoformat(timespec="microseconds")}"

    def __repr__(self) -> str:
        return f"{datetime.fromtimestamp(self.ts).isoformat(timespec="microseconds")}"


@dataclass
class PdpPrep(BaseIf):
    last_ds: str = ""
    scratch: list[RrUnion] = field(default_factory=list)

    def parse_bytes(self, b: bytes, endian: str):
        pfx = self.prefix(endian)
        offset = 0
        hdr0 = pfx+"30s2x"
        last_ds, = struct.unpack_from(hdr0, buffer=b, offset=offset)
        self.last_ds = last_ds.rstrip(b"\0").decode("utf-8")
        offset += struct.calcsize(hdr0)
        hdr1_0 = pfx+"Q"
        hdr1_1 = pfx+"d"
        step = max(struct.calcsize(hdr1_0), struct.calcsize(hdr1_1))
        for _ in range(10):
            self.scratch.append(RrUnion(b[offset:offset+step], endian))
            offset += step
        _log.debug("pdp:prep parsed: %s", self)

    def read(self, fp: io.RawIOBase, endian: str):
        self.parse_bytes(fp.read(112), endian)

    def encode_bytes(self, endian: str) -> bytes:
        pfx = self.prefix(endian)
        hdr0 = pfx+"30s2x"
        res = struct.pack(hdr0, self.last_ds.encode("utf-8"))
        for i in self.scratch:
            res += i.encode_bytes(endian)
        assert len(res) == 112
        return res

    @property
    def value(self) -> float:
        return self.scratch[1].float_val


@dataclass
class CdpPrep(BaseIf):
    scratch: list[RrUnion] = field(default_factory=list)

    def parse_bytes(self, b: bytes, endian: str):
        pfx = self.prefix(endian)
        offset = 0
        hdr1_0 = pfx+"Q"
        hdr1_1 = pfx+"d"
        step = max(struct.calcsize(hdr1_0), struct.calcsize(hdr1_1))
        for _ in range(10):
            self.scratch.append(RrUnion(b[offset:offset+step], endian))
            offset += step
        _log.debug("cdp:prep parsed: %s", self)

    def read(self, fp: io.RawIOBase, endian: str):
        self.parse_bytes(fp.read(80), endian)

    def encode_bytes(self, endian: str) -> bytes:
        res = b''
        for i in self.scratch:
            res += i.encode_bytes(endian)
        assert len(res) == 80
        return res

    @property
    def value(self) -> float:
        return self.scratch[0].float_val


@dataclass
class RraPtr(BaseIf):
    prep: int = 0

    def parse_bytes(self, b: bytes, endian: str):
        pfx = self.prefix(endian)
        offset = 0
        hdr0 = pfx+"Q"
        self.prep, = struct.unpack_from(hdr0, buffer=b, offset=offset)
        _log.debug("rraptr parsed: %s", self)

    def read(self, fp: io.RawIOBase, endian: str):
        self.parse_bytes(fp.read(8), endian)

    def encode_bytes(self, endian: str) -> bytes:
        pfx = self.prefix(endian)
        hdr0 = pfx+"Q"
        res = struct.pack(hdr0, self.prep)
        assert len(res) == 8
        return res


@dataclass
class RrdValue(BaseIf):
    value: float = 0.0

    def parse_bytes(self, b: bytes, endian: str):
        pfx = self.prefix(endian)
        offset = 0
        hdr0 = pfx+"d"
        self.value, = struct.unpack_from(hdr0, buffer=b, offset=offset)

    def read(self, fp: io.RawIOBase, endian: str):
        self.parse_bytes(fp.read(8), endian)

    def encode_bytes(self, endian: str) -> bytes:
        pfx = self.prefix(endian)
        hdr0 = pfx+"d"
        res = struct.pack(hdr0, self.value)
        assert len(res) == 8
        return res


@dataclass
class RrdFile(BaseIf):
    header: Header = field(default_factory=Header)
    dshdr: list[DsHeader] = field(default_factory=list)
    rrahdr: list[RraHeader] = field(default_factory=list)
    livehead: LiveHead = field(default_factory=LiveHead)
    pdps: list[PdpPrep] = field(default_factory=list)
    cdps: list[CdpPrep] = field(default_factory=list)
    rras: list[RraPtr] = field(default_factory=list)
    values: array.array = field(default_factory=lambda: array.array('d'))

    def parse_bytes(self, b: bytes, endian: str):
        fp = io.BytesIO(b)
        self.read(fp, endian)

    def read(self, fp: io.RawIOBase, endian: str):
        self.header.read(fp, endian)
        for _ in range(self.header.ds_cnt):
            ent = DsHeader()
            ent.read(fp, endian)
            self.dshdr.append(ent)
        for _ in range(self.header.rra_cnt):
            ent = RraHeader()
            ent.read(fp, endian)
            self.rrahdr.append(ent)
        self.livehead.read(fp, endian, version=self.header.version)
        for _ in range(self.header.ds_cnt):
            ent = PdpPrep()
            ent.read(fp, endian)
            self.pdps.append(ent)
        for _ in range(self.header.ds_cnt*self.header.rra_cnt):
            ent = CdpPrep()
            ent.read(fp, endian)
            self.cdps.append(ent)
        for _ in range(self.header.rra_cnt):
            ent = RraPtr()
            ent.read(fp, endian)
            self.rras.append(ent)
        # data body
        rows = 0
        for _ in self.dshdr:
            for i in self.rrahdr:
                rows += i.row_cnt
        self.values.fromfile(fp, rows)
        if endian not in (sys.byteorder, 'native'):
            self.values.byteswap()
        _log.debug("offset: %s", fp.tell())

    def encode_bytes(self, endian: str) -> bytes:
        res = self.header.encode_bytes(endian)
        for i in self.dshdr:
            res += i.encode_bytes(endian)
        for i in self.rrahdr:
            res += i.encode_bytes(endian)
        res += self.livehead.encode_bytes(endian, version=self.header.version)
        for i in self.pdps:
            res += i.encode_bytes(endian)
        for i in self.cdps:
            res += i.encode_bytes(endian)
        for i in self.rras:
            res += i.encode_bytes(endian)
        if endian not in (sys.byteorder, 'native'):
            self.values.byteswap()
            res += self.values.tobytes()
            self.values.byteswap()
        else:
            res += self.values.tobytes()
        return res

    def create_args(self) -> list[str]:
        res: list[str] = []
        oldest: int = 0
        res.append("--step")
        res.append(f"{self.header.pdp_step}s")
        # DS
        for i in self.dshdr:
            if i.dst not in ("COMPUTE"):
                res.append(f"DS:{i.name}:{i.dst}:{i.minimal_heartbeat}:{i.minval}:{i.maxval}")
            else:
                raise Exception(f"dst=COMPUTE is not supported: {i}")
        # RRA
        for i in self.rrahdr:
            if i.name in ("AVERAGE", "MAX", "MIN", "LAST"):
                res.append(f"RRA:{i.name}:{i.xff}:{i.pdp_cnt}:{i.row_cnt}")
                if oldest < i.pdp_cnt * i.row_cnt:
                    oldest = i.pdp_cnt*i.row_cnt
            else:
                raise Exception(f"name={i.name} is not supported: {i}")
        res.insert(0, f"now-{oldest}s")
        res.insert(0, "--start")
        return res

    def ds_names(self) -> Generator[DsHeader, None, None]:
        yield from self.dshdr

    def rra_names(self) -> Generator[RraHeader, None, None]:
        yield from self.rrahdr

    def rra_iter(self, dsname: str | None = None, rraname: str | None = None) -> \
            Generator[tuple[DsHeader, RraHeader, array.array, int], None, None]:
        n = 0
        idx = 0
        for i in self.dshdr:
            for j in self.rrahdr:
                if (dsname is None or fnmatch.fnmatch(i.name, dsname)) and \
                        (rraname is None or fnmatch.fnmatch(j.name, rraname)):
                    yield i, j, self.values[idx:idx+j.row_cnt], self.rras[n].prep
                idx += j.row_cnt
                n += 1

    def data_iter(self, dsname: str | None = None, rraname: str | None = None):
        last_ts = self.livehead.ts
        for ds, rra, data_raw, data_idx in self.rra_iter(dsname, rraname):
            step_ts = self.header.pdp_step * rra.pdp_cnt
            first_ts = int(last_ts) - (rra.row_cnt-1) * step_ts
            yield ds, rra, zip(range(first_ts, first_ts+step_ts*rra.row_cnt, step_ts),
                               data_raw[data_idx+1:] + data_raw[:data_idx+1])
