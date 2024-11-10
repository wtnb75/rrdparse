import sys
import click
import datetime
import math
import functools
import fnmatch
from .parse import RrdFile
from typing import Callable
from logging import getLogger

_log = getLogger(__name__)


@click.group(invoke_without_command=True)
@click.version_option()
@click.pass_context
def cli(ctx):
    if ctx.invoked_subcommand is None:
        print(ctx.get_help())


def verbose_option(func):
    @click.option("--verbose/--quiet")
    @functools.wraps(func)
    def _(verbose: bool | None, *args, **kwargs):
        from logging import basicConfig
        logfmt = "%(asctime)s %(levelname)s %(message)s"
        if verbose is None:
            basicConfig(level="INFO", format=logfmt)
        elif verbose:
            basicConfig(level="DEBUG", format=logfmt)
        else:
            basicConfig(level="WARNING", format=logfmt)
        return func(*args, **kwargs)

    return _


def rrdfile_options(func):
    @click.option("--endian", type=click.Choice(["little", "big", "native"]), default="native")
    @click.option("--input", type=click.File("rb"))
    @functools.wraps(func)
    def _(endian: str, input, *args, **kwargs):
        rfp = RrdFile()
        rfp.read(input, endian=endian)
        return func(*args, rrdfile=rfp, **kwargs)

    return _


def rrfilter_options(func):
    @click.option("--ds-filter")
    @click.option("--rra-filter")
    @functools.wraps(func)
    def _(ds_filter, rra_filter, *args, **kwargs):
        dsfn: Callable | None = None
        if ds_filter:
            dsfn = functools.partial(fnmatch.fnmatch, pat=ds_filter)
        rrfn: Callable | None = None
        if rra_filter:
            rrfn = functools.partial(fnmatch.fnmatch, pat=rra_filter)
        return func(*args, dsfn=dsfn, rrfn=rrfn, **kwargs)

    return _


def defjson(o):
    if hasattr(o, "isoformat"):
        return o.isoformat()
    else:
        return str(o)


@cli.command()
@verbose_option
@rrdfile_options
@rrfilter_options
@click.option("--format", type=click.Choice(["rrdcreate", "csv", "json", "json2", "rra", "rra-little", "rra-big"]))
@click.option("--output", type=click.File("wb"), default=sys.stdout)
@click.option("--tsformat", default="%Y-%m-%d %H:%M:%S")
def convert(rrdfile: RrdFile, output, format: str, tsformat: str | None, dsfn: Callable | None, rrfn: Callable | None):
    from collections import defaultdict
    import csv
    import json
    if format == "rrdcreate":
        click.echo("rrdcreate " + "__OUTPUT__.rrd " + " ".join(rrdfile.create_args()))
    elif format == "csv":
        res = defaultdict(dict)
        names: set[str] = set()
        for ds, rra, data in rrdfile.data_iter(dsfn, rrfn):
            _log.debug("ds=%s/%s/%s rra=%s/pdp=%s/row=%s", ds.name, ds.dst,
                       ds.minimal_heartbeat, rra.name, rra.pdp_cnt, rra.row_cnt)
            name = f"{ds.name}/{rra.name}"
            names.add(name)
            for ts, value in data:
                res[ts][name] = value
        fields = ["time"] + list(sorted(names))
        wr = csv.DictWriter(output, fields)
        wr.writeheader()
        for ts in sorted(res.keys()):
            tstr = datetime.fromtimestamp(ts).strftime(tsformat)
            ent = {k: v for k, v in res[ts].items() if not math.isnan(v)}
            if len(ent) != 0:
                wr.writerow({"time": tstr, **ent})
    elif format == "json":
        res: dict[str, dict[str, list[dict]]] = {}
        names: set[str] = set()
        for ds, rra, data in rrdfile.data_iter(dsfn, rrfn):
            k1 = ds.name
            k2: str = f"{rra.name}/{rra.pdp_cnt}"
            if k1 not in res:
                res[k1] = {}
            assert k2 not in res[k1]
            res[k1][k2] = []
            _log.debug("ds=%s/%s/%s rra=%s/pdp=%s/row=%s", ds.name, ds.dst,
                       ds.minimal_heartbeat, rra.name, rra.pdp_cnt, rra.row_cnt)
            for ts, value in data:
                if not math.isnan(value):
                    res[k1][k2].append({
                        "timestamp": ts,
                        "datetime": datetime.fromtimestamp(ts).isoformat(),
                        "value": value})
        json.dump(res, output, default=defjson)
    elif format == "json2":
        res = rrdfile.ds_iter(dsfn)
        json.dump({"/".join([str(y) for y in k]): v for k, v in res.items()}, output, default=defjson)
    elif format == "rra":
        rrdfile.write(output, "native")
    elif format == "rra-little":
        rrdfile.write(output, "little")
    elif format == "rra-big":
        rrdfile.write(output, "big")


def timerange(td: datetime.timedelta) -> str:
    if td.days in (1, 2):
        return "day"
    elif td.days in (7, 8, 9):
        return "week"
    elif 30 <= td.days and td.days <= 60:
        return "month"
    elif 365 <= td.days and td.days <= 365*2:
        return "year"
    return str(td)


@cli.command()
@verbose_option
@rrdfile_options
@rrfilter_options
def plot_matplotlib(rrdfile: RrdFile, dsfn, rrfn):
    import matplotlib.pyplot as plt
    import matplotlib.dates as pltdates
    data = rrdfile.ds_iter(dsfn)
    dkeys = sorted(data.keys())
    fig, ax = plt.subplots(len(dkeys), 1, layout='constrained')
    for key, v in data.items():
        idx = dkeys.index(key)
        name = timerange(v[-1]["datetime"]-v[0]["datetime"])
        ax[idx].grid(True)
        ts = [x["datetime"] for x in v]
        minv = [x["min"] for x in v]
        maxv = [x["max"] for x in v]
        avgv = [x["average"] for x in v]
        ax[idx].fill_between(ts, minv, maxv, alpha=0.3)
        ax[idx].plot(ts, avgv)
        ax[idx].set_title(name)
        ax[idx].xaxis.set_minor_formatter(pltdates.DateFormatter("%H:%M"))
        ax[idx].xaxis.set_minor_formatter(pltdates.DateFormatter("\n%Y-%m-%d"))
    plt.show()


@cli.command()
@verbose_option
@rrdfile_options
@rrfilter_options
def plot_plotly(rrdfile, dsfn, rrfn):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    data = rrdfile.ds_iter(dsfn)
    dkeys = sorted(data.keys())
    names = []
    for key in dkeys:
        v = data[key]
        td: datetime.timedelta = v[-1]["datetime"]-v[0]["datetime"]
        names.append(timerange(td))
    fig = make_subplots(rows=len(dkeys), cols=1, subplot_titles=names)
    for key, v in data.items():
        idx = dkeys.index(key)
        ts = [x["datetime"] for x in v]
        name = key[0]
        minv = [x["min"] for x in v]
        maxv = [x["max"] for x in v]
        avgv = [x["average"] for x in v]
        minmax_marker = {"color": "aqua"}
        emptyline = {"width": 0}
        colorline = {"color": "blue"}
        fig.add_trace(go.Scatter(x=ts, y=minv, fill="none", opacity=0.2, showlegend=False,
                                 marker=minmax_marker, line=emptyline), row=idx+1, col=1)
        fig.add_trace(go.Scatter(x=ts, y=maxv, fill="tonexty", opacity=0.2,
                      showlegend=False, marker=minmax_marker, line=emptyline), row=idx+1, col=1)
        fig.add_trace(go.Scatter(x=ts, y=avgv, fill="none", name=name, line=colorline), row=idx+1, col=1)
    fig.show()


if __name__ == "__main__":
    cli()
