import sys
import click
import datetime
import math
import functools
from .parse import RrdFile
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
    def _(verbose, *args, **kwargs):
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
    def _(endian, input, *args, **kwargs):
        rfp = RrdFile()
        rfp.read(input, endian=endian)
        return func(*args, rrdfile=rfp, **kwargs)

    return _


@cli.command()
@verbose_option
@rrdfile_options
@click.option("--format", type=click.Choice(["rrdcreate", "csv", "json", "rra", "rra-little", "rra-big"]))
@click.option("--output", type=click.File("wb"), default=sys.stdout)
@click.option("--tsformat", default="%Y-%m-%d %H:%M:%S")
@click.option("--ds-filter")
@click.option("--rra-filter")
def convert(rrdfile: RrdFile, output, format, tsformat, ds_filter, rra_filter):
    from collections import defaultdict
    import csv
    import json
    if format == "rrdcreate":
        click.echo("rrdcreate " + "__OUTPUT__.rrd " + " ".join(rrdfile.create_args()))
    elif format == "csv":
        res = defaultdict(dict)
        names: set[str] = set()
        for ds, rra, data in rrdfile.data_iter(ds_filter, rra_filter):
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
        for ds, rra, data in rrdfile.data_iter(ds_filter, rra_filter):
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
        json.dump(res, output)
    elif format == "rra":
        rrdfile.write(output, "native")
    elif format == "rra-little":
        rrdfile.write(output, "little")
    elif format == "rra-big":
        rrdfile.write(output, "big")


@cli.command()
@verbose_option
@rrdfile_options
@click.option("--ds-filter")
@click.option("--rra-filter")
def plot(rrdfile, ds_filter, rra_filter):
    import matplotlib.pyplot as plt
    import matplotlib.dates as pltdates
    pdpcnts = sorted(list({x.pdp_cnt for x in rrdfile.rra_names()}))
    fig, ax = plt.subplots(len(pdpcnts), 1, layout='constrained')
    ts = {x: [] for x in pdpcnts}
    for ds, rra, data in rrdfile.data_iter(ds_filter, rra_filter):
        kvs = [(k, v) for k, v in data]
        idx = pdpcnts.index(rra.pdp_cnt)
        ts = [datetime.datetime.fromtimestamp(x[0]) for x in kvs]
        vals = [x[1] for x in kvs]
        ax[idx].grid(True)
        ax[idx].plot(ts, vals, label=rra.name)
        ax[idx].set_title(f"{ds.name} / {rra.pdp_cnt}")
        ax[idx].legend(loc="upper left")
        ax[idx].xaxis.set_minor_formatter(pltdates.DateFormatter("%H:%M"))
        ax[idx].xaxis.set_minor_formatter(pltdates.DateFormatter("\n%Y-%m-%d"))
    plt.show()


if __name__ == "__main__":
    cli()
