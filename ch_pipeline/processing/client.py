import click

from . import base
from . import daily
from . import beam
from . import quarterstack

click.disable_unicode_literals_warning = True


# Map the type names to classes
_typedict = {t.type_name: t for t in base.all_subclasses(base.ProcessingType)}


class PType(click.ParamType):
    """Param type for validating and returning ProcessingType arguments."""

    name = "processing type"

    def convert(self, value, param, ctx):

        if value not in _typedict:
            self.fail(
                'processing type "%s" unknown. See `chp type list` '
                "for valid options." % value
            )

        return _typedict[value]


class PRev(click.ParamType):
    """Param type for validating and returning ProcessingType arguments."""

    name = "processing revision"

    def convert(self, value, param, ctx):
        """Parses `value` as {type}:{revision}."""

        r = value.split(":")

        typename = r[0]

        if typename not in _typedict:
            self.fail(
                'processing type "%s" unknown. See `chp type list` '
                "for valid options." % value
            )

        _type = _typedict[typename]

        if len(r) == 1:
            return _type.latest()
        else:
            rev = r[1]
            if rev not in _type.ls_rev():
                self.fail(
                    (
                        "revision not found for spec %s in type %s. "
                        "See `chp rev list %s` for valid revisions."
                    )
                    % (rev, typename, typename)
                )

            return _type(rev)


PTYPE = PType()
PREV = PRev()


@click.group()
@click.option(
    "--root",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default="/project/rpp-chime/chime/chime_processed/",  # TODO: put elsewhere
    help="Set the root directory to save processed data.",
)
def cli(root):
    """CHIME pipeline processing."""
    base.ProcessingType.root_path = root


@cli.group()
def type():
    """Types of data processing."""
    pass


@cli.group()
def rev():
    """Revisions of a processed data type."""
    pass


@cli.group()
def item():
    """Bundles of generated data."""
    pass


@type.command("list")
@click.option(
    "-a",
    "--all",
    "show_all",
    is_flag=True,
    help="Also list types for which there is no data.",
)
def type_list(show_all):
    """List known processing types."""

    for type_ in base.ProcessingType.ls_type(existing=(not show_all)):
        click.echo(type_)


@rev.command("list")
@click.argument("type_", type=PTYPE, metavar="TYPE")
def rev_list(type_):
    """List known revisions of TYPE."""

    for rev in type_.ls_rev():
        click.echo(rev)


@rev.command()
@click.argument("type_", type=PTYPE, metavar="TYPE")
def create(type_):
    """Create a new revision of TYPE."""

    rev = type_.create_rev()
    click.echo("Created {}".format(rev.revision))
    click.echo(
        (
            "You must create a virtual environment in {} "
            "before you can run any jobs."
        ).format(rev.venv_path)
    )


@item.command("list")
@click.argument("revision", type=PREV)
@click.option("-l", "--long", is_flag=True)
@click.option("-h", "--human", is_flag=True)
def item_list(revision, long, human):
    """List existing items within the REVISION (given as type:revision)."""
    for tag in revision.ls():

        if long:
            n, size = dirstats(revision.base_path / tag)
            size = humansize(size, width=10) if human else str(size)
            click.echo("{:8s}{:10d}{:10s}".format(tag, n, size))
        else:
            click.echo(tag)


@item.command()
@click.argument("revision", type=PREV)
def pending(revision):
    """List items that do not exist within REVISION
    (given as type:revision) but can be generated."""
    pending = revision.pending()
    for tag in pending:
        click.echo(tag)


@item.command()
@click.argument("revision", type=PREV)
@click.option(
    "-n",
    "--number",
    type=int,
    default=10,
    help="The maximum number of jobs to be submitted this time.",
)
@click.option(
    "-m",
    "--max-number",
    type=int,
    default=20,
    help=(
        "The maximum number of jobs to be end up in the queue. This does not include "
        " running jobs."
    ),
)
@click.option(
    "--submit/--no-submit", default=True, help="Submit the jobs to the queue (or not)"
)
@click.option(
    "--profile",
    is_flag=True,
    default=False,
    help=(
        "Run the job in cProfile. This will output a `profile_<rank>.prof` file per "
        "MPI rank."
    ),
)
@click.option(
    "-f",
    "--fairshare",
    type=float,
    default=None,
    help="Only submit jobs if the account LevelFS is above this threshold.",
)
@click.option(
    "--user-fairshare",
    type=float,
    default=None,
    help="Only submit jobs if the user LevelFS is above this threshold.",
)
def generate(revision, number, max_number, submit, profile, fairshare, user_fairshare):
    """Submit pending jobs for REVISION (given as type:revision)."""

    if fairshare or user_fairshare:
        # TODO: find a better way of supplying the account
        fs = base.slurm_fairshare("rpp-chime_cpu")

    if fairshare and fairshare > fs[0]:
        click.echo(
            f"Current fairshare {fs[0]} is lower than threshold {fairshare}. Skipping."
        )
        return

    if user_fairshare and user_fairshare > fs[1]:
        click.echo(
            f"Current user fairshare {fs[1]} is lower than threshold {user_fairshare}. "
            "Skipping."
        )
        return

    number_in_queue, number_running = [len(l) for l in revision.queued()]
    number_to_submit = max(
        min(number, max_number - number_in_queue - number_running), 0
    )

    click.echo(
        f"Generating {number_to_submit} jobs ({number_in_queue} jobs already queued)."
    )
    revision.generate(max=number_to_submit, submit=submit, profile=profile)


def dirstats(path):
    """Get stats for the specified directory.

    Parameters
    ----------
    path : Path
        Directory to calculate stats of.

    Returns
    -------
    num_files : int
        Number of files under the directory (recursive).
    total_size : int
        Total size in bytes of all files under the directory (recursive).
    """

    total_size = 0
    num_files = 0

    if not path.is_dir():
        raise ValueError("path %s must be an exisiting directory" % path)

    for p in path.rglob("*"):
        num_files += 1
        total_size += p.stat().st_size

    return num_files, total_size


def humansize(num, suffix="", precision=1, width=5):
    """Human readable file size.

    Parameters
    ----------
    num : int
        Size in bytes.

    Returns
    -------
    size : str
        The size as a human readable string.
    """
    for unit in ["B", "K", "M", "G", "T", "P"]:
        if abs(num) < 1024.0:
            break
        num /= 1024.0

    return "{:{width}.{precision}f}{}{}".format(
        num, unit, suffix, width=width, precision=precision
    )


if __name__ == "__main__":
    cli()
