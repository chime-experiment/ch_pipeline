"""Command line client for Chime Pipeline processing."""

import click

from . import base, beam, daily, quarterstack  # noqa: F401

click.disable_unicode_literals_warning = True


# Map the type names to classes
_typedict = {t.type_name: t for t in base.all_subclasses(base.ProcessingType)}


class PType(click.ParamType):
    """Param type for validating and returning ProcessingType arguments."""

    name = "processing type"

    def convert(self, value, param, ctx):
        """Get a processing type from a string."""
        if value not in _typedict:
            self.fail(
                f"processing type '{value}' unknown. See `chp type list` "
                "for valid options."
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
                f"processing type '{value}' unknown. See `chp type list` "
                "for valid options."
            )

        _type = _typedict[typename]

        if len(r) == 1:
            return _type.latest()

        rev = r[1]
        if rev not in _type.ls_rev():
            self.fail(
                f"revision not found for spec {rev} in type {typename}. "
                f"See `chp rev list {typename}` for valid revisions."
            )

        return _type(rev)


PTYPE = PType()
PREV = PRev()


@click.group()
@click.option(
    "--root",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=base.DEFAULT_ROOT,
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
    click.echo(f"Created {rev.revision}")
    click.echo(
        f"You must create a virtual environment in {rev.venv_path} "
        "before you can run any jobs."
    )


@item.command("list")
@click.argument("revision", type=PREV)
@click.option(
    "-l", "--long", is_flag=True, help="Display directory size and number of files."
)
@click.option(
    "-h",
    "--human",
    is_flag=True,
    help="Display directory size in human-readable format.",
)
@click.option("-t", "--time", is_flag=True, help="Sort item be time, newest first.")
def item_list(revision, long, human, time):
    """List existing items within the REVISION (given as type:revision)."""
    for tag in revision.ls(time):
        if long:
            n, size = dirstats(revision.base_path / tag)
            size = humansize(size, width=10) if human else str(size)
            click.echo(f"{tag:8s}{n:10d}{size:10s}")
        else:
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
    "-f",
    "--fairshare",
    type=float,
    default=0.0,
    help="Only submit jobs if the account LevelFS is above this threshold.",
)
@click.option(
    "--user-fairshare",
    type=float,
    default=0.0,
    help="Only submit jobs if the user LevelFS is above this threshold.",
)
@click.option(
    "-p",
    "--priority-fairshare",
    type=float,
    default=0.0,
    help=(
        "Fairshare threshold for priority jobs. If `--fairshare` is also given, "
        "and is lower than the priority fairshare given here, this option is ignored."
    ),
)
@click.option("--check-failed", is_flag=True, help="Try to requeue failed jobs.")
def generate(
    revision,
    number,
    max_number,
    submit,
    fairshare,
    user_fairshare,
    priority_fairshare,
    check_failed,
):
    """Submit pending jobs for REVISION (given as type:revision)."""
    priority_only = False

    fs = base.slurm_fairshare("rpp-chime_cpu")

    # If a fairshare limit has been set, check that the current fairshare
    # is below it.
    if fairshare > fs[0]:
        click.echo(
            f"Current fairshare {fs[0]} is lower than threshold {fairshare}. "
            "Skipping non-priority."
        )
        priority_only = True

    # If a user fairshare limit has been set, check that the current user
    # fairshare is below it.
    if user_fairshare > fs[1]:
        click.echo(
            f"Current user fairshare {fs[1]} is lower than threshold {user_fairshare}. "
            "Skipping non-priority."
        )
        priority_only = True

    # If a priority fairshare limit has been set, check if priority jobs
    # can still be processed
    if priority_only and priority_fairshare > fs[0]:
        click.echo(
            f"Current fairshare {fs[0]} is lower than priority threshold {priority_fairshare}. "
            "Skipping all."
        )
        return

    number_in_queue, number_running = (len(l) for l in revision.queued())
    number_to_submit = max(
        min(number, max_number - number_in_queue - number_running), 0
    )

    click.echo(
        f"Generating {number_to_submit} jobs ({number_in_queue} jobs already queued)."
    )
    revision.generate(
        max=number_to_submit,
        submit=submit,
        priority_only=priority_only,
        check_failed=check_failed,
    )


@item.command("update")
@click.argument("revision", type=PREV)
@click.option(
    "--clear",
    is_flag=True,
    help="Remove data files which are no longer needed by this revision.",
)
@click.option(
    "--retrieve",
    is_flag=True,
    help="Submit a request to move required files to project storage.",
)
def update_files(revision, clear, retrieve):
    """Manage files between project and long-term storage spaces."""
    nfiles = revision.update_files(retrieve=retrieve, clear=clear)

    if retrieve:
        click.echo(f"Retrieving {nfiles['nretrieve']} files.")
    if clear:
        click.echo(f"Clearing {nfiles['nclear']} files.")


@item.command("status")
@click.argument("revision", type=PREV)
@click.option(
    "-u",
    "--user",
    type=str,
    default="chime",
    help="User to check jobs for.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Show all tags in each category instead of just a count.",
)
@click.option("-t", "--time", is_flag=True, help="Sort tags by time.")
@click.option(
    "-c",
    "--category",
    type=str,
    default=None,
    help=(
        "Only return metrics for jobs with this status. Valid categories "
        "include: available, not_available, not_yet_submitted, pending, "
        "running, successful, failed"
    ),
)
def status(revision, user, verbose, time, category):
    """Show metrics about currently running jobs for REVISION.

    Given as (type:revision)
    """
    fs = base.slurm_fairshare("rpp-chime_cpu")
    tag_status = revision.status(user, time)

    if category is not None:
        if category not in tag_status:
            click.echo(f"Invalid category: {category}. See --help for valid options.")
            return
        category = [category]
    else:
        category = tag_status.keys()

    click.echo(f"fairshare: {fs[0]}")
    for c in category:
        tags = tag_status[c]
        click.echo(f"{c}: {len(tags)}")
        if verbose:
            for tag in tags:
                click.echo(tag)


@item.command("failed")
@click.argument("revision", type=PREV)
@click.option(
    "-u",
    "--user",
    type=str,
    default="chime",
    help="User to check jobs for.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
)
@click.option("-t", "--time", is_flag=True)
def crashed(revision, user, verbose, time):
    """List crashed tags for REVISION, associated with available category matches."""
    failed = revision.failed(user, time)

    for k, tags in failed.items():
        if not tags:
            continue

        click.echo(f"{len(tags)} job(s) failed for reason: {k.upper()}")

        if verbose:
            for tag in tags:
                click.echo(tag)


@item.command("run")
@click.argument("revision", type=PREV)
@click.option("--update", type=int, default=None, help="Refresh time in minutes.")
@click.option(
    "-m",
    "--max-number",
    type=int,
    default=None,
    help="The maximum number of jobs to end up in the queue.",
)
@click.option(
    "-f",
    "--fairshare",
    type=float,
    default=None,
    help="Only submit jobs above this threshold.",
)
@click.option(
    "-u", "--user", type=str, default=None, help="User to use when checking slurm jobs."
)
@click.option(
    "--run-indefinitely",
    is_flag=True,
    help="If set, the runner should continue running even if there are no jobs remaining.",
)
def run_pipeline(revision, update, max_number, fairshare, user, run_indefinitely):
    """Run the pipeline service for this revision."""
    from . import runner

    # Set up and update the config
    config = {
        "update_minutes": update,
        "max_jobs_number": max_number,
        "min_fairshare_number": fairshare,
        "run_indefinitely": run_indefinitely,
    }
    if user:
        existing_user = runner.GLOBAL_CONFIG.get("user")
        if existing_user is not None:
            click.echo(
                f"Existing user {existing_user} found. "
                f"Job will run as user {existing_user}."
            )
            user = existing_user
        # Set the global user
        runner.GLOBAL_CONFIG["user"] = user

    # Set up this revision
    runner.setup(revision, config)
    # Run the server
    runner.run()


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
        raise ValueError(f"path {path!s} must be an exisiting directory")

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
    suffix : str
        unit suffix to include
    precision : int
        number of decimal places to include
    width : int
        minimum number of characters to include

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
