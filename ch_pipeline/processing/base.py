import re
import warnings
import yaml
import os
import subprocess as sp
import tempfile
from typing import Dict, Union, Tuple
from pathlib import Path


DEFAULT_SCRIPT = """
cluster:
  name: {jobname}

  directory: {dir}
  temp_directory: {tempdir}

  venv: {venv}
"""


DESC_HEAD = """# Revision `{}` of type `{}`
Please describe the purpose/changes of this revision here.
"""


class ProcessingType(object):
    """Baseclass for a pipeline processing type.

    Parameters
    ----------
    revision : str
        Revision to use.
    create : bool, optional
        Create the revision if it isn't found.
    root_path : str, optional
        Override the path to the processing root.
    """

    # Must be set externally before using
    root_path = None

    # Defined in sub-classses
    default_params = {}
    default_script = DEFAULT_SCRIPT

    def __init__(self, revision, create=False, root_path=None):
        self.revision = revision

        if root_path:
            self.root_path = root_path

        # Run the create hook if specified
        if create:
            self._create()

        # Run the load hook if specified
        self._load()

    def _create(self):
        """Save default parameters and pipeline config for this revision."""

        # Subclass hook
        self._create_hook()

        # Write default configuration into directory
        with (self.revconfig_path).open("w") as fh:
            yaml.safe_dump(
                self.default_params,
                encoding="utf-8",
                allow_unicode=True,
                sort_keys=False,
                stream=fh,
            )
        with (self.jobtemplate_path).open("w") as fh:
            fh.write(self.default_script)

        # Ensure working directory and venv paths are created
        _ = self.workdir_path
        _ = self.venv_path

        # Open description file in editor
        desc_path = self.base_path / "description.md"
        with desc_path.open("w") as fh:
            fh.write(DESC_HEAD.format(self.revision, self.type_name))
        os.system(r"${EDITOR:-vi} " + str(desc_path))

    def _create_hook(self):
        """Implement to add custom behaviour when a revision is created.

        This is called *before* the revision configuration is written out.
        """
        pass

    def _load(self):
        """Load default parameters and pipeline config for this revision."""

        # Load config from file
        with (self.revconfig_path).open() as fc:
            self._revparams = yaml.safe_load(fc)

        # Load the jobscript
        with (self.jobtemplate_path).open() as fp:
            self._jobconfig = fp.read()

        # Subclass hook
        self._load_hook()

    def _load_hook(self):
        """Implement to add custom behaviour when a revision is loaded.

        This is called *after* the object has had it's configuration loaded.
        """
        pass

    def job_script(self, tag):
        """The slurm job script to queue up."""

        jobparams = dict(self._revparams)
        jobparams.update(
            {
                "jobname": self.job_name(tag),
                "dir": str(self.base_path / tag),
                "tempdir": str(self.workdir_path / tag),
                "tag": tag,
            }
        )

        jobparams = self._finalise_jobparams(tag, jobparams)

        if not (self.venv_path / "bin/activate").exists():
            raise ValueError(
                "Couldn't find a virtualenv script in {}.".format(self.venv_path)
            )
        jobparams.update({"venv": self.venv_path})

        return self._jobconfig.format(**jobparams)

    def _finalise_jobparams(self, tag, jobparams):
        """Implement to edit job params based on the given tag before they are submitted."""
        return jobparams

    def ls(self, time_sort: bool = False) -> list:
        """Find all matching data.

        Parameters
        ----------
        time_sort
            true if tags should be sorted by time (newest first)

        Returns
        -------
        tags
            Return the tags of all outputs found.
        """

        base = self.base_path

        if not base.exists():
            raise ValueError("Base path %s does not exist." % base)

        file_regex = re.compile(f"^{self.tag_pattern}$")

        entries = [path for path in base.glob("*") if file_regex.match(path.name)]

        if time_sort:
            # Return the entries reverse sorted by time
            tags, times = zip(
                *[
                    (path.name, (path / "job/STATUS").stat().st_mtime)
                    for path in entries
                ]
            )
            return [x for _, x in sorted(zip(times, tags), reverse=True)]
        else:
            return sorted([path.name for path in entries])

    @classmethod
    def ls_type(cls, existing: bool = True) -> list:
        """List all processing types found.

        Parameters
        ----------
        existing
            Only return types that have existing data.

        Returns
        -------
        type_names
            list of processing types found
        """

        type_names = [t.type_name for t in all_subclasses(cls)]

        if existing:
            base = Path(cls.root_path)
            return sorted([t.name for t in base.glob("*") if t.name in type_names])
        else:
            return type_names

    @classmethod
    def ls_rev(cls) -> list:
        """List all existing revisions of this type.

        Returns
        -------
        rev
            List of revision names
        """

        base = Path(cls.root_path) / cls.type_name

        # Revisions are labelled by a two digit code
        # TODO: decide if two digits (i.e. 100 revisions max is enough)
        rev_regex = re.compile("^rev_\d{2}$")

        return sorted([t.name for t in base.glob("*") if rev_regex.match(t.name)])

    @classmethod
    def create_rev(cls):
        """Create a new revision of this type."""

        revisions = cls.ls_rev()

        if revisions:
            last_rev = revisions[-1].split("_")[-1]
            new_rev = "rev_%02i" % (int(last_rev) + 1)
        else:
            new_rev = "rev_00"

        (Path(cls.root_path) / cls.type_name / new_rev).mkdir(parents=True)

        return cls(new_rev, create=True)

    def queued(self, user: str = None) -> Tuple[list, list]:
        """Get the queued and running jobs of this type.

        Parameters
        ----------
        user
            user to find running jobs for. If not provided, this
            will default to the current user

        Returns
        -------
        waiting
            List of jobs that are waiting to run.
        running
            List of running jobs.
        """

        job_regex = re.compile(f"^{self.job_name(self.tag_pattern)}$")

        # Find matching jobs
        jobs = [job for job in slurm_jobs(user=user) if job_regex.match(job["NAME"])]

        running = [job["NAME"].split("/")[-1] for job in jobs if job["ST"] == "R"]
        waiting = [job["NAME"].split("/")[-1] for job in jobs if job["ST"] == "PD"]

        return waiting, running

    def job_name(self, tag: str) -> str:
        """The job name used to run the tag.

        Parameters
        ----------
        tag
            Tag for the job.

        Returns
        -------
        jobname
        """
        return "chp/%s/%s/%s" % (self.type_name, self.revision, tag)

    @property
    def base_path(self):
        """Base path for this processed data type."""

        base_path = Path(self.root_path) / self.type_name / self.revision

        return base_path

    @property
    def workdir_path(self):
        """Path to the working directory."""
        workdir_path = self.base_path / "working"
        workdir_path.mkdir(exist_ok=True)

        return workdir_path

    @property
    def revconfig_path(self):
        """Path to default config for this revision."""
        config_path = self.base_path / "config"
        config_path.mkdir(exist_ok=True)

        return config_path / "revconfig.yaml"

    @property
    def jobtemplate_path(self):
        """Path to template pipeline job config for this revision."""
        config_path = self.base_path / "config"
        config_path.mkdir(exist_ok=True)

        return config_path / "jobtemplate.yaml"

    @property
    def venv_path(self):
        """Path to virtual environment for this revision."""
        venv_path = self.base_path / "venv"
        venv_path.mkdir(exist_ok=True)

        return venv_path

    def available(self) -> list:
        """Return the list of tags that we can generate given current data.

        This can (and should) include tags that have already been processed
        if the prerequites are still available, but will exclude any tags
        currently present in the working directory.

        Returns
        -------
        tags
            A list of all the tags that could be generated.
        """
        tags = self._available_tags()

        # Filter out entries that already exist in the working directory
        working_tags = [path.name for path in self.workdir_path.glob("*")]
        tags = [tag for tag in tags if tag not in working_tags]

        return tags

    def _available_tags(self) -> list:
        """Return the list of tags available for processing."""
        return []

    @classmethod
    def latest(cls):
        """Create an instance to manage the latest revision.

        Returns
        -------
        pt : cls
            An instance of the processing type for the latest revision.
        """

        rev = cls.ls_rev()

        if not rev:
            raise RuntimeError("No revisions of type %s exist." % cls.type_name)

        # Create instance and set the revision
        return cls(rev[-1])

    def generate(self, max: int = 10, submit: bool = True, user: str = None):
        """Queue up jobs that are available to run.

        Parameters
        ----------
        max
            The maximum number of jobs to submit at once.
        submit
            Submit the jobs to the queue.
        user
            user to find running jobs for. If not provided, this
            will default to the current user
        """

        to_run = self._generate_hook(user=user)[:max]

        for tag in to_run:
            queue_job(self.job_script(tag), submit=submit)

    def _generate_hook(self, user: str = None) -> list:
        """Override to add custom behaviour when jobs are queued."""
        return self.status(user=user)["not_yet_submitted"]

    def pending(self, user: str = None):
        """Jobs available to run."""
        warnings.warn(
            "'pending' method is deprecated. Call 'status()['not_yet_submitted']'."
        )
        return self.status(user)["not_yet_submitted"]

    def failed(self, user: str = None, time_sort: bool = False) -> Dict[str, list]:
        """Categorize failed jobs.

        Parameters
        ----------
        user
            user to find running jobs for. If not provided, this
            will default to the current user
        time_sort
            true if tags should be sorted by time (newest first)

        Returns
        -------
        crashed
            tags associated with each category
        """
        # Get a set of all tags to check for cause of failure
        failed_tags = self.status(user, time_sort)["failed"]
        # Regex patterns associated with common sources of failure
        # Keys can be inserted in order of priority/specificity
        # as of Python 3.7+, as dicts will preserve insertion order.
        # This could be buggy on versions of python below 3.7, as the
        # broad 'unknown' key could match before a more specfic one
        patterns = {
            # Find crashes due to unwanted NaN/Inf values
            "nan_or_inf": [
                re.compile(r"ValueError(.*?)infs(.*?)NaNs"),
            ],
            # Find out of memory errors
            "out_of_memory": [
                re.compile(r"numpy.core._exceptions._ArrayMemoryError(.*?)MPI_ABORT")
            ],
            # Error connecting to chimedb database
            "chimedb_error": [
                re.compile(r"chimedb.core.exceptions.ConnectionError(.*?)MPI_ABORT")
            ],
            # Find crashes due to job timeout
            "time_limit": [
                re.compile(r"slurmstepd: error:(.*?)CANCELLED(.*?)TIME LIMIT"),
            ],
            "mpi_error": [
                re.compile(r"draco.core.misc.CheckMPIEnvironment: MPI test failed")
            ],
            # Match anything else
            "unknown": [re.compile(r"(.*?)")],
        }
        # Return all tags which match the listed patterns
        return classify_failed(self.workdir_path, failed_tags, patterns)

    def status(self, user: str = None, time_sort: bool = False) -> Dict[str, list]:
        """Find the status of existing jobs. This duplicates
        some other methods for individual status values, but
        reduces repeated method (and slurm) calls.

        Parameters
        ----------
        user
            user to find running jobs for. If not provided, this
            will default to the current user
        time_sort
            true if tags should be sorted by time (newest first)

        Returns
        -------
        status
            Return the tags associated with each status line: Available,
            Pending, Waiting, Running, Successful, Failed
        """
        file_regex = re.compile(f"^{self.tag_pattern}$")

        # Get available, finished, pending, and running jobs
        available_tags = self.available()
        finished_tags = self.ls(time_sort)
        pending_tags, running_tags = self.queued(user)

        # Get all tag paths in working directory
        working_tags = [
            path for path in self.workdir_path.glob("*") if file_regex.match(path.name)
        ]
        # Get the tags for each path in the working directory
        working_tags_set = {path.name for path in working_tags}

        # Get jobs which have not yet been submitted
        submitted_tags = set(finished_tags) | set(pending_tags) | set(running_tags)
        not_submitted_tags = [
            job for job in available_tags if job not in submitted_tags
        ]
        # Any tag in the working directory that is not running or
        # waiting should be considered as crashed. Also consider
        # finished tags to catch edge case where a tag is in the
        # process of being moved.
        crashed_tags = working_tags_set.difference(
            pending_tags + running_tags + finished_tags
        )
        # This is a workaround for an extra directory manually
        # created in some revision. It contains job directories
        # that have previously crashed. These jobs may be re-run
        # without being removed, so we have to check to include
        # only the tags that do not exist elsewhere.
        if (self.base_path / "crashed").exists():
            # Recursively search the crashed directory. Must
            # be recursive as there can sometims be sub-folders
            crashed_dir = [
                path
                for path in (self.base_path / "crashed").rglob("*")
                if file_regex.match(path.name)
            ]
            crashed_dir_tags = {path.name for path in crashed_dir}
            # Get tags in crashed folder that are not also found completed or running
            unique_crashed_dir = crashed_dir_tags.difference(
                set(working_tags_set) | set(finished_tags)
            )

            if time_sort:
                # Append any extra tag paths that need to be time sorted
                working_tags += [
                    path
                    for path in crashed_dir
                    if path.name in unique_crashed_dir - crashed_tags
                ]

            # take union of these sets in place
            crashed_tags |= unique_crashed_dir

        if time_sort:
            # Get the tag and associated completed time
            tags, times = zip(
                *[
                    (path.name, (path / "job/STATUS").stat().st_mtime)
                    for path in working_tags
                    if path.name in crashed_tags
                ]
            )
            # Get the tags sorted by completion time
            crashed_tags = [x for _, x in sorted(zip(times, tags), reverse=True)]
        else:
            crashed_tags = sorted(crashed_tags)

        # Return a dict of all status values
        tags = {
            "available": available_tags,
            "not_yet_submitted": not_submitted_tags,
            "pending": pending_tags,
            "running": running_tags,
            "successful": finished_tags,
            "failed": crashed_tags,
        }

        return tags


def find_venv():
    """Get the path of the current virtual environment

    Returns
    -------
    path : str
        Path to the venv, or `None` if we are not in a virtual environment.
    """
    import os

    return os.environ.get("VIRTUAL_ENV", None)


def queue_job(script, submit=True):
    """Queue a pipeline script given as a string."""

    import os

    with tempfile.NamedTemporaryFile("w+") as fh:
        fh.write(script)
        fh.flush()

        # TODO: do this in a better way
        if submit:
            cmd = "caput-pipeline queue %s"
        else:
            cmd = "caput-pipeline queue --nosubmit %s"
        os.system(cmd % fh.name)


def classify_failed(
    dir: Union[Path, str], tags: list, patterns: dict = {}
) -> Dict[str, list]:
    """Analyze the cause of crashed jobs.

    Parameters
    ----------
    dir
        directory to find tags
    tags
        tags to check
    patterns
        dictionary of patterns to check for. Keys are expected
        categories and values are lists of regex patterns to
        check for.

    Returns
    -------
    crashed
        tags associated with each pattern. Any unmatched tags are
        added with the key "other"
    """
    failed = {k: [] for k in list(patterns.keys())}

    for tag in tags:
        file = Path(dir) / tag / "job" / "jobout.log"
        if not file.is_file():
            continue
        # Get the end of the file. Assume an average of 100 characters
        # per line, so get around 300 lines.
        with open(file, "rb") as f:
            f.seek(-300 * 100, os.SEEK_END)
            tail = f.read().decode()

        # See if any of the patterns that we are looking for
        # exist in the stdout
        for key, regex_patterns in patterns.items():
            if any([bool(re.search(p, tail)) for p in regex_patterns]):
                failed[key].append(tag)
                break

    return failed


def slurm_jobs(user: str = None) -> list:
    """Get the jobs of the given user.

    Parameters
    ----------
    user
        User to fetch the slurm jobs of. If not set, use the current user.

    Returns
    -------
    jobs
        List of dictionaries giving the jobs state.
    """
    if user is None:
        import getpass

        user = getpass.getuser()

    # Call squeue to get the users jobs and get it's stdout
    try:
        process = sp.Popen(
            ["squeue", "-u", user, "-o", "%all"],
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            shell=False,
            universal_newlines=True,
        )
        proc_stdout, proc_stderr = process.communicate()
        lines = proc_stdout.split("\n")
    except OSError:
        warnings.warn('Failure running "squeue".')
        return []

    # Extract the headers
    header_line = lines.pop(0)
    header_cols = header_line.split("|")

    def slurm_split(line):
        # Split an squeue line accounting for the partitions

        tokens = line.split("|")

        fields = []

        t = None
        for token in tokens:
            t = token if t is None else t + "|" + token

            # Check if the token is balanced with square brackets
            br = t.count("[") - t.count("]")

            # If balanced keep the whole token, otherwise we keep will just
            # continue to see if the next token balances it
            if br == 0:
                fields.append(t)
                t = None

        return fields

    # Iterate over the following entries and parse them into queue jobs
    entries = []
    error_lines = []  # do something with this later
    for line in lines:
        parts = slurm_split(line)
        d = {}

        if len(parts) != len(header_cols):
            error_lines.append((len(parts), line, parts))
        else:
            for i, key in enumerate(header_cols):
                d[key] = parts[i]
            entries.append(d)

    return entries


def slurm_fairshare(account: str, user: str = None) -> Tuple[str, str]:
    """Get the LevelFS for the current user and account.

    Parameters
    ----------
    account
        The account to check.
    user
        The user on the account to check for.

    Returns
    -------
    account_fs
        The LevelFS for the whole account, i.e. the priority relative to all other
        accounts on the cluster.
    user_fs
        The LevelFS for the user, i.e. the priority compared to all other users on
        the account.
    """
    cmd = ["sshare", "-A", account, "-o", "LevelFS", "-n"]

    if user is not None:
        cmd += ["-u", user]

    # Call sshare to get the level fairshares
    try:
        process = sp.Popen(
            cmd,
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            shell=False,
            universal_newlines=True,
        )
        proc_stdout, proc_stderr = process.communicate()
        lines = proc_stdout.split("\n")
    except OSError as e:
        raise RuntimeError('Failure running "sshare".') from e

    # Filter empty lines
    lines = [line for line in lines if line]

    if len(lines) != 2:
        raise RuntimeError('Could not parse output from "sshare".')

    return tuple(float(line) for line in lines)


def all_subclasses(cls):
    """Recursively find all subclasses of cls."""

    subclasses = []

    stack = [cls]
    while stack:
        cls = stack.pop()

        for c in cls.__subclasses__():
            subclasses.append(c)
            stack.append(c)

    return subclasses
