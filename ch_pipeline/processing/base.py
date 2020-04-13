# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import re
import yaml
import os
from subprocess import check_call
import tempfile

# TODO: Python 3 workaround
try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path


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
    """Baseclass for a pipeline processing type."""

    # Must be set externally before using
    root_path = None

    # Defined in sub-classses
    default_params = {}
    default_script = DEFAULT_SCRIPT

    def __init__(self, revision, create=False):

        self.revision = revision

        # Run the create hook if specified
        if create:
            self._create()

        # Run the load hook if specified
        self._load()

    def _create(self):
        """Save default parameters and pipeline config for this revision."""

        # Write default configuration into directory
        with (self.revconfig_path).open("w") as fh:
            dump = yaml.safe_dump(
                self.default_params, encoding="utf-8", allow_unicode=True
            )
            fh.write(str(dump))  # TODO: Python 3 - str needed
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

        # Subclass hook
        self._create_hook()

    def _create_hook(self):
        """Implement to add custom behaviour when a revision is created."""
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
        """Implement to add custom behaviour when a revision is loaded."""
        pass

    def job_script(self, tag):
        """The slurm job script to queue up."""

        jobparams = dict(self._revparams)
        jobparams.update(
            {
                "jobname": self.job_name(tag),
                "dir": str(self.base_path / tag),
                "tempdir": str(self.workdir_path / tag),
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

    def ls(self):
        """Find all matching data.

        Returns
        -------
        tags : list
            Return the tags of all outputs found.
        """

        base = self.base_path

        if not base.exists():
            raise ValueError("Base path %s does not exist." % base)

        file_regex = re.compile("^%s$" % self.tag_pattern)

        entries = [path.name for path in base.glob("*") if file_regex.match(path.name)]

        return sorted(entries)

    @classmethod
    def ls_type(cls, existing=True):
        """List all processing types found.

        Parameters
        ----------
        existing : bool, optional
            Only return types that have existing data.

        Returns
        -------
        type_names : list
        """

        type_names = [t.type_name for t in all_subclasses(cls)]

        if existing:
            base = Path(cls.root_path)
            return sorted([t.name for t in base.glob("*") if t.name in type_names])
        else:
            return type_names

    @classmethod
    def ls_rev(cls):
        """List all existing revisions of this type.

        Returns
        -------
        rev : list
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

    def queued(self):
        """Get the queued and running jobs of this type.

        Returns
        -------
        waiting : list
            List of jobs that are waiting to run.
        running : list
            List of running jobs.
        """

        job_regex = re.compile("^%s$" % self.job_name(self.tag_pattern))

        # Find matching jobs
        jobs = [job for job in slurm_jobs() if job_regex.match(job["NAME"])]

        running = [job["NAME"].split("/")[-1] for job in jobs if job["ST"] == "R"]
        waiting = [job["NAME"].split("/")[-1] for job in jobs if job["ST"] == "PD"]

        return waiting, running

    def job_name(self, tag):
        """The job name used to run the tag.

        Parameters
        ----------
        tag : str
            Tag for the job.

        Returns
        -------
        jobname : str
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

    def available(self):
        """Return the list of tags that we can generate given current data.

        This can (and should) include tags that have already been processed
        if the prerequites are still available, but will exclude any tags
        currently present in the working directory.

        Returns
        -------
        tags : list of strings
            A list of all the tags that could be generated.
        """
        tags = self._available_tags()

        # Filter out entries that already exist in the working directory
        working_tags = [path.name for path in self.workdir_path.glob("*")]
        tags = [tag for tag in tags if tag not in working_tags]

        return tags

    def _available_tags(self):
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

    def generate(self, max=10, submit=True):
        """Queue up jobs that are available to run.

        Parameters
        ----------
        max : int, optional
            The maximum number of jobs to submit at once.
        submit : bool, optional
            Submit the jobs to the queue.
        """

        to_run = self.pending()[:max]

        for tag in to_run:
            queue_job(self.job_script(tag), submit=submit)

    def pending(self):
        """Jobs available to run."""

        waiting, running = self.queued()
        pending = set(self.available()).difference(self.ls(), waiting, running)

        return sorted(list(pending))


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
    import tempfile

    with tempfile.NamedTemporaryFile("w+") as fh:
        fh.write(script)
        fh.flush()

        # TODO: do this in a better way
        if submit:
            cmd = "caput-pipeline queue %s"
        else:
            cmd = "caput-pipeline queue --nosubmit %s"
        os.system(cmd % fh.name)


def slurm_jobs(user=None):
    """Get the jobs of the given user.

    Parameters
    ----------
    user : str, optional
        User to fetch the slurm jobs of. If not set, use the current user.

    Returns
    -------
    jobs : list
        List of dictionaries giving the jobs state.
    """

    import subprocess as sp

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
        import warnings

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
