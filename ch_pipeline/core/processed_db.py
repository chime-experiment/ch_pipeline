"""Database of processed data files

Tasks and functions for interacting with the database of processed data files.
For the moment, the "database" is implemented as a YAML file containing a list
of files and associated metadata.

Tasks
=====

RegisterProcessedFiles
"""

from caput import config, mpiutil
from draco.core import task
import yaml
from os import path
import git


class RegisterProcessedFiles(task.SingleTask):
    """Register a file in the processed data database.

    For now this is implemented as a YAML file specified in the
    config parameters.

    For most cases, this class should be inherited and the process method
    overridden to save the information relevant to the type of processed data
    being saved.

    The 'product_type' parameter is meant to identify the type of processing and
    the 'tag' could be used to distinguish between runs.
    """

    output_root = config.Property(proptype=str, default=None)
    product_type = config.Property(proptype=str, default=None)
    tag = config.Property(proptype=str, default=None)
    db_fname = config.Property(proptype=str)

    def setup(self):
        """Extract git tags for a few important pipeline packages: `ch_util`,
        `caput`, `ch_pipeline`, `draco`, and `driftscan`.
        """

        # Extract config, git tags
        self.git_tags = {
            "caput": {},
            "ch_util": {},
            "ch_pipeline": {},
            "draco": {},
            "driftscan": {},
        }
        for k in self.git_tags.keys():
            try:
                module = __import__(k)
            except ImportError:
                continue
            try:
                self.git_tags[k]["version"] = module.__version__
            except AttributeError:
                self.git_tags[k]["version"] = None
            try:
                repo = git.Repo(
                    path.dirname(module.__file__), search_parent_directories=True
                )
                self.git_tags[k]["branch"] = repo.active_branch.name
                self.git_tags[k]["commit"] = repo.active_branch.commit.hexsha
            except git.InvalidGitRepositoryError:
                continue
        # TODO: figure out how to get access to config

    def process(self, output):
        """Writes out a file and registers it in the database. Saves the file path,
        software git tags, product type, and tag. The file name is prefixed with
        the 'output_root' config parameter.

        This should be overriden to register information relevant to specific
        types of processed data if necessary.

        Parameters
        ----------
        output: caput.memh5.MemDiskGroup
            Data container to save and register.
        """

        # Create a tag for the output file name
        tag = output.attrs["tag"] if "tag" in output.attrs else self._count

        # Construct the filename
        outfile = self.output_root + str(tag) + ".h5"

        # Expand any variables in the path
        outfile = path.expanduser(outfile)
        outfile = path.expandvars(outfile)

        self.write_output(outfile, output)

        if mpiutil.rank0:
            # Add entry in database
            # TODO: check for duplicates ?
            append_product(
                self.db_fname,
                outfile,
                self.product_type,
                config=None,
                tag=self.tag,
                git_tags=self.git_tags,
            )

        return None


def append_product(
    db_fname, prod_fname, prod_type, config, tag=None, git_tags=[], **kwargs
):
    """Utility function to append a processed file to the YAML-based database.

    Parameters
    ----------
    db_fname: str
        Path to database YAML file.
    prod_fname: str
        Path to processed file to register.
    prod_type: str
        Type of processing.
    config: dict
        The pipeline config used to generate the data.
    tag: str
        A tag for this processing run.
    git_tags: dict
        Git tags for relevant software packages.
    **kwargs: dict
        Any other metadata to register with this file.
    """

    with open(db_fname, "r") as fh:
        entries = yaml.load(fh)
    if type(entries) is not list:
        raise "Could not parse YAML for processed data record."
    new_entry = {
        "filename": prod_fname,
        "type": prod_type,
        "tag": tag,
        "git_tags": git_tags,
        "pipeline_config": config,
    }
    new_entry.update(kwargs)
    entries.append(new_entry)
    with open(db_fname, "w") as fh:
        yaml.dump(entries, fh)
