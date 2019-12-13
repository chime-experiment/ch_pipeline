from caput import config, mpiutil
from draco.core import task
import yaml
from os import path
import git


class RegisterProcessedFiles(task.SingleTask):

    output_root = config.Property(proptype=str, default=None)
    product_type = config.Property(proptype=str, default=None)
    tag = config.Property(proptype=str, default=None)
    db_fname = config.Property(proptype=str)

    def setup(self):
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


def get_proc_transits(db_fname):
    with open(db_fname, "r") as fh:
        entries = yaml.load(fh)
    entries_filt = []
    for e in entries:
        if isinstance(e, dict) and "holobs_id" in e.keys():
            entries_filt.append(e)
    return entries_filt
