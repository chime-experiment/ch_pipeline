<h1 align="center">CHIME Analysis Pipeline</h1>

This is the repository for storing the CHIME Analysis Pipeline. For CHIME members, more information about
the pipeline can be found in the [Pipeline Repo](https://github.com/chime-experiment/Pipeline) and
[Pipeline Wiki](https://github.com/chime-experiment/Pipeline/wiki/Overview).

Development should be done along the lines of [Github Flow](https://guides.github.com/introduction/flow/).

Important notes:

 - *Don't* develop directly in `main`, use a feature branch for any change, and merge back into *main* promptly. Merging should be done by filing a Pull Request.
 - *Do* install the `virtualenv` with `mkchimeenv`, located [here](https://github.com/chime-experiment/mkchimeenv).

# Development Guidelines

The idea behind this repository is to keep track of the CHIME pipeline development, such that the union of the input data and this repository always gives the same
output. This requires that we keep track of not only the code and scripts in this repository, but also any dependencies (discussed below).

The underlying pipeline code uses the pipeline task module `caput.pipeline` ([doc](https://caput.readthedocs.io/en/latest/autoapi/caput/pipeline/index.html)).

## Structure

Tasks should go into the appropriate subdirectory of `ch_pipeline/`. Ask for clarification if not clear.

## Coding Standards

Code should adhere to the [CHIME contribution guidelines](https://github.com/chime-experiment/Pipeline/blob/main/CONTRIBUTING.md).
If you haven't looked at it please do so.

Code should be well documented, with a docstring expected for each public
function, class or method. These should be done according to [Numpy docstring style](https://numpydoc.readthedocs.io/en/latest/format.html).

Code should also be well commented (not the same as well documented). Try and
put in comments that explain logically what each section of code is trying to
achieve, with specifics about any non-obvious parts.

## Branches

As mentioned above, this should be done a la Github Flow. Development should
generally be done around reasonably small, self contained, features, and should
be tracked in specific feature branches. These should not be long lived, as soon
as the feature is finished and tested, file a Pull Request, get it reviewed by
someone else, and when given the okay, merge the code and delete the branch. Any
new development should be branched off from `main` again.

Please don't use long lived, person specific branches e.g. `richard-dev`.

## Dependencies

Dependencies should be python packages installable with `pip`.  The list of
dependencies is kept in the `pyproject.toml` file.  This dependency list can
contain references to exact versions of dependencies, using both version tags,
and commit hashes. An example dependency list that you might find in
`pyproject.toml` is given below:

```toml
dependencies = [
    "caput @ git+https://github.com/radiocosmology/caput.git@ee1c55ea4cf8cb7857af2ef3adcb2439d876768d",
    "ch_util @ git+https://github.com/chime-experiment/ch_util.git@e33b174696509b158c15cf0bfc27f4cb2b0c6406#egg=ch_util",
    "cora @ git+https://github.com/radiocosmology/cora.git@v1.0.0",
    "driftscan @ git+https://github.com/radiocosmology/driftscan.git@v1.0.0"
]
```

Here, the first two requirements specify an exact git hash, whereas the second two use git tags as a shorthand.

These dependencies can be installed using:
```bash
pip install .
```
This is automatically done by the `mkchimeenv` script.

Additional, optional dependencies can also be specified in `pyproject.toml`, in the
`[project.optional-dependencies]` section.  Each list of optional dependencies needs a tag
and should be specified using syntax similar to this:

```toml
[project.optional-dependencies]
my_tag = [
    "my_optional_dependency",
    "another_optional_dependency"
]
```

These optional dependencies may be installed with pip by using the tag, e.g.:
```bash
pip install .[my_tag]
```

## Virtualenv

The script `mkchimeenv`, located [here](https://github.com/chime-experiment/mkchimeenv), will automatically install a
[virtualenv](http://www.virtualenv.org/) containing all the pipeline
dependencies from the `requirements.txt` file. This gives a fresh, self-contained installation of the pipeline to work with. Before use, you should activate it using:
```bash
$ source VENV_NAME/venv/bin/activate
```
where `VENV_NAME` was the name you specified when invoking `mkchimeenv`.

# Running the Pipeline
`ch_pipeline` provides the `chp` command-line utility, which is a **manager** for CHIME pipeline types, revisions, and items. The underlying pipelines
are run using [caput-pipeline](https://github.com/radiocosmology/caput/tree/main/caput/pipeline) - for more information, see the
[docs](https://caput.readthedocs.io/en/latest/autoapi/caput/pipeline/index.html).

The `chp` management hierarchy is:
- `type`: a broad processing category, implemented in `ch_pipeline/processing` (eg., `daily`)
- `rev`: a revision of a processing category, which will typically have the same input files but slightly different outputs and/or config options (eg., `daily:rev_01`)
- `item`: a single unit (tag, or job) of a revision, run as a single `caput-pipeline` slurm job (eg., `daily:rev_01 1878`)

`chp` has a fleshed-out help system - for more information, use the `--help` flag. For example,
```bash
$ chp --help
$ chp type --help
```

## Creating a new pipeline type
New pipeline types require significant development in the form of a new `.py` module in `ch_pipeline/processing`. See existing types for reference.

## Creating a new pipeline type revision
```bash
$ chp --root </path/to/root/directory/> rev create <type>
```

Each new revisions is one more than the current highest revision in that type's directory, up to a max of 99.

## Customizing a revision
Each new pipeline revision comes with the following:
- `/venv`
- `/config/jobtemplate.yaml`
- `/config/revconfig.yaml`

Create a new virtual environment (using `mkchimeenv`) in the `venv` directory, or symlink to an existing venv. From there, you can update
the configuration files as-needed.

### jobtemplate.yaml
This is a template of the pipeline config file provided to `caput-pipeline`. Each individual `item` is automatically created by populating
a copy of `jobtemplate.yaml` using items in `revconfig.yaml`.

Typically, you don't want to change anything in `jobtemplate.yaml` other than adding/removing pipeline tasks.

### revconfig.yaml
This contains all of the revision configuration information - data variables, slurm job parameters, etc...

