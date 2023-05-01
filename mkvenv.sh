#!/bin/bash

# Set default values
venv_name="ch-pipeline"
venv_path="./venv"
code_path="./venv/src"
ignore_system="false"
install_chpipeline_in_code_path="false"

help_message="
    $(basename $0) [-n VENV_NAME] [-v VENV_PATH] [-e CODE_PATH] [-i] [-c] \n\n
    -n  \t Virtual environment name that will appear in your terminal prompt when activated. \n
        \t\t Defaults to '$venv_name'. \n
    -v  \t Path where virtual environment will be installed. \n
        \t\t Defaults to '$venv_path'. \n
    -e  \t Path where packages will be cloned for development. \n
        \t\t Defaults to '$code_path'. \n
    -i  \t Ignore packages already installed on your system, i.e., install all dependencies. \n
    -c  \t Also install ch_pipeline into path where other CHIME/radiocosmology packages are cloned. \n
"

# Parse any options provided by user
while getopts 'hn:v:e:ic' OPTION; do
    case "$OPTION" in
        n)
            venv_name="($OPTARG)"
            ;;

        v)
            venv_path="$OPTARG"
            ;;

        e)
            code_path="$OPTARG"
            ;;

        i)
            ignore_system="true"
            ;;

        c)
            install_chpipeline_in_code_path="true"
            ;;

        h)
            echo -e $help_message >&2
            exit 1
            ;;

        ?)
            echo -e $help_message >&2
            exit 1
            ;;
    esac
done
shift "$(($OPTIND -1))"

# Create and source the virtual environment
if ${ignore_system}; then
    virtualenv --prompt=$venv_name $venv_path
else
    virtualenv --system-site-packages --prompt=$venv_name $venv_path
fi
source $venv_path/bin/activate

# Create requirements.txt file for pip.
# We need a custom requirements file because the one bundled with ch_pipeline does
# not allow editable installs, but we'd like the radiocosmology and CHIME codes to be
# installed in an easily-accessible directory.
# If we'd like ch_pipeline to be cloned into the venv src directory, we add an
# appropriate line to the requirements file.
requirements="
    cython\n
    pytz\n
    -e git+https://github.com/radiocosmology/caput.git#egg=caput\n
    -e git+https://github.com/chime-experiment/ch_util.git#egg=ch_util\n
    -e git+https://github.com/radiocosmology/cora.git#egg=cora\n
    -e git+https://github.com/radiocosmology/driftscan.git#egg=driftscan\n
    -e git+https://github.com/radiocosmology/draco.git#egg=draco\n
"
chp_requirement="-e git+https://github.com/chime-experiment/ch_pipeline.git#egg=ch_pipeline"
if ${install_chpipeline_in_code_path}; then
    requirements="${requirements}${chp_requirement}"
fi
echo -e $requirements > ${venv_path}/venv_requirements.txt

# Install the packages
mkdir $code_path
pip install --use-deprecated=legacy-resolver --src $code_path -r ${venv_path}/venv_requirements.txt

# If not cloning ch_pipeline into the src directory, perform an editable install that
# keeps ch_pipeline where it is
if ! ${install_chpipeline_in_code_path}; then
    pip install -e .
fi
