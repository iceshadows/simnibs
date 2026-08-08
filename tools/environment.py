import argparse
from dataclasses import dataclass
import json
import platform
import subprocess
import sys
import textwrap
import urllib.error
import urllib.parse
import urllib.request


global PYTHON_VERSION
global DEFAULT_ENVIRONMENT_NAME
global DEFAULT_ENVIRONMENT_FILENAME
global DEFAULT_CONDA_CHANNELS
global DEPENDENCIES  # all defined at the end

# fmt: off
PYTHON_VERSION                  = "3.11"
DEFAULT_ENVIRONMENT_NAME        = "simnibs_env"
DEFAULT_ENVIRONMENT_FILENAME    = "environment.yml"
CONDA_CHANNELS                  = ["conda-forge"]
# fmt: on


def check_url(url):
    try:
        urllib.request.urlopen(url)
    except urllib.error.HTTPError as e:
        print("URL does not exist")
        print(url)
        raise e


@dataclass
class Dependency:
    """Class which defines basic information about a dependency/package.

    name:
        Name of package, e.g., numpy.
    version:
        Version of package, e.g., 2.0.
    platforms:
        The platform(s) on which this dependency is required.
    tag:
        A tag associated with this dependency allowing us to group
        dependencies.
    """

    name: str
    version: str | int | None = None
    platforms: str | list[str] | tuple = ("darwin", "linux", "windows")
    tag: str = "runtime"

    def build(self):
        raise NotImplementedError

    def __post_init__(self):
        if self.version is None:
            self.version = ""
        else:
            self.version = str(self.version)

        if isinstance(self.platforms, str):
            self.platforms = (self.platforms,)

        assert self.tag in ("runtime", "build", "test", "dev")

    def is_required(self, platform: str, valid_tags: list[str] | tuple | None = None):
        """Determine whether the package is required for this combination of
        platform and tags.
        """
        valid_tags = valid_tags or ("runtime",)
        return platform in self.platforms and self.tag in valid_tags


@dataclass
class CondaPackage(Dependency):
    """Class which defines a conda package.

    Paramters
    ---------
    comparison : str | None
        The comparison for the version package, e.g., "eq", "geq". Mandatory
        when a version is specified.
    """

    comparison: str | None = None

    def __post_init__(self):
        super().__post_init__()

        match self.comparison:
            case "eq":
                self.comparison = "="
            case "leq":
                self.comparison = "<="
            case "geq":
                self.comparison = ">="
            case "le":
                self.comparison = "<"
            case "ge":
                self.comparison = ">"
            case None:
                if self.version not in (None, ""):
                    raise ValueError(
                        "A comparison method must be specified when a package version is set."
                    )
                self.comparison = ""
            case _:
                raise ValueError(f"Invalid comparison method `{self.comparison}`")

    def build(self):
        return f"{self.name}{self.comparison}{self.version}"


@dataclass
class PipPackage(Dependency):
    """Class which defines a Pip package.

    Paramters
    ---------
    comparison : str | None
        The comparison for the version package, e.g., "eq", "geq". Mandatory
        when a version is specified.
    """

    comparison: str | None = None

    def __post_init__(self):
        super().__post_init__()

        match self.comparison:
            case "eq":
                self.comparison = "=="
            case "leq":
                self.comparison = "<="
            case "geq":
                self.comparison = ">="
            case "le":
                self.comparison = "<"
            case "ge":
                self.comparison = ">"
            case None:
                if self.version not in (None, ""):
                    raise ValueError(
                        "A comparison method must be specified when a package version is set."
                    )
                self.comparison = ""
            case _:
                raise ValueError(f"Invalid comparison method `{self.comparison}`")

    def build(self):
        return f"{self.name}{self.comparison}{self.version}"


@dataclass(kw_only=True)
class GitHubCommit(Dependency):
    """Class which defines a pip installable package from a GitHub repository
    and commit/tag. The commit/tag is specified as the package version.

    Parameters
    ----------
    user : str
        GitHub user hosting the package.
    repository: str | None
        GitHub repository hosting the package. If None, use the package name.
    """

    user: str
    repository: str | None = None

    def __post_init__(self):
        super().__post_init__()
        self.repository = self.repository or self.name

    def build(self):
        url = "/".join(["https://github.com", self.user, self.repository])
        check_url(url)
        return f"{self.name}@git+{url}@{self.version}"


@dataclass(kw_only=True)
class Wheel(Dependency):
    """Class which defines builds a wheel name from package information.

    platform_tags : dict
        The platform tags to use for this wheel (e.g., linux_x86_64).
    """

    platform_tags: dict
    # We use cpython so this is the python tag
    CPYTHON_VERSION_TAG = f"cp{PYTHON_VERSION.replace('.', '')}"

    def build(self, platform):
        platform_tag = self.platform_tags[platform]
        return f"{self.name}-{self.version}-{self.CPYTHON_VERSION_TAG}-{self.CPYTHON_VERSION_TAG}-{platform_tag}.whl"


@dataclass(kw_only=True)
class GitHubRelease(Wheel):
    """Class which defines a pip installable package form a GitHub release/tag.

    Parameters
    ----------
    user : str
        GitHub user hosting the package.
    repository : str | None
        GitHub repository hosting the package. If None, use the package name.
    release : str | None
        Name of the release/tag. If None, will use the package version
        prepended by "v", e.g., v1.2.3.
    """

    user: str
    repository: str | None = None
    release: str | None = None

    def __post_init__(self):
        super().__post_init__()
        self.repository = self.repository or self.name
        self.release = self.release or f"v{self.version}"

    def build(self, platform):
        url = "/".join(
            [
                "https://github.com",
                self.user,
                self.repository,
                "releases",
                "download",
                self.release,
                super().build(platform),
            ]
        )
        url = urllib.parse.quote_plus(url, ":/")
        check_url(url)
        return url


@dataclass(kw_only=True)
class IndexURL(Wheel):
    """Class which defines a pip installable package wheel from a base URL and
    package information.

    Parameters
    ----------
    url : str
        The base or index URL where the wheel is found.
    """

    url: str

    def build(self, platform):
        url = "/".join([self.url, super().build(platform)])
        url = urllib.parse.quote_plus(url, ":/")
        check_url(url)
        return url


class Dependencies(list):
    def __init__(self, *args):
        assert all(isinstance(d, Dependency) for d in args)
        super().__init__(args)

    @staticmethod
    def get_tags_for_env_type(env_type):
        match env_type:
            case "runtime":
                return ("runtime",)
            case "build":
                return ("runtime", "build")
            case "test":
                return ("runtime", "test")
            case "dev":
                return ("runtime", "build", "test")
            case _:
                raise ValueError("Invalid ")

    def get_conda_deps(self, platform: str, env_type: str):
        tags = self.get_tags_for_env_type(env_type)
        pkgs = []
        for dep in self:
            if isinstance(dep, CondaPackage) and dep.is_required(platform, tags):
                pkgs.append(dep.build())
        return pkgs

    def get_pip_deps(self, platform: str, env_type: str):
        tags = self.get_tags_for_env_type(env_type)

        pkgs = []
        for dep in self:
            if isinstance(dep, (PipPackage, GitHubCommit)) and dep.is_required(
                platform, tags
            ):
                pkgs.append(dep.build())
            elif isinstance(dep, (GitHubRelease, IndexURL)) and dep.is_required(
                platform, tags
            ):
                pkgs.append(dep.build(platform))
        return pkgs


def make_env_string(name, deps_conda, deps_pip):
    out = f"name: {name}\n"
    out += "channels:\n"
    for c in CONDA_CHANNELS:
        out += f"  - {c}\n"
    out += "dependencies:\n"
    for dep in sorted(deps_conda):
        out += f"  - {dep}\n"
    out += "  - pip:\n"
    for dep in sorted(deps_pip):
        out += f"    - {dep}\n"
    return out


def create_env(args):
    deps_conda = DEPENDENCIES.get_conda_deps(args.platform, args.type)
    deps_pip = DEPENDENCIES.get_pip_deps(args.platform, args.type)

    out = make_env_string(args.name, deps_conda, deps_pip)
    with open(args.filename, "w") as f:
        f.write(out)


def lock_env(args):
    env = subprocess.run(
        f"conda list -n {args.name} --json".split(), capture_output=True
    )
    packages = json.loads(env.stdout.decode())
    packages = {p["name"]: p["version"] for p in packages}

    locked = Dependencies()
    tags = locked.get_tags_for_env_type(args.type)

    for dep in DEPENDENCIES:
        if dep.is_required(args.platform, tags):
            if isinstance(dep, (CondaPackage, PipPackage)):
                kw = vars(dep)
                kw["version"] = packages[kw["name"]]
                kw["comparison"] = "eq"
                locked.append(dep.__class__(**kw))
            else:
                locked.append(dep)

    deps_conda = locked.get_conda_deps(args.platform, args.type)
    deps_pip = locked.get_pip_deps(args.platform, args.type)

    out = make_env_string(args.name, deps_conda, deps_pip)
    with open(args.filename, "w") as f:
        f.write(out)


def parse_args(argv):
    program = dict(
        description="""Tool for handling SimNIBS environments.""",
        epilog=textwrap.dedent("""
        EXAMPLES
        --------
        Create environment.yml

            python environment.py create runtime -n simnibs-run

        Lock simnibs-run

            python environment.py lock runtime simnibs-run

        which creates environment.lock.yml.
        """),
    )
    filename_args = ["-f", "--filename"]
    filename_kwargs = dict(type=str, help="Environment filename.")

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "type",
        choices=["runtime", "build", "test", "dev"],
        type=str,
        default="runtime",
        help="Environment type.",
    )
    parent_parser.add_argument(
        "-p",
        "--platform",
        choices=["darwin", "linux", "windows"],
        default=platform.system().lower(),
        help="Platform for which to generate environment.",
    )
    parser = argparse.ArgumentParser(
        **program, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    actions = parser.add_subparsers(
        title="commands", required=True, help="Command to run."
    )

    # create
    create = actions.add_parser(
        "create",
        parents=[parent_parser],
        help="Create an environment file of a certain type.",
    )
    create.add_argument(
        "-n",
        "--name",
        type=str,
        default=DEFAULT_ENVIRONMENT_NAME,
        help="Environment name.",
    )
    create.add_argument(
        *filename_args, **filename_kwargs, default=DEFAULT_ENVIRONMENT_FILENAME
    )
    create.set_defaults(func=create_env)

    # lock
    lock = actions.add_parser("lock", parents=[parent_parser], help="lock environment.")
    lock.add_argument(
        "name",
        type=str,
        default=DEFAULT_ENVIRONMENT_NAME,
        help="Name of the conda environment to lock dependencies from.",
    )
    lock.add_argument(*filename_args, **filename_kwargs, default="environment.lock.yml")
    lock.set_defaults(func=lock_env)

    return parser.parse_args(argv[1:])


DEFAULT_PLAT_TAGS = dict(
    darwin="macosx_14_0_arm64", linux="manylinux_2_28_x86_64", windows="win_amd64"
)
# fmt: off
DEPENDENCIES = Dependencies(
    CondaPackage("python", "3.11", comparison="eq"),

    # Runtime
    # =========================================================================
    CondaPackage("freeglut", platforms=["linux", "windows"]),
    CondaPackage("numpy", "2.3", comparison="eq"),
    CondaPackage("jsonschema"),
    CondaPackage("jupyterlab"),
    CondaPackage("libwebp"),
    CondaPackage("matplotlib-base"),
    CondaPackage("mkl","2024", comparison="eq", platforms=["linux", "windows"]),
    CondaPackage("mpfr"),
    CondaPackage("pillow"),
    CondaPackage("pip"),
    CondaPackage("pyopengl", platforms=["darwin", "windows"]),
    # on linux, newer versions include additional libraries which causes
    # compatibility issues with gmsh package
    CondaPackage("pyopengl", "3.1.7", "linux", comparison="leq"),
    CondaPackage("python-mumps"),
    CondaPackage("requests"),
    CondaPackage("tbb", platforms=["linux", "windows"]),
    CondaPackage("vs2015_runtime", platforms="windows"),
    CondaPackage("zlib"),

    # NOTE some packages which depends on numpy have been moved to pip because
    # of compatibility issues with torch on windows. torch is built against mkl
    # as is numpy on pypi but numpy on conda-forge is built against openblas
    # which seems to cause to openmp/openmpi library issues
    PipPackage("h5py"),
    PipPackage("gmsh", "4.14", comparison="eq"),
    PipPackage("nibabel"),
    PipPackage("numba"),
    PipPackage("numpy", "2.3", comparison="eq"),
    PipPackage("pygpc", "0.4", comparison="geq"),
    PipPackage("pyqt5", platforms=["darwin","linux"]),
    PipPackage("PyQt5", platforms="windows"),
    PipPackage("scipy"),
    PipPackage("taichi"),
    IndexURL("torch", "2.6.0+cpu", ["linux", "windows"], url="https://download.pytorch.org/whl/cpu", platform_tags=dict(linux="linux_x86_64", windows="win_amd64")),
    # On MacOS, torch+cpu is simply the default package
    PipPackage("torch", "2.6.0", "darwin", comparison="eq"),

    # Replace when new samseg is released...
    # PyPIPackage("samseg", "0.5a0", comparison="eq"),
    GitHubRelease("samseg", "0.5a0", platform_tags=DEFAULT_PLAT_TAGS, user="oulap", repository="samseg_wheels", release="dev"),

    # used for layer placement and general FreeSurfer compatible IO of surface
    # files
    GitHubRelease("cortech", "0.1", platform_tags=dict(darwin="macosx_11_0_arm64", linux="manylinux_2_24_x86_64.manylinux_2_28_x86_64", windows="win_amd64"), user="simnibs"),
    GitHubRelease("fmm3dpy", "1.0.4", platform_tags=DEFAULT_PLAT_TAGS, user="simnibs"),
    GitHubRelease("petsc4py", "3.22.2", platform_tags=DEFAULT_PLAT_TAGS, user="simnibs"),
    # used for brain surface prediction. brainsynth handles preprocessing,
    # brainnet does the actual prediction
    GitHubCommit("brainsynth", "v0.1", user="simnibs"),
    GitHubCommit("brainnet", "v0.2", user="simnibs"),

    # Build
    # =========================================================================
    CondaPackage("conda-pack", tag="build"),
    CondaPackage("jinja2", tag="build"),
    CondaPackage("cgal-cpp", "5.5", comparison="eq", tag="build"),
    CondaPackage("libboost-devel", "1.83", comparison="leq", tag="build"),
    CondaPackage("packaging", tag="build"),
    CondaPackage("setuptools-scm", tag="build"),
    CondaPackage("tbb-devel", platforms=["linux", "windows"], tag="build"),

    PipPackage("cython", tag="build"),
    PipPackage("build", tag="build"),

    # Test
    # =========================================================================
    CondaPackage("mock", tag="test"),
    CondaPackage("pytest"),  # runtime dep: postinstall uses pytest for smoke test
)
# fmt: on

if __name__ == "__main__":
    args = parse_args(sys.argv)
    args.func(args)
