# Contributing

Typed-dfs is licensed under the
[Apache License, version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
New issues and pull requests are welcome.
Feel free to direct a question to the authors by creating an [issue with the _question_ tag](https://github.com/dmyersturnbull/tyrannosaurus/issues/new?assignees=&labels=kind%3A+question&template=question.md).
Contributors are asked to abide by both the [GitHub community guidelines](https://docs.github.com/en/site-policy/github-terms/github-community-guidelines)
and the [Contributor Code of Conduct, version 2.0](https://www.contributor-covenant.org/version/2/0/code_of_conduct/).

#### Pull requests

Please update `CHANGELOG.md` and add your name to the contributors in `pyproject.toml`
so that you’re credited. Run `poetry lock` and `tyrannosaurus sync` to sync metadata.
Feel free to make a draft pull request and solicit feedback from the authors.

#### Publishing a new version

1. Bump the version in `tool.poetry.version` in `pyproject.toml`, following
   [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
2. Run `tyrannosaurus sync` so that the Poetry lock file is up-to-date
   and metadata are synced to pyproject.toml.
3. Create a [new release](https://github.com/dmyersturnbull/typed-dfs/releases/new)
   with both the name and tag set to something like `v1.4.13` (keep the _v_).
4. An hour later, check that the _publish on release creation_
   [workflow](https://github.com/dmyersturnbull/typed-dfs/actions) passes
   and that the PyPi, Docker Hub, and GitHub Package versions are updated as shown in the
   shields on the readme.
