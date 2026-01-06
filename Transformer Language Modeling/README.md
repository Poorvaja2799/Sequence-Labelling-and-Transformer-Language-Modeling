# CS7650 Fall 2025 Project 3

For a full description of the assignment, see the assignment handout at `docs/project3-doc.pdf`

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!). After installing uv, run the following command:

```sh
uv sync
```

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).


### How to submit

Once you finish the project, run
```
bash make_submission.sh <your_gt_usernamme>
```
to create a zip folder `<your_gt_usernamme>_submission.zip`. It should contain:

• `<your_gt_usernamme>_proj3.pdf`: Use the template slides provided (report.pptx) to
write the report. After that, convert it to the PDF format and change the file name.

• `project3_src`

• `tests/adapters.py`

• `completions.json`: your model outputs