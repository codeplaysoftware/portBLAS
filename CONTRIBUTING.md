# Contributing to SYCL BLAS

## Pull requests

We really appreciate all pull requests to SYCL BLAS, but we also have a few
guidelines in place to help keep the project consistent and reduce the noise in
commits, pull requests and code. Hopefully these guidelines are clear and easy
to follow, if not then feel free to add an issue to address this.

Before starting any development please make an issue to communicate your intent
and ensure that you are not duplicating work.

### Formatting

Please ensure that any contributions are formatted using clang-format with the
Google style. This helps to ensure consistency across the project and also
prevents any unhelpful comments in code reviews about how code should be
formatted.

To do this run: `clang-format --style=Google -i <filename>` on any C++ files
changed in your contribution.

### Pull request guidelines

* Submit any pull request to the `master` branch, unless you are really sure
  you need to push your changes elsewhere.

* Unless your pull request is trivial, expect comments and suggestions on your
  pull request. We will provide feedback as soon as possible.

* We cannot accept or test any pull request with merge conflicts, so please fix
  these before submitting your pull request.

* Please ensure pull requests are small and focussed. This means that they
  should only address a single feature, change or bug fix. Larger pull requests
  trying to fix multiple things at once should be split into smaller pull
  requests.

* All tests must pass before a pull request is accepted.

* If your pull request adds a new feature, please make sure that you add new
  test cases to cover this feature.

### Commit guidelines

* In your commit messages, the first line should be a short summary of the
  changes which is no more than 50 characters. Following this should be an
  empty line, then the remaining commit message explaining the changes in more
  detail.

* Use the present tense to say what the commit changes.

* Check for any unnecessary whitespace changes with `git diff --check` before
  committing your changes.
