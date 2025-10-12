# Contributing to duckdb-python

## General Guidelines

### **Did you find a bug?**

* **Ensure the bug was not already reported** by searching on GitHub under [Issues](https://github.com/duckdb/duckdb-python/issues).
* If you're unable to find an open issue addressing the problem, [open a new one](https://github.com/duckdb/duckdb-python/issues/new/choose). Be sure to include a **title and clear description**, as much relevant information as possible, and a **code sample** or an **executable test case** demonstrating the expected behavior that is not occurring.

### **Did you write a patch that fixes a bug?**

* Great!
* If possible, add a unit test case to make sure the issue does not occur again.
* Open a new GitHub pull request with the patch.
* Ensure the PR description clearly describes the problem and solution. Include the relevant issue number if applicable.

### Outside Contributors

* Discuss your intended changes with the core team on Github
* Announce that you are working or want to work on a specific issue
* Avoid large pull requests - they are much less likely to be merged as they are incredibly hard to review

### Pull Requests

* Do not commit/push directly to the main branch. Instead, create a fork and file a pull request.
* When maintaining a branch, merge frequently with the main.
* When maintaining a branch, submit pull requests to the main frequently.
* If you are working on a bigger issue try to split it up into several smaller issues.
* Please do not open "Draft" pull requests. Rather, use issues or discussion topics to discuss whatever needs discussing.
* We reserve full and final discretion over whether or not we will merge a pull request. Adhering to these guidelines is not a complete guarantee that your pull request will be merged.

### CI for pull requests

* Pull requests will need to pass all continuous integration checks before merging.
* For faster iteration and more control, consider running CI on your own fork or when possible directly locally.
* Submitting changes to an open pull request will move it to 'draft' state.
* Pull requests will get a complete run on the main repo CI only when marked as 'ready for review' (via Web UI, button on bottom right).

### Testing cross-platform and cross-Python

* On your fork you can [run](https://docs.github.com/en/actions/using-workflows/manually-running-a-workflow#running-a-workflow) the Packaging workflow manually for any branch. You can choose whether to build for all platforms or a subset, and to either run the full testsuite, the fast tests only, or no tests at all.

## Setting up a development environment

See the [instructions on duckdb.org](https://duckdb.org/docs/stable/dev/building/python).
