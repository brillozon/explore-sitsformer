Contributing to SITS-Former
===========================

We welcome contributions! This guide will help you get started.

Development Environment
-----------------------

1. Fork the repository on GitHub
2. Clone your fork locally:

.. code-block:: bash

    git clone https://github.com/yourusername/sitsformer.git
    cd sitsformer

3. Install development dependencies:

.. code-block:: bash

    poetry install --extras "dev docs experiment"

4. Set up pre-commit hooks:

.. code-block:: bash

    poetry run pre-commit install

Code Standards
--------------

We follow these coding standards:

* **Black** for code formatting
* **isort** for import sorting
* **flake8** for linting
* **mypy** for type checking
* **pytest** for testing

Run all checks:

.. code-block:: bash

    # Format code
    poetry run black src tests
    poetry run isort src tests

    # Lint code
    poetry run flake8 src tests

    # Type checking
    poetry run mypy src

    # Run tests
    poetry run pytest

Documentation
-------------

* Write clear docstrings using Google style
* Update documentation for new features
* Build docs locally to test:

.. code-block:: bash

    cd docs
    poetry run make html

Testing
-------

* Write tests for new functionality
* Ensure all tests pass before submitting PR
* Aim for good test coverage

.. code-block:: bash

    # Run tests with coverage
    poetry run pytest --cov=sitsformer

Pull Request Process
-------------------

1. Create a feature branch from main
2. Make your changes
3. Run all code quality checks
4. Update documentation if needed
5. Submit a pull request

Commit Messages
---------------

Use clear, descriptive commit messages:

* feat: add new feature
* fix: bug fix
* docs: documentation changes
* style: formatting changes
* refactor: code refactoring
* test: add or update tests