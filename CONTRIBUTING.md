# Contributing to WindWatts

We welcome contributions! Please follow these guidelines to ensure a smooth process.

## Getting Started

See the [Quickstart Guide](docs/02-quickstart.md) to set up your development environment.

## Pull Request Process

1.  Create a new branch for your feature or fix.
2.  Make your changes.
3.  Run tests and linters locally.
4.  Open a Pull Request against the `develop` branch.
5.  Describe your changes clearly in the PR description.

## Branch Naming

We recommend using descriptive branch names:

- `feature/my-new-feature`
- `fix/bug-description`
- `docs/documentation-update`

## Standards

- **Commits**: Use clear, descriptive commit messages. We encourage [Conventional Commits](https://www.conventionalcommits.org/).
  - `feat: add new map layer`
  - `fix: resolve api timeout`
  - `docs: update quickstart guide`
- **Code Style**: Follow the existing patterns in the codebase.
  - **Frontend**: Prettier & ESLint are configured. Run `yarn format` before committing.
  - **Backend**: PEP 8 compliance is expected.
- **Tests**: Add tests for new features or bug fixes.
