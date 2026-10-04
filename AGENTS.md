# Agents Documentation

## Setup Commands
- Install dependencies & Build: `./tools/build.sh`
- Run unit tests: `./tools/test.sh`
- Run app: `./tools/run.sh`

## Code Style
- Always use strict typing.
- Use object-oriented programming.
- Priorize smaller, modular files. One class per file, (unless they are incredibly tiny)
- Use dependency injection for easier testing.
  - *Especially* when interacting with libraries such as Pytorch, Chroma, or any web apis.
- Add codedocs to all classes and methods.
  - If you notice a class/method without a codedoc, add it.
- When adding codedocs, leave an empty line after to seperate them from the code. Example:
  ```
  def my_function():
    """This function does a thing."""

    variable = 1 + 1
    return variable
  ```
- Add comments **only** when the purpose of certain code is not obvious.

## Testing Style
- Make sure all code is tested where reasonable.
  - Small, obvious functions like `getters` do not need tests.
- *Always* test after making changes to the codebase.
- If you notice a section that is lacking tests, add a few.
- Make sure tests have meaningful descriptions/comments.
- Add or update tests for the code you change, even if nobody asked.