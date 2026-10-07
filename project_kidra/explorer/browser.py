import curses

from project_kidra.explorer.renderer import ResultRenderer

_BACKSPACE_KEYS = (curses.KEY_BACKSPACE, 127, 8)
_QUIT_KEYS = (27, 4)  # Esc, Ctrl-D


class Browser:
    """Interactive terminal UI that re-renders results on every keystroke."""

    def __init__(self, renderer: ResultRenderer) -> None:
        """Create a browser that displays output from `renderer`."""

        self._renderer = renderer

    def run(self) -> None:
        """Run the UI until the user presses Esc or Ctrl-D."""

        curses.wrapper(self._loop)

    def _loop(self, screen: "curses.window") -> None:
        """Read keys and redraw until quit."""

        query = ""
        while True:
            screen.erase()
            height, width = screen.getmaxyx()
            screen.addnstr(0, 0, f"> {query}", width - 1)
            for row, line in enumerate(self._renderer.render(query), start=2):
                if row >= height:
                    break
                screen.addnstr(row, 0, line, width - 1)
            screen.addnstr(height - 1, 0, "Esc/Ctrl-D to quit", width - 1)
            screen.move(0, min(2 + len(query), width - 1))
            screen.refresh()

            key = screen.get_wch()
            if key in _QUIT_KEYS or key == "\x1b":
                return
            if key in _BACKSPACE_KEYS or key == "\x7f":
                query = query[:-1]
            elif isinstance(key, str) and key.isprintable():
                query += key
