import curses
import os
import sys
import subprocess
import pty
import re
import select
import glob
import time
from importlib.util import find_spec
import textwrap
import shutil
from pathlib import Path
import json
import csv

import yaml

from data_generation.configuration import load_config, resolve_config
from data_generation.export.cat_csv import default_scores_dir, find_scores, list_sequences, next_step
from data_generation.dataset_creation.preflight import validate_processed_input, validate_raw_input
from data_generation.download import download_grandtour, download_wayfast, download_tartandrive, download_models as dl_models

def format_size(size_bytes) -> str:
    """
    Format a byte count as GB, or TB from 1000 GB upward.

    :param size_bytes: int | float - size in bytes
    :return: str - size with one decimal in GB, two in TB
    """
    size_gb = size_bytes / (1024**3)
    if size_gb >= 1000:
        return f"{size_gb/1024:.2f} TB"
    return f"{size_gb:.1f} GB"

def get_viewport(current_row, num_options, max_display) -> tuple[int, int]:
    """
    Compute the visible option window, centered on the current row.

    :param current_row: int - highlighted option index
    :param num_options: int - total options
    :param max_display: int - maximum visible options
    :return: tuple[int, int] - start (inclusive) and end (exclusive) index
    """
    start_idx = max(0, current_row - max_display // 2)
    end_idx = min(num_options, start_idx + max_display)
    if end_idx - start_idx < max_display:
        start_idx = max(0, end_idx - max_display)
    return start_idx, end_idx

MENU_START_Y = 2
MENU_ASCII_ART = [
    r"   ____    _  _____   ___ ____   ___  ____    ____   ___ ____   __   ",
    r"  / ___|  / \|_   _| |_ _|  _ \ / _ \/ ___|  |___ \ / _ \___ \ / /_  ",
    r" | |     / _ \ | |    | || |_) | | | \___ \    __) | | | |__) | '_ \ ",
    r" | |___ / ___ \| |    | ||  _ <| |_| |___) |  / __/| |_| / __/| (_) |",
    r"  \____/_/   \_\_|   |___|_| \_\\___/|____/  |_____|\___/_____|\___/ "
]

def get_menu_layout(h, options, current_row, description, dynamic_desc_callback, total_size_str) -> tuple[list[str], int, int, int]:
    """
    Compute description lines, first option row and visible range of a menu frame.

    :param h: int - screen height in rows
    :param options: list[str] - option labels
    :param current_row: int - highlighted option index
    :param description: str | list[str] - static description
    :param dynamic_desc_callback: Callable[[str], str | list[str]] | None - per-option description override
    :param total_size_str: str - summary line above the options
    :return: tuple[list[str], int, int, int] - description lines, first option row, visible start (inclusive) and end (exclusive) index
    """

    current_desc = description
    if dynamic_desc_callback:
        current_desc = dynamic_desc_callback(options[current_row])
    if not current_desc:
        current_desc = []
    elif isinstance(current_desc, str):
        current_desc = [current_desc]

    desc_offset = len(current_desc) + 3 if current_desc else 0
    if total_size_str:
        desc_offset += 2

    options_start_y = MENU_START_Y + len(MENU_ASCII_ART) + 4 + desc_offset
    max_display = max(3, h - options_start_y - 4)
    start_idx, end_idx = get_viewport(current_row, len(options), max_display)
    return current_desc, options_start_y, start_idx, end_idx
def draw_menu(stdscr, title, options, current_row, description="", dynamic_desc_callback=None, multi_select=False, selected_items=None, total_size_str="", error_msg="", deletable=False) -> tuple[int, int, int]:
    """
    Draw the menu: banner, description, options and footer.

    :param stdscr: curses.window - screen
    :param title: str - menu title
    :param options: list[str] - option labels, empty string = blank separator
    :param current_row: int - highlighted option index
    :param description: str | list[str] - description text, "**" toggles bold
    :param dynamic_desc_callback: Callable[[str], str | list[str]] | None - per-option description override
    :param multi_select: bool - draw checkboxes
    :param selected_items: set[str] | None - checked options in multi-select mode
    :param total_size_str: str - summary line above the options
    :param error_msg: str - error line near the bottom
    :param deletable: bool - footer advertises the delete key
    :return: tuple[int, int, int] - first option row and visible option index range
    """
    stdscr.erase()
    h, w = stdscr.getmaxyx()

    try:
        start_y = MENU_START_Y
        for i, line in enumerate(MENU_ASCII_ART):
            x = w//2 - len(line)//2
            stdscr.attron(curses.color_pair(1) | curses.A_BOLD)
            stdscr.addstr(start_y + i, max(0, x), line)
            stdscr.attroff(curses.color_pair(1) | curses.A_BOLD)

        subtitle = "--- Main Menu ---" if title == "Main Menu - CAT IROS 2026" else f"--- {title} ---"
        title_x = w//2 - len(subtitle)//2
        stdscr.attron(curses.color_pair(1))
        stdscr.addstr(start_y + len(MENU_ASCII_ART) + 1, max(0, title_x), subtitle)
        stdscr.attroff(curses.color_pair(1))

        current_desc, options_start_y, start_idx, end_idx = get_menu_layout(h, options, current_row, description, dynamic_desc_callback, total_size_str)

        if current_desc:
            max_len = max(len(line.replace("**", "")) for line in current_desc)
            box_width = max(80, max_len + 4)

            top_border = "╭" + "─" * box_width + "╮"
            bottom_border = "╰" + "─" * box_width + "╯"

            stdscr.attron(curses.color_pair(3))

            border_x = w//2 - len(top_border)//2
            stdscr.addstr(start_y + len(MENU_ASCII_ART) + 3, max(0, border_x), top_border)

            for i, line in enumerate(current_desc):
                raw_line = line.replace("**", "")
                pad_total = box_width - 4 - len(raw_line)
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left

                formatted_line = f"│  {' ' * pad_left}{line}{' ' * pad_right}  │"

                current_x = border_x
                parts = formatted_line.split("**")
                for j, part in enumerate(parts):
                    if j % 2 == 1:
                        stdscr.attron(curses.A_BOLD)
                    stdscr.addstr(start_y + len(MENU_ASCII_ART) + 4 + i, max(0, current_x), part)
                    if j % 2 == 1:
                        stdscr.attroff(curses.A_BOLD)
                    current_x += len(part)

            stdscr.addstr(start_y + len(MENU_ASCII_ART) + 4 + len(current_desc), max(0, border_x), bottom_border)
            stdscr.attroff(curses.color_pair(3))

        if total_size_str:
            size_x = w//2 - len(total_size_str)//2
            stdscr.attron(curses.color_pair(4) | curses.A_BOLD)
            stdscr.addstr(options_start_y - 2, max(0, size_x), total_size_str)
            stdscr.attroff(curses.color_pair(4) | curses.A_BOLD)

        button_width = min(80, max(40, int(w * 0.6)))

        for i, idx in enumerate(range(start_idx, end_idx)):
            option = options[idx]
            x = w//2 - button_width//2
            y = options_start_y + i

            if option == "":
                continue

            if idx == current_row:
                stdscr.attron(curses.color_pair(2))
            else:
                stdscr.attron(curses.color_pair(3))

            display_text = option

            if multi_select:
                if option == "Select All":
                    valid_opts = [o for o in options if o != "" and o != "Select All"]
                    all_selected = len(selected_items) == len(valid_opts) and len(valid_opts) > 0
                    box = "[*] " if all_selected else "[ ] "
                    display_text = box + display_text
                else:
                    box = "[X] " if selected_items and option in selected_items else "[ ] "
                    display_text = box + display_text

            tag = None
            if "  [RAW" in display_text:
                m = re.search(r"  \[RAW.*?\]", display_text)
                if m:
                    tag = m.group(0)
                    color_norm = curses.color_pair(4) | curses.A_BOLD
                    color_sel = curses.color_pair(14) | curses.A_BOLD
                    display_text = display_text.replace(tag, "")
            elif "  [PROCESSED]" in display_text:
                tag = "  [PROCESSED]"
                color_norm = curses.color_pair(7) | curses.A_BOLD
                color_sel = curses.color_pair(15) | curses.A_BOLD
                display_text = display_text.replace(tag, "")
            elif "  [READY FOR USE" in display_text:
                m = re.search(r"  \[READY FOR USE.*?\]", display_text)
                if m:
                    tag = m.group(0)
                    color_norm = curses.color_pair(6) | curses.A_BOLD
                    color_sel = curses.color_pair(16) | curses.A_BOLD
                    display_text = display_text.replace(tag, "")

            tag_len = len(tag) if tag else 0
            if len(display_text) + tag_len > button_width - 4:
                display_text = display_text[:button_width - 7 - tag_len] + "..."

            text = f" {display_text}{tag if tag else ''} "
            padded_text = text.center(button_width)

            if tag:
                parts = padded_text.split(tag)
                stdscr.addstr(y, max(0, x), parts[0])
                if idx == current_row:
                    stdscr.attroff(curses.color_pair(2))
                    stdscr.attron(color_sel)
                    stdscr.addstr(tag)
                    stdscr.attroff(color_sel)
                    stdscr.attron(curses.color_pair(2))
                else:
                    stdscr.attroff(curses.color_pair(3))
                    stdscr.attron(color_norm)
                    stdscr.addstr(tag)
                    stdscr.attroff(color_norm)
                    stdscr.attron(curses.color_pair(3))
                stdscr.addstr(parts[1])
            else:
                stdscr.addstr(y, max(0, x), padded_text)

            if idx == current_row:
                stdscr.attroff(curses.color_pair(2))
            else:
                stdscr.attroff(curses.color_pair(3))

        if start_idx > 0:
            stdscr.addstr(options_start_y - 1, w//2 - 1, "▲", curses.color_pair(1) | curses.A_BOLD)
        if end_idx < len(options):
            stdscr.addstr(options_start_y + (end_idx - start_idx), w//2 - 1, "▼", curses.color_pair(1) | curses.A_BOLD)

        if multi_select:
            instructions_left = "Arrows/Mouse: Move | SPACE: Toggle | ENTER: Confirm"
        else:
            instructions_left = "Arrows/Mouse: Move | ENTER: Select"

        if deletable:
            instructions_right = "D: Delete | ESC: Go Back / Exit"
        else:
            instructions_right = "ESC: Go Back / Exit"

        # avoid out-of-bounds footer writes on narrow terminals
        padding = 5
        if h > 2 and w > len(instructions_left) + len(instructions_right) + padding * 2:
            stdscr.addstr(h-2, padding, instructions_left)
            right_x = w - len(instructions_right) - padding
            stdscr.addstr(h-2, right_x, instructions_right)

        if error_msg and h > 3:
            err_x = w//2 - len(error_msg)//2
            stdscr.attron(curses.color_pair(5) | curses.A_BOLD)
            stdscr.addstr(h-3, max(0, err_x), error_msg)
            stdscr.attroff(curses.color_pair(5) | curses.A_BOLD)

    except curses.error:
        pass

    return options_start_y, start_idx, end_idx

def run_menu(stdscr, title, options, description="", dynamic_desc_callback=None, multi_select=False, total_size_callback=None, deletable=False) -> str | list[str] | tuple[str, str] | None:
    """
    Block on a menu until the user chooses, cancels or requests deletion.

    :param stdscr: curses.window - screen
    :param title: str - menu title
    :param options: list[str] - option labels, empty string = blank separator
    :param description: str | list[str] - static description
    :param dynamic_desc_callback: Callable[[str], str | list[str]] | None - per-option description
    :param multi_select: bool - toggle several options with SPACE
    :param total_size_callback: Callable[[set[str]], str] | None - builds the total-size line
    :param deletable: bool - D requests deletion of the highlighted option
    :return: str | list[str] | tuple[str, str] | None - choice, selection, ("__DELETE__", label) delete request, or on ESC [] in multi-select mode and None otherwise
    """
    curses.curs_set(0)
    # keep escape responsive while still letting arrow-key escape sequences arrive
    curses.set_escdelay(35)
    curses.mouseinterval(0)
    curses.mousemask(curses.BUTTON1_PRESSED)

    if multi_select and "Select All" not in options:
        options = ["Select All", ""] + options

    selected_items = set()

    try:
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_CYAN, -1)
        curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_BLUE)
        curses.init_pair(3, curses.COLOR_WHITE, -1)
        curses.init_pair(4, curses.COLOR_YELLOW, -1)
        curses.init_pair(5, curses.COLOR_RED, -1)
        curses.init_pair(6, curses.COLOR_GREEN, -1)
        curses.init_pair(7, curses.COLOR_BLUE, -1)
        curses.init_pair(14, curses.COLOR_YELLOW, curses.COLOR_BLUE)
        curses.init_pair(15, curses.COLOR_CYAN, curses.COLOR_BLUE)
        curses.init_pair(16, curses.COLOR_GREEN, curses.COLOR_BLUE)

        current_row = 0
        error_msg = ""

        while True:
            total_size_str = ""
            if multi_select and total_size_callback and len(selected_items) > 1:
                total_size_str = total_size_callback(selected_items)
            options_start_y, start_idx, end_idx = draw_menu(stdscr, title, options, current_row, description, dynamic_desc_callback, multi_select, selected_items, total_size_str, error_msg, deletable=deletable)

            key = stdscr.getch()
            error_msg = ""

            if key == curses.KEY_UP and current_row > 0:
                current_row -= 1
                if current_row >= 0 and options[current_row] == "":
                    current_row = max(0, current_row - 1)
            elif key == curses.KEY_DOWN and current_row < len(options) - 1:
                current_row += 1
                if current_row < len(options) and options[current_row] == "":
                    current_row = min(len(options) - 1, current_row + 1)
            elif key == ord(' ') and multi_select:
                if options[current_row] == "Select All":
                    valid_opts = [o for o in options if o != "" and o != "Select All"]
                    if len(selected_items) == len(valid_opts):
                        selected_items.clear()
                    else:
                        selected_items.update(valid_opts)
                elif options[current_row] != "":
                    if options[current_row] in selected_items:
                        selected_items.remove(options[current_row])
                    else:
                        selected_items.add(options[current_row])
            elif key in [ord('d'), ord('D')] and deletable:
                if options[current_row] != "" and options[current_row] != "Select All":
                    return ("__DELETE__", options[current_row])
            elif key == curses.KEY_ENTER or key in [10, 13]:
                if multi_select:
                    if not selected_items:
                        error_msg = "Error: Please select files with SPACE before continuing."
                    else:
                        return [opt for opt in options if opt in selected_items]
                else:
                    if options[current_row] != "":
                        return options[current_row]
            elif key == 27:
                return [] if multi_select else None
            elif key == curses.KEY_RESIZE:
                pass
            elif key == curses.KEY_MOUSE:
                try:
                    _, mx, my, _, bstate = curses.getmouse()
                    _, w = stdscr.getmaxyx()
                    button_width = min(80, max(40, int(w * 0.6)))
                    x_start = w//2 - button_width//2

                    for i, idx in enumerate(range(start_idx, end_idx)):
                        y = options_start_y + i
                        if my == y and x_start <= mx < x_start + button_width:
                            current_row = idx
                            if bstate & curses.BUTTON1_PRESSED or bstate & curses.BUTTON1_CLICKED:
                                if multi_select:
                                    if options[current_row] == "Select All":
                                        valid_opts = [o for o in options if o != "" and o != "Select All"]
                                        if len(selected_items) == len(valid_opts):
                                            selected_items.clear()
                                        else:
                                            selected_items.update(valid_opts)
                                    elif options[current_row] != "":
                                        if options[current_row] in selected_items:
                                            selected_items.remove(options[current_row])
                                        else:
                                            selected_items.add(options[current_row])
                                else:
                                    if options[current_row] != "":
                                        return options[current_row]
                            break
                except curses.error:
                    pass
    finally:
        curses.mousemask(0)

def draw_progress_ui(stdscr, title, percent, status_text, footer_text=None, info_lines=None) -> None:
    """
    Draw a status page with an optional progress bar; title "error" draws it in red.

    :param stdscr: curses.window - screen
    :param title: str - page title
    :param percent: float | None - completion 0-100, None = no bar
    :param status_text: str - message below the bar
    :param footer_text: str | None - text near the bottom edge
    :param info_lines: list[str] | None - lines between the title and the bar
    """
    stdscr.clear()
    h, w = stdscr.getmaxyx()

    try:
        ASCII_ART = MENU_ASCII_ART

        y_offset = 2
        if info_lines:
            y_offset = 1

        is_error = title.lower() == "error"
        color_idx = 5 if is_error else 1

        curses.init_pair(5, curses.COLOR_RED, -1)

        start_y = 2
        for i, line in enumerate(ASCII_ART):
            x = w//2 - len(line)//2
            stdscr.attron(curses.color_pair(color_idx) | curses.A_BOLD)
            stdscr.addstr(start_y + i, max(0, x), line)
            stdscr.attroff(curses.color_pair(color_idx) | curses.A_BOLD)

        subtitle = f"--- {title} ---"
        title_x = w//2 - len(subtitle)//2
        stdscr.attron(curses.color_pair(color_idx))
        stdscr.addstr(start_y + len(ASCII_ART) + 1, max(0, title_x), subtitle)
        stdscr.attroff(curses.color_pair(color_idx))

        y_offset = start_y + len(ASCII_ART) + 3
        if info_lines:
            for i, line in enumerate(info_lines):
                stdscr.attron(curses.color_pair(3))
                stdscr.addstr(y_offset + i, max(0, w//2 - len(line)//2), line)
                stdscr.attroff(curses.color_pair(3))
            y_offset += len(info_lines) + 1

        bar_y = max(y_offset + 1, start_y + len(ASCII_ART) + 6)

        if percent is not None:
            bar_width = min(60, max(20, w - 20))
            filled = int(bar_width * percent / 100)
            bar = "█" * filled + "░" * (bar_width - filled)

            bar_x = w//2 - (bar_width + 6)//2

            bar_color = 5 if is_error else 2

            stdscr.attron(curses.color_pair(bar_color))
            stdscr.addstr(bar_y, max(0, bar_x), f"{percent:3.0f}% ")
            stdscr.attroff(curses.color_pair(bar_color))

            if is_error:
                stdscr.attron(curses.color_pair(5))
                stdscr.addstr(bar_y, max(0, bar_x + 5), bar)
                stdscr.attroff(curses.color_pair(5))
            else:
                stdscr.addstr(bar_y, max(0, bar_x + 5), bar)

            status_y = bar_y + 2
        else:
            status_y = bar_y

        if is_error:
            stdscr.attron(curses.color_pair(5) | curses.A_BOLD)
            stdscr.addstr(status_y, max(0, w//2 - len(status_text)//2), status_text)
            stdscr.attroff(curses.color_pair(5) | curses.A_BOLD)
        else:
            stdscr.addstr(status_y, max(0, w//2 - len(status_text)//2), status_text)

        if footer_text:
            stdscr.attron(curses.color_pair(4))
            stdscr.addstr(h - 2, max(0, w//2 - len(footer_text)//2), footer_text)
            stdscr.attroff(curses.color_pair(4))

    except curses.error:
        pass

    stdscr.refresh()

def confirm_action_ui(stdscr, title, message) -> bool:
    """
    Block on a yes/no question.

    :param stdscr: curses.window - screen
    :param title: str - page title
    :param message: str - question
    :return: bool - True for Y, False for N or ESC
    """
    draw_progress_ui(stdscr, title, None, message)
    curses.flushinp()
    while True:
        c = stdscr.getch()
        if c in [ord('y'), ord('Y')]:
            return True
        elif c in [ord('n'), ord('N'), 27]:
            return False

def show_input_warning(stdscr, message, title="Dataset input warning") -> None:
    """
    Show a scrollable, word-wrapped message until ENTER or ESC.

    :param stdscr: curses.window - screen
    :param message: object - message, converted with str
    :param title: str - heading
    """
    offset = 0
    curses.flushinp()
    while True:
        stdscr.clear()
        height, width = stdscr.getmaxyx()
        lines = []
        for paragraph in str(message).splitlines():
            lines.extend(textwrap.wrap(paragraph, max(1, width - 4)) or [""])
        visible = max(1, height - 5)
        offset = min(offset, max(0, len(lines) - visible))
        stdscr.addnstr(0, 0, title, max(0, width - 1), curses.A_BOLD)
        for index, line in enumerate(lines[offset:offset + visible], start=2):
            if index < height - 2:
                stdscr.addnstr(index, 1, line, max(0, width - 2))
        stdscr.addnstr(height - 1, 0, "Up/Down: scroll | ENTER/ESC: return", max(0, width - 1))
        stdscr.refresh()
        key = stdscr.getch()
        if key in (10, 13, 27):
            return
        if key == curses.KEY_DOWN:
            offset = min(offset + 1, max(0, len(lines) - visible))
        elif key == curses.KEY_UP:
            offset = max(0, offset - 1)

def check_processed_selection(stdscr, selected_datasets, require_masks=False, config=None) -> bool:
    """
    Validate selected processed datasets; warn on the first invalid one.

    :param stdscr: curses.window - screen for the warning
    :param selected_datasets: list[str] - menu labels, text before the first double space = folder under dataset_processed
    :param require_masks: bool - datasets must already contain masks
    :param config: dict | None - generation configuration
    :return: bool - True if all are valid
    """
    config = resolve_config(config)
    try:
        for selected in selected_datasets:
            validate_processed_input(
                os.path.join(config["bag_processing"]["output_dataset_dir"], selected.split("  ")[0]),
                require_masks=require_masks, config=config,
            )
    except ValueError as exc:
        show_input_warning(stdscr, exc)
        return False
    return True

def run_process_with_ui(stdscr, cmd, title, info_lines=None, progress_dir=None, total_frames=0, startup_status=None, config=None) -> bool:
    """
    Run a subprocess with a progress screen; ESC terminates it.

    :param stdscr: curses.window - screen
    :param cmd: list[str] - command and arguments
    :param title: str - progress page title
    :param info_lines: list[str] | None - lines above the progress bar
    :param progress_dir: str | None - directory whose generated TIFF masks track progress; masks already present at start are not counted as new
    :param total_frames: int - expected frames, for the percentage
    :param startup_status: str | None - status shown until the first output
    :param config: dict | None - generation configuration
    :return: bool - True if exit code 0 and not cancelled
    """
    mask_prefix = resolve_config(config)["data_format"]["positive_mask_prefix"]
    master, slave = pty.openpty()
    p = subprocess.Popen(cmd, stdout=slave, stderr=slave, close_fds=True)
    os.close(slave)

    footer_text = None

    pattern = re.compile(r"Processed frame (\d+)/(\d+)")

    buf = ""
    last_lines = []
    start_time = time.time()
    last_startup_update = 0
    received_output = False
    cancelled = False

    initial_processed = 0
    if progress_dir and os.path.exists(progress_dir):
        initial_processed = len([f for f in os.listdir(progress_dir) if f.startswith(mask_prefix) and f.endswith(".tif")])
    progress_start_time = None

    stdscr.nodelay(True)
    try:
        while True:
            c = stdscr.getch()
            if c == 27:
                p.terminate()
                try: p.wait(timeout=1)
                except subprocess.TimeoutExpired: p.kill()
                draw_progress_ui(stdscr, title, None, "Process interrupted by user.", footer_text, info_lines)
                time.sleep(1)
                cancelled = True
                break

            rlist, _, _ = select.select([master], [], [], 0.1)

            if startup_status and not received_output:
                elapsed = int(time.time() - start_time)
                if elapsed >= last_startup_update:
                    draw_progress_ui(stdscr, title, None, f"{startup_status} ({elapsed}s)", footer_text, info_lines)
                    last_startup_update = elapsed + 1
            elif progress_dir and os.path.exists(progress_dir) and total_frames > 0:
                try:
                    processed = len([f for f in os.listdir(progress_dir) if f.startswith(mask_prefix) and f.endswith(".tif")])
                    new_processed = max(0, processed - initial_processed)
                    percent = (processed / total_frames) * 100 if total_frames > 0 else 0

                    if new_processed and progress_start_time is None:
                        progress_start_time = time.time()
                    elapsed = time.time() - progress_start_time if progress_start_time is not None else 0
                    fps = new_processed / elapsed if elapsed > 0 else 0

                    status = f"[{processed}/{total_frames} | {fps:.1f} FPS | {new_processed} new]"
                    draw_progress_ui(stdscr, title, percent, status[:120], footer_text, info_lines)
                except Exception:
                    pass

            if master in rlist:
                try:
                    data = os.read(master, 1024).decode('utf-8', errors='replace')
                except OSError:
                    break
                if not data:
                    break
                received_output = True

                for char in data:
                    if char in ['\r', '\n']:
                        if not progress_dir:
                            if buf:
                                m = pattern.search(buf)
                                if m:
                                    current = int(m.group(1))
                                    total = int(m.group(2))
                                    percent = (current / total) * 100 if total > 0 else 0
                                    draw_progress_ui(stdscr, title, percent, buf.strip(), footer_text, info_lines)
                                else:
                                    draw_progress_ui(stdscr, title, 0, buf.strip()[:80], footer_text, info_lines)

                        if buf:
                            last_lines.append(buf.strip())
                            if len(last_lines) > 30:
                                last_lines.pop(0)

                        buf = ""
                    else:
                        buf += char
            elif p.poll() is not None:
                break
    finally:
        stdscr.nodelay(False)

    if buf:
        last_lines.append(buf.strip())

    p.wait()
    os.close(master)

    if p.returncode != 0 and not cancelled:
        error_msg = "\n".join(last_lines) if last_lines else "The process produced no diagnostic output."
        show_input_warning(
            stdscr, f"{title} failed (exit code {p.returncode}).\n{error_msg}",
            title="Processing failed",
        )

    return not cancelled and p.returncode == 0

def session_config_ui(stdscr, session_cfg) -> bool:
    """
    Show the Grounding SAM session config screen; edits apply in place, even on ESC.

    :param stdscr: curses.window - screen
    :param session_cfg: dict[str, bool | int | float] - parameters by name, modified in place
    :return: bool - True for Save & Start, False for ESC
    """
    current_row = 0
    keys = list(session_cfg.keys())

    while True:
        stdscr.clear()
        h, w = stdscr.getmaxyx()

        title = "--- Live Session Configs ---"
        stdscr.attron(curses.color_pair(1) | curses.A_BOLD)
        stdscr.addstr(2, max(0, w//2 - len(title)//2), title)
        stdscr.attroff(curses.color_pair(1) | curses.A_BOLD)

        info_text = "Before starting the segmentation, please configure the inference parameters for this session."
        stdscr.addstr(3, max(0, w//2 - len(info_text)//2), info_text)

        desc = "Use LEFT/RIGHT arrows (< >) to change values. ENTER on 'Save & Start' to begin, or ESC to abort."
        stdscr.addstr(5, max(0, w//2 - len(desc)//2), desc)

        for i, key in enumerate(keys):
            val = session_cfg[key]

            if isinstance(val, bool):
                display_val = "< True >" if val else "< False >"
            elif isinstance(val, float):
                display_val = f"< {val:.2f} >"
            elif isinstance(val, int):
                display_val = f"< {val} >"
            else:
                display_val = f"< {val} >"

            text = f"{key.replace('_', ' ').title()}: {display_val}"

            x = w//2 - len(text)//2
            y = 8 + i * 2

            if i == current_row:
                stdscr.attron(curses.color_pair(2))
                stdscr.addstr(y, max(0, x), text)
                stdscr.attroff(curses.color_pair(2))
            else:
                stdscr.addstr(y, max(0, x), text)

        ret_text = "Save & Start"
        if current_row == len(keys):
            stdscr.attron(curses.color_pair(2))
            stdscr.addstr(8 + len(keys)*2 + 2, max(0, w//2 - len(ret_text)//2), ret_text)
            stdscr.attroff(curses.color_pair(2))
        else:
            stdscr.addstr(8 + len(keys)*2 + 2, max(0, w//2 - len(ret_text)//2), ret_text)

        explanations = {
            'box_threshold': "Box Threshold: Minimum confidence (0-1) to detect an object's bounding box.",
            'text_threshold': "Text Threshold: Minimum confidence (0-1) that the detected object matches your text prompts.",
            'step_positive': "Step Positive: Frame interval to re-trigger Grounding DINO to refresh tracking.",
            'batch_size': "Batch Size: Number of frames to load into GPU memory simultaneously.",
            'human_in_the_loop': "HIL: Pause and ask for human interaction when the AI is uncertain.",
            'resume_processing': "Resume: Continue at the first frame missing its generated mask."
        }

        if current_row < len(keys):
            k = keys[current_row]
            exp_text = explanations.get(k, "")
            stdscr.attron(curses.color_pair(4))
            stdscr.addstr(h - 4, max(0, w//2 - len(exp_text)//2), exp_text)
            stdscr.attroff(curses.color_pair(4))

        stdscr.refresh()

        c = stdscr.getch()

        if c == curses.KEY_UP and current_row > 0:
            current_row -= 1
        elif c == curses.KEY_DOWN and current_row < len(keys):
            current_row += 1
        elif c == curses.KEY_LEFT and current_row < len(keys):
            k = keys[current_row]
            if isinstance(session_cfg[k], bool):
                session_cfg[k] = not session_cfg[k]
            elif isinstance(session_cfg[k], float):
                session_cfg[k] = max(0.0, round(session_cfg[k] - 0.01, 2))
            elif isinstance(session_cfg[k], int):
                session_cfg[k] = max(1, session_cfg[k] - 10)
        elif c == curses.KEY_RIGHT and current_row < len(keys):
            k = keys[current_row]
            if isinstance(session_cfg[k], bool):
                session_cfg[k] = not session_cfg[k]
            elif isinstance(session_cfg[k], float):
                session_cfg[k] = min(1.0, round(session_cfg[k] + 0.01, 2))
            elif isinstance(session_cfg[k], int):
                session_cfg[k] = min(5000, session_cfg[k] + 10)
        elif c in [10, 13]:
            if current_row == len(keys):
                break
        elif c == 27:
            return False

    return True

def dataset_config_ui(stdscr, dataset_cfg) -> bool:
    """
    Show the dataset processing config screen; edits apply in place, even on ESC.

    :param stdscr: curses.window - screen
    :param dataset_cfg: dict[str, bool | int | float | str] - parameters by name, modified in place
    :return: bool - True for Save & Start Processing, False for ESC
    """
    current_row = 0
    keys = list(dataset_cfg.keys())

    while True:
        stdscr.clear()
        h, w = stdscr.getmaxyx()

        title = "--- Dataset Processing Configs ---"
        stdscr.attron(curses.color_pair(1) | curses.A_BOLD)
        stdscr.addstr(2, max(0, w//2 - len(title)//2), title)
        stdscr.attroff(curses.color_pair(1) | curses.A_BOLD)

        desc = "Use LEFT/RIGHT arrows (< >) to change values. ENTER on 'Save & Start Processing' to begin."
        stdscr.addstr(4, max(0, w//2 - len(desc)//2), desc)

        for i, key in enumerate(keys):
            val = dataset_cfg[key]

            if isinstance(val, bool):
                display_val = "< True >" if val else "< False >"
            elif isinstance(val, int):
                display_val = f"< {val} >"
            elif isinstance(val, str):
                display_val = f"< {val} >"
            else:
                display_val = f"< {val} >"

            if key == 'ts_robot_width':
                label = "TerraSentia Robot Width"
            elif key == 'anymal_robot_width':
                label = "ANYmal-D Robot Width"
            else:
                label = key.replace('_', ' ').title()
            text = f"{label}: {display_val}"

            x = w//2 - len(text)//2
            y = 7 + i * 2

            if i == current_row:
                stdscr.attron(curses.color_pair(2))
                stdscr.addstr(y, max(0, x), text)
                stdscr.attroff(curses.color_pair(2))
            else:
                stdscr.addstr(y, max(0, x), text)

        ret_text = "Save & Start Processing"
        if current_row == len(keys):
            stdscr.attron(curses.color_pair(2))
            stdscr.addstr(7 + len(keys)*2 + 2, max(0, w//2 - len(ret_text)//2), ret_text)
            stdscr.attroff(curses.color_pair(2))
        else:
            stdscr.addstr(7 + len(keys)*2 + 2, max(0, w//2 - len(ret_text)//2), ret_text)

        explanations = {
            'image_width': "Image Width: Resize raw image width (e.g. 480, 640, 1024).",
            'image_height': "Image Height: Resize raw image height.",
            'remove_reverse': "Remove Reverse: Automatically drop frames where robot's velocity is negative.",
            'trajectory_shape': "Trajectory Shape: 'point' (scattered dots) or 'polygon' (filled path).",
            'tartan_robot_width': "Tartan Robot Width: Average width (meters) of Tartan ATV. Default: 1.80",
            'ts_robot_width': "TerraSentia Robot Width: Average width (meters). Default: 0.30",
            'anymal_robot_width': "ANYmal-D Robot Width: Average width (meters) of ANYmal-D. Default: 0.50"
        }

        if current_row < len(keys):
            k = keys[current_row]
            exp_text = explanations.get(k, "")
            stdscr.attron(curses.color_pair(4))
            stdscr.addstr(h - 4, max(0, w//2 - len(exp_text)//2), exp_text)
            stdscr.attroff(curses.color_pair(4))

        stdscr.refresh()

        c = stdscr.getch()

        if c == curses.KEY_UP and current_row > 0:
            current_row -= 1
        elif c == curses.KEY_DOWN and current_row < len(keys):
            current_row += 1
        elif c == curses.KEY_LEFT and current_row < len(keys):
            k = keys[current_row]
            if isinstance(dataset_cfg[k], bool):
                dataset_cfg[k] = not dataset_cfg[k]
            elif isinstance(dataset_cfg[k], int):
                dataset_cfg[k] = max(100, dataset_cfg[k] - 10)
            elif isinstance(dataset_cfg[k], float):
                dataset_cfg[k] = max(0.1, round(dataset_cfg[k] - 0.1, 2))
            elif k == 'trajectory_shape':
                dataset_cfg[k] = "polygon" if dataset_cfg[k] == "point" else "point"
        elif c == curses.KEY_RIGHT and current_row < len(keys):
            k = keys[current_row]
            if isinstance(dataset_cfg[k], bool):
                dataset_cfg[k] = not dataset_cfg[k]
            elif isinstance(dataset_cfg[k], int):
                dataset_cfg[k] = min(2000, dataset_cfg[k] + 10)
            elif isinstance(dataset_cfg[k], float):
                dataset_cfg[k] = min(3.0, round(dataset_cfg[k] + 0.1, 2))
            elif k == 'trajectory_shape':
                dataset_cfg[k] = "polygon" if dataset_cfg[k] == "point" else "point"
        elif c in [10, 13]:
            if current_row == len(keys):
                break
        elif c == 27:
            return False

    return True

def segformer_config_ui(stdscr, seg_cfg) -> bool:
    """
    Show the Segformer training config screen; edits apply in place, even on ESC.

    :param stdscr: curses.window - screen
    :param seg_cfg: dict[str, bool | int | float] - parameters by name, modified in place
    :return: bool - True for Save & Continue, False for ESC
    """
    current_row = 0
    keys = list(seg_cfg.keys())

    while True:
        stdscr.clear()
        h, w = stdscr.getmaxyx()

        title = "--- Segformer Training Configs ---"
        stdscr.attron(curses.color_pair(1) | curses.A_BOLD)
        stdscr.addstr(2, max(0, w//2 - len(title)//2), title)
        stdscr.attroff(curses.color_pair(1) | curses.A_BOLD)

        desc = "Use LEFT/RIGHT arrows (< >) to change values. ENTER on 'Save & Continue' to proceed."
        stdscr.addstr(4, max(0, w//2 - len(desc)//2), desc)

        for i, key in enumerate(keys):
            val = seg_cfg[key]

            if isinstance(val, bool):
                display_val = "< True >" if val else "< False >"
            elif isinstance(val, int):
                display_val = f"< {val} >"
            elif isinstance(val, float):
                if key == 'learning_rate':
                    display_val = f"< {val:.1e} >"
                else:
                    display_val = f"< {val:.4f} >"
            else:
                display_val = f"< {val} >"

            text = f"{key.replace('_', ' ').title()}: {display_val}"

            x = w//2 - len(text)//2
            y = 7 + i * 2

            if i == current_row:
                stdscr.attron(curses.color_pair(2))
                stdscr.addstr(y, max(0, x), text)
                stdscr.attroff(curses.color_pair(2))
            else:
                stdscr.addstr(y, max(0, x), text)

        ret_text = "Save & Continue"
        if current_row == len(keys):
            stdscr.attron(curses.color_pair(2))
            stdscr.addstr(7 + len(keys)*2 + 2, max(0, w//2 - len(ret_text)//2), ret_text)
            stdscr.attroff(curses.color_pair(2))
        else:
            stdscr.addstr(7 + len(keys)*2 + 2, max(0, w//2 - len(ret_text)//2), ret_text)

        explanations = {
            'total_epochs': "Total Epochs: Number of full passes through the dataset.",
            'batch_size': "Batch Size: Number of images processed at once (Memory dependent).",
            'learning_rate': "Learning Rate: Optimization step size (e.g., 6.0e-05).",
            'early_stopping': "Early Stopping: Stop training if validation loss doesn't improve.",
            'patience': "Patience: Number of epochs to wait without improvement before stopping."
        }

        if current_row < len(keys):
            k = keys[current_row]
            exp_text = explanations.get(k, "")
            stdscr.attron(curses.color_pair(4))
            stdscr.addstr(h - 4, max(0, w//2 - len(exp_text)//2), exp_text)
            stdscr.attroff(curses.color_pair(4))

        stdscr.refresh()

        c = stdscr.getch()

        if c == curses.KEY_UP and current_row > 0:
            current_row -= 1
        elif c == curses.KEY_DOWN and current_row < len(keys):
            current_row += 1
        elif c == curses.KEY_LEFT and current_row < len(keys):
            k = keys[current_row]
            if isinstance(seg_cfg[k], bool):
                seg_cfg[k] = not seg_cfg[k]
            elif isinstance(seg_cfg[k], int):
                if k == 'total_epochs':
                    seg_cfg[k] = max(1, seg_cfg[k] - 5)
                elif k == 'batch_size':
                    seg_cfg[k] = max(1, seg_cfg[k] - 2)
                elif k == 'patience':
                    seg_cfg[k] = max(1, seg_cfg[k] - 1)
            elif isinstance(seg_cfg[k], float):
                if k == 'learning_rate':
                    seg_cfg[k] = max(1e-6, seg_cfg[k] / 2)
        elif c == curses.KEY_RIGHT and current_row < len(keys):
            k = keys[current_row]
            if isinstance(seg_cfg[k], bool):
                seg_cfg[k] = not seg_cfg[k]
            elif isinstance(seg_cfg[k], int):
                if k == 'total_epochs':
                    seg_cfg[k] = min(500, seg_cfg[k] + 5)
                elif k == 'batch_size':
                    seg_cfg[k] = min(128, seg_cfg[k] + 2)
                elif k == 'patience':
                    seg_cfg[k] = min(50, seg_cfg[k] + 1)
            elif isinstance(seg_cfg[k], float):
                if k == 'learning_rate':
                    seg_cfg[k] = min(1e-2, seg_cfg[k] * 2)
        elif c in [10, 13]:
            if current_row == len(keys):
                break
        elif c == 27:
            return False

    return True

def main(stdscr) -> None:
    """
    Run the main menu loop and dispatch to sub-screens.

    :param stdscr: curses.window - screen
    """
    pipeline_config = load_config()
    processed_root = pipeline_config["bag_processing"]["output_dataset_dir"]
    main_title = "Main Menu - CAT IROS 2026"
    main_options = [
        "Download Datasets",
        "Process Raw Dataset",
        "Grounding SAM Segmentation",
        "Segformer Segmentation",
        "Export CAT Dataset",
        "",
        "Manage Local Datasets",
        "Data Generation Workflow (Mind Map)",
        "Dataset Folder Structure (Docs)",
        "Status Tags Explanation (Docs)",
    ]

    next_menu = None

    session_cfg = {
        "box_threshold": pipeline_config["grounding_dino"]["box_threshold"],
        "text_threshold": pipeline_config["grounding_dino"]["text_threshold"],
        "step_positive": pipeline_config["grounding_dino"]["step_positive"],
        "batch_size": pipeline_config["processing"]["batch_size"],
        "human_in_the_loop": pipeline_config["processing"]["human_in_the_loop"],
        "resume_processing": True,
    }
    segformer_cfg = {
        "total_epochs": pipeline_config["segformer"]["total_epochs"],
        "batch_size": pipeline_config["segformer"]["batch_size"],
        "learning_rate": pipeline_config["segformer"]["learning_rate"],
        "early_stopping": pipeline_config["segformer"]["early_stopping"],
        "patience": pipeline_config["segformer"]["patience"],
    }

    while True:
        if next_menu:
            selected_main = next_menu
            next_menu = None
        else:
            main_desc = [
                "Welcome to the **CAT** Data Generation Pipeline.",
                "🌍 ** Website:** https://capability-aware-traversability.github.io/  |  📜 ** Paper:** https://arxiv.org/pdf/2607.20679",
                "",
                "This tool centralizes the workflow to download raw off-road robotics datasets",
                "(like **TartanDrive**, **TerraSentia**, and **Grandtour**) and process them through advanced",
                "perception models (**Grounding SAM** & **Segformer**) to generate semantic training data."
            ]
            selected_main = run_menu(stdscr, main_title, main_options, description=main_desc)

        if not selected_main:
            break

        if selected_main == "Manage Local Datasets":

            while True:
                local_options = []
                local_info = {}

                td_dir = os.path.join("dataset_raw", "tartandrive", "bags")
                if os.path.exists(td_dir):
                    for d in sorted(os.listdir(td_dir)):
                        if os.path.isdir(os.path.join(td_dir, d)) and len(os.listdir(os.path.join(td_dir, d))) > 0:
                            size = sum(os.path.getsize(os.path.join(dp, f)) for dp, dn, filenames in os.walk(os.path.join(td_dir, d)) for f in filenames)
                            name = f"{d}  [RAW TartanDrive]"
                            local_options.append(name)
                            local_info[name] = {"type": "Raw TartanDrive", "size": size, "path": os.path.join(td_dir, d)}

                gt_dir = os.path.join("dataset_raw", "grandtour")
                if os.path.exists(gt_dir):
                    for d in sorted(os.listdir(gt_dir)):
                        if os.path.isdir(os.path.join(gt_dir, d)) and len(os.listdir(os.path.join(gt_dir, d))) > 0:
                            size = sum(os.path.getsize(os.path.join(dp, f)) for dp, dn, filenames in os.walk(os.path.join(gt_dir, d)) for f in filenames)
                            name = f"{d}  [RAW Grandtour]"
                            local_options.append(name)
                            local_info[name] = {"type": "Raw Grandtour", "size": size, "path": os.path.join(gt_dir, d)}

                wf_dir = os.path.join("dataset_raw", "wayfaster", "bags")
                if os.path.isdir(wf_dir):
                    for bag in sorted(Path(wf_dir).rglob("*.bag")):
                        name = f"{bag.relative_to(wf_dir)}  [RAW WayFASTER]"
                        local_options.append(name)
                        local_info[name] = {"type": "Raw WayFASTER", "size": bag.stat().st_size, "path": str(bag)}

                processed_dir = processed_root
                if os.path.exists(processed_dir):
                    for d in sorted(os.listdir(processed_dir)):
                        p = os.path.join(processed_dir, d)
                        if d.startswith("dataset_") and os.path.isdir(p) and len(os.listdir(p)) > 0:
                            size = sum(os.path.getsize(os.path.join(dp, f)) for dp, dn, filenames in os.walk(p) for f in filenames)

                            tag = "  [PROCESSED]"
                            is_sam = os.path.exists(os.path.join(p, "sam_completed.txt"))
                            is_seg = os.path.exists(os.path.join(p, "segformer_completed.txt"))
                            if is_sam and is_seg:
                                tag = "  [READY FOR USE (GDINO & SEGFORMER)]"
                            elif is_sam:
                                tag = "  [READY FOR USE (GDINO)]"
                            elif is_seg:
                                tag = "  [READY FOR USE (SEGFORMER)]"

                            name = f"{d}{tag}"
                            local_options.append(name)
                            local_info[name] = {"type": "Processed Dataset", "size": size, "path": p}

                if not local_options:
                    stdscr.clear()
                    draw_progress_ui(stdscr, "Manage Local Datasets", None, "No downloaded datasets found. Press ENTER.")
                    curses.flushinp()
                    while True:
                        if stdscr.getch() in [10, 13]: break
                    break

                def local_desc(opt) -> list[str]:
                    """
                    Build description lines for a local dataset option.

                    :param opt: str - menu label
                    :return: list[str] - name, type, size, path and delete hint lines
                    """
                    info = local_info.get(opt, {})
                    return [
                        f"**{opt.split('  [')[0]}**",
                        f"Type: {info.get('type', 'Unknown')}",
                        f"Size: **{format_size(info.get('size', 0))}**",
                        f"Path: {info.get('path', 'Unknown')}",
                        "",
                        "Press D to delete this dataset."
                    ]

                sel = run_menu(
                    stdscr,
                    "Manage Local Datasets",
                    local_options,
                    description="View and manage all downloaded datasets.",
                    dynamic_desc_callback=local_desc,
                    deletable=True
                )

                if isinstance(sel, tuple) and sel[0] == "__DELETE__":
                    info = local_info.get(sel[1], {})
                    path = info.get("path")
                    if path and os.path.exists(path):
                        if confirm_action_ui(stdscr, "Confirm Deletion", f"Are you sure you want to PERMANENTLY delete\n'{sel[1]}'? (y/n)"):
                            if os.path.isdir(path):
                                shutil.rmtree(path)
                            else:
                                os.remove(path)
                    continue
                else:
                    break
            continue

        elif selected_main == "Download Datasets":
            dl_title = "Download Datasets"
            dl_options = ["TartanDrive (Yamaha Viking ATV)", "WayFASTER (TerraSentia)", "Grandtour (ANYmal-D)"]

            dl_desc = "Select a **specific robot dataset** to download to the local storage."
            selected_dl = run_menu(stdscr, dl_title, dl_options, description=dl_desc)

            if not selected_dl:
                continue

            if selected_dl == "Grandtour (ANYmal-D)":
                gt_metadata = {}
                meta_path = "assets/metadata/grandtour_metadata.json"
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, 'r') as mf:
                            gt_metadata = json.load(mf)
                    except: pass

                gt_missions = list(gt_metadata.keys())
                gt_missions_raw = list(gt_metadata.keys())
                gt_missions_raw.sort(reverse=True)

                if not gt_missions_raw:
                    stdscr.clear()
                    draw_progress_ui(stdscr, "Error", None, "No missions found in grandtour_metadata.json. Press ENTER.")
                    curses.flushinp()
                    while True:
                        if stdscr.getch() in [10, 13]: break
                    next_menu = "Download Datasets"
                    continue

                gt_missions = []
                for m in gt_missions_raw:
                    target = os.path.join("dataset_raw", "grandtour", m)
                    if os.path.isfile(os.path.join(target, ".download_complete")):
                        gt_missions.append(f"{m}  [Installed]")
                    else:
                        gt_missions.append(m)

                stdscr.clear()

                def get_gt_desc(mission_name) -> list[str]:
                    """
                    Build description lines for a Grandtour mission option.

                    :param mission_name: str - menu label, optional installed tag
                    :return: list[str] - name, size, files and platform lines
                    """
                    raw_name = mission_name.replace("  [Installed]", "")
                    size_bytes = gt_metadata.get(raw_name, 0)
                    size_str = format_size(size_bytes)

                    return [
                        f"**Mission: {raw_name}**",
                        f"Total Size: **{size_str}**",
                        "Files: .yaml, .zgroup, .tar (extracted automatically)",
                        "Platform: ANYmal-D Quadruped"
                    ]
                def gt_total_size(selected_items) -> str:
                    """
                    Build the total-size label for selected Grandtour missions.

                    :param selected_items: set[str] - selected menu labels
                    :return: str - label with the summed size
                    """
                    total_bytes = 0
                    for item in selected_items:
                        raw_name = item.replace("  [Installed]", "")
                        total_bytes += gt_metadata.get(raw_name, 0)
                    return f"Total Selected Size: {format_size(total_bytes)}"

                selected_missions = run_menu(stdscr, "Grandtour Downloads", gt_missions, dynamic_desc_callback=get_gt_desc, multi_select=True, total_size_callback=gt_total_size)
                if not selected_missions:
                    continue

                try:

                    for selected_mission in selected_missions:
                        def ui_callback(percent, text) -> None:
                            """
                            Draw download progress for the current Grandtour mission.

                            :param percent: float - completion 0-100
                            :param text: str - status text
                            """
                            selected_raw = selected_mission.replace("  [Installed]", "")
                            display_mission = selected_raw[:30] + "..." if len(selected_raw) > 30 else selected_raw
                            draw_progress_ui(stdscr, f"Downloading {display_mission}", percent, text)

                        selected_raw = selected_mission.replace("  [Installed]", "")
                        download_grandtour.download_grandtour(selected_raw, ui_callback)

                    draw_progress_ui(stdscr, "Grandtour Dataset", None, "Download and extraction complete! Press ENTER.")
                except Exception as e:
                    draw_progress_ui(stdscr, "Error", None, f"Error: {e}. Press ENTER to continue.")

                curses.flushinp()
                while True:
                    key = stdscr.getch()
                    if key in [10, 13]:
                        break

                stdscr.clear()
                stdscr.refresh()
                next_menu = "Download Datasets"
                continue
            elif selected_dl == "WayFASTER (TerraSentia)":
                archive_error = "No session ZIPs found on Hugging Face."

                try:
                    stdscr.clear()
                    def archive_progress(percent, text) -> None:
                        """
                        Draw progress while listing Hugging Face sessions.

                        :param percent: float - completion 0-100
                        :param text: str - status text
                        """
                        draw_progress_ui(stdscr, "WayFAST/WayFASTER Hugging Face sessions", percent, text)
                    curses.flushinp()
                    wf_sizes = download_wayfast.list_datasets(archive_progress, pipeline_config)
                    wf_options_raw = list(wf_sizes)
                except Exception as e:
                    wf_options_raw = []
                    wf_sizes = {}
                    archive_error = str(e)

                if not wf_options_raw:
                    stdscr.clear()
                    draw_progress_ui(stdscr, "Error", None, f"{archive_error} Press ENTER.")
                    curses.flushinp()
                    while True:
                        if stdscr.getch() in [10, 13]: break
                    next_menu = "Download Datasets"
                    continue

                wf_options = []
                for opt in wf_options_raw:
                    target = download_wayfast.dataset_path(opt, pipeline_config)
                    if (target / "data.csv").is_file():
                        wf_options.append(f"{opt}  [Installed]")
                    else:
                        wf_options.append(opt)

                def get_wf_desc(opt) -> list[str]:
                    """
                    Build description lines for a WayFASTER session option.

                    :param opt: str - menu label, optional installed tag
                    :return: list[str] - name, size, files, platform and download note lines
                    """
                    raw_opt = opt.replace("  [Installed]", "")
                    robot = "TerraSentia"
                    size_bytes = wf_sizes.get(raw_opt, 0)
                    size_str = format_size(size_bytes) if size_bytes > 0 else "Unknown"

                    return [
                        f"**{raw_opt}**",
                        f"Size: **{size_str}**",
                        "Files: processed session ZIP (RGB, depth, metadata and available masks)",
                        f"Platform: {robot}",
                        "",
                        "Downloads only the selected sessions from Hugging Face.",
                        "Extracts into dataset_processed/; no raw bag processing required."
                    ]

                def wf_total_size(selected_items) -> str:
                    """
                    Build the total-size label for selected WayFASTER sessions.

                    :param selected_items: set[str] - selected menu labels
                    :return: str - label with the summed size, empty if unknown
                    """
                    total_bytes = sum(wf_sizes.get(item.replace("  [Installed]", ""), 0) for item in selected_items)
                    return f"Total Selected Size: {format_size(total_bytes)}" if total_bytes > 0 else ""

                stdscr.clear()
                selected_wfs = run_menu(stdscr, "TerraSentia Downloads", wf_options, dynamic_desc_callback=get_wf_desc, multi_select=True, total_size_callback=wf_total_size)
                if not selected_wfs:
                    continue

                try:
                    for selected_wf in selected_wfs:
                        def ui_callback(percent, text) -> None:
                            """
                            Draw download progress for the current WayFASTER session.

                            :param percent: float - completion 0-100
                            :param text: str - status text
                            """
                            selected_raw = selected_wf.replace("  [Installed]", "")
                            display_name = selected_raw[:30] + "..." if len(selected_raw) > 30 else selected_raw
                            draw_progress_ui(stdscr, f"Downloading {display_name}", percent, text)

                        selected_raw = selected_wf.replace("  [Installed]", "")
                        download_wayfast.download_wayfast(selected_raw, ui_callback, pipeline_config)

                    draw_progress_ui(stdscr, "WayFAST/WayFASTER Dataset", None, "Selected sessions extracted to dataset_processed/. Ready for segmentation. Press ENTER.")
                except Exception as e:
                    draw_progress_ui(stdscr, "Error", None, f"Error: {e}. Press ENTER to continue.")

                curses.flushinp()
                while True:
                    key = stdscr.getch()
                    if key in [10, 13]:
                        break

                stdscr.clear()
                stdscr.refresh()
                next_menu = "Download Datasets"
                continue
            elif selected_dl == "TartanDrive (Yamaha Viking ATV)":
                yaml_path = os.path.join("assets", "metadata", "_tartandrive.yaml")
                try:
                    with open(yaml_path, 'r') as f:
                        content = f.read()

                    tartan_bags_raw = re.findall(r'^  kitti/all_topics/([^/]+)/:', content, re.MULTILINE)

                    if not tartan_bags_raw:
                        raise ValueError("No bags found in YAML.")

                    tartan_bags = []
                    for b in tartan_bags_raw:
                        target = os.path.join("dataset_raw", "tartandrive", "bags", b)
                        if os.path.isfile(os.path.join(target, ".download_complete")):
                            tartan_bags.append(f"{b}  [Installed]")
                        else:
                            tartan_bags.append(b)

                    stdscr.clear()

                    tartan_metadata = {}
                    meta_path = "assets/metadata/tartan_metadata.json"
                    if os.path.exists(meta_path):
                        try:
                            with open(meta_path, 'r') as mf:
                                tartan_metadata = json.load(mf)
                        except: pass

                    def get_tartan_desc(bag_name) -> list[str]:
                        """
                        Build description lines for a TartanDrive bag option.

                        :param bag_name: str - menu label, optional installed tag
                        :return: list[str] - name, size, duration, top speed, environment and keyword lines
                        """
                        raw_bag = bag_name.replace("  [Installed]", "")
                        meta = tartan_metadata.get(raw_bag, {})
                        size_bytes = meta.get("size_bytes", 0)
                        size_str = format_size(size_bytes)
                        dur = meta.get("duration", "Unknown")
                        if isinstance(dur, float): dur = f"{dur:.1f}"
                        spd = meta.get("top_speed", "Unknown")
                        if isinstance(spd, float): spd = f"{spd:.1f}"
                        env = meta.get("environment", "Unknown")

                        parts = raw_bag.split('_')
                        keywords = " ".join(parts[1:]) if len(parts) > 1 else "None"

                        return [
                            f"**{raw_bag}**",
                            f"Size: **{size_str}** | Duration: {dur}s | Top Speed: {spd} m/s",
                            f"Environment: **{env}**",
                            f"Keywords: {keywords}"
                        ]
                    def tartan_total_size(selected_items) -> str:
                        """
                        Build the total-size label for selected TartanDrive bags.

                        :param selected_items: set[str] - selected menu labels
                        :return: str - label with the summed size
                        """
                        total_bytes = 0
                        for item in selected_items:
                            raw_bag = item.replace("  [Installed]", "")
                            total_bytes += tartan_metadata.get(raw_bag, {}).get("size_bytes", 0)
                        return f"Total Selected Size: {format_size(total_bytes)}"

                    selected_bags = run_menu(stdscr, "TartanDrive Downloads", tartan_bags, dynamic_desc_callback=get_tartan_desc, multi_select=True, total_size_callback=tartan_total_size)
                    if not selected_bags:
                        continue


                    for selected_bag in selected_bags:
                        def ui_callback(percent, text) -> None:
                            """
                            Draw download progress for the current TartanDrive bag.

                            :param percent: float - completion 0-100
                            :param text: str - status text
                            """
                            selected_raw = selected_bag.replace("  [Installed]", "")
                            display_bag = selected_raw[:30] + "..." if len(selected_raw) > 30 else selected_raw
                            draw_progress_ui(stdscr, f"Downloading {display_bag}", percent, text)

                        selected_raw = selected_bag.replace("  [Installed]", "")
                        download_tartandrive.download_tartandrive(selected_raw, ui_callback, pipeline_config)

                    draw_progress_ui(stdscr, "TartanDrive Dataset", None, "Download complete! Press ENTER.")

                except Exception as e:
                    stdscr.clear()
                    draw_progress_ui(stdscr, "Error", None, f"Error: {e}. Press ENTER to continue.")

                curses.flushinp()
                while True:
                    key = stdscr.getch()
                    if key in [10, 13]:
                        break

                stdscr.clear()
                stdscr.refresh()
                next_menu = "Download Datasets"
                continue

        elif selected_main == "Process Raw Dataset":
            process_options = []

            td_dir = os.path.join("dataset_raw", "tartandrive", "bags")
            has_td = False
            if os.path.exists(td_dir):
                for d in sorted(os.listdir(td_dir)):
                    if os.path.isdir(os.path.join(td_dir, d)) and len(os.listdir(os.path.join(td_dir, d))) > 0:
                        is_ready = False
                        is_processed = os.path.exists(os.path.join(processed_root, f"dataset_tartan_{d}", "processing_completed.txt"))
                        tag = "  [READY FOR USE]" if is_ready else ("  [PROCESSED]" if is_processed else "  [RAW]")
                        process_options.append(f"{d} (Tartandrive){tag}")
                        has_td = True

            gt_dir = os.path.join("dataset_raw", "grandtour")
            has_gt = False
            if os.path.exists(gt_dir):
                for d in sorted(os.listdir(gt_dir)):
                    if os.path.isdir(os.path.join(gt_dir, d)) and len(os.listdir(os.path.join(gt_dir, d))) > 0:
                        if has_td and not has_gt:
                            process_options.append("")
                        is_ready = False
                        is_processed = os.path.exists(os.path.join(processed_root, f"dataset_grandtour_{d}", "processing_completed.txt"))
                        tag = "  [READY FOR USE]" if is_ready else ("  [PROCESSED]" if is_processed else "  [RAW]")
                        process_options.append(f"{d} (Grandtour){tag}")
                        has_gt = True

            wf_dir = os.path.join("dataset_raw", "wayfaster", "bags")
            if os.path.isdir(wf_dir):
                for bag in sorted(Path(wf_dir).rglob("*.bag")):
                    bag_name = bag.stem
                    marker = os.path.join(processed_root, f"dataset_{bag_name}", "processing_completed.txt")
                    tag = "[PROCESSED]" if os.path.isfile(marker) else "[RAW]"
                    process_options.append(f"{bag.relative_to(wf_dir)} (WayFASTER)  {tag}")
            if not process_options:
                stdscr.clear()
                draw_progress_ui(stdscr, "No raw datasets", None, "No raw recordings found. HF WayFAST sessions are already processed.", footer_text="Use Manage Local Datasets or segmentation. Press ENTER.")
                curses.flushinp()
                while True:
                    if stdscr.getch() in [10, 13]: break
                continue

            process_desc = [
                "Select **raw recordings** to extract images and trajectories.",
                "**HF WayFAST/WayFASTER sessions are already processed.**",
                "Use Manage Local Datasets or segmentation for those sessions.",
            ]
            selected_processes = run_menu(stdscr, "Process raw dataset", process_options, description=process_desc, multi_select=True, deletable=True)

            if isinstance(selected_processes, tuple) and selected_processes[0] == "__DELETE__":
                item_to_delete = selected_processes[1]
                if confirm_action_ui(stdscr, "Confirm Deletion", f"Are you sure you want to PERMANENTLY delete\n'{item_to_delete}'? (y/n)"):
                    if "(Tartandrive)" in item_to_delete:
                        raw_name = item_to_delete.split(" (Tartandrive)")[0]
                        target = os.path.join("dataset_raw", "tartandrive", "bags", raw_name)
                        if os.path.exists(target): shutil.rmtree(target)
                    elif "(Grandtour)" in item_to_delete:
                        raw_name = item_to_delete.split(" (Grandtour)")[0]
                        target = os.path.join("dataset_raw", "grandtour", raw_name)
                        if os.path.exists(target): shutil.rmtree(target)
                    elif "(WayFASTER)" in item_to_delete:
                        os.remove(os.path.join("dataset_raw", "wayfaster", "bags", item_to_delete.split(" (WayFASTER)")[0]))
                next_menu = "Process Raw Dataset"
                continue

            if not selected_processes:
                continue

            try:
                input_config = pipeline_config
                if (not isinstance(input_config, dict)
                        or not isinstance(input_config.get("bag_processing"), dict)):
                    raise ValueError("configs/config.yaml must define a bag_processing mapping.")
                for proc in selected_processes:
                    if "(Tartandrive)" in proc:
                        kind = "Tartandrive"
                        raw_name = proc.split(" (Tartandrive)")[0]
                        path = os.path.join("dataset_raw", "tartandrive", "bags", raw_name)
                    elif "(Grandtour)" in proc:
                        kind = "Grandtour"
                        raw_name = proc.split(" (Grandtour)")[0]
                        path = os.path.join("dataset_raw", "grandtour", raw_name)
                    else:
                        kind = "WayFASTER"
                        raw_name = proc.split(" (WayFASTER)")[0]
                        path = os.path.join("dataset_raw", "wayfaster", "bags", raw_name)
                    validate_raw_input(kind, path, input_config)
            except (ValueError, OSError, yaml.YAMLError) as exc:
                show_input_warning(stdscr, f"No processing started.\n{exc}")
                continue

            dataset_cfg = {
                'image_width': 640,
                'image_height': 480,
                'remove_reverse': True,
                'trajectory_shape': "polygon",
                'tartan_robot_width': 1.80,
                'ts_robot_width': pipeline_config["bag_processing"]["robot"]["width"],
                'anymal_robot_width': 0.50
            }

            if not dataset_config_ui(stdscr, dataset_cfg):
                continue

            os.makedirs(processed_root, exist_ok=True)
            cancel_process = False
            for proc in selected_processes:
                cfg_args = [
                    "--width", str(dataset_cfg['image_width']),
                    "--height", str(dataset_cfg['image_height']),
                    "--remove_reverse", "true" if dataset_cfg['remove_reverse'] else "false",
                    "--traj_shape", dataset_cfg['trajectory_shape']
                ]

                success = True
                if "(Tartandrive)" in proc:
                    raw_name = proc.split(" (Tartandrive)")[0]
                    d_path = os.path.join("dataset_raw", "tartandrive", "bags", raw_name)
                    cmd = [sys.executable, "-u", "-m", "data_generation.dataset_creation.create_dataset_tartan", "--dataset", d_path, "--robot_width", str(dataset_cfg['tartan_robot_width'])] + cfg_args
                    success = run_process_with_ui(stdscr, cmd, f"Tartandrive: {raw_name}", config=pipeline_config)
                elif "(Grandtour)" in proc:
                    raw_name = proc.split(" (Grandtour)")[0]
                    d_path = os.path.join("dataset_raw", "grandtour", raw_name)
                    cmd = [sys.executable, "-u", "-m", "data_generation.dataset_creation.create_dataset_grand_tour", "--dataset", d_path, "--robot_width", str(dataset_cfg['anymal_robot_width'])] + cfg_args
                    success = run_process_with_ui(stdscr, cmd, f"Grandtour: {raw_name}", config=pipeline_config)
                elif "(WayFASTER)" in proc:
                    raw_name = proc.split(" (WayFASTER)")[0]
                    cmd = [sys.executable, "-u", "-m", "data_generation.dataset_creation.create_dataset_ts", "--dataset", os.path.join("dataset_raw", "wayfaster", "bags", raw_name), "--robot_width", str(dataset_cfg['ts_robot_width'])] + cfg_args
                    success = run_process_with_ui(stdscr, cmd, f"WayFASTER: {raw_name}", config=pipeline_config)

                if not success:
                    cancel_process = True
                    break

            if cancel_process:
                continue

            next_menu = "Process Raw Dataset"
            continue

        elif selected_main == "Grounding SAM Segmentation":

            if find_spec("torch") is None:
                draw_progress_ui(
                    stdscr,
                    "Missing Model Dependencies",
                    None,
                    "PyTorch is missing. Run 'make setup-models' from data_generation/.",
                    footer_text="Press ENTER to return; restart with 'make run'.",
                )
                curses.flushinp()
                while True:
                    if stdscr.getch() in [10, 13]: break
                continue

            try:

                def ui_callback(percent, text) -> None:
                    """
                    Draw SAM2 model download progress.

                    :param percent: float - completion 0-100
                    :param text: str - status text
                    """
                    draw_progress_ui(stdscr, "Downloading SAM2 Model", percent, text)

                dl_models.download_models(ui_callback)

                draw_progress_ui(stdscr, "Downloading SAM2 Model", None, "Download complete! Starting inference...")
                curses.flushinp()
            except Exception as e:
                stdscr.clear()
                draw_progress_ui(stdscr, "Error", None, f"Download Error: {e}. Press ENTER to continue.")
                while True:
                    if stdscr.getch() in [10, 13]: break
                curses.reset_prog_mode()
                stdscr.clear()
                stdscr.refresh()
                continue

            processed_dir = processed_root
            available_datasets = []
            if os.path.exists(processed_dir):
                for d in sorted(os.listdir(processed_dir)):
                    if os.path.isdir(os.path.join(processed_dir, d)) and d.startswith("dataset_"):
                        rgb_dir = os.path.join(processed_dir, d, "rgb")
                        mask_dir = os.path.join(processed_dir, d, "positive_masks")

                        total = 0
                        if os.path.exists(rgb_dir):
                            total = len([f for f in os.listdir(rgb_dir) if f.startswith("rgb_") and f.endswith(('.jpg', '.png', '.jpeg'))])

                        processed = 0
                        if os.path.exists(mask_dir):
                            processed = len([f for f in os.listdir(mask_dir) if f.startswith(pipeline_config["data_format"]["positive_mask_prefix"]) and f.endswith('.tif')])

                        is_sam_ready = os.path.exists(os.path.join(processed_dir, d, "sam_completed.txt"))
                        is_seg_ready = os.path.exists(os.path.join(processed_dir, d, "segformer_completed.txt"))

                        if is_sam_ready and is_seg_ready:
                            tag = "  [READY FOR USE (GDINO & SEGFORMER)]"
                        elif is_sam_ready:
                            tag = "  [READY FOR USE (GDINO)]"
                        elif is_seg_ready:
                            tag = "  [READY FOR USE (SEGFORMER)]"
                        else:
                            tag = "  [PROCESSED]"

                        available_datasets.append(f"{d}  [{processed}/{total} images]{tag}")

            if not available_datasets:
                stdscr.clear()
                draw_progress_ui(stdscr, "Error", None, "No processed datasets found in dataset_processed/. Press ENTER.")
                while True:
                    if stdscr.getch() in [10, 13]: break
                curses.reset_prog_mode()
                stdscr.clear()
                stdscr.refresh()
                continue

            if not session_config_ui(stdscr, session_cfg):
                continue

            cancel_segmentation = False
            while True:
                selected_datasets = run_menu(
                    stdscr,
                    "Select Datasets for Grounding SAM",
                    available_datasets,
                    description="Select which processed datasets you want to segment.",
                    multi_select=True,
                    deletable=True
                )

                if isinstance(selected_datasets, tuple) and selected_datasets[0] == "__DELETE__":
                    clean_sd = selected_datasets[1].split("  ")[0]
                    target = os.path.join(processed_dir, clean_sd)
                    if confirm_action_ui(stdscr, "Confirm Deletion", f"Are you sure you want to PERMANENTLY delete\n'{clean_sd}'? (y/n)"):
                        if os.path.exists(target): shutil.rmtree(target)
                    next_menu = "Grounding SAM Segmentation"
                    cancel_segmentation = True
                    break

                if not selected_datasets:
                    cancel_segmentation = True
                    break
                if not check_processed_selection(stdscr, selected_datasets, config=pipeline_config):
                    continue

                script_path = "data_generation.segmentation.gdino_sam_segment"
                if find_spec(script_path) is not None:
                    if not os.path.exists("checkpoints/sam2.1_hiera_large.pt"):
                        draw_progress_ui(stdscr, "Missing Models", None, "SAM2 model is downloaded automatically when segmentation starts. Check network access and retry. Press ENTER.")
                        curses.flushinp()
                        while True:
                            if stdscr.getch() in [10, 13]: break
                        continue

                    try:
                        yaml_cfg = pipeline_config
                        sam2_ckpt = os.path.basename(yaml_cfg["sam2"]["checkpoint"])
                        model_id = yaml_cfg["grounding_dino"]["model_id"]
                        active_prompts = yaml_cfg["prompts"]["positive"]
                    except:
                        sam2_ckpt, model_id, active_prompts = "Unknown", "Unknown", []

                    box_w = 60

                    info_lines = [
                        "="*box_w,
                        "🧠 Grounding DINO + SAM 2 Segmentation Pipeline".center(box_w),
                        "-"*box_w,
                        f"• SAM2 Checkpoint : {sam2_ckpt}".ljust(box_w),
                        f"• Grounding DINO  : {model_id}".ljust(box_w),
                    ]

                    prompts_str = ", ".join(active_prompts)
                    wrapped_prompts = textwrap.wrap(prompts_str, width=box_w - 20)
                    if not wrapped_prompts: wrapped_prompts = ["None"]

                    info_lines.append(f"• Target Prompts  : {wrapped_prompts[0]}".ljust(box_w))
                    for p_line in wrapped_prompts[1:]:
                        info_lines.append(f"                    {p_line}".ljust(box_w))

                    info_lines.extend([
                        f"• Box Threshold   : {session_cfg['box_threshold']} | Text Threshold: {session_cfg['text_threshold']}".ljust(box_w),
                        f"• Step Positive   : {session_cfg['step_positive']} | Batch Size: {session_cfg['batch_size']}".ljust(box_w),
                        f"• Resume Option   : {session_cfg['resume_processing']} | HIL: {session_cfg['human_in_the_loop']}".ljust(box_w),
                        "="*box_w
                    ])

                    draw_progress_ui(stdscr, "Grounding SAM Segmentation", 0, "Ready to start? (Loading models takes ~1 minute). Press ENTER to begin. (ESC to cancel)", info_lines=info_lines)
                    curses.flushinp()

                    cancel = False
                    while True:
                        c = stdscr.getch()
                        if c in [10, 13]: break
                        if c == 27:
                            cancel = True
                            break

                    if cancel:
                        continue

                    for sd in selected_datasets:
                        clean_sd = sd.split("  ")[0]
                        ds_path = os.path.join(processed_dir, clean_sd)
                        rgb_dir = os.path.join(ds_path, "rgb")
                        mask_dir = os.path.join(ds_path, "positive_masks")

                        total_frames_ds = 0
                        if os.path.exists(rgb_dir):
                            total_frames_ds = len([f for f in os.listdir(rgb_dir) if f.startswith("rgb_") and f.endswith(('.jpg', '.png', '.jpeg'))])

                        cmd = [
                            sys.executable, "-u", "-m", script_path,
                            "--dataset", ds_path,
                            "--box_threshold", str(session_cfg['box_threshold']),
                            "--text_threshold", str(session_cfg['text_threshold']),
                            "--step_positive", str(session_cfg['step_positive']),
                            "--batch_size", str(session_cfg['batch_size']),
                            "--hil", "true" if session_cfg['human_in_the_loop'] else "false",
                            "--resume", "true" if session_cfg['resume_processing'] else "false"
                        ]

                        success = run_process_with_ui(
                            stdscr,
                            cmd,
                            f"Segmenting: {clean_sd}",
                            info_lines,
                            progress_dir=mask_dir,
                            total_frames=total_frames_ds,
                            startup_status="Loading SAM 2 and Grounding DINO; first run may download weights.",
                            config=pipeline_config,
                        )
                        if not success:
                            cancel_segmentation = True
                            break

                    if cancel_segmentation:
                        break

                    draw_progress_ui(stdscr, "Grounding SAM Segmentation", None, "All segmentations completed. Press ENTER.")
                    curses.flushinp()
                    while True:
                        if stdscr.getch() in [10, 13]: break
                    break
                else:
                    stdscr.clear()
                    draw_progress_ui(stdscr, "Error", None, f"Script not found: {script_path}. Press ENTER.")
                    stdscr.refresh()

            if cancel_segmentation:
                continue

            curses.reset_prog_mode()
            stdscr.clear()
            stdscr.refresh()

        elif selected_main == "Dataset Folder Structure (Docs)":
            docs = [
                r" 📁 dataset_processed/                    ",
                r" └── 📁 dataset_name/                     ",
                r"     ├── 📄 data.csv                      ",
                r"     ├── 📁 rgb/                          ",
                r"     │   └── 📄 rgb_<bag>_000000.jpg      ",
                r"     ├── 📁 depth/                        ",
                r"     │   └── 📄 depth_<bag>_000000.tif    ",
                r"     ├── 📁 positive_masks/               ",
                r"     │   └── 📄 positive_rgb_<bag>_000000.tif",
                r"     ├── 📁 trajectory_masks/             ",
                r"     │   └── 📄 trajectory_<bag>_000000.tif",
                r"     └── 📁 visualization/                ",
                r"         └── 📄 vis_<bag>_000000.jpg      ",
                r"                                          ",
                r" • data.csv         : Synchronized velocity, position, and time.",
                r" • rgb/             : Raw RGB frames extracted from the rosbag.",
                r" • depth/           : Depth maps saved as TIFF files.",
                r" • positive_masks/  : Traversability masks saved as TIFF files.",
                r" • trajectory_masks/: Visual masks of the robot's future path.",
                r" • visualization/   : RGB images with masks overlaid for debugging.",
                r"",
                r"Press ENTER to return to the menu..."
            ]

            stdscr.clear()
            h, w = stdscr.getmaxyx()
            start_y = MENU_START_Y
            for i, line in enumerate(MENU_ASCII_ART):
                x = w//2 - len(line)//2
                stdscr.attron(curses.color_pair(1) | curses.A_BOLD)
                stdscr.addstr(start_y + i, max(0, x), line)
                stdscr.attroff(curses.color_pair(1) | curses.A_BOLD)
            subtitle = "--- Dataset Folder Structure ---"
            title_x = w//2 - len(subtitle)//2
            stdscr.attron(curses.color_pair(1))
            stdscr.addstr(start_y + len(MENU_ASCII_ART) + 1, max(0, title_x), subtitle)
            stdscr.attroff(curses.color_pair(1))
            msg_start_y = max(start_y + len(MENU_ASCII_ART) + 3, h // 2 - len(docs) // 2 + 3)
            for i, line in enumerate(docs):
                if msg_start_y + i < h:
                    x = max(0, w // 2 - len(line) // 2)
                    if "📁" in line or "📄" in line:
                        stdscr.attron(curses.color_pair(3))
                        stdscr.addstr(msg_start_y + i, x, line)
                        stdscr.attroff(curses.color_pair(3))
                    elif ":" in line:
                        stdscr.attron(curses.color_pair(2))
                        stdscr.addstr(msg_start_y + i, x, line)
                        stdscr.attroff(curses.color_pair(2))
                    else:
                        stdscr.addstr(msg_start_y + i, x, line)
            stdscr.refresh()
            curses.flushinp()
            while stdscr.getch() not in [10, 13]:
                pass
            curses.reset_prog_mode()
            stdscr.clear()
            stdscr.refresh()

        elif selected_main == "Status Tags Explanation (Docs)":
            docs = [
                r" [RAW]                            : Dataset downloaded but not processed. ",
                r" [PROCESSED]                      : Images and trajectories extracted successfully. ",
                r" [READY FOR USE (GDINO)]          : Grounding SAM labels are available.",
                r" [READY FOR USE (SEGFORMER)]      : SegFormer inference labels are available.",
                r" [READY FOR USE (GDINO & SEGFORMER)] : Both completion markers exist; outputs share positive_masks/.",
                r"                                          ",
                r"Press ENTER to return to the menu..."
            ]

            stdscr.clear()
            h, w = stdscr.getmaxyx()

            start_y = MENU_START_Y
            for i, line in enumerate(MENU_ASCII_ART):
                x = w//2 - len(line)//2
                stdscr.attron(curses.color_pair(1) | curses.A_BOLD)
                stdscr.addstr(start_y + i, max(0, x), line)
                stdscr.attroff(curses.color_pair(1) | curses.A_BOLD)

            subtitle = "--- Status Tags Explanation ---"
            title_x = w//2 - len(subtitle)//2
            stdscr.attron(curses.color_pair(1))
            stdscr.addstr(start_y + len(MENU_ASCII_ART) + 1, max(0, title_x), subtitle)
            stdscr.attroff(curses.color_pair(1))

            msg_start_y = start_y + len(MENU_ASCII_ART) + 3
            msg_start_y = max(msg_start_y, h // 2 - len(docs) // 2 + 3)

            for i, line in enumerate(docs):
                if msg_start_y + i < h:
                    x = max(0, w // 2 - len(line) // 2)
                    if "[RAW]" in line:
                        stdscr.attron(curses.color_pair(4) | curses.A_BOLD)
                        stdscr.addstr(msg_start_y + i, x, line)
                        stdscr.attroff(curses.color_pair(4) | curses.A_BOLD)
                    elif "[PROCESSED]" in line:
                        stdscr.attron(curses.color_pair(7) | curses.A_BOLD)
                        stdscr.addstr(msg_start_y + i, x, line)
                        stdscr.attroff(curses.color_pair(7) | curses.A_BOLD)
                    elif "[READY FOR USE" in line:
                        stdscr.attron(curses.color_pair(6) | curses.A_BOLD)
                        stdscr.addstr(msg_start_y + i, x, line)
                        stdscr.attroff(curses.color_pair(6) | curses.A_BOLD)
                    else:
                        stdscr.addstr(msg_start_y + i, x, line)

            stdscr.refresh()

            curses.flushinp()
            while True:
                if stdscr.getch() in [10, 13]: break

            curses.reset_prog_mode()
            stdscr.clear()
            stdscr.refresh()

        elif selected_main == "Segformer Segmentation":
            seg_title = "Segformer Segmentation"
            seg_options = ["Train", "Inference"]

            seg_desc = "Run Segformer **model training** or start the **inference process**."
            selected_seg = run_menu(stdscr, seg_title, seg_options, description=seg_desc)

            if not selected_seg:
                continue

            if selected_seg == "Train":
                processed_dir = processed_root
                available_datasets = []
                if os.path.exists(processed_dir):
                    for d in sorted(os.listdir(processed_dir)):
                        if os.path.isdir(os.path.join(processed_dir, d)) and d.startswith("dataset_"):
                            rgb_dir = os.path.join(processed_dir, d, "rgb")
                            mask_dir = os.path.join(processed_dir, d, "positive_masks")

                            total = 0
                            if os.path.exists(rgb_dir):
                                total = len([f for f in os.listdir(rgb_dir) if f.startswith("rgb_") and f.endswith(('.jpg', '.png', '.jpeg'))])

                            processed = 0
                            if os.path.exists(mask_dir):
                                processed = len([f for f in os.listdir(mask_dir) if f.startswith(pipeline_config["data_format"]["positive_mask_prefix"]) and f.endswith('.tif')])

                            is_sam_ready = os.path.exists(os.path.join(processed_dir, d, "sam_completed.txt"))
                            if is_sam_ready:
                                is_seg_ready = os.path.exists(os.path.join(processed_dir, d, "segformer_completed.txt"))
                                if is_seg_ready:
                                    tag = "  [READY FOR USE (GDINO & SEGFORMER)]"
                                else:
                                    tag = "  [READY FOR USE (GDINO)]"
                                available_datasets.append(f"{d}  [{processed}/{total} images]{tag}")

                if not available_datasets:
                    stdscr.clear()
                    draw_progress_ui(stdscr, "Error", None, "No [READY FOR USE] datasets found! Run Grounding SAM first. Press ENTER.")
                    while True:
                        if stdscr.getch() in [10, 13]: break
                    continue

                if not segformer_config_ui(stdscr, segformer_cfg):
                    continue

                selected_datasets = run_menu(
                    stdscr,
                    "Select Datasets for Segformer Training",
                    available_datasets,
                    description=[
                        "Select which processed datasets you want to use for training.",
                        "**NOTE:** Only datasets marked as [READY FOR USE] are eligible!"
                    ],
                    multi_select=True,
                    deletable=True
                )

                if isinstance(selected_datasets, tuple) and selected_datasets[0] == "__DELETE__":
                    clean_sd = selected_datasets[1].split("  ")[0]
                    target = os.path.join(processed_dir, clean_sd)
                    if confirm_action_ui(stdscr, "Confirm Deletion", f"Are you sure you want to PERMANENTLY delete\n'{clean_sd}'? (y/n)"):
                        if os.path.exists(target): shutil.rmtree(target)
                    next_menu = "Segformer Segmentation"
                    continue

                if not selected_datasets:
                    continue
                if not check_processed_selection(stdscr, selected_datasets, require_masks=True, config=pipeline_config):
                    continue

                script_path = "data_generation.segmentation.train_segformer"

                if find_spec(script_path) is not None:
                    clean_datasets = [sd.split("  ")[0] for sd in selected_datasets]
                    cmd = [
                        sys.executable, "-u", "-m", script_path,
                        "--dataset"
                    ] + clean_datasets + [
                        "--epochs", str(segformer_cfg['total_epochs']),
                        "--batch_size", str(segformer_cfg['batch_size']),
                        "--lr", str(segformer_cfg['learning_rate']),
                        "--early_stopping", "true" if segformer_cfg['early_stopping'] else "false",
                        "--patience", str(segformer_cfg['patience'])
                    ]

                    info_lines = [
                        "============================================================",
                        "         🚀 Training Segformer for Semantic Segmentation",
                        "------------------------------------------------------------"
                    ]
                    if len(clean_datasets) > 0:
                        info_lines.append(f"• Datasets : {clean_datasets[0]}")
                        for ds in clean_datasets[1:]:
                            info_lines.append(f"             {ds}")

                    info_lines.extend([
                        f"• Configs  : {segformer_cfg['total_epochs']} Epochs | {segformer_cfg['batch_size']} Batch | LR: {segformer_cfg['learning_rate']:.1e}",
                        "============================================================"
                    ])

                    success = run_process_with_ui(stdscr, cmd, "Training Segformer", info_lines, config=pipeline_config)

                    if success:
                        draw_progress_ui(stdscr, "Training Complete", None, "Segformer training finished successfully. Press ENTER.")
                        curses.flushinp()
                        while True:
                            if stdscr.getch() in [10, 13]: break
                else:
                    stdscr.clear()
                    draw_progress_ui(stdscr, "Error", None, f"Script not found: {script_path}. Press ENTER.")
                    stdscr.refresh()
                    while True:
                        if stdscr.getch() in [10, 13]: break

            elif selected_seg == "Inference":
                processed_dir = processed_root
                available_datasets = []
                if os.path.exists(processed_dir):
                    for d in sorted(os.listdir(processed_dir)):
                        if os.path.isdir(os.path.join(processed_dir, d)) and d.startswith("dataset_"):
                            rgb_dir = os.path.join(processed_dir, d, "rgb")

                            total = 0
                            if os.path.exists(rgb_dir):
                                total = len([f for f in os.listdir(rgb_dir) if f.startswith("rgb_") and f.endswith(('.jpg', '.png', '.jpeg'))])

                            if total > 0:
                                is_sam_ready = os.path.exists(os.path.join(processed_dir, d, "sam_completed.txt"))
                                is_seg_ready = os.path.exists(os.path.join(processed_dir, d, "segformer_completed.txt"))

                                if is_sam_ready and is_seg_ready:
                                    tag = "  [READY FOR USE (GDINO & SEGFORMER)]"
                                elif is_sam_ready:
                                    tag = "  [READY FOR USE (GDINO)]"
                                elif is_seg_ready:
                                    tag = "  [READY FOR USE (SEGFORMER)]"
                                else:
                                    tag = "  [PROCESSED]"

                                available_datasets.append(f"{d}  [{total} images]{tag}")

                if not available_datasets:
                    stdscr.clear()
                    draw_progress_ui(stdscr, "Error", None, "No datasets with RGB images found in dataset_processed/. Press ENTER.")
                    while True:
                        if stdscr.getch() in [10, 13]: break
                    continue

                selected_datasets = run_menu(
                    stdscr,
                    "Select Datasets for Segformer Inference",
                    available_datasets,
                    description="Select which processed datasets you want to run inference on.",
                    multi_select=True,
                    deletable=True
                )

                if isinstance(selected_datasets, tuple) and selected_datasets[0] == "__DELETE__":
                    clean_sd = selected_datasets[1].split("  ")[0]
                    target = os.path.join(processed_dir, clean_sd)
                    if confirm_action_ui(stdscr, "Confirm Deletion", f"Are you sure you want to PERMANENTLY delete\n'{clean_sd}'? (y/n)"):
                        if os.path.exists(target): shutil.rmtree(target)
                    next_menu = "Segformer Segmentation"
                    continue

                if not selected_datasets:
                    continue
                if not check_processed_selection(stdscr, selected_datasets, config=pipeline_config):
                    continue

                script_path = "data_generation.segmentation.inference_segformer"

                if find_spec(script_path) is not None:
                    if not os.path.exists("checkpoints/segformer_finetuned"):
                        draw_progress_ui(stdscr, "Missing Model", None, "Segformer model not found! Please run 'Train Segformer' first. Press ENTER.")
                        curses.flushinp()
                        while True:
                            if stdscr.getch() in [10, 13]: break
                        continue

                    clean_datasets = [sd.split("  ")[0] for sd in selected_datasets]
                    existing_nonempty_masks = []
                    from PIL import Image
                    mask_prefix = pipeline_config["data_format"]["positive_mask_prefix"]
                    for dataset_name in clean_datasets:
                        mask_dir = os.path.join(processed_dir, dataset_name, "positive_masks")
                        if os.path.isdir(mask_dir):
                            for mask_name in os.listdir(mask_dir):
                                if mask_name.startswith(mask_prefix) and mask_name.endswith(".tif"):
                                    mask_path = os.path.join(mask_dir, mask_name)
                                    with Image.open(mask_path) as mask:
                                        if mask.getbbox():
                                            existing_nonempty_masks.append(mask_path)
                    overwrite_masks = False
                    if existing_nonempty_masks:
                        overwrite_masks = confirm_action_ui(
                            stdscr,
                            "Confirm Mask Replacement",
                            f"Inference will replace existing nonempty labels ({len(existing_nonempty_masks)} masks). "
                            "This is destructive. Press Y to overwrite; N/ESC to cancel.",
                        )
                        if not overwrite_masks:
                            continue
                    cmd = [
                        sys.executable, "-u", "-m", script_path,
                        "--dataset"
                    ] + clean_datasets
                    if overwrite_masks:
                        cmd.append("--overwrite")

                    info_lines = [
                        "============================================================",
                        "         🚀 Running Segformer Inference",
                        "------------------------------------------------------------"
                    ]
                    if len(clean_datasets) > 0:
                        info_lines.append(f"• Datasets : {clean_datasets[0]}")
                        for ds in clean_datasets[1:]:
                            info_lines.append(f"             {ds}")

                    info_lines.extend([
                        "============================================================"
                    ])

                    success = run_process_with_ui(stdscr, cmd, "Segformer Inference", info_lines, config=pipeline_config)

                    if success:
                        draw_progress_ui(stdscr, "Inference Complete", None, "Segformer inference finished successfully. Press ENTER.")
                        curses.flushinp()
                        while True:
                            if stdscr.getch() in [10, 13]: break
                else:
                    stdscr.clear()
                    draw_progress_ui(stdscr, "Error", None, f"Script not found: {script_path}. Press ENTER.")
                    stdscr.refresh()
                    while True:
                        if stdscr.getch() in [10, 13]: break
        elif selected_main == "Export CAT Dataset":
            scores_dir = default_scores_dir(pipeline_config)
            available_datasets = []
            for d in list_sequences(processed_root):
                # labeled = data.csv has positive mask references
                csv_path = os.path.join(processed_root, d, "data.csv")
                labeled = False
                if os.path.exists(csv_path):
                    with open(csv_path, newline="") as f:
                        rows = list(csv.DictReader(f))
                    labeled = any(
                        row.get("positive_mask_name")
                        and os.path.isfile(os.path.join(processed_root, d, row["positive_mask_name"]))
                        for row in rows
                    )
                scored = find_scores(scores_dir, d) is not None
                label_tag = "[LABELED]" if labeled else "[NOT LABELED]"
                score_tag = "[SCORES]" if scored else "[NO SCORES]"
                available_datasets.append(f"{d}  {label_tag} {score_tag}")

            if not available_datasets:
                draw_progress_ui(stdscr, "Error", None, f"No datasets found in {processed_root}/. Press ENTER.")
                curses.flushinp()
                while True:
                    if stdscr.getch() in [10, 13]: break
                continue

            selected_datasets = run_menu(
                stdscr,
                "Select Datasets for CAT",
                available_datasets,
                description=f"Merges labeled sequences into one CAT training CSV. Score YAMLs are read from {scores_dir}/.",
                multi_select=True,
            )
            if not selected_datasets:
                continue

            clean_datasets = [sd.split("  ")[0] for sd in selected_datasets]
            cmd = [sys.executable, "-u", "-m", "data_generation.export.cat_csv", "--dataset"] + clean_datasets

            # sequences without their own scores need a fallback yaml
            unscored = [d for d in clean_datasets if find_scores(scores_dir, d) is None]
            if unscored:
                candidates = sorted(glob.glob(os.path.join(scores_dir, "*.yaml")))
                if not candidates:
                    show_input_warning(
                        stdscr,
                        f"No score YAML in {scores_dir}/ for:\n" + "\n".join(unscored) +
                        "\n\nAdd <sequence>.yaml files there; see the Handoff to CAT section of data_generation/README.md.",
                        title="Missing robot scores",
                    )
                    continue
                fallback = run_menu(
                    stdscr,
                    "Select Fallback Scores",
                    [os.path.basename(c) for c in candidates],
                    description=f"{len(unscored)} selected sequence(s) have no score YAML. Choose the file to use for them.",
                )
                if not fallback:
                    continue
                cmd += ["--fallback_scores", os.path.join(scores_dir, fallback)]

            info_lines = [
                "============================================================",
                "         🚀 Exporting CAT Dataset",
                "------------------------------------------------------------",
                f"• Datasets : {len(clean_datasets)} sequence(s)",
                "============================================================",
            ]
            if run_process_with_ui(stdscr, cmd, "CAT Dataset Export", info_lines, config=pipeline_config):
                root = os.path.abspath(processed_root)
                csv_path = os.path.join(root, pipeline_config["data_format"]["cat_csv_name"])
                show_input_warning(
                    stdscr,
                    f"Wrote {csv_path}.\n\n{next_step(root, csv_path, pipeline_config)}",
                    title="CAT dataset exported",
                )
        elif selected_main == "Data Generation Workflow (Mind Map)":
            mind_map = [
                r"    [ Download Raw Dataset ]  (TartanDrive, TerraSentia, Grandtour)",
                r"           │                                             ",
                r"           ▼                                             ",
                r"  [ Data Processing ]  (Extraction, Sync, Formatting)    ",
                r"           │                                             ",
                r"           ├──> Extracted RGB Images                     ",
                r"           ├──> Depth Maps                           ",
                r"           └──> Aligned Odometry / Velocity / Position ",
                r"           │                                             ",
                r"           ▼                                             ",
                r"   [ Grounding SAM ]  (Zero-shot Auto-Labeling)          ",
                r"           │                                             ",
                r"           ├──> Text Prompts (e.g. 'dirt', 'grass')      ",
                r"           ├──> Generates Semantic Masks                 ",
                r"           └──> Human-in-the-loop (Correction/QA)        ",
                r"           │                                             ",
                r"           ▼                                             ",
                r" [ Segformer Pipeline ]  (Supervised Mask Training)      ",
                r"           │                                             ",
                r"           ├──> Fine-tunes on Grounding SAM masks        ",
                r"           └──> Inference generates predicted masks       ",
                r"           │                                             ",
                r"           ▼                                             ",
                r"  [ Export CAT Dataset ]  (Merged CSV + Robot Scores)    ",
                r"           │                                             ",
                r"           ▼                                             ",
                r"  [ CAT Embedding Extraction & Training ]                ",
                "",
                "Press ENTER to return to the menu..."
            ]

            stdscr.clear()
            h, w = stdscr.getmaxyx()

            start_y = MENU_START_Y
            for i, line in enumerate(MENU_ASCII_ART):
                x = w//2 - len(line)//2
                stdscr.attron(curses.color_pair(1) | curses.A_BOLD)
                stdscr.addstr(start_y + i, max(0, x), line)
                stdscr.attroff(curses.color_pair(1) | curses.A_BOLD)

            subtitle = "--- Data Generation Workflow ---"
            title_x = w//2 - len(subtitle)//2
            stdscr.attron(curses.color_pair(1))
            stdscr.addstr(start_y + len(MENU_ASCII_ART) + 1, max(0, title_x), subtitle)
            stdscr.attroff(curses.color_pair(1))

            msg_start_y = start_y + len(MENU_ASCII_ART) + 3

            msg_start_y = max(msg_start_y, h // 2 - len(mind_map) // 2 + 3)

            for i, line in enumerate(mind_map):
                if msg_start_y + i < h:
                    x = max(0, w // 2 - len(line) // 2)
                    if "[" in line and "]" in line:
                        stdscr.attron(curses.color_pair(4) | curses.A_BOLD)
                        stdscr.addstr(msg_start_y + i, x, line)
                        stdscr.attroff(curses.color_pair(4) | curses.A_BOLD)
                    elif "▼" in line or "│" in line or "─" in line:
                        stdscr.attron(curses.color_pair(1))
                        stdscr.addstr(msg_start_y + i, x, line)
                        stdscr.attroff(curses.color_pair(1))
                    else:
                        stdscr.addstr(msg_start_y + i, x, line)

            stdscr.refresh()
            curses.flushinp()
            while True:
                ch = stdscr.getch()
                if ch in [10, 13, 27]:
                    break

            stdscr.clear()
            stdscr.refresh()

def run() -> None:
    """
    Launch the data generation TUI.
    """
    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        pass
    finally:
        print("\033[?1003l", end="")
        sys.stdout.flush()

if __name__ == "__main__":
    run()
