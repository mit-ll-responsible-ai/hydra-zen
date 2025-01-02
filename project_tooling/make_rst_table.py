from itertools import zip_longest


def make_table(*cols: str) -> str:
    """Given the string representations of consecutive columns in a table, generates
    the rst-string for that table."""
    col_lines = [[line.rstrip() for line in col.splitlines()] for col in cols]
    longest = [max(len(line) for line in col) for col in col_lines]

    lines = []

    divider = "+" + "+".join("-" * (longest_i + 2) for longest_i in longest) + "+"

    lines.append(divider)
    for ll in zip_longest(*col_lines):
        lines.append(
            "| "
            + "| ".join(
                (li if li else "") + " " * (longest[n] - len(li if li else "") + 1)
                for n, li in enumerate(ll)
            )
            + "|"
        )

    lines.append(divider)
    return "\n".join(lines)
