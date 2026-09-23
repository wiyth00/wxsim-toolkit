"""Wrap a draft NWS text product to NWS line width (69 chars) without touching its structure.

usage: python -m wxsim.nws_text draft.txt [-o product.txt] [--width 69]

Write the draft with each paragraph on ONE long line. Rules:
  * lines already <= width are left alone (headers, tables, LAT...LON blocks, sign-offs)
  * "  - bullet" and "  1) item" lines wrap with a hanging indent under the text
  * "* Watch block" lines wrap with a 2-space continuation; indented threat lines with 4 spaces
  * everything else wraps flush-left
"""
import argparse, re, textwrap


def wrap_product(text, width=69):
    out = []
    for line in text.split('\n'):
        if len(line) <= width or line.startswith('LAT...LON'):
            out.append(line); continue
        m = re.match(r'^(\s+(?:-|\d+\))\s)', line)
        if m:
            ind = ' ' * len(m.group(1))
        elif line.startswith('* '):
            ind = '  '
        elif line.startswith('  '):
            ind = '    '
        else:
            ind = ''
        out += textwrap.wrap(line, width, subsequent_indent=ind, break_on_hyphens=False, break_long_words=False)
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('draft'); ap.add_argument('-o', '--output'); ap.add_argument('--width', type=int, default=69)
    a = ap.parse_args()
    txt = wrap_product(open(a.draft).read(), a.width)
    if a.output:
        open(a.output, 'w').write(txt + ('' if txt.endswith('\n') else '\n'))
        print('saved', a.output)
    else:
        print(txt)


if __name__ == '__main__':
    main()
