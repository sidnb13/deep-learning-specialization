#!/usr/local/bin/python3

import os, sys

PREAMBLE = lambda title: f'''---
title: {title}
author: Sidharth Baskaran
date: July 2021
graphics: true
header-includes:
- \graphicspath{{{"./images/"}}}
---
'''

if __name__ == '__main__':
    filePath = sys.argv[1]
    with open(filePath,'w') as f:
        f.write(PREAMBLE(sys.argv[2]))
    os.system(f'code {filePath}')