#!/usr/local/bin/python3

import os, sys, re
from colorama import Fore

if __name__ == "__main__":
    folderSrc = sys.argv[2]
    folderPdf = sys.argv[3]
    files = [{'name': str(file), 'num': int(re.sub('\D+','',str(file)))} for file in os.listdir(f'./{folderSrc}/')]
    files = sorted(files, key=lambda i: i['num'])

    def convert_pdf(md_file):
        fileName = str(md_file).replace('.md','')
        os.system(f'pandoc ./{folderSrc}/{md_file} --pdf-engine=pdflatex -o ./{folderPdf}/{fileName}.pdf -V geometry:margin=1in')

    if len(sys.argv) > 2:
        pageRanges = sys.argv[1].split(',')
    else:
        pageRanges = [f'1-{len(files)}']
        
    for token in pageRanges:
        pages = []
        if '-' in token:
            pages = range(int(token.split('-')[0]) - 1,int(token.split('-')[1]))
        else:
            pages = range(int(token) - 1,int(token))

        for i in pages:
            if i >= len(files):
                print(Fore.RED + 'Enter valid page range')
                exit(1)

            fileName = files[i]['name']

            print(f'[{pages.index(i) + 1} of {len(pages)}] Converting {fileName}')
            convert_pdf(fileName)
            print('Done')