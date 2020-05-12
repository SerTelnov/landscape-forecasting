#!/bin/bash

if [[ "$1" == "clean" ]]; then
    rm -f {bachelor-thesis}.{bib,aux,log,bbl,bcf,blg,run.xml,toc,tct,pdf,out}
else
    for i in bachelor-thesis; do
        xelatex $i
        biber   $i
        xelatex $i
        xelatex $i
    done

    rm -f {bachelor-thesis,}.{bib,aux,log,bbl,bcf,blg,run.xml,toc,tct,out}
fi
