all: main.pdf

# apt-get install texlive-full

define Show =
	-killall evince
	evince main.pdf 2>/dev/null &
endef

main.pdf: main.tex [1-6]-*.tex *.bib initial-pages.pdf
	rm -f main.aux
	latex --shell-escape main.tex
	bibtex main
	latex --shell-escape main.tex
	latex --shell-escape main.tex
	-grep --color=auto Warn main.log
	rm -f *.aux
	$(Show)

.PHONY: archive clean show

archive:
	git archive --prefix=main-h-adapt/ --output=../main-h-adapt.zip HEAD

clean:
	git clean -n -X | grep -v swp | sed 's/Would remove //' | xargs rm -f

show:
	$(Show)

