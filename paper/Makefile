# Build arXiv-style PDF (requires pdflatex + bibtex)
MAIN=sqrt_space_kv
.PHONY: all clean
all:
	pdflatex -interaction=nonstopmode $(MAIN)
	bibtex $(MAIN) || true
	pdflatex -interaction=nonstopmode $(MAIN)
	pdflatex -interaction=nonstopmode $(MAIN)
clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).toc
