FILENAME=proposal

all: $(FILENAME).pdf

$(FILENAME).pdf: $(FILENAME).tex
	pdflatex $(FILENAME).tex
	bibtex $(FILENAME)
	pdflatex $(FILENAME).tex
	pdflatex $(FILENAME).tex

.PHONY: clean
clean:
	rm -f $(FILENAME).{ps,pdf,log,aux,out,dvi,bbl,blg,toc} $(OUTPUT_PDF)
