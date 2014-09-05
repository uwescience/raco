
%.png: %.dot
	dot -Tpng $< -o $@

%.ps: %.dot
	dot -Tps $< -o $@

%.pdf: %.ps
	ps2pdf $<
	pdfcrop $@
	mv `basename $@ .pdf`-crop.pdf $@
