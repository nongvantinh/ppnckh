# Biên dịch bằng XeLaTeX (hỗ trợ Unicode tiếng Việt)
$pdf_mode = 1;
$latex = 'xelatex -synctex=1 -interaction=nonstopmode %O %S';
$bibtex = 'biber %O %B';
$clean_ext = 'synctex.gz synctex.bak run.xml bbl bcf';
