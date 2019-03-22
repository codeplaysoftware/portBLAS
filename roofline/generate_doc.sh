cd ..
rm -rf roofline/doc
pdoc3 --html --html-dir roofline roofline
mv roofline/roofline roofline/doc
cd roofline
