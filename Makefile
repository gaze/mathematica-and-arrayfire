
bulkmin.dylib: bulkmin.cpp
	c++ ./bulkmin.cpp -dynamiclib -undefined suppress -flat_namespace -L/usr/local/lib/ -I/usr/local/include -I/Applications/Mathematica.app//Contents/SystemFiles/IncludeFiles/C/ -lafcpu -std=c++11 -o bulkmin.dylib
