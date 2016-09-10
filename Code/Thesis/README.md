## Overview

C++ implementation for the thesis. The folder is organized as follows

* **doc** contains the code documentation, generated with [Doxygen][www.stack.nl/~dimitri/doxygen/], and some [UML][https://en.wikipedia.org/wiki/Unified_Modeling_Language] diagrams that illustrate the architecture used for the project and that have been created using [Umbrello][https://umbrello.kde.org/] 
* **examples** contains the executables for the project. 
* **include** contains the header files for the project.
* **src** contains the cpp files for the project. 


## Compilation Instructions

To compile the project, first create a Makefile using cmake by running

--------------------------------
cmake -DCMAKE_BUILD_TYPE=DEBUG 
cmake -DCMAKE_BUILD_TYPE=RELEASE
--------------------------------

for the Debug or the Release version respectively. Then, compile via make

----
make
----

This produces a static library `libthesis.a` and some executables in the
[examples][examples] folder. 
