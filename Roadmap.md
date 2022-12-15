Roadmap
====================


Current Status:

* BLAS LVL1 : 90% complete, only rotmg missing.
* BLAS LVL2 : Only two sample routines
* BLAS LVL3 : Only three variants of the Matrix Multiplication
* Examples :
  * Interface tests for all implemented API entries
  * CG example with multiple variants of the algorithm using kernel fusion.

Short Term QX 2017:

* Have a common class for the implementation of the different view classes
* Obtain and analyse performance bottlenecks
* Work on the Blas 2 interface

Medium Term:

* Complete the Blas 2 interface
* Work on the Blas 3 interface
* Move evaluation methods out of the expression tree into the execution tree


Long Term:

* Complete the Blas 3 interface
* Add continuous integration testing 
