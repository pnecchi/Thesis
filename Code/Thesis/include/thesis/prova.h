#ifndef INCLUDE_PROVA_H_
#define INCLUDE_PROVA_H_

#include <iostream>
#include <armadillo>

namespace spazio {

class Prova {
public:
    Prova(int cod)
    {
        codice = arma::zeros<arma::vec>(cod);;
    }

    arma::vec getCodice();
private:
    arma::vec codice;
};
}

#endif
