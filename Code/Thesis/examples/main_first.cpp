#include <iostream>
#include "thesis/prova.h"

using namespace std;
using namespace spazio;

int main(int argc, char *argv[])
{
    Prova el(5);
    cout << el.getCodice().t() << endl;
    return 0;
}
