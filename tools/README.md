# Tools
This directory contains a selection of useful tools.
 - `cli.j.`: command line interface for using `ACEhamiltonians` models.
 - `compile_model.jl`: used to pre-compile models.

# Command Line Interface
> Usage `julia cli.jl <model-file> <geometry-file> <output-file>`

Arguments:
 - `<model-file>`: JSON or binary file storing the model to be used. Model files must
   end in an appropriate extension; i.e. *.json* for JSON formatted files and *.bin*
   for binary files.
 - `<geometry-file>`: `ace` parsable geometry file storing the system upon which the
   previously specified model is to be evaluated.
 - `<output-file>`: file into which the CSV data representing the resulting matrix
   should be placed. Hamiltonian matrices are given in units of Hartrees.

The command line interface tool "`cli.jl`" allows for `ACEhamiltonians` models to be used
from the command line. This tool should be used in lieu of the `ACEhamiltonians` Python
API. The Python interface is currently too unwieldy, unstable and slow to be of general
use. 

For stability purposes the `ACEhamiltonians` models are stored in JSON files. However,
this results in a non-trivial io overhead. Thus, it is recommended to use binary models
compile via the model compilation tool rather than use JSON files directly.


# Model Compiler
> Usage `julia compile_model.jl <input-file> <output-file>`

Arguments:
 - `<input-file>`: JSON file storing the model to be compiled.
 - `<output-file>`: file in which the compiled model should be placed.

This tool allows for JSON formatted model files to be pre-compiled to help reduce the
io overhead associated with loading the model.

For the purposes of stability and reproducibility, all ACE data-structures, including
`ACEhamiltonians` models, are stored in JSON formatted files. Unfortunately, the large
and recursive nature of the `ACEhamiltonians` model means that a non-trivial amount of
time is required to parse and construct each model. Thus, it is recommended to use the
inbuilt tools to parse the model into a binary file. This cuts the load time from around
a minute to a few fractions of a second. However, care must be taken as these binary files
are very sensitive to system changes and thus must be build anew for each compute system.

