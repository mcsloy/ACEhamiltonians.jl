module ACEhamiltonians

using JuLIP, JSON, HDF5, Reexport

include("common.jl")
@reexport using ACEhamiltonians.Common

include("io.jl")
@reexport using ACEhamiltonians.DataIO

include("parameters.jl")
@reexport using ACEhamiltonians.Parameters

include("basis.jl")
@reexport using ACEhamiltonians.Bases

# This will be renamed to Data later on once conflicts
# with the existing code are removed. 
include("data.jl")
@reexport using ACEhamiltonians.DataManipulation

include("struc_setting.jl")
@reexport using ACEhamiltonians.Structure

include("dataproc.jl")
@reexport using ACEhamiltonians.DataProcess

include("fit.jl")
@reexport using ACEhamiltonians.Fitting

include("fitting.jl")
@reexport using ACEhamiltonians.Fitting2

include("predict.jl")
@reexport using ACEhamiltonians.Predict

include("dictionary.jl")
@reexport using ACEhamiltonians.Dictionary

include("interface.jl")
@reexport using ACEhamiltonians.Interface

include("tools.jl")

end
