using Test
using ACEhamiltonians.Parameters
using ACEhamiltonians.Parameters: _parameter_expander

"""
Test the ``ParaDef`` structure and its associated methods.
"""

@testset "ParaDef" begin
#######################
# _parameter_expander #
#######################
# The ``_parameter_expander`` method(s) are tested first as they are the methods upon which
# the ``ParaDef`` structure's constructors are built. 
@testset "_parameter_expander" begin

    # Helper function to test the parameter expander function.
    function expander_test_runner(name, site, basis_def, parameter, result)
        @testset "$name" begin
            output = _parameter_expander(parameter, basis_def, site)
            @test all([v == result[k] for (k, v) in output])
        end
    end

    # Note that this basis definition is not actually intended to be meaningful but rather
    # to provide a range of different test conditions.
    basis = Dict(1=>[0], 6=>[0, 0, 1], 31=>[0, 1, 0, 2])

    # Ensure on-site parameters are expanded correctly
    @testset "On-site" begin
        # Expansion of on-site global deceleration to full
        expander_test_runner("Global", "on", basis,
            1.2,
            Dict(1=>[1.2;;], 6=>fill(1.2,3,3), 31=>fill(1.2,4,4)))

        # Expansion of on-site species resolved deceleration to full
        expander_test_runner("Species", "on", basis,
            Dict(1=>1.1, 6=>1.2, 31=>1.3),
            Dict(1=>[1.1;;], 6=>fill(1.2,3,3), 31=>fill(1.3,4,4)))
        
        full = Dict(1=>[1;;], 6=>[1 1 2; 1 1 2; 2 2 3],
                    31=>[1 2 1 3; 2 4 2 5; 1 2 1 3; 3 5 3 6])
        
        # Expansion of on-site azimuthally resolved deceleration to full
        expander_test_runner("Azimuthal", "on", basis,
            Dict(1=>[1;;], 6=>[1 2; 2 3], 31=>[1 2 3; 2 4 5; 3 5 6]),
            full)

        # Full species deceleration test, essentially a null-operation   
        expander_test_runner("Full", "on", basis, full, full)

        # When only one of each shell type is present (s, p, etc.) then the code cannot
        # differentiate between azimuthal and full deceleration. This should not have
        # any effect on the result, however a test must be performed.
        expander_test_runner("Special ambiguous case", "on", Dict(1=>[0], 6=>[0, 1]),
            Dict(1=>[1;;], 6=>[1 2; 2 3]),
            Dict(1=>[1;;], 6=>[1 2; 2 3]))
    end

    # Ensure off-site parameters are expanded correctly
    @testset "Off-site" begin
        # Inputs and associated outputs for parameter expansion 
        global_in = 1.2

        species_in   = Dict((1,1)=>1.1, (1,6)=>1.2, (1,31)=>1.3,
                            (6,6)=>1.4, (6,31)=>1.5, (31,31)=>1.6)
        
        azimuthal_in = Dict((1,1)=>[1;;], (1,6)=>[1 2], (1,31)=>[1 2 3], (6,6)=>[1 2; 2 3],
                            (6,31)=>[1 2 3; 4 5 6], (31,31)=>[1 2 3; 2 4 5; 3 5 6])
        
        full         = Dict((1,1)=>[1;;], (1,6)=>[1 1 2], (1,31)=>[1 2 1 3],
                            (6,6)=>[1 1 2; 1 1 2; 2 2 3], (6,31)=>[1 2 1 3; 1 2 1 3; 4 5 4 6],
                            (31,31)=>[1 2 1 3; 2 4 2 5; 1 2 1 3; 3 5 3 6])


        global_out   = Dict((1,1)=>[1.2;;], (1,6)=>fill(1.2,1,3), (1,31)=>fill(1.2,1,4),
                            (6,6)=>fill(1.2,3,3), (6,31)=>fill(1.2,3,4), (31,31)=>fill(1.2,4,4))
        
        species_out  = Dict((1,1)=>[1.1;;], (1,6)=>fill(1.2,1,3), (1,31)=>fill(1.3,1,4),
                            (6,6)=>fill(1.4,3,3), (6,31)=>fill(1.5,3,4), (31,31)=>fill(1.6,4,4))
            
        # Note that full_in == full_out and azimuthal_out == full 
 
        # Expansion of off-site global deceleration to full
        expander_test_runner("Global", "off", basis, global_in, global_out)

        # Expansion of off-site species resolved deceleration to full
        expander_test_runner("Species", "off", basis, species_in, species_out)

        # Expansion of off-site azimuthally resolved deceleration to full
        expander_test_runner("Azimuthal", "off", basis, azimuthal_in, full)

        # Full species deceleration test, essentially a null-operation   
        expander_test_runner("Full", "off", basis, full, full)

        # When only one of each shell type is present (s, p, etc.) then the code cannot
        # differentiate between azimuthal and full deceleration. This should not have
        # any effect on the result, however a test must be performed.
        expander_test_runner("Special ambiguous case", "off", Dict(1=>[0], 6=>[0, 1]),
                Dict((1,1)=>[1;;], (1,6)=>[1 2], (6,6)=>[1 2; 2 3]),
                Dict((1,1)=>[1;;], (1,6)=>[1 2], (6,6)=>[1 2; 2 3]))

    end

    # Ensure that error handling works 
    @testset "Error Handling" begin

        # An exception should be raised when attempting to expand global parameters
        # with site-autodetect enabled "auto". This test is site agnostic and thus
        # does not need to be repeated in the off-site test section.  
        @testset "Global/site-autodetection incompatibility" begin
            @test_throws AssertionError _parameter_expander(1, basis, "auto")
        end

        @testset "Malformed parameter matrix specification" begin
        # Ensure that malformed species deceleration are caught and an error is thrown.
            for (site, key) in zip(("on", "off"),(6, (6,6)))
                @testset "$(uppercasefirst(site))-site" begin
                    @test_throws AssertionError _parameter_expander(
                        Dict(key=>ones(3,3)), Dict(6=>[0, 1]), site)
                end
            end
        end

        @testset "Site validity key check" begin
            @test_throws AssertionError _parameter_expander(1, basis, "invalid_key")
        end

    end

    # All other tests than don't fit anywhere else
    @testset "General" begin
        # Check that vectors are converted into matrices
        @testset "Vector to matrix conversion" begin
            @test all([valtype(_parameter_expander(Dict(1=>[i]), Dict(1=>[1])))<:Matrix
                       for i in (1, [1])])
        end
    end

end



########################
# ParaDef Constructors #
########################
# Ensure ParaDef's constructor methods function as expected.
@testset "Constructors" begin
    # Note that the dictionary constructor is not tested here as it is tested
    # in the io section.

    # Check the operational performance of the base constructors
    @testset "Base Constructors" begin
        @testset "On-site" begin
            # As on-site interactions are symmetric, i.e. sp≡ps, there is only one model
            # used to represent them. Meaning that asymmetric parameter matrices are not
            # valid. Thus an error should be raised if one is encountered. 
            @testset "Block asymmetric parameter matrices" begin
                p_i, p_f = Dict(6=>[1 2; 3 4]), Dict(6=>[1. 2.; 3. 4.])
                @test_throws AssertionError ParaDef(p_i, p_i, p_f, p_f)
            end
        end

        @testset "Off-site" begin
            
            # If an asymmetric parameter matrix is supplied for a homo-atomic interactions
            # then a warning should be issued to the user informing them that this is not
            # a good idea.
            @testset "Warn against asymmetric homo-atomic parameter matrices" begin
                p_i, p_f = Dict((6,6)=>[1 2; 3 4]), Dict((6,6)=>[1. 2.; 3. 4.])
                msg = "Non-symmetric off-site parameter matrices are ill-advised for homoatomic interactions: (6, 6)"
                @test_logs (:warn, msg) ParaDef(p_i, p_i, p_f, p_f, p_f, p_f, p_f)
            end

            # If interaction parameters are accidentally specified twice then an exception
            # should be raised; i.e. if (1, 6) is present then (6, 1) should not be as they
            # are symmetrically equivalent.
            @testset "Block symmetrically equivalent interactions" begin
                p_i, p_f = Dict((6,1)=>[1;;], (1,6)=>[1;;]), Dict((6,1)=>[1.;;], (1,6)=>[1.;;])
                @test_throws AssertionError ParaDef(p_i, p_i, p_f, p_f, p_f, p_f, p_f)
            end

            # Keys should be sorted so that the lowest atomic number comes first. This
            # makes the lower level code easier. A check is performed to ensure that
            # the constructor performs this action.
            @testset "Sort interaction keys" begin
                p_i, p_f = Dict((6,1)=>[1;;]), Dict((6,1)=>[1.;;])
                @test haskey(ParaDef(p_i, p_i, p_f, p_f, p_f, p_f, p_f).ν, (1, 6))
            end
        end
    end

    # Ensure the parameter expansion constructor operates as intended
    @testset "Expansion Constructor" begin
        # Note that only a few test are actually required here as this constructor is
        # built atop of _parameter_expander and base constructor methods.

        # Check that both on- and off-sites ParaDefs can be constructed.
        @testset "On-site operability" begin
            @test keytype(ParaDef(Dict(1=>[0]), 1, 1, site="on").ν)<:Real
        end

        @testset "Off-site operability" begin
            @test keytype(ParaDef(Dict(1=>[0]), 1, 1, site="off").ν)<:Tuple
        end

        # Warnings should be issued if λ terms are given for on-site interactions.
        @testset "Warn when using off-site arguments for on-site parameters" begin
            msg = "Arguments bond_cut, λₙ & λₗ are off-site exclusive and will be ignored."
            @test_logs (:warn, msg) ParaDef(Dict(1=>[0]), 1, 1, 1.0, 1.0, 1.0, 1.0, site="on")
        end

        # Auto site key with global only
        @testset "Global/site-autodetection incompatibility" begin
            @test_throws AssertionError ParaDef(Dict(1=>[0]), 1, 1)
        end

        # Invalid value given for site key
        @testset "Site validity key check" begin
            @test_throws AssertionError ParaDef(Dict(1=>[0]), 1, 1, site="invalid_key")
        end
    end
end

###################################
# Miscellaneous support functions #
###################################
# All other miscellaneous functions are tested here.
@testset "Support functions" begin
    # Ensure the `ison` method operates correctly
    @testset "ison" begin
        @test ison(ParaDef(Dict(1=>[0]), 1, 1, site="on")) && ~ison(ParaDef(Dict(1=>[0]), 1, 1, site="off"))
    end
    
    # Verify that ParaDef instance can be converted to, and instantiated from, dictionaries.
    @testset "ParaDef⇄Dict interconversion" begin
        @testset "On-sites" begin
            pd_in = ParaDef(Dict(1=>[0], 6=>[0, 0, 1]), 1, 2, site="on")
            pd_out = read_dict(Val{:ParaDef}(), write_dict(pd_in))
            @test pd_in == pd_out
        end
        @testset "Off-sites" begin
            pd_in = ParaDef(Dict(1=>[0], 6=>[0, 0, 1]), 1, 2, site="off")
            pd_out = read_dict(Val{:ParaDef}(), write_dict(pd_in))
            @test pd_in == pd_out
        end
    end
    
end

end;
