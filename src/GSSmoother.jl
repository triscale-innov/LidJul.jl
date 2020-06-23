export GSSmoother,smooth
using SparseArrays
using LinearAlgebra
using BenchmarkTools


struct GSSmoother
    invh2::Float64
    bc::NTuple{2,NTuple{2,BoundaryCondition}}
    finest::Bool
    function GSSmoother(invh2,nrows,ncols,bc,finest)
        new(invh2,bc,finest)
    end
end

@inline function sij(denom,i,j,sl,rl,ih2,nx,ny)
    @inbounds denom*(rl[i,j]+ih2*(sl[i,j-1]+sl[i-1,j]+sl[i+1,j]+sl[i,j+1]))
end



@inline function smooth_line(nrm1,j,i1,sl,rl,ih2,denom)
    @fastmath @inbounds @simd for i=i1:2:nrm1
        sl[i,j]=denom*(rl[i,j]+ih2*(sl[i,j-1]+sl[i-1,j]+sl[i+1,j]+sl[i,j+1]))
    end
end


@inline function smooth_seq(nrows,ncols,sl,rl,ih2,)
    denom=1/(4ih2)
    nrm1=nrows-1

    @inbounds for j=2:2:ncols-1
        smooth_line(nrm1,j,2,sl,rl,ih2,denom)
        smooth_line(nrm1,j+1,3,sl,rl,ih2,denom)
    end
    @inbounds for j=2:2:ncols-1
        smooth_line(nrm1,j,3,sl,rl,ih2,denom)
        smooth_line(nrm1,j+1,2,sl,rl,ih2,denom)
    end

end


function smooth(sl,rl,a::GSSmoother)
    ih2=a.invh2
    (nrows,ncols)=size(sl)
    @assert(size(rl)==size(sl))
    denom=1/(4ih2)
    smooth_seq(nrows,ncols,sl,rl,ih2)
    return
end
