using LinearAlgebra
using SparseArrays

export Laplacian2D,sparse_corr

struct Laplacian2D
    lpx::Laplacian1D
    lpy::Laplacian1D
    nx::Int
    ny::Int
    Lx::Float64
    Ly::Float64
    bc::NTuple{2,NTuple{2,BoundaryCondition}}
    allneumann::Bool
    function Laplacian2D(nx,ny,Lx,Ly,bcleft,bcright,bcbottom,bctop)
        bc=((bcleft,bcright),(bcbottom,bctop))
        allneumann = bc[1][1]==bc[1][1]==bc[1][1]==bc[1][1]==neumann
        allneumann && println("allneumann !!")
        new(Laplacian1D(nx,Lx,bcleft,bcright),Laplacian1D(ny,Ly,bcbottom,bctop),nx,ny,Lx,Ly,bc,allneumann)
    end


end
function push_element!(Is,Js,Vs,I,J,v)
    push!(Is,I)
    push!(Js,J)
    push!(Vs,v)
end


function SparseArrays.sparse(a::Laplacian2D)
    Is=Int[]
    Js=Int[]
    Vs=Float64[]
    lpx,lpy=a.lpx,a.lpy
    nx,ny=lpx.n_,lpy.n_

    Is=Int[]
    Js=Int[]
    Vs=Float64[]

    for i=1:nx
        for j=1:ny
            I=i+nx*(j-1)
            push_element!(Is,Js,Vs,I,I,lpx[i,i]+lpy[j,j])# sp[I,I]=Lx[i,i]

            i>1  && push_element!(Is,Js,Vs,I,I-1,lpx[i,i-1])#(sp[I,I-1]=Lx[i,i-1])
            i<nx && push_element!(Is,Js,Vs,I,I+1,lpx[i,i+1])#(sp[I,I+1]=Lx[i,i+1])
            j>1 &&  push_element!(Is,Js,Vs,I,I-ny,lpy[j,j-1])#(sp[I,I-ny]+=Ly[j,j-1])
            j<ny && push_element!(Is,Js,Vs,I,I+ny,lpy[j,j+1])#(sp[I,I+ny]+=Ly[j,j+1])

        end
    end
    sp=sparse(Is,Js,Vs)
    sp
end

function sparse_corr(a::Laplacian2D)
    sp=SparseArrays.sparse(a::Laplacian2D)
    if a.allneumann
        sp[1,1]=(3/2)sp[1][1]
    end
    sp
end


