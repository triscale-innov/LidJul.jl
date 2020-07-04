using BenchmarkTools
using LoopVectorization
using LinearAlgebra
using Random

#Original functions (smooth_line and smooth_seq using @simd)
@inline function smooth_line(nrm1,j,i1,sl,rl,ih2,denom)
    @fastmath @inbounds @simd for i=i1:2:nrm1
            # @show i,j,denom,sl[i,j],rl[i,j],sl[i,j-1],sl[i,j+1],sl[i-1,j],sl[i+1,j]
            sl[i,j]=denom*(rl[i,j]+ih2*(sl[i,j-1]+sl[i-1,j]+sl[i+1,j]+sl[i,j+1]))
    end
end


@inline function smooth_seq(nrows,ncols,sl,rl,ih2)
    denom=1/(4ih2)
    nrm1=nrows-1
    #even (i+j)%2==0 from odds
    @inbounds for j=2:2:ncols-1
        smooth_line(nrm1,j,2,sl,rl,ih2,denom)
        smooth_line(nrm1,j+1,3,sl,rl,ih2,denom)
    end
    # #odds (i+j)%2==1 from evens
    @inbounds for j=2:2:ncols-1
        smooth_line(nrm1,j,3,sl,rl,ih2,denom)
        smooth_line(nrm1,j+1,2,sl,rl,ih2,denom)
    end

end

#RBArray is a Type acting like a 2D array with a data layout suitable to
# Red Black Gauss-Seidel algorithms
#  contiguous red (iseven(i+j)) values are stored in [i2,1,j] 
#  contiguous odd (isodd(i+j)) values are stores in [i2,2,j] 
struct RBArray
    data::Array{Float64,3}
end
#Ctor : should use an inner Ctor to ensure that nx is even
#Note that >>> means unsigned >>. Hence x >>> 1 returns div(x,2) is x is a positive integer
RBArray(nx,ny) = RBArray(Array{Float64,3}(undef,nx>>>1,2,ny))
#Helper functions to acess the the RB arrays with 2 indexes rb[i,j] 
@inline Base.getindex(rb::RBArray,i,j) = rb.data[(i+1)>>>1,1+isodd(i+j),j]
@inline Base.setindex!(rb::RBArray,value,i,j) = rb.data[(i+1)>>>1,1+isodd(i+j),j]=value
Base.size(rb::RBArray) = (size(rb.data)[1]*2,size(rb.data)[3])
Base.fill!(rb::RBArray,val) = fill!(rb.data,val)
Base.eltype(b::RBArray) = Float64


#RBArray from Array
function RBArray(a::Array{Float64,2}) 
    nx,ny=size(a)
    @assert iseven(nx)
    nxs2=nx>>>1
    rb=RBArray(nx,ny)
    for i in 1:nx,j in 1:ny
        rb[i,j]=a[i,j]
    end
    for j in 1:2:ny 
        #j odd
        for i2 in 0:nxs2-1
            rb.data[i2+1,1,j]=a[2i2+1,j] #i=2i2+1 is odd  => i+j is even (rb.data[:,1,:])
            rb.data[i2+1,2,j]=a[2i2+2,j] #i=2i2+2 is even => i+j is odd (rb.data[:,2,:])
        end
        #j+1 even  
        for i2 in 0:nxs2-1 #i=2i2 is even
            rb.data[i2+1,2,j+1]=a[2i2+1,j+1] #i=2i2+1 is odd  => i+j is even (rb.data[:,2,:])
            rb.data[i2+1,1,j+1]=a[2i2+2,j+1] #i=2i2+2 is even => i+j is odd (rb.data[:,1,:])
        end  
    end
    rb
end

#Array from RBArray
function Array{Float64,2}(rb::RBArray)
    nxs2,two,ny=size(rb.data) 
    nx=2nxs2
    a=Array{Float64,2}(undef,nx,ny)
    #The following blocks works but is probably slow
    # for i in 1:nx,j in 1:ny
    #     a[i,j]=rb[i,j]
    # end
    for j in 1:2:ny 
        #j odd
        for i2 in 0:nxs2-1 #i=2i2 is even
            a[2i2+1,j]=rb.data[i2+1,1,j]
            a[2i2+2,j]=rb.data[i2+1,2,j]
        end
        #j+1 even  
        for i2 in 0:nxs2-1 #i=2i2 is even
            a[2i2+1,j+1]=rb.data[i2+1,2,j+1]
            a[2i2+2,j+1]=rb.data[i2+1,1,j+1]
        end  
    end
    a
end



@inline function smooth_line_rb(nrm1s2,j,di,rbout,rbin,sl,rl,ih2,denom)
    @avx for i2 in 1:nrm1s2
        sl[i2+di,rbout,j]=denom*(rl[i2+di,rbout,j]+ih2*(sl[i2+di,rbin,j-1]+sl[i2+di,rbin,j+1]+sl[i2,rbin,j]+sl[i2+1,rbin,j]))
    end
end

# #Version with broadcast (slow but suitable to test CuArrays)
@inline function smooth_line_rb_brcst(nrm1s2,j,di,rbout,rbin,sl,rl,ih2,denom)
    ib=1+di
    ie=nrm1s2+di
    @inbounds @fastmath @views @. sl[ib:ie,rbout,j]=denom*(rl[ib:ie,rbout,j]+ih2*(sl[ib:ie,rbin,j-1]+sl[ib:ie,rbin,j+1]+sl[1:nrm1s2,rbin,j]+sl[2:nrm1s2+1,rbin,j]))
end





#Specialized smooth_seq method for RedBlack arrays.
#Note that non specialized smooth_seq method should produce the same result(but slowly)
@inline function smooth_seq(nrows,ncols,slrb::RBArray,rlrb::RBArray,ih2)
    denom=1/(4ih2)
    nrm1,ncm1=nrows-1,ncols-1
    nrm1s2=nrm1 >>> 1
    sl,rl=slrb.data,rlrb.data

    #evens from odds
    @inbounds for j in 2:2:ncm1 
        smooth_line_rb(nrm1s2,j  ,0,1,2,sl,rl,ih2,denom)
        smooth_line_rb(nrm1s2,j+1,1,1,2,sl,rl,ih2,denom)
    end
     #odds from evens
     @inbounds for j in 2:2:ncm1 
        smooth_line_rb(nrm1s2,j  ,1,2,1,sl,rl,ih2,denom)
        smooth_line_rb(nrm1s2,j+1,0,2,1,sl,rl,ih2,denom)
    end
end




#function that call the benchmarks
function benchmark_smooth_gs_rb(n)
    @assert iseven(n)
    # initialize two 2D arrays
    sl=rand(n+2,n+2)
    rl=rand(n+2,n+2)
    # the following blocks check if smooth_seq and smooth_seq_avx
    # return the same result 
    sl_ref=deepcopy(sl)
    slrb=RBArray(sl)
    rlrb=RBArray(rl)
    (nrows,ncols)=size(sl)
    ih2=rand()
    smooth_seq(nrows,ncols,sl,rl,ih2)
    smooth_seq(nrows,ncols,slrb,rlrb,ih2)
    res=norm(sl-Array{Float64,2}(slrb))
 
    # #The actual timings
    # bs=@btime smooth_seq($nrows,$ncols,$slrb,$rlrb,$ih2)
    tsimd=@belapsed smooth_seq($nrows,$ncols,$sl,$rl,$ih2)
    trb=@belapsed smooth_seq($nrows,$ncols,$slrb,$rlrb,$ih2)
    @show n,res,tsimd,trb,tsimd/trb
    nothing
end

function interpolate_correct!(uf,uc)
    (nrows,ncols)=size(uc)
    # @show size(uc)
    ncm1=ncols-1
    nrm1=nrows-1

    @inbounds @fastmath @simd for j=2:ncm1
        fj=2j-1
        for i=2:nrm1
            fi=2i-1
             v=uc[i,j]
             uf[fi  ,fj  ]+=v
             uf[fi-1,fj  ]+=v
             uf[fi  ,fj-1]+=v
             uf[fi-1,fj-1]+=v
        end
    end

end


@inline function scatter!(df,v,i,fj)
    df[i-1,1,fj-1]+=v 
    df[i,1,fj]+=v
    df[i-1,2,fj]+=v
    df[i,2,fj-1]+=v
end

function interpolate_correct!(uf::RBArray,uc::RBArray)
    (nrows,ncols)=size(uc)
    # @show size(uc)
    ncm1=ncols-1
    nrm1=nrows-1

    df,dc=uf.data,uc.data

    ncm1s2=ncm1>>>1
    nrm1s2=nrm1>>>1

    @inbounds @fastmath for j2=1:ncm1s2
         for i2=1:nrm1s2
            # @inbounds @simd for j2=1:ncm1>>>1
            #     @fastmath @simd for i2=1:nrm1>>>1
            v1,v2,v3,v4=dc[i2,1,2j2],dc[i2+1,2,2j2],dc[i2,2,2j2+1],dc[i2+1,1,2j2+1]

            df[2i2-1,1,4j2-2]+=v1
            df[2i2  ,1,4j2-2]+=v2

            df[2i2  ,1,4j2-1]+=v1 
            df[2i2+1,1,4j2-1]+=v2
            
            df[2i2-1,1,4j2  ]+=v3
            df[2i2  ,1,4j2  ]+=v4

            df[2i2  ,1,4j2+1]+=v3
            df[2i2+1,1,4j2+1]+=v4



            df[2i2-1,2,4j2-1]+=v1
            df[2i2,  2,4j2-2]+=v1

            df[2i2  ,2,4j2-1]+=v2
            df[2i2+1,2,4j2-2]+=v2

            df[2i2-1,2,4j2+1]+=v3
            df[2i2,  2,4j2  ]+=v3

            df[2i2  ,2,4j2+1]+=v4
            df[2i2+1,2,4j2  ]+=v4

       
        end     
    end

    # @inbounds @simd for j2=1:ncm1>>>1
    #     @fastmath @simd for i2=1:nrm1>>>1
    #         v1,v2,v3,v4=dc[i2,1,2j2],dc[i2+1,2,2j2],dc[i2,2,2j2+1],dc[i2+1,1,2j2+1]

    #         df[2i2-1,1,4j2-2]+=v1
    #         df[2i2  ,1,4j2-1]+=v1
    #         df[2i2-1,2,4j2-1]+=v1
    #         df[2i2,  2,4j2-2]+=v1

    #         df[2i2  ,1,4j2-2]+=v2
    #         df[2i2+1,1,4j2-1]+=v2
    #         df[2i2  ,2,4j2-1]+=v2
    #         df[2i2+1,2,4j2-2]+=v2

        
    #         df[2i2-1,1,4j2  ]+=v3
    #         df[2i2  ,1,4j2+1]+=v3
    #         df[2i2-1,2,4j2+1]+=v3
    #         df[2i2,  2,4j2  ]+=v3

             
    #         df[2i2  ,1,4j2  ]+=v4
    #         df[2i2+1,1,4j2+1]+=v4
    #         df[2i2  ,2,4j2+1]+=v4
    #         df[2i2+1,2,4j2  ]+=v4

    #         # scatter!(df,v1,2i2,4j2-1)
    #         # scatter!(df,v2,2i2+1,4j2-1)
    #         # scatter!(df,v3,2i2,4j2+1)
    #         # scatter!(df,v4,2i2+1,4j2+1)
    #     end     
    # end




    # @inbounds @simd for j2=1:ncm1>>>1
    #     @fastmath @simd for i2=1:nrm1>>>1
    #         v1,v2,v3,v4=dc[i2,1,2j2],dc[i2+1,2,2j2],dc[i2,2,2j2+1],dc[i2+1,1,2j2+1]

    #         scatter!(df,v1,2i2,4j2-1)
    #         scatter!(df,v2,2i2+1,4j2-1)
    #         scatter!(df,v3,2i2,4j2+1)
    #         scatter!(df,v4,2i2+1,4j2+1)
    #     end     
    # end

end

#function that call the benchmarks
function benchmark_interpolate_rb(nf)
    @assert iseven(nf)
    # initialize two 2D arrays (fine and coarse)
    nc=nf>>>1
    uf=rand(nf+2,nf+2)
    uc=rand(nc+2,nc+2)
    # the following blocks check if smooth_seq and smooth_seq_avx
    # return the same result 
    ufrb=RBArray(uf)
    ucrb=RBArray(uc)
    interpolate_correct!(uf,uc)
    interpolate_correct!(ufrb,ucrb)
    res=norm(uf-Array{Float64,2}(ufrb))
    @show res
    # #The actual timings
    tsimd=@belapsed interpolate_correct!($uf,$uc)
    trb=@belapsed interpolate_correct!($ufrb,$ucrb)
    @show "interp",nf,res,tsimd,trb,tsimd/trb
    nothing
end

@inline function fres(x,y,i,j,ih2,nx,ny)
    # @show x[i,j]+ih2*(y[i+1,j]+y[i,j+1]+y[i,j-1]+y[i-1,j]-eltype(x)(4)*y[i,j])
    # @show x[i,j]+ih2*(y[i+1,j]+y[i,j+1]+y[i,j-1]+y[i-1,j]-eltype(x)(4)*y[i,j])
    # x[i,j]+ih2*(y[i+1,j]+y[i,j+1])
    @inbounds x[i,j]+ih2*(y[i+1,j]+y[i,j+1]+y[i,j-1]+y[i-1,j]-eltype(x)(4)*y[i,j])
end

#rlc = rl coarse
function restrict_residual!(rlc,slf,rlf,ih2)
   
    (nrows,ncols) = size(rlc)
    (nx,ny) = (nrows-1,ncols-1)
    fill!(rlc,0.)

    @inline finer(i,j)=fres(rlf,slf,i,j,ih2,nx,ny)
    oneOverFour=eltype(rlc)(1/4)

    @simd for j=2:ncols-1
        fj=2j-2
        for i=2:nrows-1
            fi=2i-2
            # rlc[i,j]=oneOverFour*(finer(fi,fj)+finer(fi+1,fj))
            # rlc[i,j]=oneOverFour*(finer(fi+1,fj+1))
             @inbounds rlc[i,j]=oneOverFour*(finer(fi,fj)+finer(fi+1,fj)+finer(fi,fj+1)+finer(fi+1,fj+1))
        end
    end
end

@inline fres_ee(x,y,i,fj,ih2,nx,ny) = @inbounds x[i,1,fj]+ih2*(y[i  ,2,fj]+y[i,2,fj+1]+y[i,2,fj-1]+y[i+1,2,fj]-4.0*y[i,1,fj])
@inline fres_oe(x,y,i,fj,ih2,nx,ny) = @inbounds x[i,2,fj]+ih2*(y[i  ,1,fj]+y[i,1,fj+1]+y[i,1,fj-1]+y[i-1,1,fj]-4.0*y[i,2,fj])
@inline fres_eo(x,y,i,fj,ih2,nx,ny) = @inbounds x[i,2,fj]+ih2*(y[i+1,1,fj]+y[i,1,fj+1]+y[i,1,fj-1]+y[i  ,1,fj]-4.0*y[i,2,fj])
@inline fres_oo(x,y,i,fj,ih2,nx,ny) = @inbounds x[i,1,fj]+ih2*(y[i  ,2,fj]+y[i,2,fj+1]+y[i,2,fj-1]+y[i-1,2,fj]-4.0*y[i,1,fj])

#rlc = rl coarse
function restrict_residual!(rlc::RBArray,slf::RBArray,rlf::RBArray,ih2)
   
    (nrows,ncols) = size(rlc)
    (nx,ny) = (nrows-1,ncols-1)
    fill!(rlc,0.)

    @inline finer(i,j)=fres(rlf,slf,i,j,ih2,nx,ny)
    oneOverFour=eltype(rlc)(1/4)
    four=eltype(rlc)(4)
    x,y=rlf.data,slf.data


    @simd for j=2:ncols-1
        fj=2j-2
        for i=2:nrows-1
            fi=2i-2
            @inbounds rlc[i,j]=oneOverFour*(
                fres_ee(x,y,i-1,fj,ih2,nx,ny)+
                fres_oe(x,y,i,fj,ih2,nx,ny)+
                fres_eo(x,y,i-1,fj+1,ih2,nx,ny)+
                fres_oo(x,y,i,fj+1,ih2,nx,ny)
                # fres_e(x,y,i,fj+1,ih2,nx,ny)
                # rlf[fi,fj]+ih2*(slf[fi+1,fj]+slf[fi,fj+1]+slf[fi,fj-1]+slf[fi-1,fj]-four*slf[fi,fj])+
                # rlf[fi+1,fj]+ih2*(slf[fi+2,fj]+slf[fi+1,fj+1]+slf[fi+1,fj-1]+slf[fi  ,fj]-four*slf[fi+1,fj])+
                # rlf[fi,fj+1]+ih2*(slf[fi+1,fj+1]+slf[fi,fj+2]+slf[fi,fj  ]+slf[fi-1,fj+1]-four*slf[fi,fj+1])+
                # rlf[fi+1,fj+1]+ih2*(slf[fi+2,fj+1]+slf[fi+1,fj+2]+slf[fi+1,fj  ]+slf[fi  ,fj+1]-four*slf[fi+1,fj+1])
            )
        end
    end
end

#function that call the benchmarks
function benchmark_restrict_rb(nf)
    @assert iseven(nf)
    # initialize two 2D arrays (fine and coarse)
    nc=nf>>>1
    Random.seed!(1234);

    slf=rand(nf+2,nf+2)
    rlf=rand(nf+2,nf+2)
    rlc=rand(nc+2,nc+2)
    ih2=rand()
    # the following blocks check if smooth_seq and smooth_seq_avx
    # return the same result 
    slfrb=RBArray(slf)
    rlfrb=RBArray(rlf)
    rlcrb=RBArray(rlc)
    restrict_residual!(rlc,slf,rlf,ih2)
    restrict_residual!(rlcrb,slfrb,rlfrb,ih2)
    res=norm(rlc-Array{Float64,2}(rlcrb))
    @show res
    # #The actual timings
    tsimd=@belapsed restrict_residual!($rlc,$slf,$rlf,$ih2)
    trb=@belapsed restrict_residual!($rlcrb,$slfrb,$rlfrb,$ih2)
    @show "restrict",nf,res,tsimd,trb,tsimd/trb
    nothing
end


# for n in [32,1024]
#     benchmark_smooth_gs_rb(n)
# end

# for n in [32,1024]
#     benchmark_interpolate_rb(n)
# end

for n in [128]
    benchmark_restrict_rb(n)
end





