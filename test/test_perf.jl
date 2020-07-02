using BenchmarkTools
using LoopVectorization
using LinearAlgebra

#Original functions (smooth_line and smooth_seq using @simd)
@inline function smooth_line(nrm1,j,i1,sl,rl,ih2,denom)
    @fastmath @inbounds @simd for i=i1:2:nrm1
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
    #odds (i+j)%2==1 from evens
    @inbounds for j=2:2:ncols-1
        smooth_line(nrm1,j,3,sl,rl,ih2,denom)
        smooth_line(nrm1,j+1,2,sl,rl,ih2,denom)
    end

end

#@avx'ed functions (smooth_line_avx and smooth_seq_avx using @simd)

@inline function smooth_line_avx(nrm1,j,i1,sl,rl,ih2,denom)
    #The trick to avoid strided loop and use @avx
    ni2=div(nrm1,2)-1

    @inbounds @simd for i2=0:ni2
    # @avx for i2=0:ni2
        sl[i1+2i2,j]=denom*(rl[i1+2i2,j]+ih2*(sl[i1+2i2,j-1]+sl[i1+2i2-1,j]+sl[i1+2i2+1,j]+sl[i1+2i2,j+1]))
    end
end

@inline function smooth_seq_avx(nrows,ncols,sl,rl,ih2)
    denom=1/(4ih2)
    nrm1=nrows-1

    @inbounds for j=2:2:ncols-1
        smooth_line_avx(nrm1,j,2,sl,rl,ih2,denom)
        smooth_line_avx(nrm1,j+1,3,sl,rl,ih2,denom)
    end
    @inbounds for j=2:2:ncols-1
        smooth_line_avx(nrm1,j,3,sl,rl,ih2,denom)
        smooth_line_avx(nrm1,j+1,2,sl,rl,ih2,denom)
    end

end

function smooth_line_dim3_even(nrm1,j,sl,rl,ih2,denom)
    @fastmath @inbounds @simd for i= 1:nrm1 >>> 1
        # @avx for i= 1:nrm1 >>> 1
            sl[i,2,j]=denom*(rl[i,2,j]+ih2*(sl[i,2,j-1]+sl[i,1,j]+sl[i+1,1,j]+sl[i,2,j+1]))
    end
end

function smooth_line_dim3_odd(nrm1,j,sl,rl,ih2,denom)
    @fastmath @inbounds @simd for i = 2:nrm1 >>> 1
        # @avx for i = 2:nrm1 >>> 1
            sl[i,1,j]=denom*(rl[i,1,j]+ih2*(sl[i,1,j-1]+sl[i-1,2,j]+sl[i,2,j]+sl[i,1,j+1]))
    end
end

@inline function smooth_seq_new(nrows,ncols,sl,rl,ih2)
    denom=1/(4ih2)
    nrm1=nrows-1

    @inbounds for j=2:2:ncols-1
        smooth_line_dim3_even(nrm1,j,sl,rl,ih2,denom)
        smooth_line_dim3_even(nrm1,j+1,sl,rl,ih2,denom)

    end
    @inbounds for j=2:2:ncols-1
        smooth_line_dim3_odd(nrm1,j,sl,rl,ih2,denom)
        smooth_line_dim3_odd(nrm1,j+1,sl,rl,ih2,denom)

    end

end


#function that call the benchmarks
function go(n)
    # initialize two 2D arrays
    sl=rand(n+2,n+2)
    rl=rand(n+2,n+2)
    # the following blocks check if smooth_seq and smooth_seq_avx
    # return the same result 
    sl_ref=deepcopy(sl)
    (nrows,ncols)=size(sl)
    ih2=rand()
    smooth_seq(nrows,ncols,sl_ref,rl,ih2)
    smooth_seq_avx(nrows,ncols,sl,rl,ih2)
    @show norm(sl-sl_ref)
    #The actual timings
    tsimd=@belapsed smooth_seq($nrows,$ncols,$sl,$rl,$ih2)
    tavx=@belapsed smooth_seq_avx($nrows,$ncols,$sl,$rl,$ih2)
    @show tsimd,tavx

    slnew = permutedims(reshape(sl, (2, size(sl,1) ÷ 2, size(sl,2))), (2,1,3));
    rlnew = permutedims(reshape(rl, (2, size(sl,1) ÷ 2, size(sl,2))), (2,1,3));

    nrm1=nrows-1

    smooth_line(nrm1,2,2,sl,rl,ih2,denom)
    smooth_line(nrm1,3,3,sl,rl,ih2,denom)

    smooth_line_dim3_even(nrm1,2,slnew,rlnew,ih2,denom)
    smooth_line_dim3_odd(nrm1,3,slnew,rlnew,ih2,denom)

    slcomp = permutedims(reshape(sl, (2, size(sl,1) ÷ 2, size(sl,2))), (2,1,3));
    @show slnew ≈ slcomp # true

    t_smooth_line=@belapsed smooth_line($nrm1,2,2,$sl,$rl,$ih2,$denom)
    t_smooth_line_dim3_even=@belapsed smooth_line_dim3_even($nrm1,2,$slnew,$rlnew,$ih2,$denom)
    t_mooth_line_dim3_odd=@belapsed smooth_line_dim3_odd($nrm1,3,$slnew,$rlnew,$ih2,$denom)
    t_smooth_seq_new=@belapsed smooth_seq_new($nrows,$ncols,$slnew,$rlnew,$ih2)

    @show n,tsimd,tavx,t_smooth_seq_new
    @show n,t_smooth_line,t_smooth_line_dim3_even,t_mooth_line_dim3_odd
    tsimd,tavx
end

struct RBArray
    data::Array{Float64,3}
end

RBArray(nx,ny) = RBArray(Array{Float64,3}(undef,nx>>>1,2,ny))


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

#Helper functions to acess the the RB arrays with 2 indexes rb[i,j] 
@inline Base.getindex(rb::RBArray,i,j) = rb.data[(i+1)>>>1,1+isodd(i+j),j]
@inline Base.setindex!(rb::RBArray,value,i,j) = rb.data[(i+1)>>>1,1+isodd(i+j),j]=value


#Specialized smooth_seq method for RedBlack arrays.
#Note that non specialized smooth_seq method should work (but slowly)
@inline function smooth_seq(nrows,ncols,slrb::RBArray,rlrb::RBArray,ih2)
    denom=1/(4ih2)
    nrm1=nrows-1
    sl,rl=slrb.data,rlrb.data

    #evens from odds
    @inbounds for j in 2:2:ncols-1 
        #evens from odds
        @avx for i2 in 1:nrm1 >>> 1
            sl[i2,1,j]=denom*(rl[i2,1,j]+ih2*(sl[i2,2,j-1]+sl[i2,2,j+1]+sl[i2,2,j]+sl[i2+1,2,j]))
        end
        @avx for i2 in 2:nrm1 >>> 1
            sl[i2,1,j+1]=denom*(rl[i2,1,j+1]+ih2*(sl[i2,2,j]+sl[i2,2,j+2]+sl[i2,2,j+1]+sl[i2+1,2,j+1]))
        end
    end
     #odds from evens
    @inbounds for j in 2:2:ncols-1 
        # end
        @avx for i2 in 1:nrm1 >>> 1
            sl[i2,2,j]=denom*(rl[i2,2,j]+ih2*(sl[i2,1,j-1]+sl[i2,1,j+1]+sl[i2,1,j]+sl[i2+1,1,j]))
        end
        @avx for i2 in 2:nrm1 >>> 1
            sl[i2,2,j+1]=denom*(rl[i2,2,j+1]+ih2*(sl[i2,2,j]+sl[i2,1,j+2]+sl[i2,1,j+1]+sl[i2+1,1,j+1]))
        end
    end
end

#function that call the benchmarks
function go_rb(n)
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
    smooth_seq_avx(nrows,ncols,slrb,rlrb,ih2)
    @show norm(sl-Array{Float64,2}(slrb))
    #The actual timings
    tsimd=@belapsed smooth_seq($nrows,$ncols,$sl,$rl,$ih2)
    trb=@belapsed smooth_seq($nrows,$ncols,$slrb,$rlrb,$ih2)
    @show tsimd,trb

end

go_rb(128)
go_rb(512)




