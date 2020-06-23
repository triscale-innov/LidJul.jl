
import Plots
using PrettyTables
using JLD2
using IterativeSolvers
# export residuals,niterations
# using Formatting
# using Printf
Plots.pyplot()

residuals(hist) = hist.data[:resnorm]
residuals(x::Nothing) = nothing
niterations(x::Nothing) = nothing
residuals(x::Array) = x
niterations(x) = length(residuals(x))==0 ? nothing : length(residuals(x))


function plotbar(table,fn)
    plt=Plots.bar(Vector{String}(table[:,1]),Vector{Float64}(table[:,4]),title="Poisson Solver Duration (s)",
        xrotation=90,label="",size=(600,400),tickfont=Plots.font(12),guidefont=Plots.font(12))
    Plots.display(plt)
    Plots.savefig(plt,fn)
end

function plotallconvergence(itable,fn)
    plt=nothing

    for i=1:size(itable,1)
        resi=residuals(itable[i,7])
        ls=:solid
        if itable[i,5]>1.e-6
            # @show "nc",itable[i,1],itable[i,5]
            ls=:dot
        else
            # @show "co",itable[i,1],itable[i,5]
            ls=:solid
        end
        if i==1
            plt=Plots.plot(resi,yaxis=:log,label=itable[i,1],
                legend=:outertopright,  linewidth = 3,linestyle=ls,
                ylabel="Residual norm",xlabel="Iteration #")
        else
            plt=Plots.plot!(plt,resi,yaxis=:log,label=itable[i,1],
                legend=:outertopright,  linewidth = 3,linestyle=ls,
                size=(600,400),tickfont=Plots.font(12),guidefont=Plots.font(12),legendfont=Plots.font(11))
        end
    end
    Plots.display(plt)
    Plots.savefig(plt,fn)
end

function plotfast(itable,fn)
    markers = filter((m->begin
            m in Plots.supported_markers()
        end), Plots._shape_keys)
    markers = reshape(markers, 1, length(markers))
    # @show markers
    plt2=nothing
    for i=1:size(itable,1)
        resi=residuals(itable[i,7])
        ls=:solid
        if itable[i,6]<100
            # @show itable[i,1],itable[i,6]
            if plt2==nothing
                plt2=Plots.plot(resi,yaxis=:log,label=itable[i,1],
                    linewidth = 3,linestyle=ls,
                    ylabel="Residual norm",xlabel="Iteration #", reuse=false,
                    marker=markers[i],markersize=6)
            else
                plt2=Plots.plot!(plt2,resi,yaxis=:log,label=itable[i,1],
                    linewidth = 3,linestyle=ls,
                    size=(600,400),tickfont=Plots.font(12),guidefont=Plots.font(12),legendfont=Plots.font(11),
                    marker=markers[i],markersize=6)
            end
        end
    end
    display(plt2)
    Plots.savefig(plt2,fn)
end

function plotconvergence(itable,fn)
    markers = filter((m->begin
            m in Plots.supported_markers()
        end), Plots._shape_keys)
    markers = reshape(markers, 1, length(markers))
    # @show markers
    plt2=nothing
    for i=1:size(itable,1)
        resi=residuals(itable[i,7])
        # @show label=itable[i,1]
        ls=:solid
        # @show itable[i,1],itable[i,6],resi
        if plt2==nothing
            # @show itable[i,1],itable[i,6]
            plt2=Plots.plot(resi,yaxis=:log,label=itable[i,1],
                linewidth = 3,linestyle=ls,
                ylabel="Residual norm",xlabel="Iteration #",
                size=(600,400),tickfont=Plots.font(12),guidefont=Plots.font(12),legendfont=Plots.font(11),
                marker=markers[i],markersize=6)
        else
            # @show itable[i,1],itable[i,6]
            plt2=Plots.plot!(plt2,resi,yaxis=:log,label=itable[i,1],
                linewidth = 3,linestyle=ls,
                ylabel="Residual norm",xlabel="Iteration #",
                size=(600,400),tickfont=Plots.font(12),guidefont=Plots.font(12),legendfont=Plots.font(11),
                marker=markers[i],markersize=6)
        end
    end
    display(plt2)
    Plots.savefig(plt2,fn)
end


function measureplot(msm,sname)
    header=keys(msm[1])[1:end]
    nrows=length(msm)
    ncols=length(header)
    table=Matrix{Any}(undef,nrows,ncols)
    iterations=Vector{Union{Nothing,Vector{Float64}}}(undef,nrows)
    for i=1:nrows
        for j=1:ncols
            table[i,j]=msm[i][header[j]]
        end
        table[i]=replace(table[i],"Poisson"=>"")
        table[i]=replace(table[i],"Preconditioner"=>"")
        table[i]=replace(table[i],"SparseCG"=>"PCG")
    end
    table[7,1]="PCGAMG{SmoothA}"
    table[8,1]="PCGAMG{RugeStuben}"

    #Remove some measurements :
    removed=["GMG_CGSmoother"]
    table=table[filter(i->!(table[i,1] in removed),1:size(table,1)),:]

    #add Total Time and niter
    header=[header[1:3]...,"Total Time",header[4],"niters",header[5]]
    table=hcat(table[:,1:3],table[:,2].+table[:,3],table[:,4],niterations.(residuals(table[:,5])),table[:,5])
    #Sort
    sp=sortperm(table[:, 4])
    table=table[sp,:]
    # iterations=iterations[sp,:]
    nr,nc=size(table)
    println("all methods")
    pretty_table(table[:,1:6],collect(header[1:6]);formatters=ft_printf("%5.3E",2:5),alignment=:l)
    println("converged methods")
    ctable=table[filter(i->(!isnan(table[i,5]) && table[i,5]<1.e-6),1:size(table,1)),:]
    pretty_table(ctable[:,1:6],collect(header[1:6]);formatters=ft_printf("%5.3E",2:5),alignment=:l)

    println("non converged methods")
    nctable=table[filter(i->(isnan(table[i,5]) ||  table[i,5]>1.e-6),1:size(table,1)),:]
    pretty_table(nctable[:,1:6],collect(header[1:6]);formatters=ft_printf("%5.3E",2:5),alignment=:l)
    println("converged iterative methods")
    ictable=ctable[filter(i->ctable[i,6]!==nothing,1:size(ctable,1)),:]
    pretty_table(ictable[:,1:6],collect(header[1:6]);formatters=ft_printf("%5.3E",2:5),alignment=:l)
    println("efficient converged iterative methods")
    iectable=ictable[filter(i->ictable[i,4]<10.0,1:size(ictable,1)),:]
    pretty_table(ictable[:,1:6],collect(header[1:6]);formatters=ft_printf("%5.3E",2:5),alignment=:l)
    println("iterative methods")
    itable=table[filter(i->table[i,6]!==nothing,1:size(table,1)),:]
    pretty_table(itable[:,1:6],collect(header[1:6]);formatters=ft_printf("%5.3E",2:5),alignment=:l)
    println("fast methods")
    ftable=ictable[filter(i->(ictable[i,6]<25 && ictable[i,4]<10.0),1:size(ictable,1)),:]
    pretty_table(ftable[:,1:6],collect(header[1:6]);formatters=ft_printf("%5.3E",2:5),alignment=:l)


    sp=sortperm(itable[:, 6]+0.0001*itable[:,5])
    itable=itable[reverse(sp),:]
    pretty_table(itable[:,1:6],collect(header[1:6]);formatters=ft_printf("%5.3E",2:5),alignment=:l)

    cftable=table[filter(i->(!isnan(table[i,5]) && table[i,5]<1.e-6 && table[i,4]<10.0),1:size(table,1)),:]
    
    plotbar(cftable,"time_converge_"*sname*"_all.svg")
    plotbar(iectable,"time_converge_"*sname*"_mixed.svg")
    plotbar(ftable,"time_converge_"*sname*"_fast.svg")

    plotallconvergence(itable,"allconvergence_"*sname*".svg")
    plotfast(itable,"fastconvergence_"*sname*".svg")
    plotconvergence(ftable,"convergence_"*sname*".svg")

    table,header
end

function test(sname)
    fnjld2="msm"*sname*".jld2"
    @load fnjld2 msm
    t,h=measureplot(msm,sname)
    nothing
end
